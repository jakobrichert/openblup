//! General AI-REML for arbitrary variance structures.
//!
//! This engine handles any [`VarStruct`](crate::variance::VarStruct) on the random terms (AR1, Diagonal,
//! Unstructured, FactorAnalytic, Kronecker interactions such as
//! `FA(env) ⊗ A`) and structured residuals (AR1 ⊗ AR1 spatial models, with
//! or without missing plots). The specialised engine in [`super::AiReml`]
//! remains the fast path for the common case of scaled identity /
//! relationship-matrix terms with an IID residual and dispatches here
//! automatically otherwise.
//!
//! # Method
//!
//! Notation: `V = Σ_k Z_k G_k Z_k' + R`, `P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹`, and
//! the MME coefficient matrix `C = W'R⁻¹W + blockdiag(0, G_k⁻¹)` with
//! `W = [X Z]`. With the MME identities `P = R⁻¹ − R⁻¹WC⁻¹W'R⁻¹`,
//! `û_k = G_k Z_k'Py` and `Z_k'PZ_k = G_k⁻¹ − G_k⁻¹C^{kk}G_k⁻¹`, the REML
//! score for a parameter θ of term k, writing `Ḡ = ∂G_k⁻¹/∂θ`, is
//!
//! ```text
//! s = ½ [ −û'Ḡû − tr(Ḡ C^{kk}) + tr(G_k Ḡ) ]
//! ```
//!
//! and for a residual parameter with `D = ∂R/∂θ`
//!
//! ```text
//! s = ½ [ (R⁻¹ê)'D(R⁻¹ê) − tr(R⁻¹D) + tr(C⁻¹ (R⁻¹W)'D(R⁻¹W)) ]
//! ```
//!
//! The average information (Gilmour, Thompson & Cullis 1995) is
//! `AI_ij = ½ w_i'Pw_j` with working variates `w = −Z_k(G_kḠ)û` for term
//! parameters and `w = D R⁻¹ê` for residual parameters. Only `G_k⁻¹`,
//! `∂G_k⁻¹/∂θ` and `G_k ∂G_k⁻¹/∂θ` are ever needed for a random term, so
//! separable structures never form the full covariance.
//!
//! Newton steps are safeguarded exactly like the specialised engine:
//! backtracking when the restricted likelihood would decrease, bounded
//! growth of variance parameters, and sticky boundary handling.

use nalgebra::{DMatrix, DVector};
use sprs::TriMat;

use crate::error::{LmmError, Result};
use crate::matrix::sparse::{spmv, xt_y};
use crate::model::MixedModel;
use crate::types::SparseMat;

use super::mme::{MixedModelEquations, MmeSolution};
use super::result::{FitResult, NamedEffect, RandomEffectBlock, RemlIteration, VarianceEstimate};

/// General AI-REML solver (see the module documentation).
pub struct GeneralReml {
    max_iter: usize,
    tol: f64,
}

/// Layout of the flat parameter vector: one range per random term followed
/// by the residual parameters.
struct ParamLayout {
    term_ranges: Vec<std::ops::Range<usize>>,
    residual_range: std::ops::Range<usize>,
    total: usize,
    bounds: Vec<(f64, f64)>,
    is_variance: Vec<bool>,
}

impl ParamLayout {
    fn from_model(model: &MixedModel) -> Self {
        let mut term_ranges = Vec::with_capacity(model.random_var_structs.len());
        let mut bounds = Vec::new();
        let mut is_variance = Vec::new();
        let mut offset = 0;
        for vs in &model.random_var_structs {
            let n = vs.n_params();
            term_ranges.push(offset..offset + n);
            bounds.extend(vs.bounds());
            is_variance.extend((0..n).map(|i| vs.is_variance_param(i)));
            offset += n;
        }
        let n_res = model.residual_var_struct.n_params();
        let residual_range = offset..offset + n_res;
        bounds.extend(model.residual_var_struct.bounds());
        is_variance.extend((0..n_res).map(|i| model.residual_var_struct.is_variance_param(i)));
        Self {
            term_ranges,
            residual_range,
            total: offset + n_res,
            bounds,
            is_variance,
        }
    }

    fn apply(&self, model: &mut MixedModel, theta: &[f64]) -> Result<()> {
        for (k, vs) in model.random_var_structs.iter_mut().enumerate() {
            vs.set_params(&theta[self.term_ranges[k].clone()])?;
        }
        model
            .residual_var_struct
            .set_params(&theta[self.residual_range.clone()])
    }
}

/// Residual covariance evaluated at the current parameters.
enum Residual {
    /// `R = sigma2 I`.
    Scalar { sigma2: f64 },
    /// General dense `R` (restricted to the observed cells of a grid).
    Dense {
        r_inv: DMatrix<f64>,
        log_det: f64,
        /// `∂R/∂θ_i` restricted to the observed cells.
        d_r: Vec<DMatrix<f64>>,
    },
}

impl Residual {
    fn from_model(model: &MixedModel) -> Result<Self> {
        let vs = &model.residual_var_struct;
        let n = model.n_obs;
        if vs.name() == "Identity" && vs.n_params() == 1 && model.residual_grid.is_none() {
            return Ok(Residual::Scalar {
                sigma2: vs.params()[0],
            });
        }
        let (n_cells, cells): (usize, Vec<usize>) = match &model.residual_grid {
            Some(g) => (g.n_cells, g.cell_index.clone()),
            None => (n, (0..n).collect()),
        };
        let r_full = vs.covariance_matrix(n_cells);
        let r_obs = subset_dense(&r_full, &cells);
        let chol = r_obs.clone().cholesky().ok_or(LmmError::CholeskyFailed(
            "residual covariance is not positive definite".into(),
        ))?;
        let l = chol.l();
        let log_det = 2.0 * (0..n).map(|i| l[(i, i)].ln()).sum::<f64>();
        let r_inv = chol.inverse();
        let d_r = vs
            .derivatives_of_covariance(n_cells)
            .iter()
            .map(|d| subset_dense(d, &cells))
            .collect();
        Ok(Residual::Dense {
            r_inv,
            log_det,
            d_r,
        })
    }

    fn n_params(&self) -> usize {
        match self {
            Residual::Scalar { .. } => 1,
            Residual::Dense { d_r, .. } => d_r.len(),
        }
    }

    /// `R⁻¹ v`
    fn apply_inv(&self, v: &[f64]) -> Vec<f64> {
        match self {
            Residual::Scalar { sigma2 } => v.iter().map(|x| x / sigma2).collect(),
            Residual::Dense { r_inv, .. } => {
                (r_inv * DVector::from_column_slice(v)).as_slice().to_vec()
            }
        }
    }

    /// `R⁻¹ M`
    fn apply_inv_mat(&self, m: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Residual::Scalar { sigma2 } => m / *sigma2,
            Residual::Dense { r_inv, .. } => r_inv * m,
        }
    }

    /// `(∂R/∂θ_i) v`
    fn apply_d(&self, i: usize, v: &[f64]) -> Vec<f64> {
        match self {
            Residual::Scalar { .. } => v.to_vec(),
            Residual::Dense { d_r, .. } => (&d_r[i] * DVector::from_column_slice(v))
                .as_slice()
                .to_vec(),
        }
    }

    /// `tr(R⁻¹ ∂R/∂θ_i)`
    fn trace_rinv_d(&self, i: usize, n: usize) -> f64 {
        match self {
            Residual::Scalar { sigma2 } => n as f64 / sigma2,
            Residual::Dense { r_inv, d_r, .. } => r_inv.component_mul(&d_r[i].transpose()).sum(),
        }
    }

    /// `(R⁻¹W)' (∂R/∂θ_i) (R⁻¹W)`
    fn m_matrix(&self, i: usize, rinv_w: &DMatrix<f64>) -> DMatrix<f64> {
        match self {
            Residual::Scalar { .. } => rinv_w.transpose() * rinv_w,
            Residual::Dense { d_r, .. } => rinv_w.transpose() * (&d_r[i] * rinv_w),
        }
    }

    fn log_det(&self, n: usize) -> f64 {
        match self {
            Residual::Scalar { sigma2 } => n as f64 * sigma2.ln(),
            Residual::Dense { log_det, .. } => *log_det,
        }
    }

    fn to_sparse(&self, n: usize) -> SparseMat {
        match self {
            Residual::Scalar { sigma2 } => {
                crate::matrix::sparse::sparse_diagonal(&vec![1.0 / sigma2; n])
            }
            Residual::Dense { r_inv, .. } => dense_to_sparse(r_inv),
        }
    }
}

/// Everything computed at one point of the parameter space.
struct Evaluation {
    mme: MixedModelEquations,
    sol: MmeSolution,
    residual: Residual,
    logl: f64,
    e_hat: Vec<f64>,
}

impl GeneralReml {
    /// Create a solver with the given iteration limit and relative tolerance.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Fit the model.
    pub fn fit(&self, model: &mut MixedModel) -> Result<FitResult> {
        let n = model.n_obs;
        let layout = ParamLayout::from_model(model);
        let n_params = layout.total;
        if n_params == 0 {
            return Err(LmmError::ModelSpec(
                "The model has no variance parameters to estimate".into(),
            ));
        }

        // ---- starting values ----
        let y = &model.y;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_var: f64 =
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let n_var_params = layout.is_variance.iter().filter(|v| **v).count().max(1);
        let init_var = (y_var / n_var_params as f64).max(0.01);

        let mut theta: Vec<f64> = Vec::with_capacity(n_params);
        for vs in &model.random_var_structs {
            theta.extend(vs.initial_params());
        }
        theta.extend(model.residual_var_struct.initial_params());
        // Variance parameters left at the default value of 1.0 are scaled to
        // the data; anything else is taken as a user-supplied starting value.
        for (i, t) in theta.iter_mut().enumerate() {
            if layout.is_variance[i] && (*t - 1.0).abs() < 1e-12 {
                *t = init_var;
            }
        }

        let total_var: f64 = theta
            .iter()
            .zip(layout.is_variance.iter())
            .filter(|(_, v)| **v)
            .map(|(t, _)| *t)
            .sum::<f64>()
            .max(init_var);
        let floor = (1e-6 * total_var).max(1e-12);
        let lower: Vec<f64> = (0..n_params)
            .map(|i| {
                if layout.is_variance[i] {
                    layout.bounds[i].0.max(floor)
                } else {
                    layout.bounds[i].0
                }
            })
            .collect();
        let upper: Vec<f64> = layout.bounds.iter().map(|b| b.1).collect();
        for (i, t) in theta.iter_mut().enumerate() {
            *t = t.clamp(lower[i], upper[i]);
        }

        let mut history = Vec::new();
        let mut converged = false;
        let mut fixed = vec![false; n_params];
        let mut ev = self.evaluate(model, &layout, &theta)?;

        for iter in 0..self.max_iter {
            // Sticky boundary flags (a parameter that reached a bound stays there).
            for i in 0..n_params {
                let at_lower = theta[i] <= lower[i] * (1.0 + 1e-9) + 1e-300;
                let at_upper = upper[i].is_finite() && theta[i] >= upper[i] - 1e-9;
                if at_lower || at_upper {
                    fixed[i] = true;
                }
            }

            let logl_cur = ev.logl;
            let (score, ai) = self.scores_and_ai(model, &ev, &layout);
            let free: Vec<usize> = (0..n_params).filter(|&i| !fixed[i]).collect();
            if free.is_empty() {
                converged = true;
                history.push(RemlIteration {
                    iteration: iter + 1,
                    log_likelihood: logl_cur,
                    variance_params: theta.clone(),
                    change: 0.0,
                });
                break;
            }
            let m = free.len();
            let ai_free = DMatrix::from_fn(m, m, |a, b| ai[(free[a], free[b])]);
            let score_free = DVector::from_fn(m, |a, _| score[free[a]]);
            let ai_scale = ai_free
                .diagonal()
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, f64::max)
                .max(1e-12);

            // Candidate directions: Newton (AI^-1 s), then scaled gradient.
            let mut directions: Vec<DVector<f64>> = Vec::with_capacity(2);
            if let Some(d) = newton_direction(&ai_free, &score_free) {
                directions.push(d);
            }
            directions.push(&score_free / ai_scale);

            // Apply a direction with step length `frac`, respecting bounds and
            // per-step growth limits.
            let propose = |dir: &DVector<f64>, frac: f64| -> Vec<f64> {
                let mut trial = theta.clone();
                for (a, &i) in free.iter().enumerate() {
                    let old = theta[i];
                    let mut new = old + frac * dir[a];
                    if layout.is_variance[i] {
                        // no more than a factor of 10 growth per step
                        new = new.min(10.0 * old.max(lower[i]));
                    } else if upper[i].is_finite() && lower[i].is_finite() {
                        // correlations: at most half the range per step
                        let half = 0.5 * (upper[i] - lower[i]);
                        new = new.clamp(old - half, old + half);
                    } else {
                        // unbounded (e.g. loadings): trust-region style cap
                        let cap = 2.0 * old.abs().max(1.0);
                        new = new.clamp(old - cap, old + cap);
                    }
                    trial[i] = new.clamp(lower[i], upper[i]);
                }
                trial
            };

            // Line search: the restricted likelihood must not decrease.
            let logl_tol = 1e-10 * logl_cur.abs().max(1.0);
            let mut accepted: Option<(Vec<f64>, Evaluation)> = None;
            'dirs: for dir in &directions {
                let mut frac = 1.0;
                for _ in 0..12 {
                    let trial = propose(dir, frac);
                    if trial.iter().zip(theta.iter()).all(|(t, o)| t == o) {
                        break;
                    }
                    if let Ok(e) = self.evaluate(model, &layout, &trial) {
                        if e.logl >= logl_cur - logl_tol {
                            accepted = Some((trial, e));
                            break 'dirs;
                        }
                    }
                    frac *= 0.5;
                }
            }

            match accepted {
                Some((trial, e)) => {
                    let mut diff2 = 0.0;
                    let mut norm2 = 0.0;
                    for &i in &free {
                        diff2 += (trial[i] - theta[i]).powi(2);
                        norm2 += theta[i].powi(2);
                    }
                    let rel_change = if norm2 > 0.0 {
                        (diff2 / norm2).sqrt()
                    } else {
                        diff2.sqrt()
                    };
                    let logl_gain = e.logl - logl_cur;
                    theta = trial;
                    ev = e;
                    history.push(RemlIteration {
                        iteration: iter + 1,
                        log_likelihood: ev.logl,
                        variance_params: theta.clone(),
                        change: rel_change,
                    });
                    if iter > 0 && (rel_change < self.tol || logl_gain.abs() < logl_tol) {
                        converged = rel_change < self.tol
                            || gradient_is_small(&score_free, &theta, &free, logl_cur);
                        if converged {
                            break;
                        }
                    }
                }
                None => {
                    // No direction improves the likelihood: we are at a
                    // stationary point up to numerical precision (converged)
                    // or the surface is degenerate here (not converged).
                    converged = gradient_is_small(&score_free, &theta, &free, logl_cur);
                    history.push(RemlIteration {
                        iteration: iter + 1,
                        log_likelihood: logl_cur,
                        variance_params: theta.clone(),
                        change: 0.0,
                    });
                    break;
                }
            }
        }

        // ---- final state ----
        let at_boundary: Vec<bool> = (0..n_params)
            .map(|i| {
                fixed[i]
                    || theta[i] <= lower[i] * (1.0 + 1e-9) + 1e-300
                    || (upper[i].is_finite() && theta[i] >= upper[i] - 1e-9)
            })
            .collect();

        let mut variance_se = vec![0.0; n_params];
        let mut ai_matrix = None;
        let variance_at_floor = (0..n_params)
            .any(|i| layout.is_variance[i] && theta[i] <= lower[i] * (1.0 + 1e-9) + 1e-300);
        if !variance_at_floor {
            let (_, ai) = self.scores_and_ai(model, &ev, &layout);
            ai_matrix = Some(ai.clone());
            let free: Vec<usize> = (0..n_params).filter(|&i| !at_boundary[i]).collect();
            let m = free.len();
            let ai_free = DMatrix::from_fn(m, m, |a, b| ai[(free[a], free[b])]);
            if let Some(chol) = ai_free.cholesky() {
                let inv = chol.inverse();
                for (a, &i) in free.iter().enumerate() {
                    variance_se[i] = inv[(a, a)].abs().sqrt();
                }
            }
        }

        let mut result = self.build_result(
            model,
            &layout,
            &ev,
            &theta,
            &variance_se,
            &at_boundary,
            history,
            converged,
        )?;
        result.ai_matrix = ai_matrix;
        Ok(result)
    }

    /// Solve the MME at `theta` and compute the REML log-likelihood.
    fn evaluate(
        &self,
        model: &mut MixedModel,
        layout: &ParamLayout,
        theta: &[f64],
    ) -> Result<Evaluation> {
        layout.apply(model, theta)?;
        let n = model.n_obs;

        let g_inv_blocks: Vec<SparseMat> = model
            .random_var_structs
            .iter()
            .enumerate()
            .map(|(k, vs)| vs.inverse_covariance_matrix(model.z_blocks[k].cols()))
            .collect();
        let residual = Residual::from_model(model)?;

        let mme = match &residual {
            Residual::Scalar { sigma2 } => MixedModelEquations::assemble(
                &model.x,
                &model.z_blocks,
                &model.y,
                1.0 / sigma2,
                &g_inv_blocks,
            ),
            Residual::Dense { .. } => MixedModelEquations::assemble_structured_r(
                &model.x,
                &model.z_blocks,
                &model.y,
                &residual.to_sparse(n),
                &g_inv_blocks,
            ),
        };
        let sol = mme.solve()?;

        // residuals
        let mut fitted = spmv(&model.x, &sol.fixed_effects);
        for (k, z) in model.z_blocks.iter().enumerate() {
            let zu = spmv(z, &sol.random_effects[k]);
            for i in 0..n {
                fitted[i] += zu[i];
            }
        }
        let e_hat: Vec<f64> = model
            .y
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect();

        let rinv_y = residual.apply_inv(&model.y);
        let y_r_inv_y: f64 = model.y.iter().zip(rinv_y.iter()).map(|(a, b)| a * b).sum();
        let sol_rhs: f64 = sol
            .solution
            .iter()
            .zip(mme.rhs.iter())
            .map(|(s, r)| s * r)
            .sum();
        let y_p_y = y_r_inv_y - sol_rhs;

        let log_det_g: f64 = model
            .random_var_structs
            .iter()
            .enumerate()
            .map(|(k, vs)| vs.log_determinant(model.z_blocks[k].cols()))
            .sum();
        let n_eff = (n - mme.n_fixed) as f64;
        let log_2_pi = (2.0 * std::f64::consts::PI).ln();
        let logl =
            -0.5 * (n_eff * log_2_pi + residual.log_det(n) + log_det_g + sol.log_det_c + y_p_y);

        Ok(Evaluation {
            mme,
            sol,
            residual,
            logl,
            e_hat,
        })
    }

    /// REML scores and the average information matrix at the evaluated point.
    fn scores_and_ai(
        &self,
        model: &MixedModel,
        ev: &Evaluation,
        layout: &ParamLayout,
    ) -> (Vec<f64>, DMatrix<f64>) {
        let n = model.n_obs;
        let n_params = layout.total;
        let c_inv = ev
            .sol
            .c_inv
            .as_ref()
            .expect("dense C^-1 is always computed by the MME solve");
        let n_fixed = ev.mme.n_fixed;
        let dim = c_inv.nrows();

        let mut score = vec![0.0; n_params];
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_params);

        // ---- random terms ----
        let mut offset = n_fixed;
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            let z = &model.z_blocks[k];
            let q = z.cols();
            let u = &ev.sol.random_effects[k];
            let c_kk = c_inv.view((offset, offset), (q, q));
            let gbars = vs.derivatives_of_inverse(q);
            let g_gbars = vs.sigma_times_derivatives_of_inverse(q);
            for (i, (gbar, g_gbar)) in gbars.iter().zip(g_gbars.iter()).enumerate() {
                let gbar_u = spmv(gbar, u);
                let u_gbar_u: f64 = u.iter().zip(gbar_u.iter()).map(|(a, b)| a * b).sum();
                let tr_gbar_c: f64 = gbar.iter().map(|(v, (a, b))| v * c_kk[(b, a)]).sum();
                let tr_g_gbar: f64 = g_gbar
                    .iter()
                    .filter(|(_, (a, b))| a == b)
                    .map(|(v, _)| *v)
                    .sum();
                score[layout.term_ranges[k].start + i] = 0.5 * (-u_gbar_u - tr_gbar_c + tr_g_gbar);

                let g_gbar_u = spmv(g_gbar, u);
                let w_i: Vec<f64> = spmv(z, &g_gbar_u).iter().map(|v| -v).collect();
                w.push(w_i);
            }
            offset += q;
        }

        // ---- residual parameters ----
        let n_res = ev.residual.n_params();
        if n_res > 0 {
            let rinv_e = ev.residual.apply_inv(&ev.e_hat);
            let w_dense = dense_design(model, dim);
            let rinv_w = ev.residual.apply_inv_mat(&w_dense);
            for i in 0..n_res {
                let d_rinv_e = ev.residual.apply_d(i, &rinv_e);
                let quad: f64 = rinv_e.iter().zip(d_rinv_e.iter()).map(|(a, b)| a * b).sum();
                let tr_rinv_d = ev.residual.trace_rinv_d(i, n);
                let m = ev.residual.m_matrix(i, &rinv_w);
                let tr_cinv_m = c_inv.component_mul(&m).sum();
                score[layout.residual_range.start + i] = 0.5 * (quad - tr_rinv_d + tr_cinv_m);
                w.push(d_rinv_e);
            }
        }

        // ---- average information: AI_ij = ½ (w_i'R⁻¹w_j − t_i'C⁻¹t_j), t = W'R⁻¹w ----
        let rinv_w_vecs: Vec<Vec<f64>> = w.iter().map(|wi| ev.residual.apply_inv(wi)).collect();
        let t: Vec<DVector<f64>> = rinv_w_vecs
            .iter()
            .map(|rw| wt_apply(model, rw, dim, n_fixed))
            .collect();
        let c_inv_t: Vec<DVector<f64>> = t.iter().map(|ti| c_inv * ti).collect();

        let mut ai = DMatrix::zeros(n_params, n_params);
        for i in 0..n_params {
            for j in i..n_params {
                let wi_rinv_wj: f64 = w[i]
                    .iter()
                    .zip(rinv_w_vecs[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let value = 0.5 * (wi_rinv_wj - t[i].dot(&c_inv_t[j]));
                ai[(i, j)] = value;
                ai[(j, i)] = value;
            }
        }
        (score, ai)
    }

    fn build_result(
        &self,
        model: &MixedModel,
        layout: &ParamLayout,
        ev: &Evaluation,
        theta: &[f64],
        variance_se: &[f64],
        at_boundary: &[bool],
        history: Vec<RemlIteration>,
        converged: bool,
    ) -> Result<FitResult> {
        let n = model.n_obs;
        let n_fixed = ev.mme.n_fixed;
        let c_inv = ev.sol.c_inv.as_ref().unwrap();

        let mut variance_components = Vec::new();
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            let r = layout.term_ranges[k].clone();
            variance_components.push(VarianceEstimate {
                name: model.random_term_names[k].clone(),
                structure: vs.name().to_string(),
                parameters: vs
                    .param_names()
                    .into_iter()
                    .zip(theta[r.clone()].iter().copied())
                    .collect(),
                se: variance_se[r.clone()].to_vec(),
                at_boundary: at_boundary[r].to_vec(),
            });
        }
        let r = layout.residual_range.clone();
        variance_components.push(VarianceEstimate {
            name: "residual".to_string(),
            structure: model.residual_var_struct.name().to_string(),
            parameters: model
                .residual_var_struct
                .param_names()
                .into_iter()
                .zip(theta[r.clone()].iter().copied())
                .collect(),
            se: variance_se[r.clone()].to_vec(),
            at_boundary: at_boundary[r].to_vec(),
        });

        let fixed_cov: Vec<Vec<f64>> = (0..n_fixed)
            .map(|i| (0..n_fixed).map(|j| c_inv[(i, j)]).collect())
            .collect();
        let fixed_effects: Vec<NamedEffect> = model
            .fixed_labels
            .iter()
            .enumerate()
            .map(|(i, label)| NamedEffect {
                term: label.term.clone(),
                level: label.level.clone(),
                estimate: ev.sol.fixed_effects[i],
                se: c_inv[(i, i)].sqrt(),
            })
            .collect();

        let mut random_effects = Vec::new();
        let mut block_offset = n_fixed;
        for (k, levels) in model.random_level_names.iter().enumerate() {
            let effects: Vec<NamedEffect> = levels
                .iter()
                .enumerate()
                .map(|(j, level_name)| NamedEffect {
                    term: model.random_term_names[k].clone(),
                    level: level_name.clone(),
                    estimate: ev.sol.random_effects[k][j],
                    se: c_inv[(block_offset + j, block_offset + j)].sqrt(),
                })
                .collect();
            random_effects.push(RandomEffectBlock {
                term: model.random_term_names[k].clone(),
                effects,
            });
            block_offset += levels.len();
        }

        Ok(FitResult {
            variance_components,
            fixed_effects,
            random_effects,
            log_likelihood: ev.logl,
            n_iterations: history.len(),
            converged,
            history,
            variance_se: variance_se.to_vec(),
            residuals: ev.e_hat.clone(),
            fixed_cov,
            at_boundary: at_boundary.to_vec(),
            n_obs: n,
            n_fixed_params: n_fixed,
            n_variance_params: layout.total,
            c_inv: ev.sol.c_inv.clone(),
            ai_matrix: None,
            n_random_per_term: model.z_blocks.iter().map(|z| z.cols()).collect(),
        })
    }
}

/// Scale-aware test that the free-parameter gradient is negligible.
fn gradient_is_small(score: &DVector<f64>, theta: &[f64], free: &[usize], logl: f64) -> bool {
    let scale = 1e-6 * logl.abs().max(1.0);
    score
        .iter()
        .zip(free.iter())
        .all(|(s, &i)| (s * theta[i].abs().max(1.0)).abs() < scale)
}

/// Newton direction `AI⁻¹ s`, with a small ridge if AI is not positive
/// definite; `None` if it is hopeless.
fn newton_direction(ai: &DMatrix<f64>, score: &DVector<f64>) -> Option<DVector<f64>> {
    let m = ai.nrows();
    let chol = match ai.clone().cholesky() {
        Some(c) => c,
        None => {
            let mut ridged = ai.clone();
            let ridge = 1e-6 * ai.diagonal().iter().map(|x| x.abs()).sum::<f64>() / m as f64;
            for i in 0..m {
                ridged[(i, i)] += ridge.max(1e-8);
            }
            ridged.cholesky()?
        }
    };
    let d = chol.solve(score);
    if d.iter().any(|x| !x.is_finite()) {
        None
    } else {
        Some(d)
    }
}

/// `W' v` for `W = [X Z_1 ... Z_r]`.
fn wt_apply(model: &MixedModel, v: &[f64], dim: usize, n_fixed: usize) -> DVector<f64> {
    let mut out = DVector::zeros(dim);
    let xt_v = xt_y(&model.x, v);
    out.rows_mut(0, n_fixed).copy_from_slice(&xt_v);
    let mut offset = n_fixed;
    for z in &model.z_blocks {
        let zt_v = xt_y(z, v);
        out.rows_mut(offset, z.cols()).copy_from_slice(&zt_v);
        offset += z.cols();
    }
    out
}

/// Dense `W = [X Z_1 ... Z_r]` (n x dim).
fn dense_design(model: &MixedModel, dim: usize) -> DMatrix<f64> {
    let n = model.n_obs;
    let mut w = DMatrix::zeros(n, dim);
    for (v, (i, j)) in model.x.iter() {
        w[(i, j)] += *v;
    }
    let mut offset = model.x.cols();
    for z in &model.z_blocks {
        for (v, (i, j)) in z.iter() {
            w[(i, offset + j)] += *v;
        }
        offset += z.cols();
    }
    w
}

/// Dense `[cells, cells]` sub-matrix of a sparse matrix.
fn subset_dense(m: &SparseMat, cells: &[usize]) -> DMatrix<f64> {
    let n = cells.len();
    let mut pos = vec![usize::MAX; m.rows()];
    for (obs, &cell) in cells.iter().enumerate() {
        pos[cell] = obs;
    }
    let mut out = DMatrix::zeros(n, n);
    for (v, (i, j)) in m.iter() {
        let (a, b) = (pos[i], pos[j]);
        if a != usize::MAX && b != usize::MAX {
            out[(a, b)] += *v;
        }
    }
    out
}

fn dense_to_sparse(m: &DMatrix<f64>) -> SparseMat {
    let mut tri = TriMat::new((m.nrows(), m.ncols()));
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            let v = m[(i, j)];
            if v != 0.0 {
                tri.add_triplet(i, j, v);
            }
        }
    }
    tri.to_csc()
}
