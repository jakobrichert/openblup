use crate::diagnostics::{
    fixed_cov_derivatives_scaled_identity, kenward_roger_terms_scaled_identity,
};
use crate::error::Result;
use crate::matrix::sparse::spmv;
use crate::model::MixedModel;
use crate::types::SparseMat;

use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use super::mme::{MmeInverse, SparseMixedModelEquations, SparseMmeStructure};
use super::result::{FitResult, NamedEffect, RandomEffectBlock, RemlIteration, VarianceEstimate};

/// REML engine using the Average Information algorithm (Gilmour, Thompson &
/// Cullis 1995) for variance component estimation.
///
/// The AI-REML algorithm uses Newton-Raphson updates with the Average
/// Information matrix as an approximation to the expected Hessian of the
/// REML log-likelihood.  This provides quadratic convergence near the
/// optimum, unlike the linear convergence of EM-REML.
///
/// Newton (AI) updates are taken from the first iteration, as in ASReml and
/// airemlf90; an optional number of EM-REML burn-in steps can be requested
/// with [`em_initial_steps`](Self::em_initial_steps). A Newton update that
/// decreases the log-likelihood is backtracked (step-halving) and followed by
/// a few EM steps, which increase the likelihood monotonically.
pub struct AiReml {
    max_iter: usize,
    tol: f64,
    em_initial_steps: usize,
    /// MME structure of the last model evaluated by
    /// [`log_likelihood_at`](Self::log_likelihood_at), keyed by a fingerprint
    /// of the model's data.
    structure_cache: Mutex<Option<(u64, Arc<SparseMmeStructure>)>>,
}

impl AiReml {
    /// Create a new AI-REML solver.
    ///
    /// * `max_iter` - Maximum total iterations (EM + AI combined).
    /// * `tol`      - Relative convergence tolerance on variance parameters.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self {
            max_iter,
            tol,
            em_initial_steps: 0,
            structure_cache: Mutex::new(None),
        }
    }

    /// Set the number of EM burn-in iterations before the first Newton step
    /// (default 0: every iteration costs a factorization and the selected
    /// inversion, so burn-in steps only add iterations on well-posed models).
    pub fn em_initial_steps(mut self, n: usize) -> Self {
        self.em_initial_steps = n;
        self
    }

    /// The REML log-likelihood at `theta` without fitting.
    ///
    /// `theta` is the flat parameter vector in the order of
    /// [`FitResult::variance_components`] (one variance per random term and
    /// the residual variance for scaled-identity models; the structures'
    /// native parameters otherwise), i.e. the order of
    /// [`RemlIteration::variance_params`](super::RemlIteration). Models that
    /// need the general engine are evaluated there. The model's parameters
    /// are left unchanged.
    pub fn log_likelihood_at(&self, model: &mut MixedModel, theta: &[f64]) -> Result<f64> {
        if model.needs_general_engine() {
            return super::GeneralReml::new(self.max_iter, self.tol)
                .log_likelihood_at(model, theta);
        }
        let k = model.random_var_structs.len();
        let bounds: Vec<(f64, f64)> = model
            .random_var_structs
            .iter()
            .flat_map(|vs| vs.bounds())
            .chain(model.residual_var_struct.bounds())
            .collect();
        super::general_reml::check_theta(theta, &bounds)?;
        let saved: Vec<f64> = model
            .random_var_structs
            .iter()
            .map(|vs| vs.params()[0])
            .chain([model.residual_var_struct.params()[0]])
            .collect();
        let structure = self.cached_structure(model);
        let logl = self
            .solve_at(model, &structure, &theta[..k], theta[k])
            .map(|(mme, sol)| self.log_likelihood(model, &mme, &sol, &theta[..k], theta[k]));
        for (vs, s) in model.random_var_structs.iter_mut().zip(&saved) {
            vs.set_params(&[*s])?;
        }
        model.residual_var_struct.set_params(&[saved[k]])?;
        logl
    }

    /// The MME structure of `model`, reused while the model data are unchanged.
    fn cached_structure(&self, model: &MixedModel) -> Arc<SparseMmeStructure> {
        let key = model_fingerprint(model);
        let mut cache = self
            .structure_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match cache.as_ref() {
            Some((k, structure)) if *k == key => Arc::clone(structure),
            _ => {
                let structure = Arc::new(SparseMmeStructure::for_model(model));
                *cache = Some((key, Arc::clone(&structure)));
                structure
            }
        }
    }

    /// Fit the model using AI-REML (with EM fallback).
    ///
    /// Models with multi-parameter variance structures (AR1, Diagonal,
    /// Unstructured, FactorAnalytic, Kronecker interactions) or a structured
    /// residual are handled by the [`GeneralReml`](super::GeneralReml) engine.
    pub fn fit(&self, model: &mut MixedModel) -> Result<FitResult> {
        if model.needs_general_engine() {
            return super::GeneralReml::new(self.max_iter, self.tol).fit(model);
        }
        let n = model.n_obs;
        let n_random_terms = model.random_var_structs.len();
        let n_params = n_random_terms + 1; // random variances + residual
                                           // Pattern, ordering and symbolic factorization are shared by all
                                           // iterations.
        let structure = SparseMmeStructure::for_model(model);

        // ---- initialise variance parameters ----
        let y = &model.y;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_var: f64 =
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        let init_var = (y_var / n_params as f64).max(0.01);

        let mut sigma2_random: Vec<f64> = vec![init_var; n_random_terms];
        let mut sigma2_e = init_var;

        // Override with user-provided starting values if non-default
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            let p = vs.params()[0];
            if (p - 1.0).abs() > 1e-10 {
                sigma2_random[k] = p;
            }
        }
        {
            let p = model.residual_var_struct.params()[0];
            if (p - 1.0).abs() > 1e-10 {
                sigma2_e = p;
            }
        }

        // Variance parameters may not drop below this floor. A parameter that
        // reaches it is treated as fixed at the boundary of the parameter
        // space (what ASReml reports as "B") and is excluded from the
        // convergence criterion.
        let floor = boundary_floor(&sigma2_random, sigma2_e);

        let mut history = Vec::new();
        let mut converged = false;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut prev_params: (Vec<f64>, f64) = (sigma2_random.clone(), sigma2_e);
        let mut last_step_was_ai = false;
        let mut forced_em_steps = 0usize;
        // Parameters fixed at the boundary (sticky once reached).
        let mut fixed: Vec<bool> = vec![false; n_params];

        for iter in 0..self.max_iter {
            // Solve the MME at the current parameters. A failure right after
            // an AI step (e.g. a wild Newton step made C numerically
            // singular) is treated like a likelihood decrease below.
            let mut current = match self.solve_at(model, &structure, &sigma2_random, sigma2_e) {
                Ok((m, s)) => {
                    let l = self.log_likelihood(model, &m, &s, &sigma2_random, sigma2_e);
                    Some((m, s, l))
                }
                Err(e) if last_step_was_ai => {
                    log::debug!("MME solve failed after AI step ({}); backtracking", e);
                    None
                }
                Err(e) => return Err(e),
            };

            // ---- safeguard: a Newton (AI) step must not decrease logL ----
            // When an AI step overshoots, backtrack along the step (halving
            // the step length) until the likelihood no longer decreases; if
            // even a tiny step fails, go back to the previous parameters and
            // continue with a few EM steps (which increase the likelihood
            // monotonically) before trying AI again.
            let logl_tol = 1e-8 * prev_logl.abs().max(1.0);
            let step_rejected = last_step_was_ai
                && current
                    .as_ref()
                    .is_none_or(|(_, _, l)| *l < prev_logl - logl_tol);
            if step_rejected {
                let (prev_random, prev_e) = prev_params.clone();
                let full_random = sigma2_random.clone();
                let full_e = sigma2_e;
                let mut accepted = false;
                let mut frac = 0.5;
                for _ in 0..6 {
                    let trial_random: Vec<f64> = prev_random
                        .iter()
                        .zip(full_random.iter())
                        .map(|(p, f)| (p + frac * (f - p)).max(floor))
                        .collect();
                    let trial_e = (prev_e + frac * (full_e - prev_e)).max(floor);
                    if let Ok((m, s)) = self.solve_at(model, &structure, &trial_random, trial_e) {
                        let l = self.log_likelihood(model, &m, &s, &trial_random, trial_e);
                        if l >= prev_logl - logl_tol {
                            sigma2_random = trial_random;
                            sigma2_e = trial_e;
                            current = Some((m, s, l));
                            accepted = true;
                            break;
                        }
                    }
                    frac *= 0.5;
                }
                if !accepted {
                    sigma2_random = prev_random;
                    sigma2_e = prev_e;
                    let (m, s) = self.solve_at(model, &structure, &sigma2_random, sigma2_e)?;
                    current = Some((m, s, prev_logl));
                    forced_em_steps = 3;
                }
            }
            let (mme, sol, logl) = current.expect("MME solved at accepted parameters");

            let c_inv = sol.inverse()?;
            let n_fixed = mme.n_fixed;
            let residuals = self.residuals(model, &sol);

            // Parameters that reached the floor (in an accepted step) are
            // fixed at the boundary from now on.
            for k in 0..n_random_terms {
                if sigma2_random[k] <= floor * 1.000001 {
                    fixed[k] = true;
                    sigma2_random[k] = floor;
                }
            }
            if sigma2_e <= floor * 1.000001 {
                fixed[n_random_terms] = true;
                sigma2_e = floor;
            }

            // Save old params
            let old_sigma2_random = sigma2_random.clone();
            let old_sigma2_e = sigma2_e;

            // With a parameter at the boundary the MME become badly scaled
            // (a block of C is ~1/floor) and the MME-based AI matrix suffers
            // catastrophic cancellation, so the remaining parameters are
            // estimated by EM in that case.
            let any_fixed = fixed.iter().any(|f| *f);
            let use_em = iter < self.em_initial_steps || forced_em_steps > 0 || any_fixed;
            forced_em_steps = forced_em_steps.saturating_sub(1);

            let mut took_ai_step = false;
            if !use_em {
                if let Some((new_random, new_e)) = self.ai_update(
                    model,
                    &sol,
                    c_inv,
                    n_fixed,
                    n,
                    &sigma2_random,
                    sigma2_e,
                    &residuals,
                    floor,
                    &fixed,
                ) {
                    sigma2_random = new_random;
                    sigma2_e = new_e;
                    took_ai_step = true;
                }
            }
            if !took_ai_step {
                // EM update (burn-in, forced after a rejected AI step, or
                // fallback when the AI matrix is singular).
                self.em_update(
                    model,
                    &sol,
                    c_inv,
                    n_fixed,
                    n,
                    &mut sigma2_random,
                    &mut sigma2_e,
                );
            }
            // Keep every parameter inside the parameter space and hold the
            // boundary parameters at the floor.
            for k in 0..n_random_terms {
                sigma2_random[k] = if fixed[k] {
                    floor
                } else {
                    sigma2_random[k].max(floor)
                };
            }
            sigma2_e = if fixed[n_random_terms] {
                floor
            } else {
                sigma2_e.max(floor)
            };

            last_step_was_ai = took_ai_step;
            prev_logl = logl;
            prev_params = (old_sigma2_random.clone(), old_sigma2_e);

            // ---- convergence check (free parameters only) ----
            let rel_change = relative_change(
                &old_sigma2_random,
                old_sigma2_e,
                &sigma2_random,
                sigma2_e,
                &fixed,
            );

            let mut all_params_new = sigma2_random.clone();
            all_params_new.push(sigma2_e);
            history.push(RemlIteration {
                iteration: iter + 1,
                log_likelihood: logl,
                variance_params: all_params_new,
                change: rel_change,
            });

            if iter > 0 && rel_change < self.tol {
                converged = true;
                break;
            }
        }

        // ---- final solve with converged parameters ----
        let (mme, sol) = self.solve_at(model, &structure, &sigma2_random, sigma2_e)?;
        let c_inv = sol.inverse()?;
        let residuals = self.residuals(model, &sol);

        let mut var_params: Vec<f64> = sigma2_random.clone();
        var_params.push(sigma2_e);
        // A parameter pushed to the floor by the very last step is on the
        // boundary too, even though the sticky flag was not set yet.
        let at_boundary: Vec<bool> = var_params
            .iter()
            .zip(fixed.iter())
            .map(|(v, f)| *f || *v <= floor * 1.000001)
            .collect();

        // Approximate SEs of the variance components from the inverse of the
        // Average Information matrix at convergence. With a parameter on the
        // boundary the AI matrix is unreliable, so no SEs are reported.
        let mut variance_se = vec![0.0; n_params];
        let mut ai_matrix = None;
        if !at_boundary.iter().any(|b| *b) {
            let ai = self.average_information(
                model,
                &sol,
                c_inv,
                mme.n_fixed,
                &sigma2_random,
                sigma2_e,
                &residuals,
            )?;
            if let Some(chol) = ai.clone().cholesky() {
                let ai_inv = chol.inverse();
                for (i, se) in variance_se.iter_mut().enumerate() {
                    *se = ai_inv[(i, i)].abs().sqrt();
                }
            }
            ai_matrix = Some(ai);
        }

        let mut result = self.build_result(
            model,
            &sol,
            &mme,
            &var_params,
            &variance_se,
            &at_boundary,
            &history,
            converged,
        )?;
        result.ai_matrix = ai_matrix;
        if result.ai_matrix.is_some() {
            // Derivatives of the fixed-effects covariance for Satterthwaite df.
            let kinv: Vec<Option<&SparseMat>> =
                model.ginv_matrices.iter().map(|g| g.as_ref()).collect();
            result.fixed_cov_derivatives = fixed_cov_derivatives_scaled_identity(
                c_inv,
                &model.x,
                &model.z_blocks,
                &kinv,
                &var_params,
            )
            .iter()
            .map(|m| {
                (0..m.nrows())
                    .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
                    .collect()
            })
            .collect();
            // Kenward-Roger terms (P_i, Q_ij) through the random block of
            // the MME; a failure here only disables the Kenward-Roger tests.
            let ginv_scaled: Vec<SparseMat> = (0..n_random_terms)
                .map(|k| {
                    let q = model.z_blocks[k].cols();
                    match model.ginv_matrices[k] {
                        Some(ref g) => g.map(|v| v / sigma2_random[k]),
                        None => {
                            crate::matrix::sparse::sparse_diagonal(&vec![1.0 / sigma2_random[k]; q])
                        }
                    }
                })
                .collect();
            result.kenward_roger_terms = kenward_roger_terms_scaled_identity(
                &model.x,
                &model.z_blocks,
                &ginv_scaled,
                &var_params,
            )
            .ok();
        }
        Ok(result)
    }

    /// Assemble and solve the MME at the given variance parameters.
    fn solve_at(
        &self,
        model: &mut MixedModel,
        structure: &SparseMmeStructure,
        sigma2_random: &[f64],
        sigma2_e: f64,
    ) -> Result<(SparseMixedModelEquations, super::mme::MmeSolution)> {
        for (k, vs) in model.random_var_structs.iter_mut().enumerate() {
            vs.set_params(&[sigma2_random[k]])?;
        }
        model.residual_var_struct.set_params(&[sigma2_e])?;

        let scales: Vec<f64> = sigma2_random.iter().map(|s| 1.0 / s).collect();
        let mme = structure.assemble(1.0 / sigma2_e, &scales);
        let sol = mme.solve()?;
        Ok((mme, sol))
    }

    /// REML log-likelihood (up to a constant) at the current parameters.
    fn log_likelihood(
        &self,
        model: &MixedModel,
        mme: &SparseMixedModelEquations,
        sol: &super::mme::MmeSolution,
        sigma2_random: &[f64],
        sigma2_e: f64,
    ) -> f64 {
        let n = model.n_obs;
        let n_eff = (n - mme.n_fixed) as f64;
        let y_r_inv_y = model.y.iter().map(|yi| yi * yi).sum::<f64>() / sigma2_e;
        let sol_rhs: f64 = sol
            .solution
            .iter()
            .zip(mme.rhs.iter())
            .map(|(s, r)| s * r)
            .sum();
        let y_p_y = y_r_inv_y - sol_rhs;

        let log_det_r = n as f64 * sigma2_e.ln();
        let log_det_g: f64 = sigma2_random
            .iter()
            .enumerate()
            .map(|(k, s)| {
                model.z_blocks[k].cols() as f64 * s.ln()
                    + model.random_var_structs[k].relationship_log_det()
            })
            .sum();
        let log_2_pi = (2.0 * std::f64::consts::PI).ln();
        -0.5 * (n_eff * log_2_pi + log_det_r + log_det_g + sol.log_det_c + y_p_y)
    }

    /// Residuals ê = y - Xb - Zu at the current solution.
    fn residuals(&self, model: &MixedModel, sol: &super::mme::MmeSolution) -> Vec<f64> {
        let n = model.n_obs;
        let mut fitted = spmv(&model.x, &sol.fixed_effects);
        for (k, z) in model.z_blocks.iter().enumerate() {
            let zu = spmv(z, &sol.random_effects[k]);
            for i in 0..n {
                fitted[i] += zu[i];
            }
        }
        model
            .y
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect()
    }

    /// For each random term k: `(u_k' K_k^{-1} u_k, tr(K_k^{-1} C^{uu_k}))`
    /// where `C^{uu_k}` is the k-th random block of C^{-1}.
    fn quadratic_and_trace(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &MmeInverse,
        n_fixed: usize,
    ) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(model.z_blocks.len());
        let mut block_start = n_fixed;
        for (k, z) in model.z_blocks.iter().enumerate() {
            let q_k = z.cols();
            let u_k = &sol.random_effects[k];

            let (u_quadratic, trace_term) = if let Some(ref ginv_k) = model.ginv_matrices[k] {
                let kinv_u = spmv(ginv_k, u_k);
                let uq = u_k
                    .iter()
                    .zip(kinv_u.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                // tr(K^{-1} C^{uu}) = sum_{ij} K^{-1}_{ij} C^{uu}_{ji}; K^{-1} is sparse.
                let tr = c_inv.trace_block(ginv_k, block_start);
                (uq, tr)
            } else {
                let uq = u_k.iter().map(|u| u * u).sum::<f64>();
                let tr = sol.c_inv_diag()[block_start..block_start + q_k]
                    .iter()
                    .sum::<f64>();
                (uq, tr)
            };
            out.push((u_quadratic, trace_term));
            block_start += q_k;
        }
        out
    }

    /// EM-REML update step (identical to the EmReml implementation).
    fn em_update(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &MmeInverse,
        n_fixed: usize,
        n: usize,
        sigma2_random: &mut [f64],
        sigma2_e: &mut f64,
    ) {
        let n_eff = (n - n_fixed) as f64;

        // EM update for residual variance: y'ê / (n - p)
        let mut y_e_hat = model.y.iter().map(|y| y * y).sum::<f64>();
        let xty = crate::matrix::sparse::xt_y(&model.x, &model.y);
        for i in 0..n_fixed {
            y_e_hat -= sol.fixed_effects[i] * xty[i];
        }
        for (k, z) in model.z_blocks.iter().enumerate() {
            let zty = crate::matrix::sparse::xt_y(z, &model.y);
            for j in 0..z.cols() {
                y_e_hat -= sol.random_effects[k][j] * zty[j];
            }
        }
        *sigma2_e = (y_e_hat / n_eff).max(1e-10);

        // EM update for each random variance component:
        // sigma^2_k = (u_k'K_k^{-1}u_k + tr(K_k^{-1} C^{uu_k})) / q_k
        for (k, (u_quadratic, trace_term)) in self
            .quadratic_and_trace(model, sol, c_inv, n_fixed)
            .into_iter()
            .enumerate()
        {
            let q_k = model.z_blocks[k].cols() as f64;
            sigma2_random[k] = ((u_quadratic + trace_term) / q_k).max(1e-10);
        }
    }

    /// REML score vector (gradient of the restricted log-likelihood with
    /// respect to the variance parameters `[sigma2_1, ..., sigma2_r, sigma2_e]`).
    ///
    /// With `G_k = sigma2_k K_k`, `R = sigma2_e I`, `u_k` the BLUPs and
    /// `C^{uu_k}` the k-th random block of C^{-1} (Johnson & Thompson 1995):
    ///
    /// ```text
    /// s_k = 1/2 [ u_k'K_k^{-1}u_k / sigma2_k^2 - (q_k - tr(K_k^{-1}C^{uu_k})/sigma2_k) / sigma2_k ]
    /// s_e = 1/2 [ e'e / sigma2_e^2 - (n - p - q + sum_k tr(K_k^{-1}C^{uu_k})/sigma2_k) / sigma2_e ]
    /// ```
    fn reml_scores(
        &self,
        model: &MixedModel,
        n_fixed: usize,
        n: usize,
        sigma2_random: &[f64],
        sigma2_e: f64,
        residuals: &[f64],
        quad_trace: &[(f64, f64)],
    ) -> Vec<f64> {
        let n_random_terms = sigma2_random.len();
        let mut score = vec![0.0; n_random_terms + 1];

        let mut q_total = 0usize;
        let mut trace_kinv_cuu_over_sigma = 0.0;
        for (k, &(u_quadratic, trace_term)) in quad_trace.iter().enumerate() {
            let q_k = model.z_blocks[k].cols();
            let s2 = sigma2_random[k];
            score[k] = 0.5 * (u_quadratic / (s2 * s2) - (q_k as f64 - trace_term / s2) / s2);
            q_total += q_k;
            trace_kinv_cuu_over_sigma += trace_term / s2;
        }

        let e_e: f64 = residuals.iter().map(|e| e * e).sum();
        // n - p - q can be negative when there are more random levels than
        // residual degrees of freedom (e.g. an animal model with few records).
        let trace_p_scaled = n as f64 - n_fixed as f64 - q_total as f64 + trace_kinv_cuu_over_sigma;
        score[n_random_terms] = 0.5 * (e_e / (sigma2_e * sigma2_e) - trace_p_scaled / sigma2_e);
        score
    }

    /// Average Information matrix (Gilmour, Thompson & Cullis 1995):
    ///
    /// ```text
    /// AI_ij = 1/2 w_i' P w_j,   w_i = (dV/dtheta_i) P y
    /// ```
    ///
    /// with `w_k = Z_k u_k / sigma2_k` for a random term and `w_e = e / sigma2_e`
    /// for the residual, and `P w = (w - W C^{-1} W'w / sigma2_e) / sigma2_e`
    /// evaluated through the MME inverse.
    fn average_information(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &MmeInverse,
        n_fixed: usize,
        sigma2_random: &[f64],
        sigma2_e: f64,
        residuals: &[f64],
    ) -> Result<nalgebra::DMatrix<f64>> {
        let n_random_terms = sigma2_random.len();
        let n_params = n_random_terms + 1;
        let dim = c_inv.dim();

        // Working variates w_i (length n).
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_params);
        for (k, z) in model.z_blocks.iter().enumerate() {
            let scaled_u: Vec<f64> = sol.random_effects[k]
                .iter()
                .map(|u| u / sigma2_random[k])
                .collect();
            w.push(spmv(z, &scaled_u));
        }
        w.push(residuals.iter().map(|e| e / sigma2_e).collect());

        // t_i = W' w_i with W = [X Z_1 ... Z_r], and C^{-1} t_i.
        let mut t: Vec<nalgebra::DVector<f64>> = Vec::with_capacity(n_params);
        let mut c_inv_t: Vec<nalgebra::DVector<f64>> = Vec::with_capacity(n_params);
        for wi in &w {
            let mut ti = nalgebra::DVector::zeros(dim);
            let xt_wi = crate::matrix::sparse::xt_y(&model.x, wi);
            for j in 0..n_fixed {
                ti[j] = xt_wi[j];
            }
            let mut offset = n_fixed;
            for z in &model.z_blocks {
                let zt_wi = crate::matrix::sparse::xt_y(z, wi);
                for j in 0..z.cols() {
                    ti[offset + j] = zt_wi[j];
                }
                offset += z.cols();
            }
            c_inv_t.push(nalgebra::DVector::from_vec(c_inv.apply(ti.as_slice())?));
            t.push(ti);
        }

        // AI_ij = 1/2 ( w_i'w_j / sigma2_e - t_i' C^{-1} t_j / sigma2_e^2 )
        let mut ai = nalgebra::DMatrix::zeros(n_params, n_params);
        for i in 0..n_params {
            for j in i..n_params {
                let wi_wj: f64 = w[i].iter().zip(w[j].iter()).map(|(a, b)| a * b).sum();
                let ti_cinv_tj = t[i].dot(&c_inv_t[j]);
                let value = 0.5 * (wi_wj / sigma2_e - ti_cinv_tj / (sigma2_e * sigma2_e));
                ai[(i, j)] = value;
                ai[(j, i)] = value;
            }
        }
        Ok(ai)
    }

    /// AI-REML Newton-Raphson update over the free (non-boundary) parameters.
    ///
    /// Returns `Some((new_random, new_e))` on success, or `None` if the AI
    /// matrix is singular (or no parameter is free) and the caller should
    /// fall back to EM. Parameters that the Newton step would push below
    /// `floor` are clamped to the floor (boundary of the parameter space).
    fn ai_update(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &MmeInverse,
        n_fixed: usize,
        n: usize,
        sigma2_random: &[f64],
        sigma2_e: f64,
        residuals: &[f64],
        floor: f64,
        fixed: &[bool],
    ) -> Option<(Vec<f64>, f64)> {
        let n_random_terms = sigma2_random.len();
        let n_params = n_random_terms + 1;
        let free: Vec<usize> = (0..n_params).filter(|&i| !fixed[i]).collect();
        if free.is_empty() {
            return None;
        }

        let quad_trace = self.quadratic_and_trace(model, sol, c_inv, n_fixed);
        let score = self.reml_scores(
            model,
            n_fixed,
            n,
            sigma2_random,
            sigma2_e,
            residuals,
            &quad_trace,
        );
        let ai = self
            .average_information(
                model,
                sol,
                c_inv,
                n_fixed,
                sigma2_random,
                sigma2_e,
                residuals,
            )
            .ok()?;

        // ---- Newton step over the free parameters: delta = AI^{-1} * score ----
        let m = free.len();
        let ai_free = nalgebra::DMatrix::from_fn(m, m, |a, b| ai[(free[a], free[b])]);
        let score_free = nalgebra::DVector::from_fn(m, |a, _| score[free[a]]);

        let ai_chol = match ai_free.clone().cholesky() {
            Some(c) => c,
            None => {
                // AI is not positive definite (e.g. collinear working
                // variates); try a small ridge before giving up.
                let mut ai_ridge = ai_free.clone();
                let ridge =
                    1e-6 * ai_free.diagonal().iter().map(|x| x.abs()).sum::<f64>() / m as f64;
                for i in 0..m {
                    ai_ridge[(i, i)] += ridge.max(1e-8);
                }
                ai_ridge.cholesky()?
            }
        };

        let delta = ai_chol.solve(&score_free);
        if delta.iter().any(|d| !d.is_finite()) {
            return None;
        }

        // Parameters that the step would push out of the parameter space are
        // clamped to the boundary, and no parameter may grow by more than a
        // factor of 10 in one step (the quadratic model is unreliable that
        // far away). A rejected (likelihood-decreasing) step is caught by the
        // safeguard in `fit`.
        let bounded = |old: f64, d: f64| (old + d).clamp(floor, 10.0 * old.max(floor));
        let mut new_random = sigma2_random.to_vec();
        let mut new_e = sigma2_e;
        for (a, &i) in free.iter().enumerate() {
            if i < n_random_terms {
                new_random[i] = bounded(sigma2_random[i], delta[a]);
            } else {
                new_e = bounded(sigma2_e, delta[a]);
            }
        }

        Some((new_random, new_e))
    }

    /// Build the final FitResult from converged solution.
    fn build_result(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        mme: &SparseMixedModelEquations,
        var_params: &[f64],
        variance_se: &[f64],
        at_boundary: &[bool],
        history: &[RemlIteration],
        converged: bool,
    ) -> Result<FitResult> {
        let n = model.n_obs;
        let sigma_e2 = model.residual_var_struct.params()[0];
        let r_inv_scale = 1.0 / sigma_e2;

        // Compute REML log-likelihood
        let y_r_inv_y = r_inv_scale * model.y.iter().map(|yi| yi * yi).sum::<f64>();
        let sol_rhs: f64 = sol
            .solution
            .iter()
            .zip(mme.rhs.iter())
            .map(|(s, r)| s * r)
            .sum();
        let y_p_y = y_r_inv_y - sol_rhs;

        let log_det_r = n as f64 * sigma_e2.ln();
        let mut log_det_g = 0.0;
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            let q = model.z_blocks[k].cols();
            log_det_g += vs.log_determinant(q);
        }
        let n_fixed = mme.n_fixed;
        let n_eff = n as f64 - n_fixed as f64;
        let log_2_pi = (2.0 * std::f64::consts::PI).ln();
        let logl = -0.5 * (n_eff * log_2_pi + log_det_r + log_det_g + sol.log_det_c + y_p_y);

        // Variance component estimates
        let mut variance_components = Vec::new();
        let n_terms = model.random_var_structs.len();
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            variance_components.push(VarianceEstimate {
                name: model.random_term_names[k].clone(),
                structure: vs.name().to_string(),
                parameters: vec![("sigma2".to_string(), vs.params()[0])],
                se: vec![variance_se.get(k).copied().unwrap_or(0.0)],
                at_boundary: vec![at_boundary.get(k).copied().unwrap_or(false)],
            });
        }
        variance_components.push(VarianceEstimate {
            name: "residual".to_string(),
            structure: model.residual_var_struct.name().to_string(),
            parameters: vec![("sigma2".to_string(), sigma_e2)],
            se: vec![variance_se.get(n_terms).copied().unwrap_or(0.0)],
            at_boundary: vec![at_boundary.get(n_terms).copied().unwrap_or(false)],
        });

        // Fixed effects with SEs from C^{-1}
        let c_inv = sol.inverse()?;
        let fixed_block = c_inv.fixed_block(n_fixed);
        let fixed_cov: Vec<Vec<f64>> = (0..n_fixed)
            .map(|i| (0..n_fixed).map(|j| fixed_block[(i, j)]).collect())
            .collect();
        let fixed_effects: Vec<NamedEffect> = model
            .fixed_labels
            .iter()
            .enumerate()
            .map(|(i, label)| NamedEffect {
                term: label.term.clone(),
                level: label.level.clone(),
                estimate: sol.fixed_effects[i],
                se: sol.c_inv_diag()[i].sqrt(),
            })
            .collect();

        // Random effects with SEs
        let mut random_effects = Vec::new();
        let mut block_offset = n_fixed;
        for (k, levels) in model.random_level_names.iter().enumerate() {
            let effects: Vec<NamedEffect> = levels
                .iter()
                .enumerate()
                .map(|(j, level_name)| NamedEffect {
                    term: model.random_term_names[k].clone(),
                    level: level_name.clone(),
                    estimate: sol.random_effects[k][j],
                    se: sol.c_inv_diag()[block_offset + j].sqrt(),
                })
                .collect();
            random_effects.push(RandomEffectBlock {
                term: model.random_term_names[k].clone(),
                effects,
            });
            block_offset += levels.len();
        }

        // Residuals: e = y - Xb - Zu
        let mut fitted = vec![0.0; n];
        let xb = spmv(&model.x, &sol.fixed_effects);
        for i in 0..n {
            fitted[i] += xb[i];
        }
        for (k, z) in model.z_blocks.iter().enumerate() {
            let zu = spmv(z, &sol.random_effects[k]);
            for i in 0..n {
                fitted[i] += zu[i];
            }
        }
        let residuals: Vec<f64> = model
            .y
            .iter()
            .zip(fitted.iter())
            .map(|(y, f)| y - f)
            .collect();

        Ok(FitResult {
            variance_components,
            fixed_effects,
            random_effects,
            log_likelihood: logl,
            n_iterations: history.len(),
            converged,
            history: history.to_vec(),
            variance_se: variance_se.to_vec(),
            residuals,
            fixed_cov,
            at_boundary: at_boundary.to_vec(),
            n_obs: n,
            n_fixed_params: n_fixed,
            n_variance_params: var_params.len(),
            c_inv: sol.c_inv.clone(),
            ai_matrix: None,
            n_random_per_term: model.z_blocks.iter().map(|z| z.cols()).collect(),
            fixed_cov_derivatives: Vec::new(),
            kenward_roger_terms: None,
        })
    }
}

/// Lower bound for variance parameters: a small fraction of the total
/// variance at the start of the iterations.
fn boundary_floor(sigma2_random: &[f64], sigma2_e: f64) -> f64 {
    let total: f64 = sigma2_random.iter().sum::<f64>() + sigma2_e;
    (1e-8 * total).max(1e-12)
}

/// Relative change in the variance parameters, ignoring parameters that are
/// fixed at the boundary.
fn relative_change(
    old_random: &[f64],
    old_e: f64,
    new_random: &[f64],
    new_e: f64,
    fixed: &[bool],
) -> f64 {
    let mut diff2 = 0.0;
    let mut norm2 = 0.0;
    let pairs = old_random
        .iter()
        .zip(new_random.iter())
        .map(|(o, n)| (*o, *n))
        .chain(std::iter::once((old_e, new_e)));
    for (i, (o, n)) in pairs.enumerate() {
        if fixed.get(i).copied().unwrap_or(false) {
            continue;
        }
        diff2 += (n - o).powi(2);
        norm2 += o * o;
    }
    if norm2 <= 0.0 {
        return 0.0;
    }
    (diff2 / norm2).sqrt()
}

/// A hash of everything the MME structure is built from.
fn model_fingerprint(model: &MixedModel) -> u64 {
    fn sparse(h: &mut impl Hasher, m: &SparseMat) {
        m.shape().hash(h);
        m.indptr().raw_storage().hash(h);
        m.indices().hash(h);
        for v in m.data() {
            v.to_bits().hash(h);
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in &model.y {
        v.to_bits().hash(&mut h);
    }
    sparse(&mut h, &model.x);
    for z in &model.z_blocks {
        sparse(&mut h, z);
    }
    for g in &model.ginv_matrices {
        match g {
            Some(g) => sparse(&mut h, g),
            None => 0u8.hash(&mut h),
        }
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFrame;
    use crate::model::MixedModelBuilder;
    use crate::variance::Identity;
    use approx::assert_relative_eq;

    /// Create a balanced trial dataset: 5 genotypes x 3 reps = 15 observations.
    /// Perfectly additive data (yield = rep + genotype, no residual): the
    /// REML residual variance is exactly zero, i.e. on the boundary.
    fn additive_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column(
            "yield",
            vec![
                10.0, 8.0, 6.0, 7.0, 9.0, // R1
                12.0, 10.0, 8.0, 9.0, 11.0, // R2
                11.0, 9.0, 7.0, 8.0, 10.0, // R3
            ],
        )
        .unwrap();
        add_factors(&mut df);
        df
    }

    /// Simulated yield data with genotype and rep effects plus residual
    /// noise, so that both variance components are strictly positive.
    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column(
            "yield",
            vec![
                10.3, 7.6, 6.4, 7.1, 8.7, // R1
                11.8, 10.4, 7.7, 9.5, 11.2, // R2
                11.1, 8.6, 7.3, 7.9, 10.4, // R3
            ],
        )
        .unwrap();
        add_factors(&mut df);
        df
    }

    fn add_factors(df: &mut DataFrame) {
        df.add_factor_column(
            "genotype",
            &[
                "G1", "G2", "G3", "G4", "G5", "G1", "G2", "G3", "G4", "G5", "G1", "G2", "G3", "G4",
                "G5",
            ],
        )
        .unwrap();
        df.add_factor_column(
            "rep",
            &[
                "R1", "R1", "R1", "R1", "R1", "R2", "R2", "R2", "R2", "R2", "R3", "R3", "R3", "R3",
                "R3",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_ai_reml_basic_convergence() {
        let df = sample_df();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(50)
            .convergence(1e-6)
            .build()
            .unwrap();

        let solver = AiReml::new(50, 1e-6);
        let result = solver.fit(&mut model).unwrap();

        assert!(result.converged, "AI-REML should converge");
        assert!(
            result.n_iterations < 50,
            "Should converge before max iterations"
        );
        assert!(result.variance_components.len() == 2); // genotype + residual
    }

    #[test]
    fn test_ai_reml_matches_em_approximately() {
        let df = sample_df();

        // Fit with AI-REML
        let mut model_ai = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(50)
            .convergence(1e-8)
            .build()
            .unwrap();
        let solver_ai = AiReml::new(50, 1e-8);
        let result_ai = solver_ai.fit(&mut model_ai).unwrap();

        // Fit with EM-REML
        let mut model_em = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(200)
            .convergence(1e-8)
            .build()
            .unwrap();
        let solver_em = super::super::reml::EmReml::new(200, 1e-8);
        let result_em = solver_em.fit(&mut model_em).unwrap();

        // Both should converge to the same variance components
        let sigma2_g_ai = result_ai.variance_components[0].parameters[0].1;
        let sigma2_e_ai = result_ai.variance_components[1].parameters[0].1;
        let sigma2_g_em = result_em.variance_components[0].parameters[0].1;
        let sigma2_e_em = result_em.variance_components[1].parameters[0].1;

        assert_relative_eq!(sigma2_g_ai, sigma2_g_em, epsilon = 1e-3);
        assert_relative_eq!(sigma2_e_ai, sigma2_e_em, epsilon = 1e-3);

        // Log-likelihoods should match
        assert_relative_eq!(
            result_ai.log_likelihood,
            result_em.log_likelihood,
            epsilon = 1e-2
        );
    }

    #[test]
    fn test_ai_reml_variance_se_computed() {
        let df = sample_df();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(50)
            .convergence(1e-6)
            .build()
            .unwrap();

        let solver = AiReml::new(50, 1e-6);
        let result = solver.fit(&mut model).unwrap();

        // Variance SEs vector should have correct length
        assert_eq!(result.variance_se.len(), 2);
        // The AI matrix is positive definite at the optimum, so every SE is
        // strictly positive and finite.
        for &se in &result.variance_se {
            assert!(
                se > 0.0 && se.is_finite(),
                "Variance SE should be positive, got {}",
                se
            );
        }
        assert_eq!(result.at_boundary, vec![false, false]);
    }

    /// The Average Information matrix must be symmetric positive definite
    /// (it is 1/2 W'PW for the working variates) and the REML scores must
    /// vanish at the converged solution.
    #[test]
    fn test_ai_matrix_is_positive_definite_and_scores_vanish() {
        let df = sample_df();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .build()
            .unwrap();
        let solver = AiReml::new(100, 1e-10);
        let result = solver.fit(&mut model).unwrap();
        assert!(result.converged);

        let sigma2_random = vec![result.variance_components[0].parameters[0].1];
        let sigma2_e = result.variance_components[1].parameters[0].1;
        let structure = SparseMmeStructure::for_model(&model);
        let (mme, sol) = solver
            .solve_at(&mut model, &structure, &sigma2_random, sigma2_e)
            .unwrap();
        let c_inv = sol.c_inv.as_ref().unwrap();
        let residuals = solver.residuals(&model, &sol);

        let ai = solver
            .average_information(
                &model,
                &sol,
                c_inv,
                mme.n_fixed,
                &sigma2_random,
                sigma2_e,
                &residuals,
            )
            .unwrap();
        assert_relative_eq!(ai[(0, 1)], ai[(1, 0)], epsilon = 1e-10);
        assert!(
            ai[(0, 0)] > 0.0 && ai[(1, 1)] > 0.0,
            "AI diagonal: {:?}",
            ai
        );
        assert!(ai.clone().cholesky().is_some(), "AI must be PD: {:?}", ai);

        let qt = solver.quadratic_and_trace(&model, &sol, c_inv, mme.n_fixed);
        let score = solver.reml_scores(
            &model,
            mme.n_fixed,
            model.n_obs,
            &sigma2_random,
            sigma2_e,
            &residuals,
            &qt,
        );
        for s in &score {
            assert!(
                s.abs() < 1e-5,
                "score should vanish at the optimum: {:?}",
                score
            );
        }
    }

    /// Perfectly additive data: the REML residual variance is zero, so the
    /// engine must converge with that parameter flagged at the boundary
    /// instead of running out of iterations.
    #[test]
    fn test_ai_reml_boundary_parameter_is_flagged() {
        let df = additive_df();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .build()
            .unwrap();
        let result = AiReml::new(200, 1e-6).fit(&mut model).unwrap();
        assert!(
            result.converged,
            "should converge with a boundary parameter ({} iterations)",
            result.n_iterations
        );
        assert_eq!(result.at_boundary, vec![false, true]);
        assert!(result.summary().contains("[boundary]"));
        // Genotype variance is still estimated: the genotype effects are
        // 2, 0, -2, -1, 1 around the mean.
        let sigma2_g = result.variance_components[0].parameters[0].1;
        assert!(sigma2_g > 1.0 && sigma2_g < 5.0, "sigma2_g = {}", sigma2_g);
        // BLUP ranking G1 > G5 > G2 > G4 > G3
        let blups = &result.random_effects[0].effects;
        let est = |l: &str| blups.iter().find(|e| e.level == l).unwrap().estimate;
        assert!(est("G1") > est("G5") && est("G5") > est("G2"));
        assert!(est("G2") > est("G4") && est("G4") > est("G3"));
    }

    #[test]
    fn test_ai_reml_faster_than_em() {
        let df = sample_df();

        let mut model_ai = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(100)
            .convergence(1e-8)
            .build()
            .unwrap();
        let solver_ai = AiReml::new(100, 1e-8);
        let result_ai = solver_ai.fit(&mut model_ai).unwrap();

        let mut model_em = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random("genotype", Identity::new(1.0), None)
            .max_iterations(200)
            .convergence(1e-8)
            .build()
            .unwrap();
        let solver_em = super::super::reml::EmReml::new(200, 1e-8);
        let result_em = solver_em.fit(&mut model_em).unwrap();

        // AI-REML must converge in strictly fewer iterations than EM-REML:
        // the Newton steps converge quadratically.
        assert!(
            result_ai.n_iterations < result_em.n_iterations,
            "AI-REML ({} iters) should take fewer iterations than EM-REML ({} iters)",
            result_ai.n_iterations,
            result_em.n_iterations,
        );
        assert!(
            result_ai.n_iterations <= 15,
            "AI-REML took {} iterations on a tiny balanced problem",
            result_ai.n_iterations
        );
        // Both reach the same optimum.
        assert_relative_eq!(
            result_ai.log_likelihood,
            result_em.log_likelihood,
            epsilon = 1e-4
        );
    }
}
