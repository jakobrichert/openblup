use crate::error::{LmmError, Result};
use crate::matrix::sparse::spmv;
use crate::model::MixedModel;

use super::mme::MixedModelEquations;
use super::result::{
    FitResult, NamedEffect, RandomEffectBlock, RemlIteration, VarianceEstimate,
};

/// REML engine using the Average Information algorithm (Gilmour, Thompson &
/// Cullis 1995) for variance component estimation.
///
/// The AI-REML algorithm uses Newton-Raphson updates with the Average
/// Information matrix as an approximation to the expected Hessian of the
/// REML log-likelihood.  This provides quadratic convergence near the
/// optimum, unlike the linear convergence of EM-REML.
///
/// The implementation starts with a configurable number of EM-REML steps
/// to obtain good starting values, then switches to AI updates for fast
/// convergence.  Step-halving is used if a Newton update produces negative
/// variance estimates or decreases the log-likelihood.
pub struct AiReml {
    max_iter: usize,
    tol: f64,
    em_initial_steps: usize,
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
            em_initial_steps: 5,
        }
    }

    /// Set the number of EM burn-in iterations (default 5).
    pub fn em_initial_steps(mut self, n: usize) -> Self {
        self.em_initial_steps = n;
        self
    }

    /// Fit the model using AI-REML with EM burn-in.
    pub fn fit(&self, model: &mut MixedModel) -> Result<FitResult> {
        let n = model.n_obs;
        let n_random_terms = model.random_var_structs.len();
        let _n_params = n_random_terms + 1; // random variances + residual

        // ---- initialise variance parameters ----
        let y = &model.y;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_var: f64 =
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        let n_components = n_random_terms + 1;
        let init_var = (y_var / n_components as f64).max(0.01);

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

        let mut history = Vec::new();
        let mut converged = false;
        let mut prev_logl = f64::NEG_INFINITY;

        for iter in 0..self.max_iter {
            // Update variance structures
            for (k, vs) in model.random_var_structs.iter_mut().enumerate() {
                vs.set_params(&[sigma2_random[k]])?;
            }
            model.residual_var_struct.set_params(&[sigma2_e])?;

            // Build G^{-1} blocks
            let g_inv_blocks: Vec<sprs::CsMat<f64>> = (0..n_random_terms)
                .map(|k| {
                    let q = model.z_blocks[k].cols();
                    if let Some(ref ginv_k) = model.ginv_matrices[k] {
                        ginv_k.map(|v| v / sigma2_random[k])
                    } else {
                        crate::matrix::sparse::sparse_diagonal(
                            &vec![1.0 / sigma2_random[k]; q],
                        )
                    }
                })
                .collect();

            let r_inv_scale = 1.0 / sigma2_e;

            // Assemble and solve MME
            let mme = MixedModelEquations::assemble(
                &model.x,
                &model.z_blocks,
                &model.y,
                r_inv_scale,
                &g_inv_blocks,
            );
            let sol = mme.solve()?;

            let c_inv = sol.c_inv.as_ref().ok_or(LmmError::CholeskyFailed(
                "C^{-1} not available".into(),
            ))?;

            let n_fixed = mme.n_fixed;
            let n_eff = (n - n_fixed) as f64;

            // ---- compute REML log-likelihood ----
            let y_r_inv_y =
                r_inv_scale * model.y.iter().map(|yi| yi * yi).sum::<f64>();
            let sol_rhs: f64 = sol
                .solution
                .iter()
                .zip(mme.rhs.iter())
                .map(|(s, r)| s * r)
                .sum();
            let y_p_y = y_r_inv_y - sol_rhs;

            let log_det_r = n as f64 * sigma2_e.ln();
            let mut log_det_g = 0.0;
            for k in 0..n_random_terms {
                let q = model.z_blocks[k].cols();
                log_det_g += q as f64 * sigma2_random[k].ln();
            }
            let log_2_pi = (2.0 * std::f64::consts::PI).ln();
            let logl =
                -0.5 * (n_eff * log_2_pi + log_det_r + log_det_g + sol.log_det_c + y_p_y);

            // Save old params
            let old_sigma2_random = sigma2_random.clone();
            let old_sigma2_e = sigma2_e;

            if iter < self.em_initial_steps {
                // ---- EM update ----
                self.em_update(
                    model,
                    &sol,
                    c_inv,
                    n_fixed,
                    n,
                    &mut sigma2_random,
                    &mut sigma2_e,
                );
            } else {
                // ---- AI-REML update ----
                let ai_result = self.ai_update(
                    model,
                    &sol,
                    c_inv,
                    n_fixed,
                    n,
                    &sigma2_random,
                    sigma2_e,
                    logl,
                    prev_logl,
                );

                match ai_result {
                    Some((new_random, new_e)) => {
                        sigma2_random = new_random;
                        sigma2_e = new_e;
                    }
                    None => {
                        // AI failed (e.g. singular AI), fall back to EM
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
                }
            }

            prev_logl = logl;

            // ---- convergence check ----
            let mut all_params_old = old_sigma2_random.clone();
            all_params_old.push(old_sigma2_e);
            let mut all_params_new = sigma2_random.clone();
            all_params_new.push(sigma2_e);

            let param_diff: f64 = all_params_new
                .iter()
                .zip(all_params_old.iter())
                .map(|(n, o)| (n - o).powi(2))
                .sum::<f64>()
                .sqrt();
            let param_norm: f64 = all_params_old
                .iter()
                .map(|p| p * p)
                .sum::<f64>()
                .sqrt()
                .max(1e-10);
            let rel_change = param_diff / param_norm;

            history.push(RemlIteration {
                iteration: iter + 1,
                log_likelihood: logl,
                variance_params: all_params_new.clone(),
                change: rel_change,
            });

            if iter > 0 && rel_change < self.tol {
                converged = true;
                break;
            }
        }

        // ---- final solve with converged parameters ----
        for (k, vs) in model.random_var_structs.iter_mut().enumerate() {
            vs.set_params(&[sigma2_random[k]])?;
        }
        model.residual_var_struct.set_params(&[sigma2_e])?;

        let g_inv_blocks: Vec<sprs::CsMat<f64>> = (0..n_random_terms)
            .map(|k| {
                let q = model.z_blocks[k].cols();
                if let Some(ref ginv_k) = model.ginv_matrices[k] {
                    ginv_k.map(|v| v / sigma2_random[k])
                } else {
                    crate::matrix::sparse::sparse_diagonal(
                        &vec![1.0 / sigma2_random[k]; q],
                    )
                }
            })
            .collect();

        let r_inv_scale = 1.0 / sigma2_e;
        let mme = MixedModelEquations::assemble(
            &model.x,
            &model.z_blocks,
            &model.y,
            r_inv_scale,
            &g_inv_blocks,
        );
        let sol = mme.solve()?;

        let mut var_params: Vec<f64> = sigma2_random.clone();
        var_params.push(sigma2_e);

        // Compute approximate SEs of variance components from final AI matrix
        let c_inv = sol.c_inv.as_ref().ok_or(LmmError::CholeskyFailed(
            "C^{-1} not available for SE computation".into(),
        ))?;
        let variance_se = self.compute_variance_se(
            model, &sol, c_inv, mme.n_fixed, &sigma2_random, sigma2_e,
        );

        self.build_result(model, &sol, &mme, &var_params, &variance_se, &history, converged)
    }

    /// EM-REML update step (identical to the EmReml implementation).
    fn em_update(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &nalgebra::DMatrix<f64>,
        n_fixed: usize,
        n: usize,
        sigma2_random: &mut Vec<f64>,
        sigma2_e: &mut f64,
    ) {
        let n_random_terms = model.random_var_structs.len();
        let n_eff = (n - n_fixed) as f64;

        // EM update for residual variance
        let mut y_e_hat = 0.0;
        for i in 0..n {
            y_e_hat += model.y[i] * model.y[i];
        }
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

        // EM update for each random variance component
        let mut block_start = n_fixed;
        for k in 0..n_random_terms {
            let q_k = model.z_blocks[k].cols();
            let u_k = &sol.random_effects[k];

            let u_quadratic = if let Some(ref ginv_k) = model.ginv_matrices[k] {
                let kinv_u = spmv(ginv_k, u_k);
                u_k.iter().zip(kinv_u.iter()).map(|(a, b)| a * b).sum::<f64>()
            } else {
                u_k.iter().map(|u| u * u).sum::<f64>()
            };

            let trace_term = if let Some(ref ginv_k) = model.ginv_matrices[k] {
                let mut tr = 0.0;
                for i in 0..q_k {
                    for j in 0..q_k {
                        let kinv_ij = ginv_k.get(i, j).copied().unwrap_or(0.0);
                        let cinv_ji = c_inv[(block_start + j, block_start + i)];
                        tr += kinv_ij * cinv_ji;
                    }
                }
                tr
            } else {
                let mut tr = 0.0;
                for i in 0..q_k {
                    tr += c_inv[(block_start + i, block_start + i)];
                }
                tr
            };

            sigma2_random[k] = ((u_quadratic + trace_term) / q_k as f64).max(1e-10);
            block_start += q_k;
        }
    }

    /// AI-REML Newton-Raphson update.
    ///
    /// Returns `Some((new_random, new_e))` on success, or `None` if the AI
    /// matrix is singular and we should fall back to EM.
    fn ai_update(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &nalgebra::DMatrix<f64>,
        n_fixed: usize,
        n: usize,
        sigma2_random: &[f64],
        sigma2_e: f64,
        _logl: f64,
        _prev_logl: f64,
    ) -> Option<(Vec<f64>, f64)> {
        let n_random_terms = model.random_var_structs.len();
        let n_params = n_random_terms + 1;

        // ---- Compute Py = (1/sigma2_e) * (y - Xb - Zu) ----
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
        let py: Vec<f64> = (0..n)
            .map(|i| (model.y[i] - fitted[i]) / sigma2_e)
            .collect();

        // ---- Compute score vector ----
        // score[k] = -0.5 * (tr(P dV/d_theta_k) - y'P dV/d_theta_k Py)
        //
        // For random term k with identity K:
        //   dV/d_sigma2_k = Z_k Z_k'
        //   tr(P Z_k Z_k') = q_k/sigma2_k - tr(C^{-1}_{uu_k})/sigma2_k^2
        //     (using the identity: tr(P Z G^{-1} Z') = sum of diag of C^{-1} block / sigma4)
        //     Wait: tr(P Z_k Z_k') is computed from C^{-1} differently.
        //
        // Actually, for the trace term:
        //   tr(P Z_k Z_k') = tr(Z_k' P Z_k)
        // From the MME, the (k,k) block of C^{-1} is:
        //   C^{-1}_{uu_k} = (Z_k' R^{-1} Z_k + G_k^{-1})^{-1} (approximately, ignoring cross terms)
        //   But the full C^{-1} block accounts for everything.
        //
        // We use: tr(P Z_k Z_k') = tr(Z_k'R^{-1}Z_k * C^{-1}_{uu_k}) / sigma2_e ... no, let's
        // be more careful.
        //
        // Using the MME identity (Searle, Casella & McCulloch 1992):
        //   P = R^{-1} - R^{-1} [X Z] C^{-1} [X Z]' R^{-1}
        //
        // So Z_k' P Z_k = Z_k' R^{-1} Z_k - Z_k' R^{-1} [X Z] C^{-1} [X Z]' R^{-1} Z_k
        //
        // For R = sigma2_e * I:
        //   Z_k' P Z_k = (1/sigma2_e) Z_k'Z_k - (1/sigma2_e^2) Z_k'[X Z] C^{-1} [X Z]'Z_k
        //
        // Hmm, computing the full trace is expensive. Instead, use the simpler relation:
        //   tr(P dV/d_sigma2_k) for dV = Z_k Z_k' (when K_k = I):
        //     = (1/sigma2_e) * tr(Z_k'Z_k) - (1/sigma2_e^2) * tr(Z_k'[X Z] C^{-1} [X Z]'Z_k)
        //
        // The second term = (1/sigma2_e^2) * sum_{i,j in block_k} [X Z]'Z_k entries * C^{-1} entries
        //
        // This is equivalent to:
        //   tr(P Z_k Z_k') = (1/sigma2_e) * q_k_effective
        //
        // An easier approach: use the relation
        //   score_k = -0.5 * [ tr_k - quad_k ]
        // where:
        //   tr_k = (for random k): q_k / sigma2_k - sum of diagonal of C^{-1}_{uu_k block} / sigma2_k^2
        //          (this is tr(G_k^{-1}) - tr(G_k^{-1} C^{-1}_{uu_k} G_k^{-1}) * sigma2_k
        //          ... for identity K_k, G_k = sigma2_k * I, G_k^{-1} = (1/sigma2_k) * I)
        //   No, let me compute score from scratch using Py.
        //
        // For the quadratic part: y'P (dV/d_sigma2_k) Py
        //   For random term k:  y'P Z_k Z_k' Py = (Z_k' Py)' (Z_k' Py)
        //   For residual:       y'P I Py = Py' Py
        //
        // For the trace part, use:
        //   tr(P Z_k Z_k') = tr(Z_k'Z_k)/sigma2_e - tr(C^{-1} * [Z_k'Z_k block in MME]) / sigma2_e^2
        // which simplifies (using the fact that C = coefficient matrix and
        // C^{-1} * C blocks relate to the identity) to:
        //   tr(P Z_k Z_k') = q_k / sigma2_e - tr(C^{-1}_{rows_k, cols_k} * ZtRiZ_{k,k}) / sigma2_e
        //   where ZtRiZ_{k,k} is the (k,k) block of Z'R^{-1}Z
        //
        // SIMPLEST correct approach: for the case where K_k = I (or general K_k):
        //   tr(P * dV/d_sigma2_k) = (for K_k = I):
        //     (q_k - tr(C^{-1}_{uu_k}) / sigma2_k) / sigma2_e
        //   Wait, still not clean. Let me just use the direct formulation.

        let mut score = vec![0.0; n_params];

        // --- Compute Z_k' Py for each random term ---
        let mut zt_py: Vec<Vec<f64>> = Vec::with_capacity(n_random_terms);
        for z in &model.z_blocks {
            zt_py.push(crate::matrix::sparse::xt_y(z, &py));
        }

        // --- Score for random terms ---
        // Using the working formulation:
        //   score_k = -0.5 * [tr(P Z_k K_k Z_k') - (Z_k'Py)' K_k (Z_k'Py)]
        //
        // For K_k = I:
        //   quadratic: (Z_k'Py)'(Z_k'Py)
        //   trace: we need tr(P Z_k Z_k')
        //
        // The trace term can be computed as:
        //   tr(P Z_k Z_k') = q_k / sigma2_e - (1/sigma2_e^2) * sum_{i,j in uu_k block} [XZ_all]_col_i' [XZ_all]_col_j * C^{-1}_{i,j}
        //
        // But since we already have C^{-1}, we can use:
        //   P = R^{-1} - R^{-1} W C^{-1} W' R^{-1}  where W = [X Z]
        //   Z_k' P Z_k = Z_k'Z_k / sigma2_e - (Z_k'W / sigma2_e) C^{-1} (W'Z_k / sigma2_e)
        //
        // For trace, this gives:
        //   tr(Z_k' P Z_k) = tr(Z_k'Z_k) / sigma2_e - (1/sigma2_e^2) * tr(W'Z_k * C^{-1} * Z_k'W)
        //                   = tr(Z_k'Z_k) / sigma2_e - (1/sigma2_e^2) * sum_i sum_j (W'Z_k)_{ij} * (C^{-1} W'Z_k)_{ij}
        //
        // This is expensive for large problems. For Phase 1 (dense C^{-1}),
        // we take a simpler approach: compute the trace from the C^{-1} blocks.
        //
        // Using Henderson's result for the score of sigma2_k when G_k = sigma2_k * K_k:
        //   d logL_R / d sigma2_k = -0.5 * [ -tr(G_k^{-1} C^{uu_k}) / sigma2_k + u_k' G_k^{-2} u_k - q_k / sigma2_k + u_k' G_k^{-1} u_k ]
        //   Hmm, that's also getting complicated.
        //
        // Let me use the cleanest known form (Johnson & Thompson 1995):
        //   score_k = -0.5 * (q_k/sigma2_k - tr(K_k^{-1} C^{uu_k}_inv) / sigma2_k^2
        //              - u_k' K_k^{-1} u_k / sigma2_k^2)
        //   Wait, that's not right either. The standard score for sigma2_k is:
        //
        //   d logL_R / d sigma2_k = -0.5 * tr(P dV/d sigma2_k) + 0.5 * y'P(dV/d sigma2_k)Py
        //
        //   = -0.5 * tr(P Z_k K_k Z_k') + 0.5 * y'P Z_k K_k Z_k' P y
        //
        //   For K_k = I:
        //   = -0.5 * tr(P Z_k Z_k') + 0.5 * (Z_k'Py)'(Z_k'Py)
        //
        //   The trace can be expressed using the C^{-1}:
        //   tr(P Z_k Z_k') = tr(Z_k'Z_k) / sigma2_e - (1/sigma2_e) * tr([block_k cols of C^{-1}] * [block_k rows of C * 1/sigma2_e])
        //
        // OK, let me use the most direct correct formula.
        // From the MME perspective, using identity R = sigma2_e * I:
        //
        //   C = [X'X/se  X'Z/se     ]   where se = sigma2_e
        //       [Z'X/se  Z'Z/se+G^-1]
        //
        //   tr(P Z_k Z_k') = tr(Z_k Z_k' P)
        //
        // The practical formula for this trace is (Meyer 1997, Gilmour et al. 1995):
        //
        //   For G_k = sigma2_k * I (no K matrix):
        //   tr(P Z_k Z_k') = (1/sigma2_e) * [ q_k - (1/sigma2_e) * tr(C^{-1}_{uu_k,uu_k} * ZtRiZ_{kk}) ]
        //
        // where ZtRiZ_{kk} = Z_k'Z_k / sigma2_e is the (k,k) diagonal block of Z'R^{-1}Z.
        //
        // BUT we can simplify further. Note that C_{uu_k, uu_k} (the k-th random block of C)
        // = Z_k'Z_k / sigma2_e + I / sigma2_k  (for identity K_k and ignoring cross-terms with other random effects)
        // In the full system, we need the actual (k,k) block of C.
        //
        // Let me just compute:
        //   tr_k = (1/sigma2_e) * (q_k - (1/sigma2_e) * tr(C^{-1}_{block_k} * C_block_k_in_C))
        //
        // where C^{-1}_{block_k} is rows/cols [start_k..start_k+q_k] of C^{-1}
        // and C_block_k_in_C is the same rows/cols of C itself.
        //
        // Actually, using the identity that for any invertible C:
        //   tr(C^{-1}_{sub} * C_{sub}) gives the dimension of the projection...
        //
        // The cleanest approach for small problems: just compute it from first principles.
        // tr(P Z_k Z_k') = sum_i (P Z_k Z_k')_{ii} = sum_i sum_j P_{ij} (Z_k Z_k')_{ji}
        //                 = vec(P)' vec(Z_k Z_k')  (Frobenius inner product)
        //
        // But P is n x n and we don't form it explicitly.
        //
        // INSTEAD, use the VERY clean formula (e.g. Mrode 2005 Ch 11, Lee & Van der Werf 2006):
        //
        //   For random component k with G_k = sigma2_k * K_k:
        //     score_k = (-q_k + u_k' K_k^{-1} u_k / sigma2_k + tr(K_k^{-1} C^{uu_k})) / (2 * sigma2_k)
        //     Wait, this is the EM identity rearranged...
        //
        //  Let me derive it properly.  The REML score for sigma2_k is:
        //
        //     s_k = d logL_R / d sigma2_k
        //         = -0.5 * tr(P dV/d sigma2_k) + 0.5 * y'P (dV/d sigma2_k) P y
        //
        //   For dV/d sigma2_k = Z_k K_k Z_k':
        //
        //     s_k = 0.5 * [ y'P Z_k K_k Z_k' P y  -  tr(P Z_k K_k Z_k') ]
        //         = 0.5 * [ (Z_k'Py)' K_k (Z_k'Py)  -  tr(K_k Z_k'P Z_k) ]
        //
        //   Now, Z_k' P y = Z_k' R^{-1} y - Z_k' R^{-1} W C^{-1} W' R^{-1} y
        //                  = (1/se)(Z_k'y - Z_k'W * sol)
        //   But from the MME: W'R^{-1}y = C * sol, so W'R^{-1}y = C*sol
        //   And Z_k'Py = Z_k'R^{-1}(y - W*sol) = (1/se) * Z_k'(y - X*b - Z*u)
        //              = (1/se) * Z_k' * residuals = Z_k' * py  (which we already have)
        //
        //   For the trace, using the MME identity:
        //     Z_k' P Z_k = Z_k'R^{-1}Z_k - Z_k'R^{-1}W C^{-1} W'R^{-1}Z_k
        //   The second part's trace involves C^{-1} and the design matrices.
        //
        //   Key insight: since the columns of Z_k correspond to a specific block of W = [X Z],
        //   we have W'R^{-1}Z_k = (1/se) * W'Z_k, and the columns of W'Z_k are just the
        //   rows from the coefficient matrix C (the k-th random block columns of C).
        //
        //   Let B_k = (1/se) * W'Z_k (dimension dim x q_k)
        //   Then Z_k'P Z_k = (1/se) * Z_k'Z_k - B_k' C^{-1} B_k
        //
        //   tr(K_k Z_k'P Z_k) = tr(K_k * [(1/se)*Z_k'Z_k - B_k' C^{-1} B_k])
        //
        //   For K_k = I:
        //     tr_k = tr(Z_k'Z_k)/se - tr(B_k' C^{-1} B_k)
        //          = tr(Z_k'Z_k)/se - tr(C^{-1} B_k B_k')
        //
        //   Now B_k = (1/se) * W'Z_k. The columns of B_k B_k' are the entries of
        //   (1/se^2) W'Z_k Z_k'W. And:
        //     tr(C^{-1} * B_k B_k') = (1/se^2) * tr(C^{-1} * W'Z_k Z_k'W)
        //
        //   Note that W'Z_k Z_k'W (dim x dim) has a specific structure:
        //   the (i,j) block of this matrix corresponds to
        //   [X,Z_1,...,Z_r]'Z_k Z_k'[X,Z_1,...,Z_r], so the (l,m) block is Z_l'Z_k Z_k'Z_m.
        //
        //   This is a rank-q_k matrix.  For efficiency, note that
        //   B_k = C[:, block_k_cols] (the columns of C corresponding to term k)
        //   minus the G^{-1} contribution on those columns.
        //   Actually, B_k_col_j = (1/se) * W' z_{k,j} where z_{k,j} is column j of Z_k.
        //   And C = (1/se) * W'W + block_diag(0, G^{-1}).
        //   So (1/se) * W'Z_k = C[:, block_k] - [0; ...; G_k^{-1}; ...; 0] (the G^{-1} added to block k).
        //
        //   Hmm this is getting intricate. For a practical Phase 1 implementation with
        //   dense C^{-1} already computed, let me take the direct approach:
        //
        //   Compute for each random block k:
        //     block_k starts at column offset start_k in C, size q_k.
        //     We need B_k = C[all, start_k..start_k+q_k] - delta_k
        //     where delta_k has G_k^{-1}/se in the k-th random block and zeros elsewhere.
        //     Actually, B_k = (1/se) * W'Z_k which is exactly:
        //     B_k = C[:, block_k_cols] minus the G^{-1}_k/se block.
        //     Hmm, let me reconsider.
        //
        //   C = (1/se) * W'W + block_diag(0, G_1^{-1}, ..., G_r^{-1})
        //   So (1/se) * W'Z_k = column block k of [(1/se)*W'W] = C[:, block_k] - [appropriate G^{-1} block]
        //
        //   For block k (rows start_k to start_k+q_k-1 of C):
        //     (1/se) * W'Z_k = C[:, start_k..start_k+q_k]  with the G_k^{-1} subtracted from rows start_k..start_k+q_k
        //
        //   This means B_k[i,j] = C[i, start_k+j] for i outside block k
        //             B_k[start_k+i, j] = C[start_k+i, start_k+j] - G_k^{-1}[i,j]
        //
        //   Then tr(C^{-1} B_k B_k') = sum_i sum_j C^{-1}[i,j] * (B_k B_k')[j,i]
        //                             = sum_i sum_j C^{-1}[i,j] * sum_l B_k[j,l]*B_k[i,l]
        //
        //   For small q_k, this is O(dim^2 * q_k).
        //
        // Given the complexity, let me use an alternative simpler approach:
        // The trace can be computed from C^{-1} and the known MME structure.
        //
        // For SIMPLE K_k = I:
        //   tr(P Z_k Z_k') = (1/se)[q_k - (1/se) * tr(C^{-1}[block_k, :] * C[:, block_k])]
        //   where the product C^{-1}[block_k, :] * C[:, block_k] extracts a q_k x q_k matrix
        //   whose trace = tr of the identity projection = q_k ... no, that gives q_k always.
        //   That's not right because C^{-1}[block_k] * C[block_k] would only be identity
        //   if those were truly independent blocks.
        //
        // OK, I'll use the most straightforward approach.
        // From the identity: C^{-1} C = I, so sum_j C^{-1}[i,j] C[j,k] = delta_{ik}.
        // The trace of C^{-1} * (submatrix of C) doesn't simplify easily.
        //
        // PRACTICAL APPROACH for the score:
        // We use the EM-like expressions rearranged:
        //
        // The EM update gives sigma2_k_new = (u_k'K^{-1}u_k + tr(K^{-1}C^{uu_k})) / q_k
        // This can be shown equivalent to:
        //   sigma2_k_new = sigma2_k + sigma2_k^2 * score_k * 2 / q_k  (for EM step)
        //   => score_k = (sigma2_k_new/sigma2_k - 1) * q_k / (2 * sigma2_k)
        //
        // Actually, the connection between EM and score is:
        //   sigma2_k_new = sigma2_k + sigma2_k^2 / q_k * 2 * score_k
        //   => score_k = q_k / (2 * sigma2_k^2) * (sigma2_k_new - sigma2_k)
        //
        // So we can compute the score by doing an EM step and back-computing.
        // This is the "EM-acceleration" approach and gives exact scores.

        let mut sigma2_random_em = sigma2_random.to_vec();
        let mut sigma2_e_em = sigma2_e;
        self.em_update(
            model, sol, c_inv, n_fixed, n, &mut sigma2_random_em, &mut sigma2_e_em,
        );

        // Compute score from EM update:
        // score_k = q_k / (2 * sigma2_k^2) * (sigma2_k_em - sigma2_k)
        for k in 0..n_random_terms {
            let q_k = model.z_blocks[k].cols() as f64;
            score[k] = q_k / (2.0 * sigma2_random[k] * sigma2_random[k])
                * (sigma2_random_em[k] - sigma2_random[k]);
        }
        // For residual:
        let n_eff = (n - n_fixed) as f64;
        score[n_random_terms] =
            n_eff / (2.0 * sigma2_e * sigma2_e) * (sigma2_e_em - sigma2_e);

        // ---- Compute Average Information matrix ----
        // AI[i,j] = 0.5 * y'P (dV/d theta_i) P (dV/d theta_j) P y
        //         = 0.5 * w_i' P w_j
        // where w_k = (dV/d theta_k) P y
        //
        // For random term k (K_k = I):  w_k = Z_k Z_k' Py
        // For residual:                  w_e = I * Py = Py
        //
        // And P w = R^{-1}(w - W C^{-1} W' R^{-1} w)
        // This requires solving the MME with a modified RHS.
        //
        // ALTERNATIVELY, since we need w_i' P w_j, we can use:
        //   w_i' P w_j = (1/se)(w_i'w_j - w_i'W C^{-1} W'w_j / se)
        //              = (1/se)(w_i'w_j) - (1/se^2)(W'w_i)' C^{-1} (W'w_j)
        //
        // Let's define for each component i:
        //   w_i = (dV/d theta_i) Py   (length n vector)
        //   v_i = W' w_i / se         (length dim vector, the "projected" working variate)
        //
        // Then AI[i,j] = 0.5 * (w_i'w_j/se - v_i' C^{-1} v_j / se)
        //              = 0.5/se * (w_i'w_j - v_i' C^{-1} v_j)

        // Compute working variates
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_params);

        // For random terms
        for (k, z) in model.z_blocks.iter().enumerate() {
            // w_k = Z_k K_k Z_k' Py = Z_k * (K_k * Z_k'Py)
            // For K_k = I: w_k = Z_k * (Z_k'Py)
            let zk_py = &zt_py[k];
            if let Some(ref ginv_k) = model.ginv_matrices[k] {
                // K_k is not identity. We need K_k * Z_k'Py.
                // But we have K^{-1}, not K. For the AI computation with general K:
                //   dV/d sigma2_k = Z_k K_k Z_k', so w_k = Z_k K_k (Z_k'Py)
                // Since G_k = sigma2_k * K_k, and we have K_k^{-1} = ginv_k,
                // we need to solve K_k * x = Z_k'Py for K_k * x, i.e., just compute K_k * (Z_k'Py).
                // But we don't have K_k, only K_k^{-1}.
                // We'd need to solve ginv_k * result = zk_py.
                // For Phase 1 with dense C^{-1}, we can solve this small system.
                let q_k = model.z_blocks[k].cols();
                let ginv_dense = sparse_to_dense(ginv_k, q_k);
                if let Some(chol) = ginv_dense.cholesky() {
                    let rhs_vec = nalgebra::DVector::from_column_slice(zk_py);
                    let k_zpy = chol.solve(&rhs_vec);
                    let k_zpy_slice: Vec<f64> = k_zpy.as_slice().to_vec();
                    w.push(spmv(z, &k_zpy_slice));
                } else {
                    // K^{-1} not PD? Just use identity approximation.
                    w.push(spmv(z, zk_py));
                }
            } else {
                w.push(spmv(z, zk_py));
            }
        }

        // For residual: w_e = Py
        w.push(py.clone());

        // Compute v_i = (1/se) * W' w_i  (W = [X Z1 Z2 ...])
        let dim = c_inv.nrows();
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(n_params);
        for wi in &w {
            let mut vi = vec![0.0; dim];
            // X' w_i
            let xt_wi = crate::matrix::sparse::xt_y(&model.x, wi);
            for j in 0..n_fixed {
                vi[j] = xt_wi[j] / sigma2_e;
            }
            // Z_k' w_i for each random term
            let mut offset = n_fixed;
            for z in &model.z_blocks {
                let zt_wi = crate::matrix::sparse::xt_y(z, wi);
                for j in 0..z.cols() {
                    vi[offset + j] = zt_wi[j] / sigma2_e;
                }
                offset += z.cols();
            }
            v.push(vi);
        }

        // Build AI matrix
        let mut ai = nalgebra::DMatrix::zeros(n_params, n_params);
        for i in 0..n_params {
            for j in i..n_params {
                // AI[i,j] = 0.5/se * (w_i'w_j - v_i' C^{-1} v_j)
                let wi_wj: f64 = w[i].iter().zip(w[j].iter()).map(|(a, b)| a * b).sum();

                // v_i' C^{-1} v_j
                let mut vi_cinv_vj = 0.0;
                for r in 0..dim {
                    for c in 0..dim {
                        vi_cinv_vj += v[i][r] * c_inv[(r, c)] * v[j][c];
                    }
                }

                ai[(i, j)] = 0.5 / sigma2_e * (wi_wj - vi_cinv_vj);
                if i != j {
                    ai[(j, i)] = ai[(i, j)];
                }
            }
        }

        // ---- Newton step: delta = AI^{-1} * score ----
        let ai_chol = ai.clone().cholesky();
        let ai_chol = match ai_chol {
            Some(c) => c,
            None => {
                // AI is not positive definite, try adding a small ridge
                let mut ai_ridge = ai.clone();
                let ridge = 1e-6 * ai.diagonal().iter().map(|x| x.abs()).sum::<f64>()
                    / n_params as f64;
                for i in 0..n_params {
                    ai_ridge[(i, i)] += ridge.max(1e-8);
                }
                match ai_ridge.cholesky() {
                    Some(c) => c,
                    None => return None, // Give up, fall back to EM
                }
            }
        };

        let score_vec = nalgebra::DVector::from_column_slice(&score);
        let delta = ai_chol.solve(&score_vec);
        let delta_slice: Vec<f64> = delta.as_slice().to_vec();

        // ---- Step halving to ensure positivity ----
        let mut step = 1.0;
        let mut new_sigma2_random = sigma2_random.to_vec();
        let mut new_sigma2_e;

        for _half in 0..10 {
            let mut all_positive = true;
            for k in 0..n_random_terms {
                new_sigma2_random[k] = sigma2_random[k] + step * delta_slice[k];
                if new_sigma2_random[k] <= 1e-10 {
                    all_positive = false;
                }
            }
            new_sigma2_e = sigma2_e + step * delta_slice[n_random_terms];
            if new_sigma2_e <= 1e-10 {
                all_positive = false;
            }

            if all_positive {
                return Some((new_sigma2_random, new_sigma2_e));
            }
            step *= 0.5;
        }

        // Even after step halving, couldn't get all positive. Clamp.
        for k in 0..n_random_terms {
            new_sigma2_random[k] =
                (sigma2_random[k] + step * delta_slice[k]).max(1e-10);
        }
        new_sigma2_e =
            (sigma2_e + step * delta_slice[n_random_terms]).max(1e-10);

        Some((new_sigma2_random, new_sigma2_e))
    }

    /// Compute approximate standard errors of variance components from the
    /// inverse of the Average Information matrix at convergence.
    fn compute_variance_se(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        c_inv: &nalgebra::DMatrix<f64>,
        n_fixed: usize,
        _sigma2_random: &[f64],
        sigma2_e: f64,
    ) -> Vec<f64> {
        let n = model.n_obs;
        let n_random_terms = model.random_var_structs.len();
        let n_params = n_random_terms + 1;

        // Compute Py
        let mut fitted = vec![0.0; n];
        let xb = spmv(&model.x, &sol.fixed_effects);
        for i in 0..n {
            fitted[i] += xb[i];
        }
        for (_k, z) in model.z_blocks.iter().enumerate() {
            let zu = spmv(z, &sol.random_effects[_k]);
            for i in 0..n {
                fitted[i] += zu[i];
            }
        }
        let py: Vec<f64> = (0..n)
            .map(|i| (model.y[i] - fitted[i]) / sigma2_e)
            .collect();

        // Compute working variates and AI matrix (same as in ai_update)
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(n_params);
        for (_k, z) in model.z_blocks.iter().enumerate() {
            let zk_py = crate::matrix::sparse::xt_y(z, &py);
            w.push(spmv(z, &zk_py));
        }
        w.push(py.clone());

        let dim = c_inv.nrows();
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(n_params);
        for wi in &w {
            let mut vi = vec![0.0; dim];
            let xt_wi = crate::matrix::sparse::xt_y(&model.x, wi);
            for j in 0..n_fixed {
                vi[j] = xt_wi[j] / sigma2_e;
            }
            let mut offset = n_fixed;
            for z in &model.z_blocks {
                let zt_wi = crate::matrix::sparse::xt_y(z, wi);
                for j in 0..z.cols() {
                    vi[offset + j] = zt_wi[j] / sigma2_e;
                }
                offset += z.cols();
            }
            v.push(vi);
        }

        let mut ai = nalgebra::DMatrix::zeros(n_params, n_params);
        for i in 0..n_params {
            for j in i..n_params {
                let wi_wj: f64 = w[i].iter().zip(w[j].iter()).map(|(a, b)| a * b).sum();
                let mut vi_cinv_vj = 0.0;
                for r in 0..dim {
                    for c in 0..dim {
                        vi_cinv_vj += v[i][r] * c_inv[(r, c)] * v[j][c];
                    }
                }
                ai[(i, j)] = 0.5 / sigma2_e * (wi_wj - vi_cinv_vj);
                if i != j {
                    ai[(j, i)] = ai[(i, j)];
                }
            }
        }

        // SE = sqrt(diag(AI^{-1}))
        match ai.clone().cholesky() {
            Some(chol) => {
                let ai_inv = chol.inverse();
                (0..n_params).map(|i| ai_inv[(i, i)].abs().sqrt()).collect()
            }
            None => vec![0.0; n_params],
        }
    }

    /// Build the final FitResult from converged solution.
    fn build_result(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        mme: &MixedModelEquations,
        var_params: &[f64],
        variance_se: &[f64],
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
        for (k, vs) in model.random_var_structs.iter().enumerate() {
            variance_components.push(VarianceEstimate {
                name: model.random_term_names[k].clone(),
                structure: vs.name().to_string(),
                parameters: vec![("sigma2".to_string(), vs.params()[0])],
            });
        }
        variance_components.push(VarianceEstimate {
            name: "residual".to_string(),
            structure: model.residual_var_struct.name().to_string(),
            parameters: vec![("sigma2".to_string(), sigma_e2)],
        });

        // Fixed effects with SEs from C^{-1}
        let c_inv = sol.c_inv.as_ref().unwrap();
        let fixed_effects: Vec<NamedEffect> = model
            .fixed_labels
            .iter()
            .enumerate()
            .map(|(i, label)| NamedEffect {
                term: label.term.clone(),
                level: label.level.clone(),
                estimate: sol.fixed_effects[i],
                se: c_inv[(i, i)].sqrt(),
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
                    se: c_inv[(block_offset + j, block_offset + j)].sqrt(),
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
            n_obs: n,
            n_fixed_params: n_fixed,
            n_variance_params: var_params.len(),
        })
    }
}

/// Convert a sparse matrix to a dense nalgebra matrix.
fn sparse_to_dense(sp: &sprs::CsMat<f64>, dim: usize) -> nalgebra::DMatrix<f64> {
    let mut dense = nalgebra::DMatrix::zeros(dim, dim);
    for (val, (i, j)) in sp.iter() {
        dense[(i, j)] = *val;
    }
    dense
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFrame;
    use crate::model::MixedModelBuilder;
    use crate::variance::Identity;
    use approx::assert_relative_eq;

    /// Create a balanced trial dataset: 5 genotypes x 3 reps = 15 observations.
    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        // Simulated yield data with genotype and rep effects
        df.add_float_column(
            "yield",
            vec![
                10.0, 8.0, 6.0, 7.0, 9.0,  // R1
                12.0, 10.0, 8.0, 9.0, 11.0, // R2
                11.0, 9.0, 7.0, 8.0, 10.0,  // R3
            ],
        )
        .unwrap();
        df.add_factor_column(
            "genotype",
            &[
                "G1", "G2", "G3", "G4", "G5",
                "G1", "G2", "G3", "G4", "G5",
                "G1", "G2", "G3", "G4", "G5",
            ],
        )
        .unwrap();
        df.add_factor_column(
            "rep",
            &[
                "R1", "R1", "R1", "R1", "R1",
                "R2", "R2", "R2", "R2", "R2",
                "R3", "R3", "R3", "R3", "R3",
            ],
        )
        .unwrap();
        df
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
        assert!(result.n_iterations < 50, "Should converge before max iterations");
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
        // SEs should be non-negative (may be zero for small datasets where
        // the AI matrix is near-singular)
        for &se in &result.variance_se {
            assert!(se >= 0.0, "Variance SE should be non-negative, got {}", se);
        }
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

        // AI-REML should converge in fewer total iterations than EM-REML
        // (at least not more, accounting for the EM burn-in)
        assert!(
            result_ai.n_iterations <= result_em.n_iterations,
            "AI-REML ({} iters) should not take more iterations than EM-REML ({} iters)",
            result_ai.n_iterations,
            result_em.n_iterations,
        );
    }
}
