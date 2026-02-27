use crate::error::{LmmError, Result};
use crate::matrix::sparse::spmv;
use crate::model::MixedModel;

use super::mme::MixedModelEquations;
use super::result::{
    FitResult, NamedEffect, RandomEffectBlock, RemlIteration, VarianceEstimate,
};

/// REML engine using EM algorithm for variance component estimation.
///
/// EM-REML update formulas (Henderson Method III / Mrode 2005):
///
///   sigma^2_k = (u_k'K_k^{-1}u_k + tr(K_k^{-1} C^{-1}_{uu_k})) / q_k
///   sigma^2_e = y'e_hat / (n - p)
///
/// where e_hat = y - X*b_hat - Z*u_hat and C^{-1} is the inverse of the
/// MME coefficient matrix.
pub struct AiReml {
    max_iter: usize,
    tol: f64,
}

impl AiReml {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Fit the model using EM-REML.
    pub fn fit(&self, model: &mut MixedModel) -> Result<FitResult> {
        let n = model.n_obs;
        let n_random_terms = model.random_var_structs.len();

        // Initialize variance parameters from data variance
        let y = &model.y;
        let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
        let y_var: f64 =
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;

        // Split total variance equally among components
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

        for iter in 0..self.max_iter {
            // Update variance structures with current parameters
            for (k, vs) in model.random_var_structs.iter_mut().enumerate() {
                vs.set_params(&[sigma2_random[k]])?;
            }
            model.residual_var_struct.set_params(&[sigma2_e])?;

            // Build G^{-1} blocks: G_k^{-1} = K_k^{-1} / sigma^2_k
            let g_inv_blocks: Vec<sprs::CsMat<f64>> = (0..n_random_terms)
                .map(|k| {
                    let q = model.z_blocks[k].cols();
                    if let Some(ref ginv_k) = model.ginv_matrices[k] {
                        ginv_k.map(|v| v / sigma2_random[k])
                    } else {
                        crate::matrix::sparse::sparse_diagonal(&vec![1.0 / sigma2_random[k]; q])
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

            // Save old params for convergence check
            let old_sigma2_random = sigma2_random.clone();
            let old_sigma2_e = sigma2_e;

            // --- EM update for residual variance ---
            // sigma^2_e = y'e_hat / (n - p) where e_hat = y - X*b - Z*u
            // Equivalently: y'e = y'y - b'X'y - sum_k u_k'Z_k'y
            let n_fixed = mme.n_fixed;
            let n_eff = (n - n_fixed) as f64;

            let mut y_e_hat = 0.0;
            for i in 0..n {
                y_e_hat += model.y[i] * model.y[i]; // y'y
            }
            // Subtract b'X'y
            let xty = crate::matrix::sparse::xt_y(&model.x, &model.y);
            for i in 0..n_fixed {
                y_e_hat -= sol.fixed_effects[i] * xty[i];
            }
            // Subtract u_k'Z_k'y
            for (k, z) in model.z_blocks.iter().enumerate() {
                let zty = crate::matrix::sparse::xt_y(z, &model.y);
                for j in 0..z.cols() {
                    y_e_hat -= sol.random_effects[k][j] * zty[j];
                }
            }

            sigma2_e = (y_e_hat / n_eff).max(1e-10);

            // --- EM update for each random variance component ---
            // sigma^2_k = (u_k'K_k^{-1}u_k + tr(K_k^{-1} C^{-1}_{uu_k})) / q_k
            let mut block_start = n_fixed;
            for k in 0..n_random_terms {
                let q_k = model.z_blocks[k].cols();
                let u_k = &sol.random_effects[k];

                // u_k' K_k^{-1} u_k
                let u_quadratic = if let Some(ref ginv_k) = model.ginv_matrices[k] {
                    // ginv_k = K^{-1}, so u'K^{-1}u
                    let kinv_u = spmv(ginv_k, u_k);
                    u_k.iter().zip(kinv_u.iter()).map(|(a, b)| a * b).sum::<f64>()
                } else {
                    // K = I, so u'u
                    u_k.iter().map(|u| u * u).sum::<f64>()
                };

                // tr(K_k^{-1} C^{-1}_{uu_k})
                let trace_term = if let Some(ref ginv_k) = model.ginv_matrices[k] {
                    // tr(K^{-1} C^{-1}_{uu}) = sum_{i,j} K^{-1}_{ij} C^{-1}_{uu,ji}
                    // For efficiency in Phase 1, use the full matrices
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
                    // K = I, so tr(C^{-1}_{uu}) = sum of diagonal
                    let mut tr = 0.0;
                    for i in 0..q_k {
                        tr += c_inv[(block_start + i, block_start + i)];
                    }
                    tr
                };

                sigma2_random[k] = ((u_quadratic + trace_term) / q_k as f64).max(1e-10);

                block_start += q_k;
            }

            // Compute log-likelihood for monitoring
            let y_r_inv_y = (1.0 / old_sigma2_e) * model.y.iter().map(|yi| yi * yi).sum::<f64>();
            let sol_rhs: f64 = sol
                .solution
                .iter()
                .zip(mme.rhs.iter())
                .map(|(s, r)| s * r)
                .sum();
            let y_p_y = y_r_inv_y - sol_rhs;

            let log_det_r = n as f64 * old_sigma2_e.ln();
            let mut log_det_g = 0.0;
            for k in 0..n_random_terms {
                let q = model.z_blocks[k].cols();
                log_det_g += q as f64 * old_sigma2_random[k].ln();
            }
            let log_2_pi = (2.0 * std::f64::consts::PI).ln();
            let logl =
                -0.5 * (n_eff * log_2_pi + log_det_r + log_det_g + sol.log_det_c + y_p_y);

            // Convergence criterion: relative change in parameters
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

        // Final solve with converged parameters
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
                    crate::matrix::sparse::sparse_diagonal(&vec![1.0 / sigma2_random[k]; q])
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

        self.build_result(model, &sol, &mme, &var_params, &history, converged)
    }

    /// Build the final FitResult from converged solution.
    fn build_result(
        &self,
        model: &MixedModel,
        sol: &super::mme::MmeSolution,
        mme: &MixedModelEquations,
        var_params: &[f64],
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
            variance_se: vec![0.0; var_params.len()],
            residuals,
            n_obs: n,
            n_fixed_params: n_fixed,
            n_variance_params: var_params.len(),
        })
    }
}
