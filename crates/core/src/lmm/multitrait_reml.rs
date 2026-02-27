use nalgebra::DMatrix;

use crate::error::{LmmError, Result};
use crate::lmm::result::NamedEffect;
use crate::matrix::sparse::spmv;
use crate::model::multitrait::MultiTraitModel;

/// Result of fitting a multi-trait mixed model via REML.
#[derive(Debug, Clone)]
pub struct MultiTraitFitResult {
    /// Estimated trait genetic covariance matrix (t x t).
    pub g0: DMatrix<f64>,
    /// Estimated trait residual covariance matrix (t x t).
    pub r0: DMatrix<f64>,
    /// Genetic correlations derived from G0.
    pub genetic_correlations: DMatrix<f64>,
    /// Fixed effects (BLUE) per trait: fixed_effects[trait_idx][effect_idx].
    pub fixed_effects: Vec<Vec<NamedEffect>>,
    /// Random effects (BLUP) per trait per random term:
    /// random_effects[trait_idx][term_idx] is a Vec<f64> of BLUPs.
    pub random_effects: Vec<Vec<Vec<f64>>>,
    /// REML log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Number of REML iterations performed.
    pub n_iterations: usize,
    /// Formatted summary string.
    pub summary: String,
}

/// EM-REML engine for multi-trait models with Kronecker-structured MME.
///
/// The multi-trait MME is assembled by expanding the single-trait design
/// matrices using Kronecker products:
///
/// ```text
/// X_mt = I_t kron X     (nt x tp)
/// Z_mt = I_t kron Z     (nt x tq)
/// R_mt^{-1} = R0^{-1} kron I_n
/// G_mt^{-1} = G0^{-1} kron K^{-1}   (or G0^{-1} kron I_q when K = I)
/// ```
///
/// The EM updates for the covariance matrices are:
///
/// ```text
/// G0_new[i,j] = (u_i' K^{-1} u_j + tr(K^{-1} C^{uu}_{ij})) / q
/// R0_new[i,j] = e_i' e_j / n    (simplified residual update)
/// ```
///
/// where u_i is the vector of BLUPs for trait i, e_i is the residual vector
/// for trait i, and C^{uu}_{ij} is the (i,j) block of the MME inverse
/// corresponding to random effects.
pub struct MultiTraitReml {
    max_iter: usize,
    tol: f64,
}

impl MultiTraitReml {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Fit the multi-trait model using EM-REML.
    pub fn fit(&self, model: &mut MultiTraitModel) -> Result<MultiTraitFitResult> {
        let t = model.n_traits;
        let n = model.n_obs;
        let p = model.x_single.cols(); // fixed effects per trait
        let n_random = model.z_single_blocks.len();

        // q_k: number of random levels per term
        let q_vec: Vec<usize> = model.z_single_blocks.iter().map(|z| z.cols()).collect();
        let q_total: usize = q_vec.iter().sum();

        // Total multi-trait dimensions
        let tp = t * p;
        let tq_total = t * q_total;
        let mt_dim = tp + tq_total;

        let mut g0 = model.g0.clone();
        let mut r0 = model.r0.clone();

        let mut converged = false;
        let mut n_iterations = 0;
        let mut last_logl = f64::NEG_INFINITY;

        for iter in 0..self.max_iter {
            // Invert R0 and G0
            let r0_inv = invert_small_pd(&r0)?;
            let g0_inv = invert_small_pd(&g0)?;

            // Build multi-trait MME using Kronecker expansions.
            //
            // X_mt = I_t kron X    (nt x tp)
            // Z_mt = I_t kron Z_k  for each random term
            //
            // The coefficient matrix and RHS are built directly in dense form
            // exploiting Kronecker structure.
            let (coeff, rhs) = self.assemble_mt_mme(
                model, &r0_inv, &g0_inv, &q_vec,
            );

            // Solve the system
            let chol = coeff
                .clone()
                .cholesky()
                .ok_or(LmmError::NotPositiveDefinite)?;

            let rhs_dvec = nalgebra::DVector::from_column_slice(&rhs);
            let sol = chol.solve(&rhs_dvec);
            let sol_vec: Vec<f64> = sol.as_slice().to_vec();

            let c_inv = chol.inverse();
            let l = chol.l();
            let log_det_c = 2.0 * (0..mt_dim).map(|i| l[(i, i)].ln()).sum::<f64>();

            // Extract per-trait fixed effects (b_i) and random effects (u_i)
            // Solution layout: [b_1, b_2, ..., b_t, u_{1,1}, u_{1,2}, ..., u_{t,K}]
            // where u_{i,k} is the BLUP for trait i, random term k.

            // Fixed effects: positions 0..tp, arranged as [b_1(p), b_2(p), ..., b_t(p)]
            let b_all: Vec<Vec<f64>> = (0..t)
                .map(|i| sol_vec[i * p..(i + 1) * p].to_vec())
                .collect();

            // Random effects: positions tp..mt_dim
            // Layout: for each random term k, trait i: u_{i,k} of length q_k
            // The Kronecker expansion I_t kron Z_k means trait i's contribution
            // for term k is at offset: tp + sum_{l<k}(t*q_l) + i*q_k
            let mut u_all: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_random]; t];
            let mut offset = tp;
            for k in 0..n_random {
                for i in 0..t {
                    u_all[i][k] = sol_vec[offset..offset + q_vec[k]].to_vec();
                    offset += q_vec[k];
                }
            }

            // Compute residuals per trait: e_i = y_i - X*b_i - sum_k Z_k*u_{i,k}
            let mut e_all: Vec<Vec<f64>> = Vec::with_capacity(t);
            for i in 0..t {
                let y_i = &model.y[i * n..(i + 1) * n];
                let xb = spmv(&model.x_single, &b_all[i]);
                let mut residual: Vec<f64> = y_i.iter().zip(xb.iter()).map(|(y, xb)| y - xb).collect();
                for k in 0..n_random {
                    let zu = spmv(&model.z_single_blocks[k], &u_all[i][k]);
                    for j in 0..n {
                        residual[j] -= zu[j];
                    }
                }
                e_all.push(residual);
            }

            // Save old parameters for convergence check
            let old_g0 = g0.clone();
            let old_r0 = r0.clone();

            // --- EM update for G0 ---
            // For simplicity, we use only the first random term for G0 estimation.
            // G0_new[i,j] = (u_i' K^{-1} u_j + tr(K^{-1} C^{uu}_{ij})) / q
            // where C^{uu}_{ij} is the (i,j) sub-block in the C^{-1} corresponding
            // to random effects of term 0.
            {
                let k = 0; // first random term
                let q_k = q_vec[k];
                let mut g0_new = DMatrix::zeros(t, t);

                for i in 0..t {
                    for j in i..t {
                        // u_i' K^{-1} u_j
                        let u_kinv_u = if let Some(ref kinv) = model.kinv_matrices[k] {
                            let kinv_uj = spmv(kinv, &u_all[j][k]);
                            u_all[i][k]
                                .iter()
                                .zip(kinv_uj.iter())
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                        } else {
                            u_all[i][k]
                                .iter()
                                .zip(u_all[j][k].iter())
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                        };

                        // tr(K^{-1} C^{uu}_{ij})
                        // The C^{-1} block for random term k, traits (i,j):
                        // rows: tp + k_offset + i*q_k .. tp + k_offset + (i+1)*q_k
                        // cols: tp + k_offset + j*q_k .. tp + k_offset + (j+1)*q_k
                        // where k_offset = sum_{l<k}(t*q_l)
                        let k_offset: usize = (0..k).map(|l| t * q_vec[l]).sum();
                        let row_start = tp + k_offset + i * q_k;
                        let col_start = tp + k_offset + j * q_k;

                        let trace_term = if let Some(ref kinv) = model.kinv_matrices[k] {
                            // tr(K^{-1} * C^{-1}_{ij}) = sum_{a,b} K^{-1}_{a,b} * C^{-1}_{ij,b,a}
                            let mut tr = 0.0;
                            for a in 0..q_k {
                                for b in 0..q_k {
                                    let kinv_ab =
                                        kinv.get(a, b).copied().unwrap_or(0.0);
                                    let cinv_ba =
                                        c_inv[(col_start + b, row_start + a)];
                                    tr += kinv_ab * cinv_ba;
                                }
                            }
                            tr
                        } else {
                            // K = I: tr(C^{-1}_{ij}) = sum of diagonal
                            let mut tr = 0.0;
                            for a in 0..q_k {
                                tr += c_inv[(row_start + a, col_start + a)];
                            }
                            tr
                        };

                        let val = (u_kinv_u + trace_term) / q_k as f64;
                        g0_new[(i, j)] = val;
                        g0_new[(j, i)] = val;
                    }
                }

                // Ensure G0 stays positive definite by bending if needed
                g0 = bend_to_pd(&g0_new, 1e-6);
            }

            // --- EM update for R0 ---
            // Simplified update: R0_new[i,j] = e_i' e_j / (n - p)
            {
                let denom = (n as f64 - p as f64).max(1.0);
                let mut r0_new = DMatrix::zeros(t, t);
                for i in 0..t {
                    for j in i..t {
                        let eiej: f64 = e_all[i]
                            .iter()
                            .zip(e_all[j].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let val = eiej / denom;
                        r0_new[(i, j)] = val;
                        r0_new[(j, i)] = val;
                    }
                }
                r0 = bend_to_pd(&r0_new, 1e-6);
            }

            // Compute approximate log-likelihood for monitoring
            let sol_rhs: f64 = sol_vec.iter().zip(rhs.iter()).map(|(s, r)| s * r).sum();

            // y' R_mt^{-1} y (using old R0 for consistency)
            let r0_inv_old = invert_small_pd(&old_r0)?;
            let mut y_rinv_y = 0.0;
            for i in 0..t {
                for j in 0..t {
                    let r0inv_ij = r0_inv_old[(i, j)];
                    if r0inv_ij.abs() > 1e-15 {
                        let yi = &model.y[i * n..(i + 1) * n];
                        let yj = &model.y[j * n..(j + 1) * n];
                        y_rinv_y +=
                            r0inv_ij * yi.iter().zip(yj.iter()).map(|(a, b)| a * b).sum::<f64>();
                    }
                }
            }

            let y_p_y = y_rinv_y - sol_rhs;
            let n_eff = (n * t - tp) as f64;
            let log_det_r = n as f64 * log_det_dense(&old_r0);
            let mut log_det_g = 0.0;
            for k in 0..n_random {
                // log|G_mt_k| = q_k * log|G0| (when K = I)
                // More generally: log|G0 kron K| = q_k * log|G0| + t * log|K|
                // We ignore log|K| as it is constant.
                log_det_g += q_vec[k] as f64 * log_det_dense(&old_g0);
            }
            let log_2_pi = (2.0 * std::f64::consts::PI).ln();
            let logl = -0.5 * (n_eff * log_2_pi + log_det_r + log_det_g + log_det_c + y_p_y);

            // Convergence: relative change in G0 and R0 elements
            let param_change = matrix_rel_change(&old_g0, &g0) + matrix_rel_change(&old_r0, &r0);

            n_iterations = iter + 1;

            if iter > 0 && param_change < self.tol {
                converged = true;
                last_logl = logl;
                break;
            }
            last_logl = logl;
        }

        // Update model with final estimates
        model.g0 = g0.clone();
        model.r0 = r0.clone();

        // Final solve for extracting results
        let r0_inv = invert_small_pd(&r0)?;
        let g0_inv = invert_small_pd(&g0)?;

        let (coeff, rhs) = self.assemble_mt_mme(model, &r0_inv, &g0_inv, &q_vec);

        let chol = coeff
            .clone()
            .cholesky()
            .ok_or(LmmError::NotPositiveDefinite)?;

        let rhs_dvec = nalgebra::DVector::from_column_slice(&rhs);
        let sol = chol.solve(&rhs_dvec);
        let sol_vec: Vec<f64> = sol.as_slice().to_vec();
        let c_inv = chol.inverse();

        // Extract final per-trait effects
        let mut fixed_effects_result: Vec<Vec<NamedEffect>> = Vec::new();
        for i in 0..t {
            let b_i = &sol_vec[i * p..(i + 1) * p];
            let effects: Vec<NamedEffect> = model
                .fixed_labels
                .iter()
                .enumerate()
                .map(|(j, label)| NamedEffect {
                    term: label.term.clone(),
                    level: label.level.clone(),
                    estimate: b_i[j],
                    se: c_inv[(i * p + j, i * p + j)].max(0.0).sqrt(),
                })
                .collect();
            fixed_effects_result.push(effects);
        }

        let mut random_effects_result: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_random]; t];
        let mut offset = tp;
        for k in 0..n_random {
            for i in 0..t {
                random_effects_result[i][k] = sol_vec[offset..offset + q_vec[k]].to_vec();
                offset += q_vec[k];
            }
        }

        // Compute genetic correlations from G0
        let genetic_correlations = cov_to_corr(&g0);

        // Build summary
        let summary = build_summary(
            model,
            &g0,
            &r0,
            &genetic_correlations,
            &fixed_effects_result,
            &random_effects_result,
            last_logl,
            converged,
            n_iterations,
        );

        Ok(MultiTraitFitResult {
            g0,
            r0,
            genetic_correlations,
            fixed_effects: fixed_effects_result,
            random_effects: random_effects_result,
            log_likelihood: last_logl,
            converged,
            n_iterations,
            summary,
        })
    }

    /// Assemble the multi-trait MME coefficient matrix and RHS.
    ///
    /// The MME is:
    /// ```text
    /// [X_mt' R_mt^{-1} X_mt       X_mt' R_mt^{-1} Z_mt            ] [b]   [X_mt' R_mt^{-1} y]
    /// [Z_mt' R_mt^{-1} X_mt       Z_mt' R_mt^{-1} Z_mt + G_mt^{-1}] [u] = [Z_mt' R_mt^{-1} y]
    /// ```
    ///
    /// Using Kronecker identities:
    /// - R_mt^{-1} = R0^{-1} kron I_n
    /// - X_mt' R_mt^{-1} X_mt = R0^{-1} kron (X'X)
    /// - X_mt' R_mt^{-1} Z_k_mt = R0^{-1} kron (X'Z_k)
    /// - Z_k_mt' R_mt^{-1} Z_l_mt = R0^{-1} kron (Z_k'Z_l)
    /// - G_mt_k^{-1} = G0^{-1} kron K_k^{-1}
    fn assemble_mt_mme(
        &self,
        model: &MultiTraitModel,
        r0_inv: &DMatrix<f64>,
        g0_inv: &DMatrix<f64>,
        q_vec: &[usize],
    ) -> (DMatrix<f64>, Vec<f64>) {
        let t = model.n_traits;
        let n = model.n_obs;
        let p = model.x_single.cols();
        let n_random = model.z_single_blocks.len();
        let q_total: usize = q_vec.iter().sum();
        let tp = t * p;
        let tq_total = t * q_total;
        let dim = tp + tq_total;

        let mut c = DMatrix::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        // Precompute single-trait cross-products
        let xtx = xtx_dense_local(&model.x_single, n);
        let mut xtz_vec: Vec<DMatrix<f64>> = Vec::new();
        for k in 0..n_random {
            xtz_vec.push(xtz_dense_local(
                &model.x_single,
                &model.z_single_blocks[k],
                n,
            ));
        }
        let mut ztz_vec: Vec<Vec<DMatrix<f64>>> = vec![vec![]; n_random];
        for k in 0..n_random {
            for l in k..n_random {
                let ztz = if k == l {
                    xtx_dense_local(&model.z_single_blocks[k], n)
                } else {
                    xtz_dense_local(
                        &model.z_single_blocks[k],
                        &model.z_single_blocks[l],
                        n,
                    )
                };
                if k < ztz_vec.len() {
                    ztz_vec[k].push(ztz);
                }
            }
        }

        // Precompute X'y_i and Z_k'y_i for each trait
        let mut xty: Vec<Vec<f64>> = Vec::new();
        let mut zty: Vec<Vec<Vec<f64>>> = Vec::new();
        for i in 0..t {
            let y_i = &model.y[i * n..(i + 1) * n];
            xty.push(crate::matrix::sparse::xt_y(&model.x_single, y_i));
            let mut zty_i = Vec::new();
            for k in 0..n_random {
                zty_i.push(crate::matrix::sparse::xt_y(
                    &model.z_single_blocks[k],
                    y_i,
                ));
            }
            zty.push(zty_i);
        }

        // --- Fill coefficient matrix ---

        // Block (i,j) in the fixed-fixed part: R0_inv[i,j] * X'X
        // Position: rows [i*p..(i+1)*p], cols [j*p..(j+1)*p]
        for i in 0..t {
            for j in 0..t {
                let scale = r0_inv[(i, j)];
                if scale.abs() > 1e-15 {
                    for r in 0..p {
                        for cc in 0..p {
                            c[(i * p + r, j * p + cc)] += scale * xtx[(r, cc)];
                        }
                    }
                }
            }
        }

        // Block (i, j) in fixed-random part: R0_inv[i,j] * X'Z_k
        // For random term k, trait j, position:
        //   rows: [i*p..(i+1)*p]
        //   cols: [tp + k_offset + j*q_k .. tp + k_offset + (j+1)*q_k]
        // where k_offset = sum_{l<k}(t * q_l)
        let mut k_offset = 0;
        for k in 0..n_random {
            for i in 0..t {
                for j in 0..t {
                    let scale = r0_inv[(i, j)];
                    if scale.abs() > 1e-15 {
                        for r in 0..p {
                            for cc in 0..q_vec[k] {
                                let row_idx = i * p + r;
                                let col_idx = tp + k_offset + j * q_vec[k] + cc;
                                let val = scale * xtz_vec[k][(r, cc)];
                                c[(row_idx, col_idx)] += val;
                                c[(col_idx, row_idx)] += val; // symmetric
                            }
                        }
                    }
                }
            }
            k_offset += t * q_vec[k];
        }

        // Block (i,j) in random-random part for same term k:
        //   R0_inv[i,j] * Z_k'Z_k + delta_{ij_term} * G0_inv[i,j] * K_k^{-1}
        // For cross terms (k != l): R0_inv[i,j] * Z_k'Z_l
        let mut k_offset_row = 0;
        for k in 0..n_random {
            // Diagonal random block k
            for i in 0..t {
                for j in 0..t {
                    let scale = r0_inv[(i, j)];
                    let row_base = tp + k_offset_row + i * q_vec[k];
                    let col_base = tp + k_offset_row + j * q_vec[k];

                    // R0_inv[i,j] * Z_k'Z_k
                    if scale.abs() > 1e-15 {
                        for r in 0..q_vec[k] {
                            for cc in 0..q_vec[k] {
                                c[(row_base + r, col_base + cc)] +=
                                    scale * ztz_vec[k][0][(r, cc)];
                            }
                        }
                    }

                    // G0_inv[i,j] * K_k^{-1}
                    let g0inv_ij = g0_inv[(i, j)];
                    if g0inv_ij.abs() > 1e-15 {
                        if let Some(ref kinv) = model.kinv_matrices[k] {
                            for (&val, (r, cc)) in kinv.iter() {
                                c[(row_base + r, col_base + cc)] += g0inv_ij * val;
                            }
                        } else {
                            // K = I
                            for r in 0..q_vec[k] {
                                c[(row_base + r, col_base + r)] += g0inv_ij;
                            }
                        }
                    }
                }
            }

            // Cross terms with other random terms
            let mut l_offset = k_offset_row + t * q_vec[k];
            for l in (k + 1)..n_random {
                let ztz_kl = &ztz_vec[k][l - k]; // ztz_vec[k] has entries for l >= k
                for i in 0..t {
                    for j in 0..t {
                        let scale = r0_inv[(i, j)];
                        if scale.abs() > 1e-15 {
                            let row_base = tp + k_offset_row + i * q_vec[k];
                            let col_base = tp + l_offset + j * q_vec[l];
                            for r in 0..q_vec[k] {
                                for cc in 0..q_vec[l] {
                                    let val = scale * ztz_kl[(r, cc)];
                                    c[(row_base + r, col_base + cc)] += val;
                                    c[(col_base + cc, row_base + r)] += val;
                                }
                            }
                        }
                    }
                }
                l_offset += t * q_vec[l];
            }

            k_offset_row += t * q_vec[k];
        }

        // --- Fill RHS ---
        // Fixed part: RHS[i*p..(i+1)*p] = sum_j R0_inv[i,j] * X'y_j
        for i in 0..t {
            for j in 0..t {
                let scale = r0_inv[(i, j)];
                if scale.abs() > 1e-15 {
                    for r in 0..p {
                        rhs[i * p + r] += scale * xty[j][r];
                    }
                }
            }
        }

        // Random part: RHS[tp + k_offset + i*q_k .. ] = sum_j R0_inv[i,j] * Z_k'y_j
        let mut k_offset = 0;
        for k in 0..n_random {
            for i in 0..t {
                for j in 0..t {
                    let scale = r0_inv[(i, j)];
                    if scale.abs() > 1e-15 {
                        for r in 0..q_vec[k] {
                            rhs[tp + k_offset + i * q_vec[k] + r] += scale * zty[j][k][r];
                        }
                    }
                }
            }
            k_offset += t * q_vec[k];
        }

        (c, rhs)
    }
}

// --- Utility functions ---

/// Invert a small symmetric positive-definite matrix using Cholesky decomposition.
fn invert_small_pd(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    m.clone()
        .cholesky()
        .map(|chol| chol.inverse())
        .ok_or(LmmError::NotPositiveDefinite)
}

/// Compute log|M| for a small symmetric positive-definite matrix.
fn log_det_dense(m: &DMatrix<f64>) -> f64 {
    match m.clone().cholesky() {
        Some(chol) => {
            let l = chol.l();
            2.0 * (0..m.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>()
        }
        None => f64::NEG_INFINITY,
    }
}

/// Convert a covariance matrix to a correlation matrix.
fn cov_to_corr(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let n = cov.nrows();
    let mut corr = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let denom = (cov[(i, i)] * cov[(j, j)]).sqrt();
            if denom > 1e-15 {
                corr[(i, j)] = cov[(i, j)] / denom;
            } else {
                corr[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    corr
}

/// Compute the relative change between two matrices (Frobenius norm).
fn matrix_rel_change(old: &DMatrix<f64>, new: &DMatrix<f64>) -> f64 {
    let diff_norm = (old - new).norm();
    let old_norm = old.norm().max(1e-10);
    diff_norm / old_norm
}

/// Bend a matrix to be positive definite by adjusting negative eigenvalues.
fn bend_to_pd(m: &DMatrix<f64>, min_eigenvalue: f64) -> DMatrix<f64> {
    // Force symmetry
    let sym = (m + m.transpose()) * 0.5;

    let eigen = sym.clone().symmetric_eigen();
    let mut any_negative = false;
    for &ev in eigen.eigenvalues.iter() {
        if ev < min_eigenvalue {
            any_negative = true;
            break;
        }
    }

    if !any_negative {
        return sym;
    }

    // Reconstruct with bounded eigenvalues
    let mut d = DMatrix::zeros(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        d[(i, i)] = eigen.eigenvalues[i].max(min_eigenvalue);
    }
    let q = &eigen.eigenvectors;
    q * d * q.transpose()
}

/// Compute X'X as a dense matrix (local helper to avoid depending on n parameter).
fn xtx_dense_local(x: &sprs::CsMat<f64>, _n: usize) -> DMatrix<f64> {
    let p = x.cols();
    let mut result = DMatrix::zeros(p, p);
    let x_csc = if x.is_csc() { x.clone() } else { x.to_csc() };

    for j in 0..p {
        if let Some(col_j) = x_csc.outer_view(j) {
            for i in j..p {
                if let Some(col_i) = x_csc.outer_view(i) {
                    let dot: f64 = col_j.dot(&col_i);
                    result[(i, j)] = dot;
                    if i != j {
                        result[(j, i)] = dot;
                    }
                }
            }
        }
    }
    result
}

/// Compute X'Z as a dense matrix.
fn xtz_dense_local(x: &sprs::CsMat<f64>, z: &sprs::CsMat<f64>, _n: usize) -> DMatrix<f64> {
    let p = x.cols();
    let q = z.cols();
    let mut result = DMatrix::zeros(p, q);
    let x_csc = if x.is_csc() { x.clone() } else { x.to_csc() };
    let z_csc = if z.is_csc() { z.clone() } else { z.to_csc() };

    for i in 0..p {
        if let Some(col_x) = x_csc.outer_view(i) {
            for j in 0..q {
                if let Some(col_z) = z_csc.outer_view(j) {
                    result[(i, j)] = col_x.dot(&col_z);
                }
            }
        }
    }
    result
}

/// Build a formatted summary string.
fn build_summary(
    model: &MultiTraitModel,
    g0: &DMatrix<f64>,
    r0: &DMatrix<f64>,
    corr: &DMatrix<f64>,
    fixed_effects: &[Vec<NamedEffect>],
    random_effects: &[Vec<Vec<f64>>],
    logl: f64,
    converged: bool,
    n_iter: usize,
) -> String {
    let t = model.n_traits;
    let mut s = String::new();

    s.push_str("=== Multi-Trait Mixed Model Fit (EM-REML) ===\n\n");
    s.push_str(&format!(
        "Traits: {}   Obs/trait: {}   Total obs: {}\n",
        t,
        model.n_obs,
        t * model.n_obs
    ));
    s.push_str(&format!(
        "Converged: {}   Iterations: {}\n",
        converged, n_iter
    ));
    s.push_str(&format!("Log-likelihood: {:.4}\n\n", logl));

    s.push_str("--- Genetic Covariance (G0) ---\n");
    for i in 0..t {
        s.push_str("  ");
        for j in 0..t {
            s.push_str(&format!("{:>10.4}", g0[(i, j)]));
        }
        s.push('\n');
    }

    s.push_str("\n--- Residual Covariance (R0) ---\n");
    for i in 0..t {
        s.push_str("  ");
        for j in 0..t {
            s.push_str(&format!("{:>10.4}", r0[(i, j)]));
        }
        s.push('\n');
    }

    s.push_str("\n--- Genetic Correlations ---\n");
    for i in 0..t {
        s.push_str("  ");
        for j in 0..t {
            s.push_str(&format!("{:>10.4}", corr[(i, j)]));
        }
        s.push('\n');
    }

    for i in 0..t {
        s.push_str(&format!(
            "\n--- Trait '{}': Fixed Effects ---\n",
            model.trait_names[i]
        ));
        for ef in &fixed_effects[i] {
            s.push_str(&format!(
                "  {}.{}: {:.6} (SE: {:.6})\n",
                ef.term, ef.level, ef.estimate, ef.se
            ));
        }
    }

    for i in 0..t {
        for (k, term) in model.random_term_names.iter().enumerate() {
            s.push_str(&format!(
                "\n--- Trait '{}': Random '{}' BLUPs ---\n",
                model.trait_names[i], term
            ));
            let blups = &random_effects[i][k];
            let levels = &model.random_level_names[k];
            let mut indexed: Vec<(usize, f64)> = blups.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let show = indexed.len().min(10);
            for &(idx, val) in indexed.iter().take(show) {
                s.push_str(&format!("  {}: {:.6}\n", levels[idx], val));
            }
            if indexed.len() > 10 {
                s.push_str(&format!("  ... and {} more\n", indexed.len() - 10));
            }
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_small_pd() {
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let inv = invert_small_pd(&m).unwrap();
        let product = &m * &inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cov_to_corr() {
        let cov = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 9.0]);
        let corr = cov_to_corr(&cov);
        assert!((corr[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-10);
        // r = 2 / sqrt(4*9) = 2/6 = 1/3
        assert!((corr[(0, 1)] - 1.0 / 3.0).abs() < 1e-10);
        assert!((corr[(1, 0)] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bend_to_pd_already_pd() {
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let bent = bend_to_pd(&m, 1e-6);
        assert!((bent[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((bent[(1, 1)] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bend_to_pd_not_pd() {
        // Not positive definite: eigenvalues = 3 +/- sqrt(5) => one is negative
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let bent = bend_to_pd(&m, 0.01);
        // Should now be PD
        assert!(bent.clone().cholesky().is_some());
    }

    #[test]
    fn test_log_det_dense() {
        // det([[4, 1], [1, 3]]) = 12 - 1 = 11
        let m = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let ld = log_det_dense(&m);
        assert!((ld - 11.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rel_change() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert!(matrix_rel_change(&a, &b) < 1e-10);

        let c = DMatrix::from_row_slice(2, 2, &[1.1, 0.0, 0.0, 1.1]);
        assert!(matrix_rel_change(&a, &c) > 0.0);
    }
}
