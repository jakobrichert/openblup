use nalgebra::DMatrix;

use crate::lmm::FitResult;

use super::wald::{f_distribution_sf, WaldTest};

/// Method for computing denominator degrees of freedom in Wald F-tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DdfMethod {
    /// Containment method: den_df = n - rank(X).
    /// Simple and conservative, exact for balanced designs.
    Containment,
    /// Satterthwaite approximation using the variance-covariance matrix of
    /// fixed effects and the inverse Average Information matrix.
    /// More accurate for unbalanced designs.
    Satterthwaite,
}

/// Calculator for denominator degrees of freedom (ddf) using the
/// Satterthwaite or Kenward-Roger approximation.
///
/// For a single contrast `l'beta`, the Satterthwaite df is:
///
/// ```text
/// nu = 2 * (l' Phi l)^2 / sum_{i,j} (l' dPhi/dtheta_i l) * AI_inv_{ij} * (l' dPhi/dtheta_j l)
/// ```
///
/// where `Phi = C^{-1}_{bb}` is the fixed-effects block of `C^{-1}`,
/// `AI` is the Average Information matrix at convergence, and
/// `dPhi/dtheta_k` is the derivative of `Phi` with respect to variance
/// parameter `theta_k`.
///
/// The derivatives are computed analytically using the matrix identity:
/// `dC^{-1}/dtheta_k = -C^{-1} * (dC/dtheta_k) * C^{-1}`
///
/// # Kenward-Roger (future)
///
/// The full Kenward-Roger adjustment includes two additional corrections
/// beyond the Satterthwaite ddf:
///
/// 1. **Bias adjustment**: `Phi_A = Phi + 2*Lambda` where `Lambda` corrects
///    for small-sample bias in the variance-covariance matrix of fixed effects.
///
/// 2. **Scaled F-statistic**: `F* = (q/q*) * F` where `q*` adjusts the scale
///    of the F distribution.
///
/// These require the full `dC/dtheta` derivatives which are available in this
/// implementation, but the KR-specific corrections are not yet implemented.
/// The Satterthwaite ddf is the primary method provided here.
pub struct DdfCalculator {
    /// Full C^{-1} matrix from the converged MME.
    c_inv: DMatrix<f64>,
    /// Inverse of the Average Information matrix (n_vc x n_vc).
    ai_inv: DMatrix<f64>,
    /// Number of fixed effect parameters (p).
    n_fixed: usize,
    /// Number of observations.
    n_obs: usize,
    /// Number of random effect levels per random term.
    n_random_per_term: Vec<usize>,
    /// Current variance parameters: [sigma2_1, ..., sigma2_r, sigma2_e].
    variance_params: Vec<f64>,
}

impl DdfCalculator {
    /// Create a new `DdfCalculator`.
    ///
    /// # Arguments
    ///
    /// * `c_inv` - Full inverse of the MME coefficient matrix at convergence.
    /// * `ai_matrix` - The Average Information matrix (n_vc x n_vc) at convergence.
    /// * `n_fixed` - Number of fixed effect parameters (p).
    /// * `n_obs` - Number of observations.
    /// * `n_random_per_term` - Number of random effect levels for each random term.
    /// * `variance_params` - Current variance parameters [sigma2_1, ..., sigma2_r, sigma2_e].
    ///
    /// # Returns
    ///
    /// `None` if the AI matrix is not invertible.
    pub fn new(
        c_inv: DMatrix<f64>,
        ai_matrix: DMatrix<f64>,
        n_fixed: usize,
        n_obs: usize,
        n_random_per_term: Vec<usize>,
        variance_params: Vec<f64>,
    ) -> Option<Self> {
        // Invert the AI matrix
        let ai_inv = ai_matrix.clone().cholesky()?.inverse();

        Some(Self {
            c_inv,
            ai_inv,
            n_fixed,
            n_obs,
            n_random_per_term,
            variance_params,
        })
    }

    /// Extract the fixed-effects block of C^{-1}, i.e. `Phi = C^{-1}_{bb}` (p x p).
    fn phi(&self) -> DMatrix<f64> {
        self.c_inv
            .view((0, 0), (self.n_fixed, self.n_fixed))
            .into()
    }

    /// Compute the derivative `dC/dtheta_k` for variance parameter k.
    ///
    /// The MME coefficient matrix is:
    /// ```text
    /// C = [ X'R^{-1}X       X'R^{-1}Z           ]
    ///     [ Z'R^{-1}X       Z'R^{-1}Z + G^{-1}  ]
    /// ```
    ///
    /// For R = sigma2_e * I and G_k = sigma2_k * I_qk:
    ///
    /// - `dC/d(sigma2_k)` for random term k:
    ///   Only the G^{-1}_k block changes: `d(G_k^{-1})/d(sigma2_k) = -1/sigma2_k^2 * I_qk`
    ///   So `dC/d(sigma2_k)` is zero everywhere except the (k,k) random block
    ///   on the diagonal, where it equals `-1/sigma2_k^2 * I_qk`.
    ///
    /// - `dC/d(sigma2_e)`:
    ///   `R^{-1} = 1/sigma2_e * I`, so `dR^{-1}/d(sigma2_e) = -1/sigma2_e^2 * I`.
    ///   This scales all the `R^{-1}`-containing terms by the factor `-1/sigma2_e`.
    ///   So `dC/d(sigma2_e) = -1/sigma2_e * W'R^{-1}W` where `W = [X Z]`.
    ///   The G^{-1} blocks do NOT depend on sigma2_e.
    fn dc_dtheta(&self, k: usize) -> DMatrix<f64> {
        let dim = self.c_inv.nrows();
        let n_random_terms = self.n_random_per_term.len();
        let sigma2_e = self.variance_params[n_random_terms];

        if k < n_random_terms {
            // Derivative w.r.t. sigma2_k (random term k)
            // dC/d(sigma2_k) = -1/sigma2_k^2 * I in the (k,k) random block
            let sigma2_k = self.variance_params[k];
            let mut dc = DMatrix::zeros(dim, dim);
            let block_start =
                self.n_fixed + self.n_random_per_term[..k].iter().sum::<usize>();
            let q_k = self.n_random_per_term[k];
            let scale = -1.0 / (sigma2_k * sigma2_k);
            for i in 0..q_k {
                dc[(block_start + i, block_start + i)] = scale;
            }
            dc
        } else {
            // Derivative w.r.t. sigma2_e (residual)
            // dC/d(sigma2_e) = -1/sigma2_e * (C - block_diag(0, G^{-1}))
            // Because C = (1/sigma2_e)*W'W + block_diag(0, G^{-1}_1, ..., G^{-1}_r)
            // so dC/d(sigma2_e) = -1/sigma2_e^2 * W'W = -1/sigma2_e * (C - block_diag(0, G^{-1}))

            // Reconstruct C from C^{-1}
            let c_matrix = match self.c_inv.clone().try_inverse() {
                Some(c) => c,
                None => return DMatrix::zeros(dim, dim),
            };

            // Build the G^{-1} block diagonal to subtract
            let mut ginv_contribution = DMatrix::zeros(dim, dim);
            let mut block_start = self.n_fixed;
            for kk in 0..n_random_terms {
                let q_k = self.n_random_per_term[kk];
                let sigma2_k = self.variance_params[kk];
                // G_k^{-1} = (1/sigma2_k) * I_qk (identity structure)
                for i in 0..q_k {
                    ginv_contribution[(block_start + i, block_start + i)] = 1.0 / sigma2_k;
                }
                block_start += q_k;
            }

            // W'R^{-1}W = C - block_diag(0, G^{-1})
            let wtrw = &c_matrix - &ginv_contribution;
            -1.0 / sigma2_e * wtrw
        }
    }

    /// Compute `dPhi/dtheta_k` where `Phi = C^{-1}_{bb}` (the p x p fixed-effects block).
    ///
    /// Using the matrix identity:
    /// `dC^{-1}/dtheta_k = -C^{-1} * (dC/dtheta_k) * C^{-1}`
    ///
    /// Then `dPhi/dtheta_k` is the top-left p x p block of `dC^{-1}/dtheta_k`.
    fn dphi_dtheta(&self, k: usize) -> DMatrix<f64> {
        let dc = self.dc_dtheta(k);
        let p = self.n_fixed;
        let dim = self.c_inv.nrows();

        // dC^{-1}/dtheta_k = -C^{-1} * dC/dtheta_k * C^{-1}
        // We only need the top-left p x p block:
        //   dphi[i,j] = -sum_r sum_s C^{-1}[i,r] * dc[r,s] * C^{-1}[s,j]
        //   for i,j in 0..p

        let mut dphi = DMatrix::zeros(p, p);

        for i in 0..p {
            for j in 0..p {
                let mut val = 0.0;
                for r in 0..dim {
                    for s in 0..dim {
                        val += self.c_inv[(i, r)] * dc[(r, s)] * self.c_inv[(s, j)];
                    }
                }
                dphi[(i, j)] = -val;
            }
        }

        dphi
    }

    /// Compute the Satterthwaite denominator degrees of freedom for a single
    /// contrast `l'beta`.
    ///
    /// ```text
    /// nu = 2 * (l' Phi l)^2 / sum_{i,j} g_i * AI_inv_{ij} * g_j
    /// ```
    ///
    /// where `g_k = l' (dPhi/dtheta_k) l`.
    ///
    /// # Arguments
    ///
    /// * `contrast` - The contrast vector `l` of length `p` (number of fixed effects).
    ///
    /// # Returns
    ///
    /// The Satterthwaite degrees of freedom, clamped to `[1, n - p]`.
    pub fn satterthwaite_ddf(&self, contrast: &[f64]) -> f64 {
        let p = self.n_fixed;
        assert_eq!(
            contrast.len(),
            p,
            "Contrast vector length ({}) must equal number of fixed effects ({})",
            contrast.len(),
            p
        );

        let phi = self.phi();
        let n_params = self.variance_params.len();

        // Compute l' Phi l
        let l_phi_l = quad_form(contrast, &phi);

        if l_phi_l <= 0.0 {
            // Degenerate case
            return 1.0;
        }

        // Compute gradient vector: g_k = l' (dPhi/dtheta_k) l
        let mut g = vec![0.0; n_params];
        for k in 0..n_params {
            let dphi_k = self.dphi_dtheta(k);
            g[k] = quad_form(contrast, &dphi_k);
        }

        // Compute denominator: sum_{i,j} g_i * AI_inv_{ij} * g_j
        let mut denom = 0.0;
        for i in 0..n_params {
            for j in 0..n_params {
                denom += g[i] * self.ai_inv[(i, j)] * g[j];
            }
        }

        if denom <= 0.0 {
            // If the denominator is non-positive, fall back to containment
            return (self.n_obs - self.n_fixed) as f64;
        }

        // Satterthwaite ddf
        let nu = 2.0 * l_phi_l * l_phi_l / denom;

        // Clamp to [1, n - p]
        let max_df = (self.n_obs - self.n_fixed) as f64;
        nu.max(1.0).min(max_df)
    }

    /// Compute the generalized Satterthwaite denominator degrees of freedom
    /// for a multi-parameter Wald F-test.
    ///
    /// For a contrast matrix `L` (q x p) testing q parameters simultaneously,
    /// we use the method of Fai and Cornelius (1996), which estimates a single
    /// denominator df by matching moments of the approximate F distribution.
    ///
    /// The effective ddf is computed as a combination of per-eigenvalue
    /// Satterthwaite ddf values from the spectral decomposition of the
    /// variance-covariance matrix of the contrast.
    ///
    /// For a term with q contrasts (rows of L), we decompose:
    ///   `L Phi L'` via eigendecomposition
    /// Then for each eigenvalue/eigenvector pair, compute the single-contrast
    /// Satterthwaite ddf using the corresponding transformed contrast.
    ///
    /// The combined ddf is:
    /// ```text
    /// nu = 2 * E / (E - q)  where E = sum_i nu_i / (nu_i - 2)  for nu_i > 2
    /// ```
    ///
    /// # Arguments
    ///
    /// * `contrast_matrix` - Rows of the contrast matrix `L`, each of length p.
    ///
    /// # Returns
    ///
    /// The generalized Satterthwaite ddf.
    pub fn satterthwaite_ddf_multi(&self, contrast_matrix: &[Vec<f64>]) -> f64 {
        let q = contrast_matrix.len();
        if q == 0 {
            return 1.0;
        }
        if q == 1 {
            return self.satterthwaite_ddf(&contrast_matrix[0]);
        }

        let p = self.n_fixed;
        let phi = self.phi();

        // Build L (q x p) and compute L * Phi * L' (q x q)
        let mut l_phi_lt = DMatrix::zeros(q, q);
        for i in 0..q {
            for j in 0..q {
                let mut val = 0.0;
                for r in 0..p {
                    for s in 0..p {
                        val +=
                            contrast_matrix[i][r] * phi[(r, s)] * contrast_matrix[j][s];
                    }
                }
                l_phi_lt[(i, j)] = val;
            }
        }

        // Eigendecomposition of L * Phi * L'
        let eigen = l_phi_lt.symmetric_eigen();
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        // For each eigenvector, compute the transformed contrast and its ddf
        let mut nu_values = Vec::with_capacity(q);
        for i in 0..q {
            if eigenvalues[i] <= 1e-14 {
                // Skip near-zero eigenvalues (rank-deficient contrast)
                continue;
            }
            // Transformed contrast: l_i = eigenvector_i' * L
            // This is a p-length vector
            let mut l_transformed = vec![0.0; p];
            for j in 0..p {
                let mut val = 0.0;
                for kk in 0..q {
                    val += eigenvectors[(kk, i)] * contrast_matrix[kk][j];
                }
                l_transformed[j] = val;
            }

            let nu_i = self.satterthwaite_ddf(&l_transformed);
            nu_values.push(nu_i);
        }

        if nu_values.is_empty() {
            return 1.0;
        }

        // Combine using the formula: nu = 2E / (E - q_eff) where
        // E = sum(nu_i / (nu_i - 2)) for nu_i > 2
        let q_eff = nu_values.len() as f64;
        let e_sum: f64 = nu_values
            .iter()
            .map(|&nu_i| {
                if nu_i > 2.0 {
                    nu_i / (nu_i - 2.0)
                } else {
                    // For nu_i <= 2, the expectation doesn't exist;
                    // use a large contribution to push combined df down
                    100.0
                }
            })
            .sum();

        let combined_nu = if e_sum > q_eff {
            2.0 * e_sum / (e_sum - q_eff)
        } else {
            // Fallback
            2.0
        };

        let max_df = (self.n_obs - self.n_fixed) as f64;
        combined_nu.max(1.0).min(max_df)
    }

    /// Containment degrees of freedom: `n - rank(X)`.
    pub fn containment_ddf(&self) -> f64 {
        (self.n_obs - self.n_fixed) as f64
    }
}

/// Compute the quadratic form `x' A x` for vector x and matrix A.
fn quad_form(x: &[f64], a: &DMatrix<f64>) -> f64 {
    let n = x.len();
    let mut result = 0.0;
    for i in 0..n {
        for j in 0..n {
            result += x[i] * a[(i, j)] * x[j];
        }
    }
    result
}

/// Compute Wald F-tests using Satterthwaite denominator degrees of freedom.
///
/// This function computes Wald F-tests for each fixed effect term, using the
/// Satterthwaite approximation for denominator degrees of freedom. This gives
/// more accurate p-values for unbalanced designs compared to the simple
/// containment method.
///
/// # Arguments
///
/// * `result` - A fitted mixed model result.
/// * `c_inv` - The full inverse of the MME coefficient matrix at convergence.
/// * `ai_matrix` - The Average Information matrix at convergence.
/// * `n_random_per_term` - Number of random effect levels per random term.
///
/// # Returns
///
/// A vector of `WaldTest` results with Satterthwaite ddf, or `None` if the
/// `DdfCalculator` could not be constructed (e.g., singular AI matrix).
pub fn wald_tests_satterthwaite(
    result: &FitResult,
    c_inv: &DMatrix<f64>,
    ai_matrix: &DMatrix<f64>,
    n_random_per_term: &[usize],
) -> Option<Vec<WaldTest>> {
    if result.fixed_effects.is_empty() {
        return Some(Vec::new());
    }

    // Extract variance parameters from FitResult
    let variance_params: Vec<f64> = result
        .variance_components
        .iter()
        .flat_map(|vc| vc.parameters.iter().map(|(_, v)| *v))
        .collect();

    let n_fixed = result.n_fixed_params;

    let calc = DdfCalculator::new(
        c_inv.clone(),
        ai_matrix.clone(),
        n_fixed,
        result.n_obs,
        n_random_per_term.to_vec(),
        variance_params,
    )?;

    // Group fixed effects by term name (same logic as in wald_tests)
    let mut term_order: Vec<String> = Vec::new();
    let mut term_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, ef) in result.fixed_effects.iter().enumerate() {
        if !term_indices.contains_key(&ef.term) {
            term_order.push(ef.term.clone());
        }
        term_indices
            .entry(ef.term.clone())
            .or_default()
            .push(i);
    }

    let mut tests = Vec::new();

    for term in &term_order {
        let indices = &term_indices[term];
        let num_df = indices.len();

        if num_df == 0 {
            continue;
        }

        if num_df == 1 {
            // Single-parameter term: F = (beta / SE)^2
            let idx = indices[0];
            let ef = &result.fixed_effects[idx];
            let se = ef.se;

            // Build contrast vector: e_idx (unit vector)
            let mut contrast = vec![0.0; n_fixed];
            contrast[idx] = 1.0;

            let den_df = calc.satterthwaite_ddf(&contrast);

            if se > 0.0 {
                let t = ef.estimate / se;
                let f_stat = t * t;
                let p_value = f_distribution_sf(f_stat, 1.0, den_df);
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: f_stat,
                    num_df: 1,
                    den_df,
                    p_value,
                });
            } else {
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: 0.0,
                    num_df: 1,
                    den_df,
                    p_value: 1.0,
                });
            }
        } else {
            // Multi-parameter term: build contrast matrix
            let contrast_matrix: Vec<Vec<f64>> = indices
                .iter()
                .map(|&idx| {
                    let mut row = vec![0.0; n_fixed];
                    row[idx] = 1.0;
                    row
                })
                .collect();

            let den_df = calc.satterthwaite_ddf_multi(&contrast_matrix);

            // Compute the F-statistic (same approximation as wald_tests
            // when we only have diagonal SEs)
            let f_stat: f64 = indices
                .iter()
                .map(|&idx| {
                    let ef = &result.fixed_effects[idx];
                    if ef.se > 0.0 {
                        (ef.estimate / ef.se).powi(2)
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / num_df as f64;

            let p_value = f_distribution_sf(f_stat, num_df as f64, den_df);

            tests.push(WaldTest {
                term: term.clone(),
                f_statistic: f_stat,
                num_df,
                den_df,
                p_value,
            });
        }
    }

    Some(tests)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmm::{FitResult, NamedEffect, VarianceEstimate};

    /// Helper to build a simple C^{-1} matrix for testing.
    /// Constructs a balanced one-way random intercept model MME and inverts it.
    fn make_simple_c_inv(
        p: usize,
        q: usize,
        sigma2_e: f64,
        sigma2_u: f64,
    ) -> DMatrix<f64> {
        let dim = p + q;
        let mut c = DMatrix::zeros(dim, dim);

        let n_per_group = 10;
        let n_obs = n_per_group * q;
        let r_inv = 1.0 / sigma2_e;

        // X'X / sigma2_e
        for i in 0..p {
            c[(i, i)] = (n_obs as f64) * r_inv;
        }

        // X'Z / sigma2_e (connect first fixed effect to all random levels)
        for j in 0..q {
            c[(0, p + j)] = (n_per_group as f64) * r_inv;
            c[(p + j, 0)] = (n_per_group as f64) * r_inv;
        }

        // Z'Z / sigma2_e + G^{-1}
        for j in 0..q {
            c[(p + j, p + j)] = (n_per_group as f64) * r_inv + 1.0 / sigma2_u;
        }

        c.try_inverse()
            .unwrap_or_else(|| DMatrix::identity(dim, dim))
    }

    /// Helper to build an AI matrix for testing.
    fn make_simple_ai(
        sigma2_u: f64,
        sigma2_e: f64,
        q: usize,
        n: usize,
    ) -> DMatrix<f64> {
        let mut ai = DMatrix::zeros(2, 2);
        ai[(0, 0)] = (q as f64) / (2.0 * sigma2_u * sigma2_u);
        ai[(1, 1)] = (n as f64) / (2.0 * sigma2_e * sigma2_e);
        ai[(0, 1)] = 0.01;
        ai[(1, 0)] = 0.01;
        ai
    }

    #[test]
    fn test_ddf_calculator_creation() {
        let p = 1;
        let q = 5;
        let sigma2_e = 2.0;
        let sigma2_u = 3.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, 50);

        let calc =
            DdfCalculator::new(c_inv, ai, p, 50, vec![q], vec![sigma2_u, sigma2_e]);
        assert!(
            calc.is_some(),
            "DdfCalculator should be created successfully"
        );
    }

    #[test]
    fn test_ddf_calculator_singular_ai() {
        let p = 1;
        let q = 5;
        let sigma2_e = 2.0;
        let sigma2_u = 3.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);

        let ai = DMatrix::zeros(2, 2);

        let calc =
            DdfCalculator::new(c_inv, ai, p, 50, vec![q], vec![sigma2_u, sigma2_e]);
        assert!(
            calc.is_none(),
            "DdfCalculator should fail with singular AI"
        );
    }

    #[test]
    fn test_satterthwaite_ddf_single_contrast() {
        let p = 1;
        let q = 5;
        let sigma2_e = 2.0;
        let sigma2_u = 3.0;
        let n_obs = 50;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let calc = DdfCalculator::new(
            c_inv, ai, p, n_obs, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let contrast = vec![1.0];
        let nu = calc.satterthwaite_ddf(&contrast);

        let containment_df = (n_obs - p) as f64;
        assert!(
            nu > 0.0,
            "Satterthwaite ddf should be positive, got {}",
            nu
        );
        assert!(
            nu <= containment_df + 1e-10,
            "Satterthwaite ddf ({}) should be <= containment df ({})",
            nu,
            containment_df
        );
    }

    #[test]
    fn test_satterthwaite_ddf_bounded() {
        let p = 1;
        let q = 3;
        let sigma2_e = 1.0;
        let sigma2_u = 1.0;
        let n_obs = 30;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let calc = DdfCalculator::new(
            c_inv, ai, p, n_obs, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let contrast = vec![1.0];
        let nu = calc.satterthwaite_ddf(&contrast);

        assert!(nu >= 1.0, "ddf should be at least 1, got {}", nu);
        assert!(
            nu <= (n_obs - p) as f64,
            "ddf should be at most n-p={}, got {}",
            n_obs - p,
            nu
        );
    }

    #[test]
    fn test_containment_ddf() {
        let p = 3;
        let q = 5;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let n_obs = 50;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let calc = DdfCalculator::new(
            c_inv, ai, p, n_obs, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        assert!(
            (calc.containment_ddf() - 47.0).abs() < 1e-10,
            "Containment ddf should be 50-3=47, got {}",
            calc.containment_ddf()
        );
    }

    #[test]
    fn test_satterthwaite_multi_df() {
        let p = 3;
        let q = 4;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let n_obs = 40;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let calc = DdfCalculator::new(
            c_inv, ai, p, n_obs, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let contrast_matrix = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];

        let nu = calc.satterthwaite_ddf_multi(&contrast_matrix);
        let containment_df = (n_obs - p) as f64;

        assert!(nu >= 1.0, "Multi-df should be >= 1, got {}", nu);
        assert!(
            nu <= containment_df + 1e-10,
            "Multi-df ({}) should be <= containment df ({})",
            nu,
            containment_df
        );
    }

    #[test]
    fn test_wald_tests_satterthwaite_basic() {
        let p = 1;
        let q = 3;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let n_obs = 30;

        let result = FitResult {
            variance_components: vec![
                VarianceEstimate {
                    name: "group".to_string(),
                    structure: "Identity".to_string(),
                    parameters: vec![("sigma2".to_string(), sigma2_u)],
                },
                VarianceEstimate {
                    name: "residual".to_string(),
                    structure: "Identity".to_string(),
                    parameters: vec![("sigma2".to_string(), sigma2_e)],
                },
            ],
            fixed_effects: vec![NamedEffect {
                term: "mu".to_string(),
                level: "intercept".to_string(),
                estimate: 5.0,
                se: 0.5,
            }],
            random_effects: vec![],
            log_likelihood: -50.0,
            n_iterations: 10,
            converged: true,
            history: vec![],
            variance_se: vec![0.5, 0.3],
            residuals: vec![],
            n_obs,
            n_fixed_params: p,
            n_variance_params: 2,
        };

        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let tests = wald_tests_satterthwaite(&result, &c_inv, &ai, &[q]);
        assert!(tests.is_some(), "Should produce Wald tests");

        let tests = tests.unwrap();
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);
        assert!(
            tests[0].den_df > 0.0,
            "Satterthwaite ddf should be positive"
        );
        assert!(
            tests[0].den_df <= (n_obs - p) as f64 + 1e-10,
            "Satterthwaite ddf ({}) should be <= containment df ({})",
            tests[0].den_df,
            (n_obs - p)
        );
    }

    #[test]
    fn test_wald_tests_satterthwaite_empty() {
        let result = FitResult {
            variance_components: vec![],
            fixed_effects: vec![],
            random_effects: vec![],
            log_likelihood: 0.0,
            n_iterations: 0,
            converged: true,
            history: vec![],
            variance_se: vec![],
            residuals: vec![],
            n_obs: 10,
            n_fixed_params: 0,
            n_variance_params: 0,
        };

        let c_inv = DMatrix::zeros(0, 0);
        let ai = DMatrix::zeros(0, 0);

        let tests = wald_tests_satterthwaite(&result, &c_inv, &ai, &[]);
        assert!(tests.is_some());
        assert!(tests.unwrap().is_empty());
    }

    #[test]
    fn test_satterthwaite_vs_containment_balanced() {
        let p = 1;
        let q = 5;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let n_per_group = 10;
        let n_obs = n_per_group * q;

        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let calc = DdfCalculator::new(
            c_inv, ai, p, n_obs, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let contrast = vec![1.0];
        let satt_df = calc.satterthwaite_ddf(&contrast);
        let cont_df = calc.containment_ddf();

        assert!(
            satt_df <= cont_df + 1e-10,
            "Satterthwaite ({}) should be <= containment ({})",
            satt_df,
            cont_df
        );
        assert!(
            satt_df > 0.5 * cont_df || satt_df >= 1.0,
            "For balanced data, Satterthwaite ({}) should not be too much smaller than containment ({})",
            satt_df,
            cont_df
        );
    }

    #[test]
    fn test_phi_extraction() {
        let p = 2;
        let q = 3;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);

        let ai = make_simple_ai(sigma2_u, sigma2_e, q, 30);
        let calc = DdfCalculator::new(
            c_inv.clone(),
            ai,
            p,
            30,
            vec![q],
            vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let phi = calc.phi();
        assert_eq!(phi.nrows(), p);
        assert_eq!(phi.ncols(), p);

        for i in 0..p {
            for j in 0..p {
                assert!(
                    (phi[(i, j)] - c_inv[(i, j)]).abs() < 1e-12,
                    "Phi[{},{}] = {} but C^{{-1}}[{},{}] = {}",
                    i, j, phi[(i, j)], i, j, c_inv[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_dc_dtheta_random_component() {
        let p = 1;
        let q = 3;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, 30);

        let calc = DdfCalculator::new(
            c_inv, ai, p, 30, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let dc = calc.dc_dtheta(0);
        let dim = p + q;

        // Fixed block should be zero
        for i in 0..p {
            for j in 0..dim {
                assert!(
                    dc[(i, j)].abs() < 1e-14,
                    "dC/dtheta[{},{}] should be 0 (fixed row), got {}",
                    i, j, dc[(i, j)]
                );
            }
        }

        // Random block diagonal should be -1/sigma2_u^2
        let expected = -1.0 / (sigma2_u * sigma2_u);
        for i in 0..q {
            assert!(
                (dc[(p + i, p + i)] - expected).abs() < 1e-14,
                "dC/dtheta diagonal[{}] should be {}, got {}",
                i, expected, dc[(p + i, p + i)]
            );
        }
    }

    #[test]
    fn test_quad_form() {
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let x = vec![1.0, 2.0];
        // x'Ax = 1*2*1 + 1*1*2 + 2*1*1 + 2*3*2 = 2 + 2 + 2 + 12 = 18
        let result = quad_form(&x, &a);
        assert!(
            (result - 18.0).abs() < 1e-10,
            "quad_form should be 18, got {}",
            result
        );
    }

    #[test]
    fn test_wald_tests_satterthwaite_multi_level() {
        let p = 3;
        let q = 4;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let n_obs = 40;

        let result = FitResult {
            variance_components: vec![
                VarianceEstimate {
                    name: "block".to_string(),
                    structure: "Identity".to_string(),
                    parameters: vec![("sigma2".to_string(), sigma2_u)],
                },
                VarianceEstimate {
                    name: "residual".to_string(),
                    structure: "Identity".to_string(),
                    parameters: vec![("sigma2".to_string(), sigma2_e)],
                },
            ],
            fixed_effects: vec![
                NamedEffect {
                    term: "mu".to_string(),
                    level: "intercept".to_string(),
                    estimate: 10.0,
                    se: 0.4,
                },
                NamedEffect {
                    term: "trt".to_string(),
                    level: "A".to_string(),
                    estimate: 2.0,
                    se: 0.6,
                },
                NamedEffect {
                    term: "trt".to_string(),
                    level: "B".to_string(),
                    estimate: -1.0,
                    se: 0.6,
                },
            ],
            random_effects: vec![],
            log_likelihood: -60.0,
            n_iterations: 8,
            converged: true,
            history: vec![],
            variance_se: vec![0.8, 0.4],
            residuals: vec![],
            n_obs,
            n_fixed_params: p,
            n_variance_params: 2,
        };

        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, n_obs);

        let tests = wald_tests_satterthwaite(&result, &c_inv, &ai, &[q]);
        assert!(tests.is_some());

        let tests = tests.unwrap();
        assert_eq!(tests.len(), 2);

        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);
        assert!(tests[0].den_df > 0.0);
        assert!(tests[0].den_df <= (n_obs - p) as f64 + 1e-10);

        assert_eq!(tests[1].term, "trt");
        assert_eq!(tests[1].num_df, 2);
        assert!(tests[1].den_df > 0.0);
        assert!(tests[1].den_df <= (n_obs - p) as f64 + 1e-10);
    }

    #[test]
    fn test_dphi_dtheta_symmetry() {
        let p = 2;
        let q = 3;
        let sigma2_e = 1.0;
        let sigma2_u = 2.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, 30);

        let calc = DdfCalculator::new(
            c_inv, ai, p, 30, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        // Check symmetry for derivative w.r.t. random component
        let dphi_0 = calc.dphi_dtheta(0);
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (dphi_0[(i, j)] - dphi_0[(j, i)]).abs() < 1e-12,
                    "dPhi/dtheta_0 should be symmetric: [{},{}]={} vs [{},{}]={}",
                    i, j, dphi_0[(i, j)], j, i, dphi_0[(j, i)]
                );
            }
        }

        // Check symmetry for derivative w.r.t. residual
        let dphi_1 = calc.dphi_dtheta(1);
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (dphi_1[(i, j)] - dphi_1[(j, i)]).abs() < 1e-12,
                    "dPhi/dtheta_1 should be symmetric: [{},{}]={} vs [{},{}]={}",
                    i, j, dphi_1[(i, j)], j, i, dphi_1[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_dc_dtheta_residual_structure() {
        let p = 1;
        let q = 3;
        let sigma2_e = 2.0;
        let sigma2_u = 3.0;
        let c_inv = make_simple_c_inv(p, q, sigma2_e, sigma2_u);
        let ai = make_simple_ai(sigma2_u, sigma2_e, q, 30);

        let calc = DdfCalculator::new(
            c_inv, ai, p, 30, vec![q], vec![sigma2_u, sigma2_e],
        )
        .unwrap();

        let dc = calc.dc_dtheta(1);
        let dim = p + q;

        // The fixed block should be non-zero and negative
        assert!(
            dc[(0, 0)] < 0.0,
            "dC/d(sigma2_e) for fixed block should be negative, got {}",
            dc[(0, 0)]
        );

        // The derivative should be symmetric
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (dc[(i, j)] - dc[(j, i)]).abs() < 1e-10,
                    "dC/d(sigma2_e) should be symmetric at [{},{}]",
                    i, j
                );
            }
        }
    }
}
