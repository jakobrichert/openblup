use nalgebra::DMatrix;

use crate::lmm::MmeInverse;
use crate::matrix::sparse::spmv;
use crate::types::SparseMat;

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

/// Satterthwaite denominator degrees of freedom for Wald tests of fixed
/// effects (Giesbrecht & Burns 1985; Fai & Cornelius 1996 for multi-df
/// terms).
///
/// The calculator works from
///
/// * `Φ = C⁻¹_{bb}`, the covariance matrix of the fixed effects,
/// * `∂Φ/∂θ_k` for every variance parameter `θ_k`, and
/// * the inverse of the average information matrix, `AI⁻¹ ≈ Var(θ̂)`.
///
/// For a contrast `l'β`, `ν = 2 (l'Φl)² / (g' AI⁻¹ g)` with
/// `g_k = l' (∂Φ/∂θ_k) l`. The derivatives are
/// `∂Φ/∂θ_k = −C⁻¹_{b·} (∂C/∂θ_k) C⁻¹_{·b}`; the REML engines compute them
/// at convergence from the columns of `C⁻¹` belonging to the fixed effects,
/// so no dense `C⁻¹` is needed (see
/// [`fixed_cov_derivatives_scaled_identity`]).
///
/// Kenward-Roger's small-sample bias adjustment of `Φ` is not implemented.
pub struct DdfCalculator {
    /// `Φ = C⁻¹_{bb}` (p x p).
    phi: DMatrix<f64>,
    /// `∂Φ/∂θ_k`, one p x p matrix per variance parameter.
    dphi: Vec<DMatrix<f64>>,
    /// Inverse of the Average Information matrix (n_vc x n_vc).
    ai_inv: DMatrix<f64>,
    /// Number of fixed effect parameters (p).
    n_fixed: usize,
    /// Number of observations.
    n_obs: usize,
}

impl DdfCalculator {
    /// Create a new `DdfCalculator`.
    ///
    /// # Arguments
    ///
    /// * `phi` - Covariance matrix of the fixed effects (p x p).
    /// * `dphi` - `∂Φ/∂θ_k` for each variance parameter (p x p each).
    /// * `ai_matrix` - The Average Information matrix at convergence.
    /// * `n_obs` - Number of observations.
    ///
    /// # Returns
    ///
    /// `None` if the AI matrix is not positive definite or the number of
    /// derivatives does not match its dimension.
    pub fn new(
        phi: DMatrix<f64>,
        dphi: Vec<DMatrix<f64>>,
        ai_matrix: DMatrix<f64>,
        n_obs: usize,
    ) -> Option<Self> {
        let n_fixed = phi.nrows();
        if phi.ncols() != n_fixed
            || dphi.len() != ai_matrix.nrows()
            || dphi
                .iter()
                .any(|d| d.nrows() != n_fixed || d.ncols() != n_fixed)
        {
            return None;
        }
        let ai_inv = ai_matrix.cholesky()?.inverse();
        Some(Self {
            phi,
            dphi,
            ai_inv,
            n_fixed,
            n_obs,
        })
    }

    /// The fixed-effects covariance matrix `Φ = C^{-1}_{bb}` (p x p).
    pub fn phi(&self) -> &DMatrix<f64> {
        &self.phi
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

        let n_params = self.dphi.len();
        let l_phi_l = quad_form(contrast, &self.phi);
        if l_phi_l <= 0.0 {
            return 1.0;
        }

        let g: Vec<f64> = (0..n_params)
            .map(|k| quad_form(contrast, &self.dphi[k]))
            .collect();

        let mut denom = 0.0;
        for i in 0..n_params {
            for j in 0..n_params {
                denom += g[i] * self.ai_inv[(i, j)] * g[j];
            }
        }

        let max_df = (self.n_obs - self.n_fixed) as f64;
        if denom <= 0.0 {
            return max_df;
        }

        let nu = 2.0 * l_phi_l * l_phi_l / denom;
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
    /// variance-covariance matrix of the contrast, `L Phi L'`:
    /// ```text
    /// nu = 2 * E / (E - q)  where E = sum_i nu_i / (nu_i - 2)  for nu_i > 2
    /// ```
    ///
    /// # Arguments
    ///
    /// * `contrast_matrix` - Rows of the contrast matrix `L`, each of length p.
    pub fn satterthwaite_ddf_multi(&self, contrast_matrix: &[Vec<f64>]) -> f64 {
        let q = contrast_matrix.len();
        if q == 0 {
            return 1.0;
        }
        if q == 1 {
            return self.satterthwaite_ddf(&contrast_matrix[0]);
        }

        let p = self.n_fixed;
        let phi = &self.phi;

        // L * Phi * L' (q x q)
        let mut l_phi_lt = DMatrix::zeros(q, q);
        for i in 0..q {
            for j in 0..q {
                let mut val = 0.0;
                for r in 0..p {
                    for s in 0..p {
                        val += contrast_matrix[i][r] * phi[(r, s)] * contrast_matrix[j][s];
                    }
                }
                l_phi_lt[(i, j)] = val;
            }
        }

        let eigen = l_phi_lt.symmetric_eigen();
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        let mut nu_values = Vec::with_capacity(q);
        for i in 0..q {
            if eigenvalues[i] <= 1e-14 {
                continue;
            }
            // Transformed contrast: l_i = eigenvector_i' * L
            let mut l_transformed = vec![0.0; p];
            for j in 0..p {
                let mut val = 0.0;
                for kk in 0..q {
                    val += eigenvectors[(kk, i)] * contrast_matrix[kk][j];
                }
                l_transformed[j] = val;
            }
            nu_values.push(self.satterthwaite_ddf(&l_transformed));
        }

        if nu_values.is_empty() {
            return 1.0;
        }

        let q_eff = nu_values.len() as f64;
        let e_sum: f64 = nu_values
            .iter()
            .map(|&nu_i| {
                if nu_i > 2.0 {
                    nu_i / (nu_i - 2.0)
                } else {
                    // The expectation does not exist for nu_i <= 2; a large
                    // contribution pushes the combined df down.
                    100.0
                }
            })
            .sum();

        let combined_nu = if e_sum > q_eff {
            2.0 * e_sum / (e_sum - q_eff)
        } else {
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

/// `∂Φ/∂θ` for a model whose variance parameters are the scaled variances
/// `[σ²_1, ..., σ²_r, σ²_e]` of random terms `u_k ~ N(0, σ²_k K_k)` and an
/// IID residual.
///
/// With `F = C⁻¹_{·b}` (the `p` columns of `C⁻¹` belonging to the fixed
/// effects) and `F_k` its rows for random term `k`:
///
/// ```text
/// ∂Φ/∂σ²_k = F_k' K_k⁻¹ F_k / σ⁴_k
/// ∂Φ/∂σ²_e = (W F)' (W F) / σ⁴_e,   W = [X Z_1 ... Z_r]
/// ```
///
/// `kinv[k]` is the relationship-matrix inverse `K_k⁻¹` of term `k` (`None`
/// for `K = I`).
pub fn fixed_cov_derivatives_scaled_identity(
    c_inv: &MmeInverse,
    x: &SparseMat,
    z_blocks: &[SparseMat],
    kinv: &[Option<&SparseMat>],
    variance_params: &[f64],
) -> Vec<DMatrix<f64>> {
    let p = x.cols();
    let n = x.rows();
    let f = c_inv.fixed_columns(p);
    let n_terms = z_blocks.len();
    let mut out = Vec::with_capacity(n_terms + 1);

    let mut offset = p;
    for (k, z) in z_blocks.iter().enumerate() {
        let q = z.cols();
        let f_k = f.rows(offset, q).into_owned();
        let kinv_f = match kinv.get(k).copied().flatten() {
            Some(kinv_k) => sparse_times_dense(kinv_k, &f_k),
            None => f_k.clone(),
        };
        let s4 = variance_params[k] * variance_params[k];
        out.push(f_k.transpose() * kinv_f / s4);
        offset += q;
    }

    // W F (n x p)
    let mut wf = DMatrix::zeros(n, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| f[(i, j)]).collect();
        let mut acc = spmv(x, &col);
        let mut off = p;
        for z in z_blocks {
            let q = z.cols();
            let col_k: Vec<f64> = (0..q).map(|i| f[(off + i, j)]).collect();
            for (a, v) in spmv(z, &col_k).iter().enumerate() {
                acc[a] += v;
            }
            off += q;
        }
        for i in 0..n {
            wf[(i, j)] = acc[i];
        }
    }
    let s4e = variance_params[n_terms] * variance_params[n_terms];
    out.push(wf.transpose() * &wf / s4e);
    out
}

/// `A B` for sparse `A` and dense `B`.
fn sparse_times_dense(a: &SparseMat, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(a.rows(), b.ncols());
    for (v, (i, k)) in a.iter() {
        for j in 0..b.ncols() {
            out[(i, j)] += v * b[(k, j)];
        }
    }
    out
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
/// Uses the fixed-effects covariance matrix and its derivatives stored in
/// the [`FitResult`] together with the Average Information matrix at
/// convergence.
///
/// # Returns
///
/// A vector of `WaldTest` results with Satterthwaite ddf, or `None` if the
/// `DdfCalculator` could not be constructed (singular AI matrix, or the
/// derivatives are not available).
pub fn wald_tests_satterthwaite(
    result: &FitResult,
    ai_matrix: &DMatrix<f64>,
) -> Option<Vec<WaldTest>> {
    if result.fixed_effects.is_empty() {
        return Some(Vec::new());
    }

    let n_fixed = result.n_fixed_params;
    let phi = rows_to_matrix(&result.fixed_cov, n_fixed)?;
    let dphi: Vec<DMatrix<f64>> = result
        .fixed_cov_derivatives
        .iter()
        .map(|m| rows_to_matrix(m, n_fixed))
        .collect::<Option<_>>()?;

    let calc = DdfCalculator::new(phi.clone(), dphi, ai_matrix.clone(), result.n_obs)?;

    // Group fixed effects by term name (same logic as in wald_tests)
    let mut term_order: Vec<String> = Vec::new();
    let mut term_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, ef) in result.fixed_effects.iter().enumerate() {
        if !term_indices.contains_key(&ef.term) {
            term_order.push(ef.term.clone());
        }
        term_indices.entry(ef.term.clone()).or_default().push(i);
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

            // General Wald F using the covariance block of Phi.
            let beta: Vec<f64> = indices
                .iter()
                .map(|&idx| result.fixed_effects[idx].estimate)
                .collect();
            let k = indices.len();
            let cov = DMatrix::from_fn(k, k, |a, b| phi[(indices[a], indices[b])]);
            let (f_stat, rank) = super::wald::wald_f_general(&beta, &cov);
            let num_df = rank.max(1);

            let p_value = if rank == 0 {
                1.0
            } else {
                f_distribution_sf(f_stat, num_df as f64, den_df)
            };

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

fn rows_to_matrix(rows: &[Vec<f64>], p: usize) -> Option<DMatrix<f64>> {
    if rows.len() != p || rows.iter().any(|r| r.len() != p) {
        return None;
    }
    Some(DMatrix::from_fn(p, p, |i, j| rows[i][j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmm::{FitResult, MixedModelEquations, NamedEffect, VarianceEstimate};
    use crate::matrix::sparse::sparse_diagonal;
    use approx::assert_relative_eq;

    /// Balanced one-way random model: intercept, `q` groups with
    /// `n_per_group` observations each. Returns (X, Z, y).
    fn one_way_design(q: usize, n_per_group: usize) -> (SparseMat, SparseMat, Vec<f64>) {
        let n = q * n_per_group;
        let mut x = sprs::TriMat::new((n, 1));
        let mut z = sprs::TriMat::new((n, q));
        let mut y = Vec::with_capacity(n);
        for g in 0..q {
            for r in 0..n_per_group {
                let i = g * n_per_group + r;
                x.add_triplet(i, 0, 1.0);
                z.add_triplet(i, g, 1.0);
                y.push(10.0 + g as f64 - 0.3 * r as f64);
            }
        }
        (x.to_csc(), z.to_csc(), y)
    }

    /// Dense `C⁻¹` of the one-way model at the given variances.
    fn one_way_inverse(q: usize, n_per_group: usize, sigma2_u: f64, sigma2_e: f64) -> MmeInverse {
        let (x, z, y) = one_way_design(q, n_per_group);
        let ginv = sparse_diagonal(&vec![1.0 / sigma2_u; q]);
        let mme = MixedModelEquations::assemble(&x, &[z], &y, 1.0 / sigma2_e, &[ginv]);
        mme.solve().unwrap().c_inv.unwrap()
    }

    fn one_way_calculator(
        q: usize,
        n_per_group: usize,
        sigma2_u: f64,
        sigma2_e: f64,
        ai: DMatrix<f64>,
    ) -> Option<DdfCalculator> {
        let (x, z, _) = one_way_design(q, n_per_group);
        let c_inv = one_way_inverse(q, n_per_group, sigma2_u, sigma2_e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[sigma2_u, sigma2_e],
        );
        DdfCalculator::new(c_inv.fixed_block(1), dphi, ai, q * n_per_group)
    }

    /// Helper to build an AI matrix for testing.
    fn make_simple_ai(sigma2_u: f64, sigma2_e: f64, q: usize, n: usize) -> DMatrix<f64> {
        let mut ai = DMatrix::zeros(2, 2);
        ai[(0, 0)] = (q as f64) / (2.0 * sigma2_u * sigma2_u);
        ai[(1, 1)] = (n as f64) / (2.0 * sigma2_e * sigma2_e);
        ai[(0, 1)] = 0.01;
        ai[(1, 0)] = 0.01;
        ai
    }

    #[test]
    fn test_ddf_calculator_creation() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        assert!(one_way_calculator(q, n_per, s2u, s2e, ai).is_some());
    }

    #[test]
    fn test_ddf_calculator_singular_ai() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = DMatrix::zeros(2, 2);
        assert!(one_way_calculator(q, n_per, s2u, s2e, ai).is_none());
    }

    #[test]
    fn derivatives_match_numerical_differentiation() {
        let (q, n_per, s2u, s2e) = (6, 4, 1.7, 0.9);
        let (x, z, y) = one_way_design(q, n_per);
        let phi_at = |s2u: f64, s2e: f64| {
            let ginv = sparse_diagonal(&vec![1.0 / s2u; q]);
            let mme =
                MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
            mme.solve().unwrap().c_inv.unwrap().fixed_block(1)[(0, 0)]
        };
        let c_inv = one_way_inverse(q, n_per, s2u, s2e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[s2u, s2e],
        );
        let h = 1e-5;
        let num_u = (phi_at(s2u + h, s2e) - phi_at(s2u - h, s2e)) / (2.0 * h);
        let num_e = (phi_at(s2u, s2e + h) - phi_at(s2u, s2e - h)) / (2.0 * h);
        assert_relative_eq!(dphi[0][(0, 0)], num_u, epsilon = 1e-7, max_relative = 1e-5);
        assert_relative_eq!(dphi[1][(0, 0)], num_e, epsilon = 1e-7, max_relative = 1e-5);
    }

    #[test]
    fn derivatives_with_relationship_matrix_match_numerical_differentiation() {
        // Two related "animals" per group: K⁻¹ is not the identity.
        let q = 4;
        let n_per = 3;
        let (x, z, y) = one_way_design(q, n_per);
        let mut kinv_tri = sprs::TriMat::new((q, q));
        for i in 0..q {
            kinv_tri.add_triplet(i, i, 1.5);
        }
        kinv_tri.add_triplet(0, 1, -0.5);
        kinv_tri.add_triplet(1, 0, -0.5);
        kinv_tri.add_triplet(2, 3, -0.4);
        kinv_tri.add_triplet(3, 2, -0.4);
        let kinv = kinv_tri.to_csc();
        let (s2u, s2e) = (2.0, 1.3);
        let phi_at = |s2u: f64, s2e: f64| {
            let ginv = kinv.map(|v| v / s2u);
            let mme =
                MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
            mme.solve().unwrap().c_inv.unwrap().fixed_block(1)[(0, 0)]
        };
        let ginv = kinv.map(|v| v / s2u);
        let mme =
            MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
        let c_inv = mme.solve().unwrap().c_inv.unwrap();
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[Some(&kinv)],
            &[s2u, s2e],
        );
        let h = 1e-5;
        let num_u = (phi_at(s2u + h, s2e) - phi_at(s2u - h, s2e)) / (2.0 * h);
        let num_e = (phi_at(s2u, s2e + h) - phi_at(s2u, s2e - h)) / (2.0 * h);
        assert_relative_eq!(dphi[0][(0, 0)], num_u, epsilon = 1e-7, max_relative = 1e-5);
        assert_relative_eq!(dphi[1][(0, 0)], num_e, epsilon = 1e-7, max_relative = 1e-5);
    }

    #[test]
    fn test_satterthwaite_ddf_single_contrast() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let n_obs = q * n_per;
        let ai = make_simple_ai(s2u, s2e, q, n_obs);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu > 0.0, "Satterthwaite ddf should be positive, got {}", nu);
        assert!(nu <= (n_obs - 1) as f64);
        // The intercept of a one-way random model is estimated from the q
        // group means: its df are close to q - 1 when the group variance
        // dominates the residual.
        assert!(nu < 20.0, "ddf {} should reflect the between-group df", nu);
    }

    #[test]
    fn test_satterthwaite_ddf_bounded() {
        let (q, n_per, s2e, s2u) = (3, 5, 1.0, 0.001);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu >= 1.0, "ddf should be at least 1, got {}", nu);
        assert!(nu <= (q * n_per - 1) as f64, "ddf should be at most n - p");
    }

    #[test]
    fn test_containment_ddf() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        assert_relative_eq!(calc.containment_ddf(), (q * n_per - 1) as f64);
    }

    #[test]
    fn test_satterthwaite_multi_df() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        // A repeated contrast is rank 1: one eigenvalue is zero and the
        // multi-df value equals the single-contrast value.
        let nu_multi = calc.satterthwaite_ddf_multi(&[vec![1.0], vec![1.0]]);
        assert!(nu_multi >= 1.0);
        assert!(nu_multi <= (q * n_per - 1) as f64);
        assert_relative_eq!(nu_multi, calc.satterthwaite_ddf(&[1.0]), epsilon = 1e-9);
    }

    #[test]
    fn test_phi_extraction() {
        let (q, n_per, s2e, s2u) = (4, 5, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let expected = one_way_inverse(q, n_per, s2u, s2e).entry(0, 0);
        assert_eq!(calc.phi().nrows(), 1);
        assert_relative_eq!(calc.phi()[(0, 0)], expected, epsilon = 1e-12);
    }

    fn one_way_fit_result(q: usize, n_per: usize, s2u: f64, s2e: f64) -> FitResult {
        let (x, z, _) = one_way_design(q, n_per);
        let c_inv = one_way_inverse(q, n_per, s2u, s2e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[s2u, s2e],
        );
        let phi = c_inv.fixed_block(1);
        FitResult {
            variance_components: vec![
                VarianceEstimate {
                    name: "group".into(),
                    structure: "Identity".into(),
                    parameters: vec![("sigma2".into(), s2u)],
                    se: vec![0.0],
                    at_boundary: vec![false],
                },
                VarianceEstimate {
                    name: "residual".into(),
                    structure: "Identity".into(),
                    parameters: vec![("sigma2".into(), s2e)],
                    se: vec![0.0],
                    at_boundary: vec![false],
                },
            ],
            fixed_effects: vec![NamedEffect {
                term: "mu".into(),
                level: "intercept".into(),
                estimate: 12.0,
                se: phi[(0, 0)].sqrt(),
            }],
            random_effects: vec![],
            log_likelihood: 0.0,
            n_iterations: 1,
            converged: true,
            history: vec![],
            variance_se: vec![0.0, 0.0],
            residuals: vec![],
            fixed_cov: vec![vec![phi[(0, 0)]]],
            at_boundary: vec![false, false],
            n_obs: q * n_per,
            n_fixed_params: 1,
            n_variance_params: 2,
            c_inv: Some(c_inv),
            ai_matrix: None,
            n_random_per_term: vec![q],
            fixed_cov_derivatives: dphi.iter().map(|m| vec![vec![m[(0, 0)]]]).collect(),
        }
    }

    #[test]
    fn test_wald_tests_satterthwaite_basic() {
        let (q, n_per, s2u, s2e) = (5, 10, 3.0, 2.0);
        let result = one_way_fit_result(q, n_per, s2u, s2e);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let tests = wald_tests_satterthwaite(&result, &ai).expect("Should produce Wald tests");
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);
        assert!(tests[0].den_df >= 1.0 && tests[0].den_df <= (q * n_per - 1) as f64);
        assert!(tests[0].f_statistic > 0.0);
        assert!(tests[0].p_value >= 0.0 && tests[0].p_value <= 1.0);
        // Through the FitResult method the AI matrix is required.
        assert!(result.wald_tests_satterthwaite().is_none());
        let mut with_ai = result.clone();
        with_ai.ai_matrix = Some(ai);
        assert_eq!(with_ai.wald_tests_satterthwaite().unwrap().len(), 1);
    }

    #[test]
    fn test_wald_tests_satterthwaite_empty() {
        let mut result = one_way_fit_result(3, 4, 1.0, 1.0);
        result.fixed_effects.clear();
        let ai = make_simple_ai(1.0, 1.0, 3, 12);
        let tests = wald_tests_satterthwaite(&result, &ai);
        assert!(tests.is_some());
        assert!(tests.unwrap().is_empty());
    }

    #[test]
    fn test_satterthwaite_vs_containment_balanced() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let n_obs = q * n_per;
        let ai = make_simple_ai(s2u, s2e, q, n_obs);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu <= calc.containment_ddf());
        assert!(nu >= 1.0);
    }
}
