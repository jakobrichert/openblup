use crate::error::{LmmError, Result};
use crate::types::SparseMat;
use sprs::TriMat;

use super::traits::VarStruct;

/// Factor Analytic variance structure for multi-environment trial (MET) analysis.
///
/// Models the covariance as: Σ = ΛΛ' + Ψ
/// where Λ is a p × k loading matrix and Ψ = diag(ψ₁, ..., ψ_p).
///
/// This dramatically reduces parameters from p(p+1)/2 (unstructured) to kp + p.
/// FA1 (k=1) with 10 environments: 20 params vs 55 for unstructured.
///
/// References:
/// - Smith et al. (2001). The analysis of crop cultivar breeding and evaluation trials.
/// - Thompson et al. (2003). A sparse implementation of the Average Information algorithm.
#[derive(Debug, Clone)]
pub struct FactorAnalytic {
    n_env: usize,
    n_factors: usize,
    /// Loadings stored column-major: [λ_{0,0}, λ_{1,0}, ..., λ_{p-1,0}, λ_{0,1}, ...]
    loadings: Vec<f64>,
    /// Specific variances: [ψ_0, ψ_1, ..., ψ_{p-1}]
    psi: Vec<f64>,
}

impl FactorAnalytic {
    /// Create a new FA(k) structure for `n_env` environments with `n_factors` factors.
    pub fn new(n_env: usize, n_factors: usize) -> Self {
        assert!(n_factors > 0 && n_factors <= n_env, "n_factors must be in [1, n_env]");
        // Default: small loadings, unit specific variances
        let loadings = vec![0.1; n_env * n_factors];
        let psi = vec![1.0; n_env];
        Self { n_env, n_factors, loadings, psi }
    }

    /// Create from explicit loadings and specific variances.
    pub fn from_params(n_env: usize, n_factors: usize, loadings: Vec<f64>, psi: Vec<f64>) -> Self {
        assert_eq!(loadings.len(), n_env * n_factors);
        assert_eq!(psi.len(), n_env);
        Self { n_env, n_factors, loadings, psi }
    }

    /// Get loading λ_{i,j} (environment i, factor j).
    fn loading(&self, i: usize, j: usize) -> f64 {
        self.loadings[j * self.n_env + i]
    }

    /// Compute Σ = ΛΛ' + Ψ as dense p×p.
    fn sigma_dense(&self) -> Vec<Vec<f64>> {
        let p = self.n_env;
        let k = self.n_factors;
        let mut sigma = vec![vec![0.0; p]; p];
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                for f in 0..k {
                    sum += self.loading(i, f) * self.loading(j, f);
                }
                sigma[i][j] = sum;
            }
            sigma[i][i] += self.psi[i];
        }
        sigma
    }

    /// Compute Σ⁻¹ using Woodbury: Σ⁻¹ = Ψ⁻¹ - Ψ⁻¹Λ(I + Λ'Ψ⁻¹Λ)⁻¹Λ'Ψ⁻¹
    fn sigma_inv_dense(&self) -> Vec<Vec<f64>> {
        let p = self.n_env;
        let k = self.n_factors;

        let psi_inv: Vec<f64> = self.psi.iter().map(|&v| 1.0 / v).collect();

        // Compute Λ'Ψ⁻¹Λ (k × k)
        let mut ltpil = vec![vec![0.0; k]; k];
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for i in 0..p {
                    s += self.loading(i, a) * psi_inv[i] * self.loading(i, b);
                }
                ltpil[a][b] = s;
            }
        }

        // M = I_k + Λ'Ψ⁻¹Λ
        for a in 0..k {
            ltpil[a][a] += 1.0;
        }

        // Invert M (small k×k matrix)
        let m_inv = invert_small_matrix(&ltpil);

        // Ψ⁻¹Λ (p × k)
        let mut pil = vec![vec![0.0; k]; p];
        for i in 0..p {
            for f in 0..k {
                pil[i][f] = psi_inv[i] * self.loading(i, f);
            }
        }

        // Result = Ψ⁻¹ - (Ψ⁻¹Λ) M⁻¹ (Ψ⁻¹Λ)'
        let mut result = vec![vec![0.0; p]; p];
        for i in 0..p {
            result[i][i] = psi_inv[i];
        }

        // Subtract (Ψ⁻¹Λ) M⁻¹ (Ψ⁻¹Λ)'
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for a in 0..k {
                    for b in 0..k {
                        s += pil[i][a] * m_inv[a][b] * pil[j][b];
                    }
                }
                result[i][j] -= s;
            }
        }

        result
    }
}

/// Invert a small dense matrix using Gauss-Jordan elimination.
fn invert_small_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-15, "Singular matrix in FA inverse");
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    inv
}

impl VarStruct for FactorAnalytic {
    fn name(&self) -> &str {
        "FactorAnalytic"
    }

    fn n_params(&self) -> usize {
        self.n_env * self.n_factors + self.n_env
    }

    fn params(&self) -> Vec<f64> {
        let mut p = self.loadings.clone();
        p.extend_from_slice(&self.psi);
        p
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(LmmError::InvalidParameter(format!(
                "FA({}) with {} envs expects {} parameters, got {}",
                self.n_factors, self.n_env, expected, params.len()
            )));
        }
        let n_load = self.n_env * self.n_factors;
        self.loadings = params[..n_load].to_vec();
        self.psi = params[n_load..].to_vec();
        // Ensure specific variances are positive
        for (i, &v) in self.psi.iter().enumerate() {
            if v <= 0.0 {
                return Err(LmmError::InvalidParameter(format!(
                    "Specific variance ψ_{} must be positive, got {}", i, v
                )));
            }
        }
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(dim, self.n_env);
        let sigma = self.sigma_dense();
        let mut tri = TriMat::new((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                if sigma[i][j].abs() > 1e-15 {
                    tri.add_triplet(i, j, sigma[i][j]);
                }
            }
        }
        tri.to_csc()
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(dim, self.n_env);
        let inv = self.sigma_inv_dense();
        let mut tri = TriMat::new((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                if inv[i][j].abs() > 1e-15 {
                    tri.add_triplet(i, j, inv[i][j]);
                }
            }
        }
        tri.to_csc()
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        assert_eq!(dim, self.n_env);
        let p = self.n_env;
        let k = self.n_factors;

        // log|Σ| = log|Ψ| + log|I_k + Λ'Ψ⁻¹Λ|
        let log_psi: f64 = self.psi.iter().map(|v| v.ln()).sum();

        let psi_inv: Vec<f64> = self.psi.iter().map(|&v| 1.0 / v).collect();
        let mut m = vec![vec![0.0; k]; k];
        for a in 0..k {
            for b in 0..k {
                let mut s = 0.0;
                for i in 0..p {
                    s += self.loading(i, a) * psi_inv[i] * self.loading(i, b);
                }
                m[a][b] = s;
            }
            m[a][a] += 1.0;
        }

        // log|M| via LU-like determinant for small matrix
        let log_det_m = log_det_small(&m);

        log_psi + log_det_m
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        assert_eq!(dim, self.n_env);
        let n_params = self.n_params();
        let eps = 1e-7;
        let mut derivs = Vec::with_capacity(n_params);

        for p_idx in 0..n_params {
            let mut params_plus = self.params();
            let mut params_minus = self.params();
            params_plus[p_idx] += eps;
            params_minus[p_idx] -= eps;

            let fa_plus = FactorAnalytic::from_params(
                self.n_env, self.n_factors,
                params_plus[..self.n_env * self.n_factors].to_vec(),
                params_plus[self.n_env * self.n_factors..].to_vec(),
            );
            let fa_minus = FactorAnalytic::from_params(
                self.n_env, self.n_factors,
                params_minus[..self.n_env * self.n_factors].to_vec(),
                params_minus[self.n_env * self.n_factors..].to_vec(),
            );

            // Ensure positive specific variances for perturbed params
            let inv_plus = fa_plus.sigma_inv_dense();
            let inv_minus = fa_minus.sigma_inv_dense();

            let mut tri = TriMat::new((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    let d = (inv_plus[i][j] - inv_minus[i][j]) / (2.0 * eps);
                    if d.abs() > 1e-15 {
                        tri.add_triplet(i, j, d);
                    }
                }
            }
            derivs.push(tri.to_csc());
        }
        derivs
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut b = Vec::with_capacity(self.n_params());
        // Loadings: unconstrained
        for _ in 0..(self.n_env * self.n_factors) {
            b.push((f64::NEG_INFINITY, f64::INFINITY));
        }
        // Specific variances: positive
        for _ in 0..self.n_env {
            b.push((1e-6, f64::INFINITY));
        }
        b
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }
}

/// Log-determinant of a small dense matrix via Gaussian elimination.
fn log_det_small(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m = a.to_vec();
    let mut log_det = 0.0;
    let mut sign = 1.0_f64;

    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[max_row][col].abs() {
                max_row = row;
            }
        }
        if max_row != col {
            m.swap(col, max_row);
            sign = -sign;
        }
        let pivot = m[col][col];
        if pivot.abs() < 1e-15 {
            return f64::NEG_INFINITY;
        }
        log_det += pivot.abs().ln();
        if pivot < 0.0 {
            sign = -sign;
        }
        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            for j in (col + 1)..n {
                m[row][j] -= factor * m[col][j];
            }
        }
    }
    if sign < 0.0 {
        // Negative determinant shouldn't happen for PD matrices
        log_det
    } else {
        log_det
    }
}

/// FA1: single-factor model.
pub type FA1 = FactorAnalytic;

/// Convenience constructor for FA1.
pub fn fa1(n_env: usize) -> FactorAnalytic {
    FactorAnalytic::new(n_env, 1)
}

/// Convenience constructor for FA2.
pub fn fa2(n_env: usize) -> FactorAnalytic {
    FactorAnalytic::new(n_env, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn get_entry(mat: &SparseMat, row: usize, col: usize) -> f64 {
        for (&val, (r, c)) in mat.iter() {
            if r == row && c == col {
                return val;
            }
        }
        0.0
    }

    #[test]
    fn test_fa1_param_count() {
        let fa = fa1(4);
        assert_eq!(fa.n_params(), 4 + 4); // 4 loadings + 4 psi
    }

    #[test]
    fn test_fa2_param_count() {
        let fa = fa2(5);
        assert_eq!(fa.n_params(), 10 + 5); // 2*5 loadings + 5 psi
    }

    #[test]
    fn test_fa1_covariance() {
        // FA1 with 3 envs: Λ = [2, 1, 3]', Ψ = diag(1, 2, 0.5)
        // Σ = ΛΛ' + Ψ = [[4+1, 2, 6], [2, 1+2, 3], [6, 3, 9+0.5]]
        //              = [[5, 2, 6], [2, 3, 3], [6, 3, 9.5]]
        let fa = FactorAnalytic::from_params(3, 1, vec![2.0, 1.0, 3.0], vec![1.0, 2.0, 0.5]);
        let cov = fa.covariance_matrix(3);
        assert_relative_eq!(get_entry(&cov, 0, 0), 5.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 0, 1), 2.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 0, 2), 6.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 1, 1), 3.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 1, 2), 3.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 2, 2), 9.5, epsilon = 1e-10);
    }

    #[test]
    fn test_fa_inverse_woodbury() {
        // Check Σ * Σ⁻¹ ≈ I
        let fa = FactorAnalytic::from_params(3, 1, vec![2.0, 1.0, 3.0], vec![1.0, 2.0, 0.5]);
        let cov = fa.covariance_matrix(3);
        let inv = fa.inverse_covariance_matrix(3);

        for j in 0..3 {
            let mut ej = vec![0.0; 3];
            ej[j] = 1.0;
            let col = crate::matrix::sparse::spmv(&inv, &ej);
            let res = crate::matrix::sparse::spmv(&cov, &col);
            for i in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res[i], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_fa2_inverse() {
        let fa = FactorAnalytic::from_params(
            4, 2,
            vec![1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.1],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let cov = fa.covariance_matrix(4);
        let inv = fa.inverse_covariance_matrix(4);

        for j in 0..4 {
            let mut ej = vec![0.0; 4];
            ej[j] = 1.0;
            let col = crate::matrix::sparse::spmv(&inv, &ej);
            let res = crate::matrix::sparse::spmv(&cov, &col);
            for i in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res[i], expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_fa_log_determinant() {
        // Compare log|Σ| with direct computation
        let fa = FactorAnalytic::from_params(3, 1, vec![2.0, 1.0, 3.0], vec![1.0, 2.0, 0.5]);
        let logdet = fa.log_determinant(3);

        // Compute Σ directly and get determinant via nalgebra
        let sigma = fa.sigma_dense();
        let m = nalgebra::DMatrix::from_fn(3, 3, |i, j| sigma[i][j]);
        let expected = m.determinant().ln();

        assert_relative_eq!(logdet, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_fa_set_params() {
        let mut fa = fa1(3);
        let params = vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5];
        fa.set_params(&params).unwrap();
        assert_eq!(fa.params(), params);
    }

    #[test]
    fn test_fa_set_params_bad_psi() {
        let mut fa = fa1(3);
        let params = vec![1.0, 2.0, 3.0, 0.5, -0.1, 0.5];
        assert!(fa.set_params(&params).is_err());
    }

    #[test]
    fn test_fa_bounds() {
        let fa = fa2(3);
        let bounds = fa.bounds();
        assert_eq!(bounds.len(), 9); // 6 loadings + 3 psi
        // First 6 are loadings: unconstrained
        assert!(bounds[0].0.is_infinite() && bounds[0].0 < 0.0);
        // Last 3 are psi: positive
        assert!(bounds[6].0 > 0.0);
    }

    #[test]
    fn test_fa_name() {
        let fa = fa1(4);
        assert_eq!(fa.name(), "FactorAnalytic");
    }

    #[test]
    fn test_fa_clone_boxed() {
        let fa = FactorAnalytic::from_params(2, 1, vec![1.0, 0.5], vec![1.0, 2.0]);
        let boxed: Box<dyn VarStruct> = fa.clone_boxed();
        assert_eq!(boxed.n_params(), 4);
    }

    #[test]
    fn test_fa_symmetry() {
        let fa = FactorAnalytic::from_params(
            4, 2,
            vec![1.0, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.1],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let cov = fa.covariance_matrix(4);
        let inv = fa.inverse_covariance_matrix(4);
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(
                    get_entry(&cov, i, j), get_entry(&cov, j, i), epsilon = 1e-10
                );
                assert_relative_eq!(
                    get_entry(&inv, i, j), get_entry(&inv, j, i), epsilon = 1e-10
                );
            }
        }
    }
}
