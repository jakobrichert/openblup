use crate::error::{LmmError, Result};
use crate::matrix::sparse::sparse_diagonal;
use crate::types::SparseMat;
use sprs::TriMat;

use super::traits::VarStruct;

/// Heterogeneous diagonal variance structure: Sigma = diag(sigma^2_1, sigma^2_2, ..., sigma^2_k).
///
/// Each level has its own variance. This is useful when different groups (e.g., environments,
/// traits) have distinct variances but are assumed independent.
///
/// Parameters: [sigma^2_1, sigma^2_2, ..., sigma^2_k] (k parameters).
///
/// The inverse is trivially diag(1/sigma^2_1, 1/sigma^2_2, ..., 1/sigma^2_k).
///
/// Log-determinant: sum of ln(sigma^2_i).
#[derive(Debug, Clone)]
pub struct Diagonal {
    variances: Vec<f64>,
}

impl Diagonal {
    /// Create a new Diagonal structure with the given per-level variances.
    pub fn new(variances: Vec<f64>) -> Self {
        Self { variances }
    }

    /// Create with default starting values (all variances = 1.0).
    pub fn default_start(k: usize) -> Self {
        Self {
            variances: vec![1.0; k],
        }
    }

    /// Number of distinct variance levels.
    pub fn dim(&self) -> usize {
        self.variances.len()
    }
}

impl VarStruct for Diagonal {
    fn name(&self) -> &str {
        "Diagonal"
    }

    fn n_params(&self) -> usize {
        self.variances.len()
    }

    fn params(&self) -> Vec<f64> {
        self.variances.clone()
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != self.variances.len() {
            return Err(LmmError::InvalidParameter(format!(
                "Diagonal expects {} parameters, got {}",
                self.variances.len(),
                params.len()
            )));
        }
        for (i, &val) in params.iter().enumerate() {
            if val <= 0.0 {
                return Err(LmmError::InvalidParameter(format!(
                    "Diagonal variance[{}] must be positive, got {}",
                    i, val
                )));
            }
        }
        self.variances = params.to_vec();
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(
            dim,
            self.variances.len(),
            "Diagonal dim mismatch: structure has {} variances but dim={} requested",
            self.variances.len(),
            dim
        );
        sparse_diagonal(&self.variances)
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(
            dim,
            self.variances.len(),
            "Diagonal dim mismatch: structure has {} variances but dim={} requested",
            self.variances.len(),
            dim
        );
        let inv_diag: Vec<f64> = self.variances.iter().map(|&v| 1.0 / v).collect();
        sparse_diagonal(&inv_diag)
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        assert_eq!(
            dim,
            self.variances.len(),
            "Diagonal dim mismatch: structure has {} variances but dim={} requested",
            self.variances.len(),
            dim
        );
        self.variances.iter().map(|v| v.ln()).sum()
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        assert_eq!(
            dim,
            self.variances.len(),
            "Diagonal dim mismatch: structure has {} variances but dim={} requested",
            self.variances.len(),
            dim
        );

        // d(Sigma^{-1})/d(sigma^2_k) has only one non-zero entry at (k, k):
        // d(1/sigma^2_k)/d(sigma^2_k) = -1/sigma^4_k
        let k = self.variances.len();
        let mut derivs = Vec::with_capacity(k);
        for i in 0..k {
            let mut tri = TriMat::new((k, k));
            let val = -1.0 / (self.variances[i] * self.variances[i]);
            tri.add_triplet(i, i, val);
            derivs.push(tri.to_csc());
        }
        derivs
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(1e-10, f64::INFINITY); self.variances.len()]
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }

    fn initial_params(&self) -> Vec<f64> {
        self.variances.clone()
    }
}

impl Default for Diagonal {
    fn default() -> Self {
        Self::default_start(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::spmv;
    use approx::assert_relative_eq;

    #[test]
    fn test_diagonal_basic() {
        let diag = Diagonal::new(vec![2.0, 3.0, 5.0]);
        assert_eq!(diag.name(), "Diagonal");
        assert_eq!(diag.n_params(), 3);
        assert_eq!(diag.params(), vec![2.0, 3.0, 5.0]);
        assert_eq!(diag.dim(), 3);
    }

    #[test]
    fn test_diagonal_covariance() {
        let diag = Diagonal::new(vec![2.0, 3.0, 5.0]);
        let cov = diag.covariance_matrix(3);
        let result = spmv(&cov, &[1.0, 1.0, 1.0]);
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_inverse() {
        let diag = Diagonal::new(vec![2.0, 4.0, 8.0]);
        let inv = diag.inverse_covariance_matrix(3);
        let result = spmv(&inv, &[1.0, 1.0, 1.0]);
        assert_relative_eq!(result[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(result[1], 0.25, epsilon = 1e-10);
        assert_relative_eq!(result[2], 0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_inverse_correctness() {
        let diag = Diagonal::new(vec![2.0, 3.0, 5.0]);
        let cov = diag.covariance_matrix(3);
        let inv = diag.inverse_covariance_matrix(3);

        // Sigma * Sigma^{-1} should be I
        for j in 0..3 {
            let mut ej = vec![0.0; 3];
            ej[j] = 1.0;
            let col = spmv(&inv, &ej);
            let res = spmv(&cov, &col);
            for i in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res[i], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_diagonal_log_determinant() {
        let diag = Diagonal::new(vec![2.0, 3.0, 5.0]);
        let logdet = diag.log_determinant(3);
        let expected = 2.0_f64.ln() + 3.0_f64.ln() + 5.0_f64.ln();
        assert_relative_eq!(logdet, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_derivatives() {
        let diag = Diagonal::new(vec![2.0, 3.0, 5.0]);
        let derivs = diag.derivatives_of_inverse(3);
        assert_eq!(derivs.len(), 3);

        // d(1/sigma^2_0)/d(sigma^2_0) = -1/4
        let d0 = spmv(&derivs[0], &[1.0, 1.0, 1.0]);
        assert_relative_eq!(d0[0], -0.25, epsilon = 1e-10);
        assert_relative_eq!(d0[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d0[2], 0.0, epsilon = 1e-10);

        // d(1/sigma^2_1)/d(sigma^2_1) = -1/9
        let d1 = spmv(&derivs[1], &[1.0, 1.0, 1.0]);
        assert_relative_eq!(d1[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d1[1], -1.0 / 9.0, epsilon = 1e-10);
        assert_relative_eq!(d1[2], 0.0, epsilon = 1e-10);

        // d(1/sigma^2_2)/d(sigma^2_2) = -1/25
        let d2 = spmv(&derivs[2], &[1.0, 1.0, 1.0]);
        assert_relative_eq!(d2[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d2[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(d2[2], -1.0 / 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_set_params() {
        let mut diag = Diagonal::new(vec![1.0, 1.0]);
        diag.set_params(&[4.0, 9.0]).unwrap();
        assert_eq!(diag.params(), vec![4.0, 9.0]);
    }

    #[test]
    fn test_diagonal_set_params_validation() {
        let mut diag = Diagonal::new(vec![1.0, 1.0]);
        assert!(diag.set_params(&[1.0]).is_err()); // wrong count
        assert!(diag.set_params(&[1.0, 0.0]).is_err()); // zero variance
        assert!(diag.set_params(&[1.0, -1.0]).is_err()); // negative variance
    }

    #[test]
    fn test_diagonal_bounds() {
        let diag = Diagonal::new(vec![1.0, 2.0, 3.0]);
        let bounds = diag.bounds();
        assert_eq!(bounds.len(), 3);
        for (lo, hi) in &bounds {
            assert!(*lo > 0.0);
            assert!(hi.is_infinite());
        }
    }

    #[test]
    fn test_diagonal_clone_boxed() {
        let diag = Diagonal::new(vec![2.0, 3.0]);
        let boxed: Box<dyn VarStruct> = diag.clone_boxed();
        assert_eq!(boxed.params(), vec![2.0, 3.0]);
        assert_eq!(boxed.name(), "Diagonal");
    }
}
