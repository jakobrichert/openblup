use crate::error::{LmmError, Result};
use crate::matrix::sparse::{sparse_diagonal, sparse_identity};
use crate::types::SparseMat;

use super::traits::VarStruct;

/// Identity (IID) variance structure: G = sigma^2 * I.
///
/// This is the simplest variance structure, used for independent random effects
/// with common variance.
///
/// Parameters: [sigma^2] (1 parameter).
#[derive(Debug, Clone)]
pub struct Identity {
    sigma2: f64,
}

impl Identity {
    /// Create a new Identity structure with the given variance.
    pub fn new(sigma2: f64) -> Self {
        Self { sigma2 }
    }

    /// Create with default starting value.
    pub fn default_start() -> Self {
        Self { sigma2: 1.0 }
    }
}

impl VarStruct for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.sigma2]
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 1 {
            return Err(LmmError::InvalidParameter(format!(
                "Identity expects 1 parameter, got {}",
                params.len()
            )));
        }
        self.sigma2 = params[0];
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        sparse_diagonal(&vec![self.sigma2; dim])
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        let inv_sigma2 = 1.0 / self.sigma2;
        sparse_diagonal(&vec![inv_sigma2; dim])
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        dim as f64 * self.sigma2.ln()
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        // d(sigma^{-2} I)/d(sigma^2) = -sigma^{-4} I
        let deriv_val = -1.0 / (self.sigma2 * self.sigma2);
        vec![sparse_diagonal(&vec![deriv_val; dim])]
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(1e-10, f64::INFINITY)]
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }

    fn initial_params(&self) -> Vec<f64> {
        vec![self.sigma2]
    }
}

/// Scaled Identity for use with a relationship matrix.
///
/// When using G = sigma^2 * K (e.g., A-matrix), the G^{-1} needed for the MME
/// is (1/sigma^2) * K^{-1}. The Identity variance structure produces the scalar
/// multiplier; the relationship matrix inverse is provided separately.
impl Identity {
    /// Compute the scaling factor for G^{-1}: returns 1/sigma^2.
    pub fn inverse_scale(&self) -> f64 {
        1.0 / self.sigma2
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::default_start()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::spmv;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_basic() {
        let id = Identity::new(4.0);
        assert_eq!(id.name(), "Identity");
        assert_eq!(id.n_params(), 1);
        assert_eq!(id.params(), vec![4.0]);
    }

    #[test]
    fn test_identity_covariance() {
        let id = Identity::new(2.5);
        let cov = id.covariance_matrix(3);
        let x = vec![1.0, 1.0, 1.0];
        let result = spmv(&cov, &x);
        for val in &result {
            assert_relative_eq!(*val, 2.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity_inverse() {
        let id = Identity::new(4.0);
        let inv = id.inverse_covariance_matrix(3);
        let x = vec![1.0, 1.0, 1.0];
        let result = spmv(&inv, &x);
        for val in &result {
            assert_relative_eq!(*val, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity_log_determinant() {
        let id = Identity::new(3.0);
        let logdet = id.log_determinant(5);
        assert_relative_eq!(logdet, 5.0 * 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_identity_derivatives() {
        let id = Identity::new(2.0);
        let derivs = id.derivatives_of_inverse(3);
        assert_eq!(derivs.len(), 1);
        // d(1/sigma^2 * I)/d(sigma^2) = -1/sigma^4 * I = -0.25 * I
        let result = spmv(&derivs[0], &[1.0, 1.0, 1.0]);
        for val in &result {
            assert_relative_eq!(*val, -0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_identity_set_params() {
        let mut id = Identity::new(1.0);
        id.set_params(&[5.0]).unwrap();
        assert_eq!(id.params(), vec![5.0]);
    }

    #[test]
    fn test_identity_set_params_wrong_count() {
        let mut id = Identity::new(1.0);
        assert!(id.set_params(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_identity_bounds() {
        let id = Identity::new(1.0);
        let bounds = id.bounds();
        assert_eq!(bounds.len(), 1);
        assert!(bounds[0].0 > 0.0);
        assert!(bounds[0].1.is_infinite());
    }

    #[test]
    fn test_clone_boxed() {
        let id = Identity::new(3.0);
        let boxed: Box<dyn VarStruct> = id.clone_boxed();
        assert_eq!(boxed.params(), vec![3.0]);
    }
}
