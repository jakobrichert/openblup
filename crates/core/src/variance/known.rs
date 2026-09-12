use nalgebra::DMatrix;
use sprs::TriMat;

use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use super::traits::VarStruct;

/// A fixed, fully known covariance structure with no parameters, specified
/// through its inverse (e.g. a pedigree A⁻¹ or a genomic G⁻¹).
///
/// Its scale is carried by whatever structure it is combined with, typically
/// through a [`KroneckerStruct`](super::KroneckerStruct): `FA(env) ⊗ Known(A)`
/// or `Diagonal(env) ⊗ Known(A)`. [`Known::identity`] gives the parameter-free
/// identity used for independent inner factors.
#[derive(Debug, Clone)]
pub struct Known {
    inverse: SparseMat,
    /// log|K⁻¹| (so that log|K| = -log_det_inverse).
    log_det_inverse: f64,
}

impl Known {
    /// Build from an inverse covariance matrix (symmetric positive definite).
    ///
    /// The log-determinant is computed once with a dense Cholesky
    /// factorisation, so this is intended for matrices up to a few thousand
    /// rows.
    pub fn from_inverse(inverse: SparseMat) -> Result<Self> {
        let n = inverse.rows();
        if inverse.cols() != n {
            return Err(LmmError::DimensionMismatch {
                expected: n,
                got: inverse.cols(),
                context: "Known inverse must be square".into(),
            });
        }
        let mut dense: DMatrix<f64> = DMatrix::zeros(n, n);
        for (v, (i, j)) in inverse.iter() {
            dense[(i, j)] += *v;
        }
        let chol = dense.cholesky().ok_or(LmmError::NotPositiveDefinite)?;
        let l = chol.l();
        let log_det_inverse = 2.0 * (0..n).map(|i| l[(i, i)].ln()).sum::<f64>();
        Ok(Self {
            inverse,
            log_det_inverse,
        })
    }

    /// The parameter-free identity of the given dimension.
    pub fn identity(dim: usize) -> Self {
        let mut tri = TriMat::new((dim, dim));
        for i in 0..dim {
            tri.add_triplet(i, i, 1.0);
        }
        Self {
            inverse: tri.to_csc(),
            log_det_inverse: 0.0,
        }
    }

    /// Dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.inverse.rows()
    }

    /// The stored inverse.
    pub fn inverse(&self) -> &SparseMat {
        &self.inverse
    }
}

impl VarStruct for Known {
    fn name(&self) -> &str {
        "Known"
    }

    fn n_params(&self) -> usize {
        0
    }

    fn params(&self) -> Vec<f64> {
        Vec::new()
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.is_empty() {
            Ok(())
        } else {
            Err(LmmError::InvalidParameter(format!(
                "Known structure has no parameters, got {}",
                params.len()
            )))
        }
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(dim, self.dim(), "Known dim mismatch");
        // Rarely needed (only for the default Sigma * dSigma^-1 product, and
        // this structure has no derivatives): dense inverse of the inverse.
        let n = self.dim();
        let mut dense: DMatrix<f64> = DMatrix::zeros(n, n);
        for (v, (i, j)) in self.inverse.iter() {
            dense[(i, j)] += *v;
        }
        let cov = dense
            .try_inverse()
            .expect("Known inverse is positive definite by construction");
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            for j in 0..n {
                if cov[(i, j)].abs() > 1e-15 {
                    tri.add_triplet(i, j, cov[(i, j)]);
                }
            }
        }
        tri.to_csc()
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(dim, self.dim(), "Known dim mismatch");
        self.inverse.clone()
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        assert_eq!(dim, self.dim(), "Known dim mismatch");
        -self.log_det_inverse
    }

    fn derivatives_of_inverse(&self, _dim: usize) -> Vec<SparseMat> {
        Vec::new()
    }

    fn derivatives_of_covariance(&self, _dim: usize) -> Vec<SparseMat> {
        Vec::new()
    }

    fn sigma_times_derivatives_of_inverse(&self, _dim: usize) -> Vec<SparseMat> {
        Vec::new()
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        Vec::new()
    }

    fn param_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn fixed_dim(&self) -> Option<usize> {
        Some(self.dim())
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::spmv;

    #[test]
    fn test_known_identity() {
        let mut k = Known::identity(3);
        assert_eq!(k.n_params(), 0);
        assert_eq!(k.log_determinant(3), 0.0);
        let inv = k.inverse_covariance_matrix(3);
        assert_eq!(spmv(&inv, &[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
        assert!(k.derivatives_of_inverse(3).is_empty());
        assert!(k.set_params(&[]).is_ok());
        assert!(k.clone_boxed().set_params(&[1.0]).is_err());
    }

    #[test]
    fn test_known_from_inverse_log_det_and_covariance() {
        // K^-1 = diag(2, 4): log|K^-1| = ln 8, log|K| = -ln 8, K = diag(0.5, 0.25)
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 2.0);
        tri.add_triplet(1, 1, 4.0);
        let k = Known::from_inverse(tri.to_csc()).unwrap();
        assert!((k.log_determinant(2) + 8f64.ln()).abs() < 1e-12);
        let cov = k.covariance_matrix(2);
        assert_eq!(spmv(&cov, &[1.0, 1.0]), vec![0.5, 0.25]);
        assert_eq!(k.fixed_dim(), Some(2));
    }

    #[test]
    fn test_known_rejects_non_pd() {
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 1, -1.0);
        assert!(Known::from_inverse(tri.to_csc()).is_err());
    }
}
