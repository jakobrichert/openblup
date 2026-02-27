use crate::error::{LmmError, Result};
use crate::types::SparseMat;
use sprs::TriMat;

use super::traits::VarStruct;

/// First-order autoregressive variance structure: Sigma[i,j] = sigma^2 * rho^|i-j|.
///
/// Parameters: [sigma^2, rho] where sigma^2 > 0 and -1 < rho < 1.
///
/// The inverse has a closed-form tridiagonal structure, making it highly
/// efficient for spatial models (e.g., row or column correlations in field trials).
///
/// Inverse:
/// ```text
/// Sigma^{-1} = (1 / (sigma^2 * (1 - rho^2))) *
///   [ 1    -rho    0      0    ...  0       0    ]
///   [-rho  1+rho^2 -rho   0    ...  0       0    ]
///   [ 0    -rho    1+rho^2 -rho ...  0       0    ]
///   [ ...                                    ...  ]
///   [ 0     0      0       0   ... 1+rho^2  -rho  ]
///   [ 0     0      0       0   ... -rho      1    ]
/// ```
///
/// Log-determinant: n * ln(sigma^2) + (n-1) * ln(1 - rho^2)
#[derive(Debug, Clone)]
pub struct AR1 {
    sigma2: f64,
    rho: f64,
}

impl AR1 {
    /// Create a new AR1 structure with the given variance and autocorrelation.
    pub fn new(sigma2: f64, rho: f64) -> Self {
        Self { sigma2, rho }
    }

    /// Create with default starting values (sigma^2 = 1.0, rho = 0.5).
    pub fn default_start() -> Self {
        Self {
            sigma2: 1.0,
            rho: 0.5,
        }
    }
}

impl VarStruct for AR1 {
    fn name(&self) -> &str {
        "AR1"
    }

    fn n_params(&self) -> usize {
        2
    }

    fn params(&self) -> Vec<f64> {
        vec![self.sigma2, self.rho]
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 2 {
            return Err(LmmError::InvalidParameter(format!(
                "AR1 expects 2 parameters, got {}",
                params.len()
            )));
        }
        if params[0] <= 0.0 {
            return Err(LmmError::InvalidParameter(
                "AR1 sigma^2 must be positive".to_string(),
            ));
        }
        if params[1].abs() >= 1.0 {
            return Err(LmmError::InvalidParameter(
                "AR1 rho must satisfy |rho| < 1".to_string(),
            ));
        }
        self.sigma2 = params[0];
        self.rho = params[1];
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        let mut tri = TriMat::new((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let dist = if i > j { i - j } else { j - i };
                let val = self.sigma2 * self.rho.powi(dist as i32);
                if val.abs() > 1e-15 {
                    tri.add_triplet(i, j, val);
                }
            }
        }
        tri.to_csc()
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        if dim == 0 {
            return TriMat::new((0, 0)).to_csc();
        }
        if dim == 1 {
            let mut tri = TriMat::new((1, 1));
            tri.add_triplet(0, 0, 1.0 / self.sigma2);
            return tri.to_csc();
        }

        let rho = self.rho;
        let rho2 = rho * rho;
        let scale = 1.0 / (self.sigma2 * (1.0 - rho2));

        let mut tri = TriMat::new((dim, dim));

        // First row/col: diagonal = 1
        tri.add_triplet(0, 0, scale * 1.0);

        // Interior rows: diagonal = 1 + rho^2
        for i in 1..dim - 1 {
            tri.add_triplet(i, i, scale * (1.0 + rho2));
        }

        // Last row/col: diagonal = 1
        tri.add_triplet(dim - 1, dim - 1, scale * 1.0);

        // Off-diagonals: -rho
        for i in 0..dim - 1 {
            tri.add_triplet(i, i + 1, scale * (-rho));
            tri.add_triplet(i + 1, i, scale * (-rho));
        }

        tri.to_csc()
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        if dim == 0 {
            return 0.0;
        }
        if dim == 1 {
            return self.sigma2.ln();
        }
        let rho2 = self.rho * self.rho;
        dim as f64 * self.sigma2.ln() + (dim - 1) as f64 * (1.0 - rho2).ln()
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        if dim == 0 {
            return vec![TriMat::new((0, 0)).to_csc(), TriMat::new((0, 0)).to_csc()];
        }

        let rho = self.rho;
        let sigma2 = self.sigma2;

        // Derivative with respect to sigma^2:
        // d(Sigma^{-1})/d(sigma^2) = -Sigma^{-1} / sigma^2
        // This is simply scaling the inverse by -1/sigma^2.
        let inv = self.inverse_covariance_matrix(dim);
        let neg_inv_sigma2 = -1.0 / sigma2;
        let d_sigma2 = scale_sparse(&inv, neg_inv_sigma2);

        // Derivative with respect to rho:
        // Sigma^{-1} = (1/(sigma^2*(1-rho^2))) * T
        // where T is the tridiagonal matrix.
        // d/drho[ 1/(sigma^2*(1-rho^2)) ] = 2*rho / (sigma^2*(1-rho^2)^2)
        // d(T)/d(rho): diagonal entries: d(1)/drho = 0 at corners, d(1+rho^2)/drho = 2*rho interior
        //              off-diagonal entries: d(-rho)/drho = -1
        if dim == 1 {
            // For dim=1, Sigma^{-1} = 1/sigma^2, no rho dependence.
            let d_rho = TriMat::new((1, 1)).to_csc();
            return vec![d_sigma2, d_rho];
        }

        let rho2 = rho * rho;
        let one_minus_rho2 = 1.0 - rho2;

        // Compute d/drho of the full inverse.
        // Let s = 1/(sigma^2*(1-rho^2)), then ds/drho = 2*rho/(sigma^2*(1-rho^2)^2) = s * 2*rho/(1-rho^2).
        let s = 1.0 / (sigma2 * one_minus_rho2);
        let ds_drho = s * 2.0 * rho / one_minus_rho2;

        // dT/drho entries:
        // diagonal: 0 at corners, 2*rho at interior
        // off-diagonal: -1

        let mut tri = TriMat::new((dim, dim));

        // ds/drho * T + s * dT/drho
        // For corners (0,0) and (n-1,n-1): T[i,i] = 1, dT/drho = 0
        //   => derivative = ds/drho * 1
        tri.add_triplet(0, 0, ds_drho * 1.0);
        tri.add_triplet(dim - 1, dim - 1, ds_drho * 1.0);

        // For interior diagonal: T[i,i] = 1+rho^2, dT/drho = 2*rho
        //   => derivative = ds/drho * (1+rho^2) + s * 2*rho
        for i in 1..dim - 1 {
            let val = ds_drho * (1.0 + rho2) + s * 2.0 * rho;
            tri.add_triplet(i, i, val);
        }

        // For off-diagonal: T[i,i+1] = -rho, dT/drho = -1
        //   => derivative = ds/drho * (-rho) + s * (-1)
        for i in 0..dim - 1 {
            let val = ds_drho * (-rho) + s * (-1.0);
            tri.add_triplet(i, i + 1, val);
            tri.add_triplet(i + 1, i, val);
        }

        let d_rho = tri.to_csc();

        vec![d_sigma2, d_rho]
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(1e-10, f64::INFINITY), (-0.999, 0.999)]
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }

    fn initial_params(&self) -> Vec<f64> {
        vec![self.sigma2, self.rho]
    }
}

impl Default for AR1 {
    fn default() -> Self {
        Self::default_start()
    }
}

/// Scale every entry of a sparse matrix by a scalar.
fn scale_sparse(mat: &SparseMat, scale: f64) -> SparseMat {
    let mut tri = TriMat::new((mat.rows(), mat.cols()));
    for (val, (i, j)) in mat.iter() {
        tri.add_triplet(i, j, val * scale);
    }
    tri.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::spmv;
    use approx::assert_relative_eq;

    #[test]
    fn test_ar1_basic() {
        let ar1 = AR1::new(2.0, 0.7);
        assert_eq!(ar1.name(), "AR1");
        assert_eq!(ar1.n_params(), 2);
        assert_eq!(ar1.params(), vec![2.0, 0.7]);
    }

    #[test]
    fn test_ar1_covariance_pattern() {
        let sigma2 = 3.0;
        let rho = 0.6;
        let ar1 = AR1::new(sigma2, rho);
        let cov = ar1.covariance_matrix(4);

        // Check that Sigma[i,j] = sigma^2 * rho^|i-j|
        for i in 0..4 {
            for j in 0..4 {
                let dist = if i > j { i - j } else { j - i };
                let expected = sigma2 * rho.powi(dist as i32);
                let actual = get_entry(&cov, i, j);
                assert_relative_eq!(actual, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ar1_inverse_tridiagonal() {
        let ar1 = AR1::new(2.0, 0.7);
        let dim = 5;
        let cov = ar1.covariance_matrix(dim);
        let inv = ar1.inverse_covariance_matrix(dim);

        // Sigma * Sigma^{-1} should be approximately I.
        let product = sparse_product(&cov, &inv, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[i][j], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ar1_log_determinant() {
        let sigma2 = 2.0;
        let rho = 0.5;
        let ar1 = AR1::new(sigma2, rho);
        let dim = 4;

        let logdet = ar1.log_determinant(dim);
        let expected = dim as f64 * sigma2.ln() + (dim - 1) as f64 * (1.0 - rho * rho).ln();
        assert_relative_eq!(logdet, expected, epsilon = 1e-10);

        // Cross-check: log-determinant from product of eigenvalues via covariance matrix.
        // For a small matrix, compute det directly.
        let cov = ar1.covariance_matrix(dim);
        let dense = sparse_to_dense(&cov, dim);
        let det = nalgebra::DMatrix::from_row_slice(dim, dim, &dense.concat()).determinant();
        assert_relative_eq!(logdet, det.ln(), epsilon = 1e-8);
    }

    #[test]
    fn test_ar1_rho_zero_gives_scaled_identity() {
        let sigma2 = 4.0;
        let ar1 = AR1::new(sigma2, 0.0);
        let dim = 3;

        let cov = ar1.covariance_matrix(dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { sigma2 } else { 0.0 };
                let actual = get_entry(&cov, i, j);
                assert_relative_eq!(actual, expected, epsilon = 1e-10);
            }
        }

        let inv = ar1.inverse_covariance_matrix(dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 / sigma2 } else { 0.0 };
                let actual = get_entry(&inv, i, j);
                assert_relative_eq!(actual, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_ar1_bounds_checking() {
        let mut ar1 = AR1::new(1.0, 0.5);

        // Invalid sigma^2
        assert!(ar1.set_params(&[0.0, 0.5]).is_err());
        assert!(ar1.set_params(&[-1.0, 0.5]).is_err());

        // Invalid rho
        assert!(ar1.set_params(&[1.0, 1.0]).is_err());
        assert!(ar1.set_params(&[1.0, -1.0]).is_err());

        // Valid
        assert!(ar1.set_params(&[2.0, 0.99]).is_ok());
        assert!(ar1.set_params(&[2.0, -0.99]).is_ok());
    }

    #[test]
    fn test_ar1_set_params() {
        let mut ar1 = AR1::new(1.0, 0.5);
        ar1.set_params(&[3.0, 0.8]).unwrap();
        assert_eq!(ar1.params(), vec![3.0, 0.8]);
    }

    #[test]
    fn test_ar1_set_params_wrong_count() {
        let mut ar1 = AR1::new(1.0, 0.5);
        assert!(ar1.set_params(&[1.0]).is_err());
        assert!(ar1.set_params(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_ar1_clone_boxed() {
        let ar1 = AR1::new(2.0, 0.6);
        let boxed: Box<dyn VarStruct> = ar1.clone_boxed();
        assert_eq!(boxed.params(), vec![2.0, 0.6]);
        assert_eq!(boxed.name(), "AR1");
    }

    #[test]
    fn test_ar1_derivatives_sigma2() {
        // Numerical check: d(Sigma^{-1})/d(sigma^2) ~ (Sigma^{-1}(sigma^2+eps) - Sigma^{-1}(sigma^2-eps)) / (2*eps)
        let sigma2 = 2.0;
        let rho = 0.6;
        let dim = 4;
        let eps = 1e-6;

        let ar1 = AR1::new(sigma2, rho);
        let derivs = ar1.derivatives_of_inverse(dim);
        let d_sigma2 = &derivs[0];

        let ar1_plus = AR1::new(sigma2 + eps, rho);
        let ar1_minus = AR1::new(sigma2 - eps, rho);
        let inv_plus = ar1_plus.inverse_covariance_matrix(dim);
        let inv_minus = ar1_minus.inverse_covariance_matrix(dim);

        for i in 0..dim {
            for j in 0..dim {
                let numerical = (get_entry(&inv_plus, i, j) - get_entry(&inv_minus, i, j))
                    / (2.0 * eps);
                let analytical = get_entry(d_sigma2, i, j);
                assert_relative_eq!(analytical, numerical, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_ar1_derivatives_rho() {
        // Numerical check for rho derivative.
        let sigma2 = 2.0;
        let rho = 0.6;
        let dim = 4;
        let eps = 1e-6;

        let ar1 = AR1::new(sigma2, rho);
        let derivs = ar1.derivatives_of_inverse(dim);
        let d_rho = &derivs[1];

        let ar1_plus = AR1::new(sigma2, rho + eps);
        let ar1_minus = AR1::new(sigma2, rho - eps);
        let inv_plus = ar1_plus.inverse_covariance_matrix(dim);
        let inv_minus = ar1_minus.inverse_covariance_matrix(dim);

        for i in 0..dim {
            for j in 0..dim {
                let numerical = (get_entry(&inv_plus, i, j) - get_entry(&inv_minus, i, j))
                    / (2.0 * eps);
                let analytical = get_entry(d_rho, i, j);
                assert_relative_eq!(analytical, numerical, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_ar1_dim1() {
        let ar1 = AR1::new(3.0, 0.5);
        let cov = ar1.covariance_matrix(1);
        assert_relative_eq!(get_entry(&cov, 0, 0), 3.0, epsilon = 1e-10);

        let inv = ar1.inverse_covariance_matrix(1);
        assert_relative_eq!(get_entry(&inv, 0, 0), 1.0 / 3.0, epsilon = 1e-10);

        let logdet = ar1.log_determinant(1);
        assert_relative_eq!(logdet, 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_ar1_negative_rho() {
        let ar1 = AR1::new(1.0, -0.5);
        let dim = 4;
        let cov = ar1.covariance_matrix(dim);
        let inv = ar1.inverse_covariance_matrix(dim);

        let product = sparse_product(&cov, &inv, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[i][j], expected, epsilon = 1e-10);
            }
        }
    }

    // --- Test helpers ---

    fn get_entry(mat: &SparseMat, row: usize, col: usize) -> f64 {
        for (val, (r, c)) in mat.iter() {
            if r == row && c == col {
                return *val;
            }
        }
        0.0
    }

    fn sparse_to_dense(mat: &SparseMat, dim: usize) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; dim]; dim];
        for (val, (r, c)) in mat.iter() {
            dense[r][c] = *val;
        }
        dense
    }

    fn sparse_product(a: &SparseMat, b: &SparseMat, dim: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; dim]; dim];
        for j in 0..dim {
            let mut ej = vec![0.0; dim];
            ej[j] = 1.0;
            let col = spmv(b, &ej);
            let res_col = spmv(a, &col);
            for i in 0..dim {
                result[i][j] = res_col[i];
            }
        }
        result
    }
}
