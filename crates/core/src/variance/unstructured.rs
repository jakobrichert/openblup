use crate::error::{LmmError, Result};
use crate::types::SparseMat;
use sprs::TriMat;

use super::traits::VarStruct;

/// Unstructured (full) covariance matrix using Cholesky parameterization.
///
/// Sigma = L * L' where L is a lower triangular matrix.
///
/// Parameters: the k*(k+1)/2 elements of L stored column-major in the lower triangle.
/// For a k x k covariance matrix:
///   params = [L[0,0], L[1,0], L[2,0], ..., L[k-1,0],   // column 0
///             L[1,1], L[2,1], ..., L[k-1,1],              // column 1
///             ...
///             L[k-1,k-1]]                                  // column k-1
///
/// This parameterization guarantees positive definiteness as long as diagonal
/// elements of L are positive.
///
/// Typically used for multi-trait covariance matrices (small dimensions: 2-10).
#[derive(Debug, Clone)]
pub struct Unstructured {
    dim: usize,
    chol_params: Vec<f64>, // Lower triangle of L, column-major
}

impl Unstructured {
    /// Create a new Unstructured covariance of the given dimension.
    /// `chol_params` are the lower-triangular elements of L in column-major order.
    pub fn new(dim: usize, chol_params: Vec<f64>) -> Self {
        let expected = dim * (dim + 1) / 2;
        assert_eq!(
            chol_params.len(),
            expected,
            "Expected {} Cholesky parameters for dim={}, got {}",
            expected,
            dim,
            chol_params.len()
        );
        Self { dim, chol_params }
    }

    /// Create with identity Cholesky factor (L = I, so Sigma = I).
    pub fn default_start(dim: usize) -> Self {
        let n_params = dim * (dim + 1) / 2;
        let mut params = vec![0.0; n_params];
        // Set diagonal elements of L to 1.0
        let mut idx = 0;
        for col in 0..dim {
            params[idx] = 1.0; // L[col, col] is the first element in each column's lower tri
            idx += dim - col;
        }
        Self {
            dim,
            chol_params: params,
        }
    }

    /// Reconstruct the lower triangular Cholesky factor L as a dense matrix.
    fn cholesky_factor(&self) -> Vec<Vec<f64>> {
        let k = self.dim;
        let mut l = vec![vec![0.0; k]; k];
        let mut idx = 0;
        for col in 0..k {
            for row in col..k {
                l[row][col] = self.chol_params[idx];
                idx += 1;
            }
        }
        l
    }

    /// Compute Sigma = L * L' as a dense k x k matrix.
    fn sigma_dense(&self) -> Vec<Vec<f64>> {
        let k = self.dim;
        let l = self.cholesky_factor();
        let mut sigma = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                let mut sum = 0.0;
                for m in 0..k {
                    sum += l[i][m] * l[j][m];
                }
                sigma[i][j] = sum;
            }
        }
        sigma
    }

    /// Compute L^{-1} (lower triangular inverse via forward substitution).
    fn cholesky_inverse(&self) -> Vec<Vec<f64>> {
        let k = self.dim;
        let l = self.cholesky_factor();
        let mut l_inv = vec![vec![0.0; k]; k];

        for j in 0..k {
            // Solve L * x = e_j
            for i in 0..k {
                if i < j {
                    l_inv[i][j] = 0.0;
                } else if i == j {
                    l_inv[i][j] = 1.0 / l[i][i];
                } else {
                    let mut sum = 0.0;
                    for m in j..i {
                        sum += l[i][m] * l_inv[m][j];
                    }
                    l_inv[i][j] = -sum / l[i][i];
                }
            }
        }
        l_inv
    }
}

impl VarStruct for Unstructured {
    fn name(&self) -> &str {
        "Unstructured"
    }

    fn n_params(&self) -> usize {
        self.chol_params.len()
    }

    fn params(&self) -> Vec<f64> {
        self.chol_params.clone()
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        let expected = self.dim * (self.dim + 1) / 2;
        if params.len() != expected {
            return Err(LmmError::InvalidParameter(format!(
                "Unstructured(dim={}) expects {} parameters, got {}",
                self.dim,
                expected,
                params.len()
            )));
        }
        // Check that diagonal elements of L are positive (for identifiability).
        let mut idx = 0;
        for col in 0..self.dim {
            if params[idx] <= 0.0 {
                return Err(LmmError::InvalidParameter(format!(
                    "Cholesky diagonal element L[{},{}] must be positive, got {}",
                    col, col, params[idx]
                )));
            }
            idx += self.dim - col;
        }
        self.chol_params = params.to_vec();
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        assert_eq!(
            dim, self.dim,
            "Unstructured dim mismatch: structure has dim={} but dim={} requested",
            self.dim, dim
        );
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
        assert_eq!(
            dim, self.dim,
            "Unstructured dim mismatch: structure has dim={} but dim={} requested",
            self.dim, dim
        );

        // Sigma^{-1} = (LL')^{-1} = L'^{-1} L^{-1}
        let l_inv = self.cholesky_inverse();
        let k = self.dim;

        // Compute Sigma^{-1} = L_inv' * L_inv
        let mut sigma_inv = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                let mut sum = 0.0;
                for m in 0..k {
                    // (L_inv')[i][m] * L_inv[m][j] = L_inv[m][i] * L_inv[m][j]
                    sum += l_inv[m][i] * l_inv[m][j];
                }
                sigma_inv[i][j] = sum;
            }
        }

        let mut tri = TriMat::new((k, k));
        for i in 0..k {
            for j in 0..k {
                if sigma_inv[i][j].abs() > 1e-15 {
                    tri.add_triplet(i, j, sigma_inv[i][j]);
                }
            }
        }
        tri.to_csc()
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        assert_eq!(
            dim, self.dim,
            "Unstructured dim mismatch: structure has dim={} but dim={} requested",
            self.dim, dim
        );

        // |Sigma| = |LL'| = |L|^2 = (prod L[i,i])^2
        // log|Sigma| = 2 * sum(log(L[i,i]))
        let mut idx = 0;
        let mut logdet = 0.0;
        for col in 0..self.dim {
            logdet += self.chol_params[idx].ln();
            idx += self.dim - col;
        }
        2.0 * logdet
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        assert_eq!(
            dim, self.dim,
            "Unstructured dim mismatch: structure has dim={} but dim={} requested",
            self.dim, dim
        );

        let k = self.dim;
        let n_params = self.chol_params.len();
        let eps = 1e-7;

        // Use numerical differentiation for the Cholesky parameterization.
        // For small dimensions (2-10 traits), this is perfectly efficient.
        let mut derivs = Vec::with_capacity(n_params);

        for p in 0..n_params {
            let mut params_plus = self.chol_params.clone();
            let mut params_minus = self.chol_params.clone();
            params_plus[p] += eps;
            params_minus[p] -= eps;

            let us_plus = Unstructured::new(k, params_plus);
            let us_minus = Unstructured::new(k, params_minus);

            let inv_plus = us_plus.inverse_covariance_matrix(k);
            let inv_minus = us_minus.inverse_covariance_matrix(k);

            let mut tri = TriMat::new((k, k));
            // Central difference
            for i in 0..k {
                for j in 0..k {
                    let val_plus = get_entry(&inv_plus, i, j);
                    let val_minus = get_entry(&inv_minus, i, j);
                    let deriv = (val_plus - val_minus) / (2.0 * eps);
                    if deriv.abs() > 1e-15 {
                        tri.add_triplet(i, j, deriv);
                    }
                }
            }
            derivs.push(tri.to_csc());
        }

        derivs
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = Vec::with_capacity(self.chol_params.len());
        for col in 0..self.dim {
            // Diagonal element of L: must be positive
            bounds.push((1e-10, f64::INFINITY));
            // Off-diagonal elements: can be any real number
            for _ in (col + 1)..self.dim {
                bounds.push((f64::NEG_INFINITY, f64::INFINITY));
            }
        }
        bounds
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }

    fn initial_params(&self) -> Vec<f64> {
        self.chol_params.clone()
    }
}

/// Get an entry from a sparse matrix, returning 0 if absent.
fn get_entry(mat: &SparseMat, row: usize, col: usize) -> f64 {
    for (&val, (r, c)) in mat.iter() {
        if r == row && c == col {
            return val;
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::spmv;
    use approx::assert_relative_eq;

    #[test]
    fn test_unstructured_2x2() {
        // L = [[2, 0], [1, 3]] => Sigma = [[4, 2], [2, 10]]
        let us = Unstructured::new(2, vec![2.0, 1.0, 3.0]);
        assert_eq!(us.name(), "Unstructured");
        assert_eq!(us.n_params(), 3);

        let cov = us.covariance_matrix(2);
        assert_relative_eq!(get_entry(&cov, 0, 0), 4.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 0, 1), 2.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 1, 0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(get_entry(&cov, 1, 1), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_unstructured_2x2_inverse() {
        // L = [[2, 0], [1, 3]] => Sigma = [[4, 2], [2, 10]]
        // Sigma^{-1} = (1/36) * [[10, -2], [-2, 4]]
        let us = Unstructured::new(2, vec![2.0, 1.0, 3.0]);

        let cov = us.covariance_matrix(2);
        let inv = us.inverse_covariance_matrix(2);

        // Check Sigma * Sigma^{-1} = I
        for j in 0..2 {
            let mut ej = vec![0.0; 2];
            ej[j] = 1.0;
            let col = spmv(&inv, &ej);
            let res = spmv(&cov, &col);
            for i in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res[i], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_unstructured_3x3() {
        // L = [[1, 0, 0], [0.5, 2, 0], [0.3, 0.4, 1.5]]
        // params (column-major lower triangle): [1.0, 0.5, 0.3, 2.0, 0.4, 1.5]
        let us = Unstructured::new(3, vec![1.0, 0.5, 0.3, 2.0, 0.4, 1.5]);
        assert_eq!(us.n_params(), 6);

        let cov = us.covariance_matrix(3);
        let inv = us.inverse_covariance_matrix(3);

        // Check Sigma * Sigma^{-1} = I
        for j in 0..3 {
            let mut ej = vec![0.0; 3];
            ej[j] = 1.0;
            let col = spmv(&inv, &ej);
            let res = spmv(&cov, &col);
            for i in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(res[i], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_unstructured_log_determinant() {
        // L = [[2, 0], [1, 3]] => |Sigma| = |L|^2 = (2*3)^2 = 36
        // log|Sigma| = log(36)
        let us = Unstructured::new(2, vec![2.0, 1.0, 3.0]);
        let logdet = us.log_determinant(2);
        assert_relative_eq!(logdet, 36.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_unstructured_log_determinant_3x3() {
        // L = [[1, 0, 0], [0.5, 2, 0], [0.3, 0.4, 1.5]]
        // |L| = 1 * 2 * 1.5 = 3.0
        // |Sigma| = 9.0, log|Sigma| = log(9)
        let us = Unstructured::new(3, vec![1.0, 0.5, 0.3, 2.0, 0.4, 1.5]);
        let logdet = us.log_determinant(3);
        assert_relative_eq!(logdet, 9.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_unstructured_identity_start() {
        let us = Unstructured::default_start(3);
        assert_eq!(us.dim, 3);
        assert_eq!(us.n_params(), 6);

        // Should produce I
        let cov = us.covariance_matrix(3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(get_entry(&cov, i, j), expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_unstructured_param_count() {
        // k=1: 1 param, k=2: 3 params, k=3: 6 params, k=4: 10 params
        assert_eq!(Unstructured::default_start(1).n_params(), 1);
        assert_eq!(Unstructured::default_start(2).n_params(), 3);
        assert_eq!(Unstructured::default_start(3).n_params(), 6);
        assert_eq!(Unstructured::default_start(4).n_params(), 10);
        assert_eq!(Unstructured::default_start(5).n_params(), 15);
    }

    #[test]
    fn test_unstructured_set_params() {
        let mut us = Unstructured::default_start(2);
        us.set_params(&[2.0, 1.0, 3.0]).unwrap();
        assert_eq!(us.params(), vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn test_unstructured_set_params_validation() {
        let mut us = Unstructured::default_start(2);

        // Wrong count
        assert!(us.set_params(&[1.0, 2.0]).is_err());

        // Diagonal element non-positive (L[0,0] = 0)
        assert!(us.set_params(&[0.0, 1.0, 1.0]).is_err());

        // Diagonal element non-positive (L[1,1] = -1)
        assert!(us.set_params(&[1.0, 1.0, -1.0]).is_err());

        // Valid: off-diagonal can be negative
        assert!(us.set_params(&[1.0, -0.5, 2.0]).is_ok());
    }

    #[test]
    fn test_unstructured_bounds() {
        let us = Unstructured::default_start(3);
        let bounds = us.bounds();
        assert_eq!(bounds.len(), 6);

        // Check pattern: diagonal elements have positive lower bound,
        // off-diagonals are unconstrained.
        // col 0: L[0,0] (positive), L[1,0] (free), L[2,0] (free)
        assert!(bounds[0].0 > 0.0);
        assert!(bounds[1].0.is_infinite() && bounds[1].0 < 0.0);
        assert!(bounds[2].0.is_infinite() && bounds[2].0 < 0.0);
        // col 1: L[1,1] (positive), L[2,1] (free)
        assert!(bounds[3].0 > 0.0);
        assert!(bounds[4].0.is_infinite() && bounds[4].0 < 0.0);
        // col 2: L[2,2] (positive)
        assert!(bounds[5].0 > 0.0);
    }

    #[test]
    fn test_unstructured_clone_boxed() {
        let us = Unstructured::new(2, vec![1.0, 0.5, 2.0]);
        let boxed: Box<dyn VarStruct> = us.clone_boxed();
        assert_eq!(boxed.params(), vec![1.0, 0.5, 2.0]);
        assert_eq!(boxed.name(), "Unstructured");
    }

    #[test]
    fn test_unstructured_symmetry() {
        // Verify that Sigma and Sigma^{-1} are symmetric.
        let us = Unstructured::new(3, vec![1.0, 0.5, 0.3, 2.0, 0.4, 1.5]);

        let cov = us.covariance_matrix(3);
        let inv = us.inverse_covariance_matrix(3);

        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(
                    get_entry(&cov, i, j),
                    get_entry(&cov, j, i),
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    get_entry(&inv, i, j),
                    get_entry(&inv, j, i),
                    epsilon = 1e-10
                );
            }
        }
    }
}
