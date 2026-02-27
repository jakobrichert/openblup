use nalgebra::DMatrix;

/// Solve a symmetric positive-definite system A*x = b via Cholesky decomposition.
/// Returns x, or None if Cholesky fails (matrix not SPD).
pub fn solve_spd(a: &DMatrix<f64>, b: &[f64]) -> Option<Vec<f64>> {
    let chol = a.clone().cholesky()?;
    let b_vec = nalgebra::DVector::from_column_slice(b);
    let x = chol.solve(&b_vec);
    Some(x.as_slice().to_vec())
}

/// Compute the Cholesky factorization of a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L such that A = L * L^T.
pub fn cholesky_lower(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let chol = a.clone().cholesky()?;
    Some(chol.l())
}

/// Compute the log-determinant of an SPD matrix via Cholesky: log|A| = 2 * sum(log(diag(L))).
pub fn log_determinant_spd(a: &DMatrix<f64>) -> Option<f64> {
    let l = cholesky_lower(a)?;
    let logdet = 2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>();
    Some(logdet)
}

/// Compute the inverse of an SPD matrix via Cholesky.
pub fn inverse_spd(a: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let chol = a.clone().cholesky()?;
    Some(chol.inverse())
}

/// Compute the trace of a matrix.
pub fn trace(a: &DMatrix<f64>) -> f64 {
    (0..a.nrows().min(a.ncols())).map(|i| a[(i, i)]).sum()
}

/// Compute the Frobenius norm of a matrix.
pub fn frobenius_norm(a: &DMatrix<f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_spd() {
        // A = [[4, 2], [2, 3]], b = [1, 2]
        // Solution: x = [-1/8, 3/4]
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let b = vec![1.0, 2.0];
        let x = solve_spd(&a, &b).unwrap();
        assert_relative_eq!(x[0], -0.125, epsilon = 1e-10);
        assert_relative_eq!(x[1], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_solve_spd_identity() {
        let a = DMatrix::identity(3, 3);
        let b = vec![5.0, 6.0, 7.0];
        let x = solve_spd(&a, &b).unwrap();
        assert_relative_eq!(x[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_determinant_spd() {
        // A = [[2, 0], [0, 3]], det = 6, log(det) = ln(6)
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let logdet = log_determinant_spd(&a).unwrap();
        assert_relative_eq!(logdet, 6.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log_determinant_identity() {
        let a = DMatrix::identity(5, 5);
        let logdet = log_determinant_spd(&a).unwrap();
        assert_relative_eq!(logdet, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_spd() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 3.0]);
        let a_inv = inverse_spd(&a).unwrap();
        // A * A_inv should be identity
        let product = &a * &a_inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_trace() {
        let a = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_relative_eq!(trace(&a), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_not_spd_returns_none() {
        // Not positive definite
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        assert!(solve_spd(&a, &[1.0, 1.0]).is_none());
        assert!(cholesky_lower(&a).is_none());
    }
}
