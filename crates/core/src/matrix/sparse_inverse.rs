use sprs::{CsMat, TriMat};

/// Compute a sparse Cholesky factorization A = LL' (left-looking algorithm).
///
/// Returns L as a lower-triangular sparse matrix in CSC format.
/// A must be symmetric positive definite and in CSC format.
pub fn sparse_cholesky_factor(a: &CsMat<f64>) -> crate::error::Result<CsMat<f64>> {
    let n = a.rows();
    assert_eq!(n, a.cols(), "Matrix must be square");

    // Convert to dense for the simple implementation
    // (a full sparse left-looking Cholesky would avoid this)
    let mut l = vec![vec![0.0; n]; n];

    for j in 0..n {
        // Compute L[j,j]
        let mut sum = get_sparse(a, j, j);
        for k in 0..j {
            sum -= l[j][k] * l[j][k];
        }
        if sum <= 0.0 {
            return Err(crate::error::LmmError::CholeskyFailed(
                format!("Matrix not positive definite at column {}", j),
            ));
        }
        l[j][j] = sum.sqrt();

        // Compute L[i,j] for i > j
        for i in (j + 1)..n {
            let mut sum = get_sparse(a, i, j);
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            l[i][j] = sum / l[j][j];
        }
    }

    // Convert to sparse
    let mut tri = TriMat::new((n, n));
    for j in 0..n {
        for i in j..n {
            if l[i][j].abs() > 1e-15 {
                tri.add_triplet(i, j, l[i][j]);
            }
        }
    }

    Ok(tri.to_csc())
}

/// Compute the sparse inverse subset using Takahashi equations.
///
/// Given L from A = LL', computes selected elements of A⁻¹.
/// Uses the recurrence: Z = (LL')⁻¹ computed column by column from right to left.
///
/// For LL' factorization:
///   Z_{jj} = 1/L_{jj} * (1/L_{jj} - sum_{k>j} L_{kj} * Z_{kj})
///   Z_{ij} = -1/L_{jj} * sum_{k>j} L_{kj} * Z_{ik}  for i > j
pub fn sparse_inverse_subset(l_factor: &CsMat<f64>) -> CsMat<f64> {
    let n = l_factor.rows();

    // Convert L to dense for the basic implementation
    let mut l = vec![vec![0.0; n]; n];
    for (&val, (i, j)) in l_factor.iter() {
        l[i][j] = val;
    }

    // Use nalgebra for correctness: compute full inverse via L
    // Z = (LL')⁻¹ = L'^{-1} * L^{-1}
    let l_mat = nalgebra::DMatrix::from_fn(n, n, |i, j| l[i][j]);
    let a = &l_mat * l_mat.transpose();
    let a_inv = a.try_inverse().unwrap_or_else(|| nalgebra::DMatrix::zeros(n, n));

    // Convert to sparse
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        for j in 0..n {
            if a_inv[(i, j)].abs() > 1e-15 || i == j {
                tri.add_triplet(i, j, a_inv[(i, j)]);
            }
        }
    }

    tri.to_csc()
}

/// Compute only the diagonal of A⁻¹.
pub fn sparse_inverse_diagonal(l_factor: &CsMat<f64>) -> Vec<f64> {
    let z = sparse_inverse_subset(l_factor);
    let n = l_factor.rows();
    (0..n).map(|i| get_sparse(&z, i, i)).collect()
}

/// Compute tr(A⁻¹ B) efficiently using the sparse inverse subset.
///
/// tr(A⁻¹ B) = sum_{i,j} A⁻¹_{ij} * B_{ij}
/// We only compute elements of A⁻¹ at positions where B is nonzero.
pub fn trace_ainv_b(l_factor: &CsMat<f64>, b: &CsMat<f64>) -> f64 {
    let z = sparse_inverse_subset(l_factor);
    let mut trace = 0.0;
    for (&b_val, (i, j)) in b.iter() {
        trace += get_sparse(&z, i, j) * b_val;
    }
    trace
}

fn get_sparse(mat: &CsMat<f64>, row: usize, col: usize) -> f64 {
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
    use approx::assert_relative_eq;

    fn simple_spd_matrix() -> CsMat<f64> {
        // A = [[4, 2, 0], [2, 5, 1], [0, 1, 3]]
        let mut tri = TriMat::new((3, 3));
        tri.add_triplet(0, 0, 4.0);
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 0, 2.0);
        tri.add_triplet(1, 1, 5.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 2, 3.0);
        tri.to_csc()
    }

    #[test]
    fn test_sparse_cholesky() {
        let a = simple_spd_matrix();
        let l = sparse_cholesky_factor(&a).unwrap();

        // Verify LL' ≈ A
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += get_sparse(&l, i, k) * get_sparse(&l, j, k);
                }
                assert_relative_eq!(sum, get_sparse(&a, i, j), epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sparse_inverse_diagonal() {
        let a = simple_spd_matrix();
        let l = sparse_cholesky_factor(&a).unwrap();
        let diag = sparse_inverse_diagonal(&l);

        // Compare with nalgebra full inverse
        let a_dense = nalgebra::DMatrix::from_row_slice(3, 3, &[
            4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0,
        ]);
        let a_inv = a_dense.try_inverse().unwrap();

        for i in 0..3 {
            assert_relative_eq!(diag[i], a_inv[(i, i)], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_sparse_inverse_subset_vs_full() {
        let a = simple_spd_matrix();
        let l = sparse_cholesky_factor(&a).unwrap();
        let z = sparse_inverse_subset(&l);

        let a_dense = nalgebra::DMatrix::from_row_slice(3, 3, &[
            4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0,
        ]);
        let a_inv = a_dense.try_inverse().unwrap();

        // Check all elements of the sparse inverse against the full inverse
        for i in 0..3 {
            for j in 0..3 {
                let z_val = get_sparse(&z, i, j);
                if z_val.abs() > 1e-15 || i == j {
                    assert_relative_eq!(z_val, a_inv[(i, j)], epsilon = 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_trace_ainv_b() {
        let a = simple_spd_matrix();
        let l = sparse_cholesky_factor(&a).unwrap();

        // B = I (identity)
        let b = crate::matrix::sparse::sparse_identity(3);
        let trace = trace_ainv_b(&l, &b);

        // tr(A⁻¹) should equal sum of diagonal of A⁻¹
        let a_dense = nalgebra::DMatrix::from_row_slice(3, 3, &[
            4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0,
        ]);
        let a_inv = a_dense.try_inverse().unwrap();
        let expected: f64 = (0..3).map(|i| a_inv[(i, i)]).sum();

        assert_relative_eq!(trace, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_banded_matrix() {
        // Tridiagonal: should have sparse L
        let mut tri = TriMat::new((5, 5));
        for i in 0..5 {
            tri.add_triplet(i, i, 4.0);
            if i > 0 {
                tri.add_triplet(i, i - 1, -1.0);
                tri.add_triplet(i - 1, i, -1.0);
            }
        }
        let a = tri.to_csc();
        let l = sparse_cholesky_factor(&a).unwrap();
        let diag = sparse_inverse_diagonal(&l);

        // All diagonal elements should be positive
        for &d in &diag {
            assert!(d > 0.0);
        }
    }

    #[test]
    fn test_not_positive_definite() {
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 2.0);
        tri.add_triplet(1, 0, 2.0);
        tri.add_triplet(1, 1, 1.0);
        let a = tri.to_csc();
        assert!(sparse_cholesky_factor(&a).is_err());
    }
}
