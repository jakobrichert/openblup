//! Selected inverse of a sparse SPD matrix.
//!
//! Thin wrappers around [`SparseCholeskySolver::inverse_subset`], which runs
//! the Takahashi recurrences on the sparse Cholesky factor and returns `A⁻¹`
//! at every position of the pattern of `L + L'` (a superset of the pattern
//! of `A`) without ever forming the dense inverse.

use sprs::CsMat;

use crate::error::Result;
use crate::matrix::sparse_cholesky::SparseCholeskySolver;

/// Entries of `A⁻¹` on the pattern of the Cholesky factor of the SPD matrix
/// `A` (symmetric, both triangles stored, sorted CSC).
pub fn sparse_inverse_subset(a: &CsMat<f64>) -> Result<CsMat<f64>> {
    Ok(SparseCholeskySolver::new(a)?.inverse_subset())
}

/// Diagonal of `A⁻¹` for the SPD matrix `A`.
pub fn sparse_inverse_diagonal(a: &CsMat<f64>) -> Result<Vec<f64>> {
    Ok(SparseCholeskySolver::new(a)?.inverse_diagonal())
}

/// `tr(A⁻¹ B) = Σ_{ij} A⁻¹_{ij} B_{ji}` for a sparse `B` whose pattern is
/// contained in the pattern of `A` (e.g. a relationship-matrix inverse block
/// of the MME coefficient matrix).
pub fn trace_ainv_b(a: &CsMat<f64>, b: &CsMat<f64>) -> Result<f64> {
    let z = sparse_inverse_subset(a)?;
    Ok(trace_subset_b(&z, b))
}

/// `Σ_{ij} Z_{ij} B_{ji}` for a stored inverse subset `Z` (entries of `B`
/// outside the subset contribute nothing).
pub fn trace_subset_b(z: &CsMat<f64>, b: &CsMat<f64>) -> f64 {
    b.iter()
        .map(|(v, (i, j))| v * z.get(j, i).copied().unwrap_or(0.0))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use sprs::TriMat;

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

    fn dense_inverse() -> nalgebra::DMatrix<f64> {
        nalgebra::DMatrix::from_row_slice(3, 3, &[4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0])
            .try_inverse()
            .unwrap()
    }

    #[test]
    fn test_sparse_inverse_diagonal() {
        let a = simple_spd_matrix();
        let diag = sparse_inverse_diagonal(&a).unwrap();
        let a_inv = dense_inverse();
        for i in 0..3 {
            assert_relative_eq!(diag[i], a_inv[(i, i)], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sparse_inverse_subset_vs_full() {
        let a = simple_spd_matrix();
        let z = sparse_inverse_subset(&a).unwrap();
        let a_inv = dense_inverse();
        for (v, (i, j)) in z.iter() {
            assert_relative_eq!(*v, a_inv[(i, j)], epsilon = 1e-10);
        }
        // the pattern of A is covered
        for (_, (i, j)) in a.iter() {
            assert!(z.get(i, j).is_some());
        }
    }

    #[test]
    fn test_trace_ainv_b() {
        let a = simple_spd_matrix();
        let b = crate::matrix::sparse::sparse_identity(3);
        let trace = trace_ainv_b(&a, &b).unwrap();
        let a_inv = dense_inverse();
        let expected: f64 = (0..3).map(|i| a_inv[(i, i)]).sum();
        assert_relative_eq!(trace, expected, epsilon = 1e-10);

        // B with the pattern of A
        let trace_a = trace_ainv_b(&a, &a).unwrap();
        assert_relative_eq!(trace_a, 3.0, epsilon = 1e-10); // tr(A⁻¹A) = n
    }

    #[test]
    fn test_banded_matrix() {
        let n = 50;
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 4.0);
            if i > 0 {
                tri.add_triplet(i, i - 1, -1.0);
                tri.add_triplet(i - 1, i, -1.0);
            }
        }
        let a = tri.to_csc();
        let diag = sparse_inverse_diagonal(&a).unwrap();
        let mut d = nalgebra::DMatrix::zeros(n, n);
        for (v, (i, j)) in a.iter() {
            d[(i, j)] = *v;
        }
        let a_inv = d.try_inverse().unwrap();
        for i in 0..n {
            assert_relative_eq!(diag[i], a_inv[(i, i)], epsilon = 1e-10);
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
        assert!(sparse_inverse_subset(&a).is_err());
    }
}
