use crate::types::SparseMat;
use sprs::TriMat;

/// Compute the Kronecker product A (x) B of two sparse matrices.
///
/// The Kronecker product of an (m x n) matrix A and a (p x q) matrix B
/// is the (mp x nq) block matrix where each block (i,j) is A[i,j] * B.
///
/// This is the fundamental operation for separable spatial models:
/// if rows have covariance Sigma_r and columns have covariance Sigma_c,
/// then the full field covariance is Sigma_r (x) Sigma_c.
///
/// Properties used in mixed models:
/// - (A (x) B)^{-1} = A^{-1} (x) B^{-1}
/// - |A (x) B| = |A|^q * |B|^n (for A: n x n, B: q x q)
/// - d(A (x) B)/dtheta = (dA/dtheta) (x) B (if B is constant w.r.t. theta)
pub fn kronecker_product(a: &SparseMat, b: &SparseMat) -> SparseMat {
    let (ra, ca) = (a.rows(), a.cols());
    let (rb, cb) = (b.rows(), b.cols());
    let mut tri = TriMat::new((ra * rb, ca * cb));

    for (&va, (ia, ja)) in a.iter() {
        for (&vb, (ib, jb)) in b.iter() {
            tri.add_triplet(ia * rb + ib, ja * cb + jb, va * vb);
        }
    }

    tri.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::{sparse_diagonal, sparse_identity, spmv};
    use approx::assert_relative_eq;

    #[test]
    fn test_kronecker_identity_identity() {
        // I_2 (x) I_3 = I_6
        let i2 = sparse_identity(2);
        let i3 = sparse_identity(3);
        let result = kronecker_product(&i2, &i3);

        assert_eq!(result.rows(), 6);
        assert_eq!(result.cols(), 6);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = spmv(&result, &x);
        for i in 0..6 {
            assert_relative_eq!(y[i], x[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kronecker_known_example() {
        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        // A (x) B = [[0, 5, 0, 10],
        //            [6, 7, 12, 14],
        //            [0, 15, 0, 20],
        //            [18, 21, 24, 28]]
        let mut tri_a = TriMat::new((2, 2));
        tri_a.add_triplet(0, 0, 1.0);
        tri_a.add_triplet(0, 1, 2.0);
        tri_a.add_triplet(1, 0, 3.0);
        tri_a.add_triplet(1, 1, 4.0);
        let a = tri_a.to_csc();

        let mut tri_b = TriMat::new((2, 2));
        tri_b.add_triplet(0, 1, 5.0);
        tri_b.add_triplet(1, 0, 6.0);
        tri_b.add_triplet(1, 1, 7.0);
        let b = tri_b.to_csc();

        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 4);
        assert_eq!(result.cols(), 4);

        // Test against known result: multiply by e1 = [1,0,0,0]
        let y = spmv(&result, &[1.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10); // row 0: 0*1 + 5*0 + 0*0 + 10*0 = 0
        assert_relative_eq!(y[1], 6.0, epsilon = 1e-10); // row 1: 6*1 + 7*0 + 12*0 + 14*0 = 6
        assert_relative_eq!(y[2], 0.0, epsilon = 1e-10); // row 2: 0*1 + 15*0 + 0*0 + 20*0 = 0
        assert_relative_eq!(y[3], 18.0, epsilon = 1e-10); // row 3: 18*1 + 21*0 + 24*0 + 28*0 = 18

        // Test with e4 = [0,0,0,1]
        let y = spmv(&result, &[0.0, 0.0, 0.0, 1.0]);
        assert_relative_eq!(y[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 14.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 20.0, epsilon = 1e-10);
        assert_relative_eq!(y[3], 28.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kronecker_dimension_check() {
        let a = sparse_identity(3);
        let b = sparse_identity(4);
        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 12);
        assert_eq!(result.cols(), 12);
    }

    #[test]
    fn test_kronecker_diagonal_diagonal() {
        // diag(2, 3) (x) diag(5, 7) = diag(10, 14, 15, 21)
        let a = sparse_diagonal(&[2.0, 3.0]);
        let b = sparse_diagonal(&[5.0, 7.0]);
        let result = kronecker_product(&a, &b);

        assert_eq!(result.rows(), 4);
        assert_eq!(result.cols(), 4);

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = spmv(&result, &x);
        assert_relative_eq!(y[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 14.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 15.0, epsilon = 1e-10);
        assert_relative_eq!(y[3], 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kronecker_non_square() {
        // A: 2x3, B: 3x2 => result: 6x6
        let mut tri_a = TriMat::new((2, 3));
        tri_a.add_triplet(0, 0, 1.0);
        tri_a.add_triplet(1, 2, 2.0);
        let a = tri_a.to_csc();

        let mut tri_b = TriMat::new((3, 2));
        tri_b.add_triplet(0, 0, 3.0);
        tri_b.add_triplet(2, 1, 4.0);
        let b = tri_b.to_csc();

        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 6);
        assert_eq!(result.cols(), 6);
    }
}
