//! Sparse Cholesky solver using the faer library.
//!
//! This module wraps faer's sparse Cholesky (LLT) factorization for solving the
//! Mixed Model Equations (MME). The key design separates symbolic analysis
//! (done once when the sparsity pattern is established) from numeric factorization
//! (redone each REML iteration as variance components change).
//!
//! The factorization is always *simplicial* (column-by-column, stored as a
//! CSC lower-triangular factor with a fill-reducing AMD ordering). That keeps
//! the diagonal of `L` directly accessible for the log-determinant and lets
//! [`SparseCholeskySolver::inverse_subset`] run the Takahashi recurrences on
//! the factor to obtain every entry of `A⁻¹` within the sparsity pattern of
//! `L + L'` — which contains the pattern of `A` — without forming the dense
//! inverse. Those entries are exactly what REML needs (`tr(G⁻¹ C^{kk})`,
//! prediction error variances, leverages).
//!
//! # Usage
//! ```ignore
//! let solver = SparseCholeskySolver::new(&mme_matrix)?;
//! let solution = solver.solve(&rhs)?;
//! let log_det = solver.log_determinant();
//!
//! // On next REML iteration (same sparsity, new values):
//! solver.refactorize(&updated_mme_matrix)?;
//! ```

use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{CreationError, SparseColMat};
use faer::Index as FaerIndex; // for .zx() method on index types
use faer::Parallelism;
use faer::Side;
use sprs::TriMat;

/// Convert an sprs CsMat<f64> (CSC) to faer's SparseColMat<usize, f64>.
///
/// The sprs matrix must be in CSC (Compressed Sparse Column) format. Only the
/// upper triangle is extracted for the symmetric SPD systems we solve.
fn sprs_to_faer_upper(
    matrix: &SparseMat,
) -> std::result::Result<SparseColMat<usize, f64>, LmmError> {
    let n = matrix.rows();
    assert_eq!(n, matrix.cols(), "Matrix must be square");

    // Ensure CSC format
    let csc = if matrix.is_csc() {
        matrix.clone()
    } else {
        matrix.to_csc()
    };

    // Collect upper-triangular triplets (row <= col)
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for (val, (row, col)) in csc.iter() {
        if row <= col {
            triplets.push((row, col, *val));
        }
    }

    match SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets) {
        Ok(mat) => Ok(mat),
        Err(CreationError::Generic(e)) => Err(LmmError::CholeskyFailed(format!(
            "Failed to create faer sparse matrix: {e}"
        ))),
        Err(CreationError::OutOfBounds { row, col }) => Err(LmmError::CholeskyFailed(format!(
            "Index out of bounds: row={row}, col={col}"
        ))),
    }
}

/// Sparse Cholesky solver using faer.
///
/// Splits symbolic analysis (done once when sparsity pattern is known)
/// from numeric factorization (redone each REML iteration).
///
/// Internally uses faer's `SymbolicCholesky` + `LLT` factorization with
/// fill-reducing AMD ordering.
#[derive(Debug)]
pub struct SparseCholeskySolver {
    /// Symbolic factorization (fill-reducing permutation + elimination tree).
    symbolic: SymbolicCholesky<usize>,
    /// Numerical values of the L factor.
    l_values: Vec<f64>,
    /// Dimension of the system.
    dim: usize,
}

impl SparseCholeskySolver {
    /// Create a new solver by analyzing the sparsity pattern of the given matrix
    /// and performing the initial numeric factorization.
    ///
    /// The matrix should be symmetric positive definite. Only the upper triangle
    /// is used.
    pub fn new(matrix: &SparseMat) -> Result<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(LmmError::DimensionMismatch {
                expected: n,
                got: matrix.cols(),
                context: "SparseCholeskySolver: matrix must be square".to_string(),
            });
        }

        // Convert to faer upper-triangular CSC
        let faer_mat = sprs_to_faer_upper(matrix)?;

        // Symbolic analysis with AMD ordering. The simplicial factorization
        // is forced so that L is available column by column (see module docs).
        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        };
        let symbolic = factorize_symbolic_cholesky(
            faer_mat.symbolic(),
            Side::Upper,
            Default::default(), // SymmetricOrdering::default() = Amd
            params,
        )
        .map_err(|e| LmmError::CholeskyFailed(format!("Symbolic factorization failed: {e}")))?;

        // Allocate L values
        let len_values = symbolic.len_values();
        let mut l_values = vec![0.0f64; len_values];

        // Numeric factorization
        let parallelism = Parallelism::None;
        let req = symbolic
            .factorize_numeric_llt_req::<f64>(parallelism)
            .map_err(|e| LmmError::CholeskyFailed(format!("Memory requirement error: {e}")))?;
        let mut mem = GlobalPodBuffer::new(req);

        symbolic
            .factorize_numeric_llt(
                l_values.as_mut_slice(),
                faer_mat.as_ref(),
                Side::Upper,
                Default::default(), // LltRegularization::default()
                parallelism,
                PodStack::new(&mut mem),
            )
            .map_err(|_| LmmError::NotPositiveDefinite)?;

        Ok(Self {
            symbolic,
            l_values,
            dim: n,
        })
    }

    /// Refactorize with new numeric values (same sparsity pattern).
    ///
    /// This is much cheaper than creating a new solver, because the symbolic
    /// analysis is reused. Use this in the REML iteration loop when variance
    /// components change but the MME sparsity structure remains the same.
    pub fn refactorize(&mut self, matrix: &SparseMat) -> Result<()> {
        let n = matrix.rows();
        if n != self.dim {
            return Err(LmmError::DimensionMismatch {
                expected: self.dim,
                got: n,
                context: "SparseCholeskySolver::refactorize: dimension changed".to_string(),
            });
        }

        let faer_mat = sprs_to_faer_upper(matrix)?;

        let parallelism = Parallelism::None;
        let req = self
            .symbolic
            .factorize_numeric_llt_req::<f64>(parallelism)
            .map_err(|e| LmmError::CholeskyFailed(format!("Memory requirement error: {e}")))?;
        let mut mem = GlobalPodBuffer::new(req);

        self.symbolic
            .factorize_numeric_llt(
                self.l_values.as_mut_slice(),
                faer_mat.as_ref(),
                Side::Upper,
                Default::default(),
                parallelism,
                PodStack::new(&mut mem),
            )
            .map_err(|_| LmmError::NotPositiveDefinite)?;

        Ok(())
    }

    /// Solve the system A*x = b, returning x.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.dim {
            return Err(LmmError::DimensionMismatch {
                expected: self.dim,
                got: rhs.len(),
                context: "SparseCholeskySolver::solve: rhs dimension".to_string(),
            });
        }

        // Create the LltRef from symbolic + numeric values
        let llt = LltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);

        // Copy rhs into a faer column matrix (n x 1)
        let mut sol_data = rhs.to_vec();
        let sol_mat = faer::mat::from_column_major_slice_mut(&mut sol_data, self.dim, 1);

        // Solve in place
        let req = self
            .symbolic
            .solve_in_place_req::<f64>(1)
            .map_err(|e| LmmError::CholeskyFailed(format!("Solve memory error: {e}")))?;
        let mut mem = GlobalPodBuffer::new(req);

        llt.solve_in_place_with_conj(
            faer::Conj::No,
            sol_mat,
            Parallelism::None,
            PodStack::new(&mut mem),
        );

        Ok(sol_data)
    }

    /// Compute log|A| = 2 * sum(log(diag(L))) where A = L*L'.
    ///
    /// This is needed for the REML log-likelihood calculation.
    pub fn log_determinant(&self) -> f64 {
        let (col_ptrs, _) = self.factor_structure();
        // In the simplicial factor the diagonal entry L(i,i) is the first
        // entry of column i.
        let mut log_det = 0.0;
        for i in 0..self.dim {
            log_det += self.l_values[col_ptrs[i].zx()].ln();
        }
        2.0 * log_det
    }

    /// Column pointers and (sorted) row indices of the simplicial factor `L`
    /// in the permuted ordering; the values in `self.l_values` are stored in
    /// the same order.
    fn factor_structure(&self) -> (&[usize], &[usize]) {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Simplicial(s) => (s.col_ptrs(), s.row_indices()),
            SymbolicCholeskyRaw::Supernodal(_) => {
                unreachable!("the solver always requests a simplicial factorization")
            }
        }
    }

    /// Fill-reducing permutation as `(forward, inverse)`: the factorized
    /// matrix is `B = P A P'` with `B[inv[i], inv[j]] = A[i, j]`, i.e.
    /// `A[fwd[r], fwd[c]] = B[r, c]`.
    fn permutation(&self) -> (Vec<usize>, Vec<usize>) {
        match self.symbolic.perm() {
            Some(perm) => {
                let (fwd, inv) = perm.arrays();
                (
                    fwd.iter().map(|i| i.zx()).collect(),
                    inv.iter().map(|i| i.zx()).collect(),
                )
            }
            None => ((0..self.dim).collect(), (0..self.dim).collect()),
        }
    }

    /// Selected entries of `A⁻¹` via the Takahashi recurrences.
    ///
    /// Returns a symmetric sparse matrix (both triangles stored, CSC with
    /// sorted indices, original ordering) holding `A⁻¹` at every position
    /// in the pattern of `L + L'`. That pattern contains the pattern of `A`,
    /// so `tr(A⁻¹ B)` for any `B` with the pattern of `A` (or a sub-pattern,
    /// e.g. a relationship-matrix inverse block) and the diagonal of `A⁻¹`
    /// are exact. Entries outside the pattern are *not* zero in `A⁻¹`; they
    /// are simply not stored.
    ///
    /// Cost is `Σ_j |L_{:,j}|²` operations, i.e. proportional to the work of
    /// the factorization itself.
    pub fn inverse_subset(&self) -> SparseMat {
        let n = self.dim;
        let (col_ptrs, row_ind) = self.factor_structure();
        let l = &self.l_values;
        // Z holds the inverse entries in the storage layout of L (lower
        // triangle including the diagonal, permuted ordering).
        let mut z = vec![0.0f64; l.len()];

        // For column j with below-diagonal pattern P_j (sorted):
        //   Z_ij = −(1/L_jj) Σ_{k∈P_j} L_kj Z(k,i)      for i ∈ P_j
        //   Z_jj = (1/L_jj) (1/L_jj − Σ_{k∈P_j} L_kj Z_kj)
        // Every Z(k,i) with i, k ∈ P_j lies in a column > j (already
        // computed) and, by the closure property of the Cholesky pattern,
        // is stored in column min(i,k) at row max(i,k). Walking column k of
        // Z against P_j (both sorted) therefore yields all pairs without
        // any searching; each unordered pair contributes to both sums.
        let mut acc: Vec<f64> = Vec::new();
        for j in (0..n).rev() {
            let start = col_ptrs[j];
            let end = col_ptrs[j + 1];
            let ljj = l[start];
            let pj = &row_ind[start + 1..end];
            let lj = &l[start + 1..end];
            acc.clear();
            acc.resize(pj.len(), 0.0);
            for (b, &k) in pj.iter().enumerate() {
                let ks = col_ptrs[k];
                let ke = col_ptrs[k + 1];
                let rows_k = &row_ind[ks..ke];
                let z_k = &z[ks..ke];
                let (mut a, mut c) = (b, 0);
                while a < pj.len() && c < rows_k.len() {
                    let i = pj[a];
                    let r = rows_k[c];
                    if i == r {
                        let v = z_k[c]; // Z(i,k) = Z(k,i)
                        acc[a] += lj[b] * v;
                        if a != b {
                            acc[b] += lj[a] * v;
                        }
                        a += 1;
                        c += 1;
                    } else if i < r {
                        a += 1;
                    } else {
                        c += 1;
                    }
                }
            }
            for (a, sum) in acc.iter().enumerate() {
                z[start + 1 + a] = -sum / ljj;
            }
            let mut sum = 0.0;
            for a in 0..pj.len() {
                sum += lj[a] * z[start + 1 + a];
            }
            z[start] = (1.0 / ljj - sum) / ljj;
        }

        // Map back to the original ordering and symmetrise.
        let (fwd, _) = self.permutation();
        let mut tri = TriMat::with_capacity((n, n), 2 * l.len());
        for c in 0..n {
            for a in col_ptrs[c]..col_ptrs[c + 1] {
                let r = row_ind[a];
                let (oi, oj) = (fwd[r], fwd[c]);
                tri.add_triplet(oi, oj, z[a]);
                if r != c {
                    tri.add_triplet(oj, oi, z[a]);
                }
            }
        }
        tri.to_csc()
    }

    /// Diagonal of `A⁻¹` (original ordering), from [`inverse_subset`](Self::inverse_subset).
    pub fn inverse_diagonal(&self) -> Vec<f64> {
        let z = self.inverse_subset();
        (0..self.dim)
            .map(|i| z.get(i, i).copied().unwrap_or(0.0))
            .collect()
    }

    /// Compute the full inverse A^{-1} (dense).
    ///
    /// This is expensive (O(n^2) memory, O(n^2 * nnz) time) and intended only
    /// for small problems or Phase 1 compatibility. For large problems, prefer
    /// using `solve()` with individual right-hand sides.
    pub fn inverse(&self) -> Result<nalgebra::DMatrix<f64>> {
        let n = self.dim;

        // Solve A * X = I for each column of the identity
        let llt = LltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);

        let req = self
            .symbolic
            .solve_in_place_req::<f64>(n)
            .map_err(|e| LmmError::CholeskyFailed(format!("Inverse memory error: {e}")))?;
        let mut mem = GlobalPodBuffer::new(req);

        // Create identity matrix in column-major layout
        let mut inv_data = vec![0.0f64; n * n];
        for i in 0..n {
            inv_data[i * n + i] = 1.0;
        }

        let inv_mat = faer::mat::from_column_major_slice_mut(&mut inv_data, n, n);

        llt.solve_in_place_with_conj(
            faer::Conj::No,
            inv_mat,
            Parallelism::None,
            PodStack::new(&mut mem),
        );

        // Convert to nalgebra DMatrix (also column-major)
        Ok(nalgebra::DMatrix::from_column_slice(n, n, &inv_data))
    }

    /// Returns the dimension of the system.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored entries of the Cholesky factor `L` (including the
    /// diagonal); a measure of the fill-in produced by the ordering.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    /// Build a small SPD sparse matrix using triplet format.
    ///
    /// Matrix:
    ///   [4  1  0]
    ///   [1  3  1]
    ///   [0  1  4]
    ///
    /// This is symmetric positive definite.
    fn build_test_matrix() -> SparseMat {
        let mut tri = TriMat::new((3, 3));
        // Full symmetric matrix (sprs needs both triangles for general ops)
        tri.add_triplet(0, 0, 4.0);
        tri.add_triplet(0, 1, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(1, 1, 3.0);
        tri.add_triplet(1, 2, 1.0);
        tri.add_triplet(2, 1, 1.0);
        tri.add_triplet(2, 2, 4.0);
        tri.to_csc()
    }

    /// Build a larger SPD sparse matrix (5x5 tridiagonal-like).
    ///
    /// Matrix:
    ///   [10  2  0  0  0]
    ///   [ 2 10  2  0  0]
    ///   [ 0  2 10  2  0]
    ///   [ 0  0  2 10  2]
    ///   [ 0  0  0  2 10]
    fn build_larger_test_matrix() -> SparseMat {
        let n = 5;
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 10.0);
            if i + 1 < n {
                tri.add_triplet(i, i + 1, 2.0);
                tri.add_triplet(i + 1, i, 2.0);
            }
        }
        tri.to_csc()
    }

    #[test]
    fn test_create_solver() {
        let mat = build_test_matrix();
        let solver = SparseCholeskySolver::new(&mat);
        assert!(
            solver.is_ok(),
            "Failed to create solver: {:?}",
            solver.err()
        );
        assert_eq!(solver.unwrap().dim(), 3);
    }

    #[test]
    fn test_solve_basic() {
        let mat = build_test_matrix();
        let solver = SparseCholeskySolver::new(&mat).unwrap();

        // Solve A * x = b where b = A * [1, 2, 3]
        // A * [1, 2, 3] = [4+2+0, 1+6+3, 0+2+12] = [6, 10, 14]
        let rhs = vec![6.0, 10.0, 14.0];
        let sol = solver.solve(&rhs).unwrap();

        let tol = 1e-10;
        assert!(
            (sol[0] - 1.0).abs() < tol,
            "sol[0] = {}, expected 1.0",
            sol[0]
        );
        assert!(
            (sol[1] - 2.0).abs() < tol,
            "sol[1] = {}, expected 2.0",
            sol[1]
        );
        assert!(
            (sol[2] - 3.0).abs() < tol,
            "sol[2] = {}, expected 3.0",
            sol[2]
        );
    }

    #[test]
    fn test_solve_larger() {
        let mat = build_larger_test_matrix();
        let solver = SparseCholeskySolver::new(&mat).unwrap();

        // Known solution: x = [1, 1, 1, 1, 1]
        // A * [1,1,1,1,1] = [12, 14, 14, 14, 12]
        let rhs = vec![12.0, 14.0, 14.0, 14.0, 12.0];
        let sol = solver.solve(&rhs).unwrap();

        let tol = 1e-10;
        for (i, &s) in sol.iter().enumerate() {
            assert!((s - 1.0).abs() < tol, "sol[{i}] = {s}, expected 1.0");
        }
    }

    #[test]
    fn test_refactorize() {
        let mat = build_test_matrix();
        let mut solver = SparseCholeskySolver::new(&mat).unwrap();

        // Refactorize with the same matrix (should still work)
        assert!(solver.refactorize(&mat).is_ok());

        // Solve again to verify
        let rhs = vec![6.0, 10.0, 14.0];
        let sol = solver.solve(&rhs).unwrap();
        let tol = 1e-10;
        assert!((sol[0] - 1.0).abs() < tol);
        assert!((sol[1] - 2.0).abs() < tol);
        assert!((sol[2] - 3.0).abs() < tol);
    }

    #[test]
    fn test_log_determinant() {
        let mat = build_test_matrix();
        let solver = SparseCholeskySolver::new(&mat).unwrap();

        // The matrix is:
        //   [4  1  0]
        //   [1  3  1]
        //   [0  1  4]
        //
        // det = 4*(3*4 - 1*1) - 1*(1*4 - 1*0) + 0 = 4*11 - 4 = 40
        // log(det) = log(40) = ln(40)
        let expected_log_det = 40.0_f64.ln();
        let computed_log_det = solver.log_determinant();

        let tol = 1e-10;
        assert!(
            (computed_log_det - expected_log_det).abs() < tol,
            "log_det = {computed_log_det}, expected {expected_log_det}"
        );
    }

    #[test]
    fn test_inverse() {
        let mat = build_test_matrix();
        let solver = SparseCholeskySolver::new(&mat).unwrap();
        let inv = solver.inverse().unwrap();

        // Verify A * A^{-1} = I
        // Convert A to dense first
        let n = 3;
        let mut a_dense = nalgebra::DMatrix::zeros(n, n);
        for (val, (row, col)) in mat.iter() {
            a_dense[(row, col)] = *val;
        }

        let product = &a_dense * &inv;
        let identity = nalgebra::DMatrix::<f64>::identity(n, n);

        let tol = 1e-10;
        for i in 0..n {
            for j in 0..n {
                let diff: f64 = product[(i, j)] - identity[(i, j)];
                assert!(
                    diff.abs() < tol,
                    "product[({i},{j})] = {}, expected {}",
                    product[(i, j)],
                    identity[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_solve() {
        let mat = build_test_matrix();
        let solver = SparseCholeskySolver::new(&mat).unwrap();
        let result = solver.solve(&[1.0, 2.0]); // wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_not_positive_definite() {
        // Build a matrix that is NOT positive definite
        let mut tri = TriMat::new((2, 2));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(0, 1, 5.0);
        tri.add_triplet(1, 0, 5.0);
        tri.add_triplet(1, 1, 1.0);
        let mat = tri.to_csc();

        let result = SparseCholeskySolver::new(&mat);
        assert!(result.is_err(), "Should fail for non-SPD matrix");
    }

    /// Random-ish SPD matrix with an irregular pattern (a banded part plus a
    /// dense row/column) so that the AMD ordering is non-trivial.
    fn irregular_spd(n: usize) -> SparseMat {
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 10.0 + (i % 7) as f64);
            if i + 1 < n {
                let v = -1.0 - ((i * 13) % 5) as f64 * 0.3;
                tri.add_triplet(i, i + 1, v);
                tri.add_triplet(i + 1, i, v);
            }
            if i + 3 < n && i % 2 == 0 {
                tri.add_triplet(i, i + 3, 0.5);
                tri.add_triplet(i + 3, i, 0.5);
            }
            if i > 0 {
                // dense first row/column
                tri.add_triplet(0, i, 0.2);
                tri.add_triplet(i, 0, 0.2);
            }
        }
        tri.to_csc()
    }

    fn dense_inverse(a: &SparseMat) -> nalgebra::DMatrix<f64> {
        let n = a.rows();
        let mut d = nalgebra::DMatrix::zeros(n, n);
        for (v, (i, j)) in a.iter() {
            d[(i, j)] = *v;
        }
        d.try_inverse().unwrap()
    }

    #[test]
    fn inverse_subset_matches_dense_inverse_on_pattern() {
        let n = 40;
        let a = irregular_spd(n);
        let solver = SparseCholeskySolver::new(&a).unwrap();
        let z = solver.inverse_subset();
        let a_inv = dense_inverse(&a);

        // Every stored entry equals the dense inverse (this also pins down
        // the permutation convention).
        let mut n_checked = 0;
        for (v, (i, j)) in z.iter() {
            assert!(
                (v - a_inv[(i, j)]).abs() < 1e-9,
                "Z[{i},{j}] = {v}, dense = {}",
                a_inv[(i, j)]
            );
            n_checked += 1;
        }
        assert!(n_checked >= n);
        // Every nonzero of A is covered by the subset.
        for (_, (i, j)) in a.iter() {
            assert!(z.get(i, j).is_some(), "A[{i},{j}] not in the subset");
        }
        // Diagonal helper
        let diag = solver.inverse_diagonal();
        for i in 0..n {
            assert!((diag[i] - a_inv[(i, i)]).abs() < 1e-9);
        }
        // log-determinant against a dense Cholesky
        let mut d = nalgebra::DMatrix::zeros(n, n);
        for (v, (i, j)) in a.iter() {
            d[(i, j)] = *v;
        }
        let l = d.cholesky().unwrap();
        let expected: f64 = 2.0 * (0..n).map(|i| l.l()[(i, i)].ln()).sum::<f64>();
        assert!((solver.log_determinant() - expected).abs() < 1e-8);
    }

    #[test]
    fn test_identity_matrix() {
        // Identity should be trivial
        let n = 4;
        let mut tri = TriMat::new((n, n));
        for i in 0..n {
            tri.add_triplet(i, i, 1.0);
        }
        let mat = tri.to_csc();

        let solver = SparseCholeskySolver::new(&mat).unwrap();

        // Solve I * x = b => x = b
        let rhs = vec![1.0, 2.0, 3.0, 4.0];
        let sol = solver.solve(&rhs).unwrap();
        let tol = 1e-14;
        for i in 0..n {
            assert!(
                (sol[i] - rhs[i]).abs() < tol,
                "sol[{i}] = {}, expected {}",
                sol[i],
                rhs[i]
            );
        }

        // log|I| = 0
        let log_det = solver.log_determinant();
        assert!(
            log_det.abs() < tol,
            "log_det of identity = {log_det}, expected 0.0"
        );
    }
}
