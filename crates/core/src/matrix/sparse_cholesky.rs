//! Sparse Cholesky solver (faer) with a supernodal selected inversion.
//!
//! The mixed model equations are factorized with faer's **supernodal**
//! sparse Cholesky (`A = L L'` after a fill-reducing AMD ordering): columns
//! with the same sparsity structure are grouped into supernodes whose factor
//! blocks are dense, so the numeric work runs on dense kernels and in
//! parallel.
//!
//! The work is split in two, so repeated factorizations with the same
//! pattern (every REML iteration) only redo the numeric part:
//!
//! * [`CholeskyAnalysis`] holds the ordering and the supernodal structure of
//!   a sparsity pattern and is shared (`Arc`) by all factorizations with it;
//! * [`SparseCholeskySolver`] holds the numeric factor of one matrix and
//!   provides solves, `log|A|` and, computed on first use, the
//!   [`InverseSubset`]: every entry of `A⁻¹` on the pattern of `L + L'`
//!   (which contains the pattern of `A`). That is exactly what REML needs
//!   (`tr(G⁻¹ C^{kk})`, prediction error variances, leverages), obtained
//!   supernode by supernode with the Takahashi recurrences
//!
//!   ```text
//!   Z_RJ = -Z_RR L_RJ L_JJ⁻¹
//!   Z_JJ = L_JJ⁻ᵀ L_JJ⁻¹ - (L_RJ L_JJ⁻¹)ᵀ Z_RJ
//!   ```
//!
//!   for a supernode with columns `J` and below-diagonal rows `R`, at about
//!   the cost of the factorization itself.
//!
//! # Usage
//! ```
//! use plant_breeding_lmm_core::matrix::sparse_cholesky::CholeskyAnalysis;
//! # use sprs::TriMat;
//! # let mut t = TriMat::new((2, 2));
//! # t.add_triplet(0, 0, 4.0); t.add_triplet(1, 1, 3.0); t.add_triplet(0, 1, 1.0); t.add_triplet(1, 0, 1.0);
//! # let a = t.to_csc();
//! let analysis = CholeskyAnalysis::new(&a)?;
//! let solver = analysis.factorize(&a)?; // repeat for new values, same pattern
//! let x = solver.solve(&[1.0, 2.0])?;
//! let log_det = solver.log_determinant();
//! let a_inv_00 = solver.inverse_subset().get(0, 0);
//! # assert!(a_inv_00.is_some());
//! # Ok::<(), plant_breeding_lmm_core::LmmError>(())
//! ```

use std::sync::{Arc, OnceLock};

use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::inverse::{
    inverse as llt_inverse, inverse_scratch as llt_inverse_scratch,
};
use faer::linalg::matmul::matmul;
use faer::linalg::matmul::triangular::{matmul as triangular_matmul, BlockStructure};
use faer::linalg::triangular_solve::solve_upper_triangular_in_place;
use faer::sparse::linalg::cholesky::supernodal::SymbolicSupernodalCholesky;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Accum, Conj, MatMut, MatRef, Par, Side};
use sprs::TriMat;

/// Parallelism for a dense kernel of about `flops` floating-point
/// operations: all cores for large kernels, sequential for small ones (where
/// dispatching work to threads costs more than it saves) and always in
/// WebAssembly (no threads there).
fn par_for(flops: f64) -> Par {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = flops;
        Par::Seq
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        const PARALLEL_FLOPS: f64 = 1e7;
        if flops < PARALLEL_FLOPS {
            Par::Seq
        } else {
            Par::rayon(0)
        }
    }
}

/// Dense work of one supernode with `m` columns and `r` rows below them
/// (factorization and selected inversion are both of this order).
fn supernode_flops(m: usize, r: usize) -> f64 {
    let (m, r) = (m as f64, r as f64);
    m * m * (m + r) + m * r * r
}

/// Upper triangle of a symmetric sparse matrix as CSC arrays (sorted rows).
struct UpperCsc {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

fn upper_triangle(matrix: &SparseMat) -> Result<UpperCsc> {
    let n = matrix.rows();
    if n != matrix.cols() {
        return Err(LmmError::DimensionMismatch {
            expected: n,
            got: matrix.cols(),
            context: "sparse Cholesky: matrix must be square".to_string(),
        });
    }
    let owned;
    let csc = if matrix.is_csc() {
        matrix
    } else {
        owned = matrix.to_csc();
        &owned
    };
    let indptr = csc.indptr();
    let indptr = indptr.raw_storage();
    let (indices, data) = (csc.indices(), csc.data());
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(data.len() / 2 + n);
    let mut values = Vec::with_capacity(data.len() / 2 + n);
    col_ptr.push(0);
    for j in 0..n {
        let (start, end) = (indptr[j] - indptr[0], indptr[j + 1] - indptr[0]);
        // Rows are sorted, so the upper part of a column is a prefix; sprs
        // does not guarantee sortedness for every construction, so check.
        let mut last = None;
        for a in start..end {
            let i = indices[a];
            if i > j {
                continue;
            }
            if last.is_some_and(|l| l >= i) {
                return Err(LmmError::CholeskyFailed(
                    "sparse Cholesky: matrix indices are not sorted".to_string(),
                ));
            }
            last = Some(i);
            row_idx.push(i);
            values.push(data[a]);
        }
        col_ptr.push(row_idx.len());
    }
    Ok(UpperCsc {
        col_ptr,
        row_idx,
        values,
    })
}

/// Ordering and supernodal structure of a symmetric sparsity pattern.
#[derive(Debug)]
pub struct CholeskyAnalysis {
    symbolic: SymbolicCholesky<usize>,
    dim: usize,
    /// Upper-triangle pattern the analysis was made for.
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// `perm_inv[i]`: position of original index `i` in the factor.
    perm_inv: Vec<usize>,
    /// Supernode of every (permuted) column.
    col_supernode: Vec<usize>,
    /// Dense work (flops) of the most expensive supernode.
    max_supernode_flops: f64,
}

impl CholeskyAnalysis {
    /// Analyse the pattern of a symmetric matrix (only the pattern of its
    /// upper triangle is used).
    pub fn new(matrix: &SparseMat) -> Result<Arc<Self>> {
        let upper = upper_triangle(matrix)?;
        Self::from_upper(upper).map(|(analysis, _)| analysis)
    }

    fn from_upper(upper: UpperCsc) -> Result<(Arc<Self>, Vec<f64>)> {
        let n = upper.col_ptr.len() - 1;
        let pattern =
            SymbolicSparseColMatRef::new_checked(n, n, &upper.col_ptr, None, &upper.row_idx);
        let params = CholeskySymbolicParams {
            // `FORCE_SUPERNODAL` (0.0) still selects the simplicial variant
            // for a diagonal matrix (zero flops); the selected inversion
            // below needs the supernodal structure.
            supernodal_flop_ratio_threshold: SupernodalThreshold(-1.0),
            ..Default::default()
        };
        let symbolic =
            factorize_symbolic_cholesky(pattern, Side::Upper, SymmetricOrdering::Amd, params)
                .map_err(|e| {
                    LmmError::CholeskyFailed(format!("symbolic factorization failed: {e:?}"))
                })?;
        let perm_inv = match symbolic.perm() {
            Some(perm) => {
                let (_, inv) = perm.arrays();
                inv.to_vec()
            }
            None => (0..n).collect(),
        };
        let mut col_supernode = vec![0; n];
        let mut max_supernode_flops = 0.0f64;
        if let SymbolicCholeskyRaw::Supernodal(sn) = symbolic.raw() {
            let begin = sn.supernode_begin();
            let end = sn.supernode_end();
            for s in 0..sn.n_supernodes() {
                for c in begin[s]..end[s] {
                    col_supernode[c] = s;
                }
                let m = end[s] - begin[s];
                max_supernode_flops =
                    max_supernode_flops.max(supernode_flops(m, sn.supernode(s).pattern().len()));
            }
        }
        Ok((
            Arc::new(Self {
                symbolic,
                dim: n,
                col_ptr: upper.col_ptr,
                row_idx: upper.row_idx,
                perm_inv,
                col_supernode,
                max_supernode_flops,
            }),
            upper.values,
        ))
    }

    /// Dimension of the analysed matrix.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored entries of the factor `L`.
    pub fn factor_nnz(&self) -> usize {
        self.symbolic.len_val()
    }

    fn supernodal(&self) -> &SymbolicSupernodalCholesky<usize> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Supernodal(sn) => sn,
            SymbolicCholeskyRaw::Simplicial(_) => {
                unreachable!("the analysis always requests a supernodal factorization")
            }
        }
    }

    /// Factorize a matrix with the analysed pattern (a different pattern is
    /// analysed afresh).
    pub fn factorize(self: &Arc<Self>, matrix: &SparseMat) -> Result<SparseCholeskySolver> {
        let upper = upper_triangle(matrix)?;
        if upper.col_ptr == self.col_ptr && upper.row_idx == self.row_idx {
            SparseCholeskySolver::factorize_values(Arc::clone(self), &upper.values)
        } else {
            let (analysis, values) = Self::from_upper(upper)?;
            SparseCholeskySolver::factorize_values(analysis, &values)
        }
    }

    /// The analysed matrix (upper triangle) with the given values.
    fn matrix<'a>(&'a self, values: &'a [f64]) -> SparseColMatRef<'a, usize, f64> {
        SparseColMatRef::new(
            SymbolicSparseColMatRef::new_checked(
                self.dim,
                self.dim,
                &self.col_ptr,
                None,
                &self.row_idx,
            ),
            values,
        )
    }
}

/// Numeric sparse Cholesky factorization of one symmetric positive definite
/// matrix.
#[derive(Debug)]
pub struct SparseCholeskySolver {
    analysis: Arc<CholeskyAnalysis>,
    l_values: Vec<f64>,
    inverse: OnceLock<InverseSubset>,
}

impl SparseCholeskySolver {
    /// Analyse and factorize a symmetric positive definite matrix (only its
    /// upper triangle is used).
    pub fn new(matrix: &SparseMat) -> Result<Self> {
        let (analysis, values) = CholeskyAnalysis::from_upper(upper_triangle(matrix)?)?;
        Self::factorize_values(analysis, &values)
    }

    fn factorize_values(analysis: Arc<CholeskyAnalysis>, values: &[f64]) -> Result<Self> {
        let mut l_values = vec![0.0f64; analysis.symbolic.len_val()];
        let par = par_for(analysis.max_supernode_flops);
        let mut mem = MemBuffer::new(
            analysis
                .symbolic
                .factorize_numeric_llt_scratch::<f64>(par, Default::default()),
        );
        analysis
            .symbolic
            .factorize_numeric_llt(
                &mut l_values,
                analysis.matrix(values),
                Side::Upper,
                Default::default(),
                par,
                MemStack::new(&mut mem),
                Default::default(),
            )
            .map_err(|_| LmmError::NotPositiveDefinite)?;
        Ok(Self {
            analysis,
            l_values,
            inverse: OnceLock::new(),
        })
    }

    /// Refactorize with new values (the analysis is reused when the pattern
    /// is unchanged).
    pub fn refactorize(&mut self, matrix: &SparseMat) -> Result<()> {
        if matrix.rows() != self.dim() {
            return Err(LmmError::DimensionMismatch {
                expected: self.dim(),
                got: matrix.rows(),
                context: "SparseCholeskySolver::refactorize: dimension changed".to_string(),
            });
        }
        *self = self.analysis.factorize(matrix)?;
        Ok(())
    }

    /// The shared analysis.
    pub fn analysis(&self) -> &Arc<CholeskyAnalysis> {
        &self.analysis
    }

    fn solve_matrix(&self, rhs: MatMut<'_, f64>) {
        // Supernodal solves are memory-bound and made of many small kernels.
        let par = Par::Seq;
        let symbolic = &self.analysis.symbolic;
        let mut mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(rhs.ncols(), par));
        LltRef::<'_, usize, f64>::new(symbolic, &self.l_values).solve_in_place_with_conj(
            Conj::No,
            rhs,
            par,
            MemStack::new(&mut mem),
        );
    }

    /// Solve `A x = b`.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        if rhs.len() != self.dim() {
            return Err(LmmError::DimensionMismatch {
                expected: self.dim(),
                got: rhs.len(),
                context: "SparseCholeskySolver::solve: rhs dimension".to_string(),
            });
        }
        let mut x = rhs.to_vec();
        self.solve_matrix(MatMut::from_column_major_slice_mut(&mut x, rhs.len(), 1));
        Ok(x)
    }

    /// Solve `A X = B` for several right-hand sides at once (`B` is
    /// `dim x k`).
    pub fn solve_many(&self, rhs: &nalgebra::DMatrix<f64>) -> Result<nalgebra::DMatrix<f64>> {
        if rhs.nrows() != self.dim() {
            return Err(LmmError::DimensionMismatch {
                expected: self.dim(),
                got: rhs.nrows(),
                context: "SparseCholeskySolver::solve_many: rhs rows".to_string(),
            });
        }
        let mut x = rhs.clone();
        let (n, k) = x.shape();
        self.solve_matrix(MatMut::from_column_major_slice_mut(x.as_mut_slice(), n, k));
        Ok(x)
    }

    /// `log|A| = 2 Σ log L_ii`.
    pub fn log_determinant(&self) -> f64 {
        let sn = self.analysis.supernodal();
        let (begin, end) = (sn.supernode_begin(), sn.supernode_end());
        let val_ptr = sn.col_ptr_for_val();
        let mut log_det = 0.0;
        for s in 0..sn.n_supernodes() {
            let m = end[s] - begin[s];
            let rows = m + sn.supernode(s).pattern().len();
            let block = &self.l_values[val_ptr[s]..val_ptr[s + 1]];
            for c in 0..m {
                log_det += block[c + c * rows].ln();
            }
        }
        2.0 * log_det
    }

    /// Selected entries of `A⁻¹` (computed on first use, then cached): every
    /// entry on the pattern of `L + L'`, which contains the pattern of `A`.
    /// Entries outside that pattern are not zero in `A⁻¹`, just not stored.
    pub fn inverse_subset(&self) -> &InverseSubset {
        self.inverse.get_or_init(|| self.selected_inversion())
    }

    fn selected_inversion(&self) -> InverseSubset {
        let a = &*self.analysis;
        let sn = a.supernodal();
        let (begin, end) = (sn.supernode_begin(), sn.supernode_end());
        let val_ptr = sn.col_ptr_for_val();
        let mut z = vec![0.0f64; self.l_values.len()];
        // Work buffers, allocated once at their largest size.
        let (max_r, max_mr) = (0..sn.n_supernodes())
            .map(|s| {
                let r = sn.supernode(s).pattern().len();
                (r, (end[s] - begin[s]) * r)
            })
            .fold((0, 0), |(a, b), (r, mr)| (a.max(r), b.max(mr)));
        let mut z_rr_buf = vec![0.0f64; max_r * max_r];
        let mut u_t_buf = vec![0.0f64; max_mr];

        for s in (0..sn.n_supernodes()).rev() {
            let m = end[s] - begin[s];
            let pattern = sn.supernode(s).pattern();
            let r = pattern.len();
            let par = par_for(supernode_flops(m, r));
            let l_block = MatRef::from_column_major_slice(
                &self.l_values[val_ptr[s]..val_ptr[s + 1]],
                m + r,
                m,
            );
            let l_jj = l_block.submatrix(0, 0, m, m);
            let l_rj = l_block.submatrix(m, 0, r, m);

            let mut u_t = MatMut::from_column_major_slice_mut(&mut u_t_buf[..m * r], m, r);
            let mut z_rr = MatMut::from_column_major_slice_mut(&mut z_rr_buf[..r * r], r, r);
            if r > 0 {
                // Uᵀ = (L_RJ L_JJ⁻¹)ᵀ = L_JJ⁻ᵀ L_RJᵀ
                u_t.copy_from(l_rj.transpose());
                solve_upper_triangular_in_place(l_jj.transpose(), u_t.as_mut(), par);

                // Z_RR from the supernodes of the rows in R (all later).
                for (a_idx, &k) in pattern.iter().enumerate() {
                    let t = a.col_supernode[k];
                    let (ft, lt) = (begin[t], end[t]);
                    let t_pattern = sn.supernode(t).pattern();
                    let t_rows = (lt - ft) + t_pattern.len();
                    let t_block = &z[val_ptr[t]..val_ptr[t + 1]];
                    let col_off = (k - ft) * t_rows;
                    let mut pos = 0usize;
                    for (b_idx, &i) in pattern.iter().enumerate().skip(a_idx) {
                        let row = if i < lt {
                            i - ft
                        } else {
                            // i is in t's pattern (closure of the supernodal
                            // structure); gallop forward from the last hit.
                            let found =
                                gallop(&t_pattern[pos..], i).expect("supernodal pattern is closed");
                            pos += found;
                            (lt - ft) + pos
                        };
                        let v = t_block[col_off + row];
                        z_rr[(b_idx, a_idx)] = v;
                        z_rr[(a_idx, b_idx)] = v;
                    }
                }
            }

            let out =
                MatMut::from_column_major_slice_mut(&mut z[val_ptr[s]..val_ptr[s + 1]], m + r, m);
            let (mut z_jj, mut z_rj) = out.split_at_row_mut(m);
            // Z_JJ = (L_JJ L_JJᵀ)⁻¹ (lower triangle); its m x m scratch is
            // only large for the few big (root) supernodes.
            let mut mem = MemBuffer::new(llt_inverse_scratch::<f64>(m, par));
            llt_inverse(z_jj.as_mut(), l_jj, par, MemStack::new(&mut mem));
            drop(mem);
            if r > 0 {
                // Z_RJ = -Z_RR U,  Z_JJ -= Uᵀ Z_RJ
                matmul(
                    z_rj.as_mut(),
                    Accum::Replace,
                    z_rr.as_ref(),
                    u_t.as_ref().transpose(),
                    -1.0,
                    par,
                );
                triangular_matmul(
                    z_jj.as_mut(),
                    BlockStructure::TriangularLower,
                    Accum::Add,
                    u_t.as_ref(),
                    BlockStructure::Rectangular,
                    z_rj.as_ref(),
                    BlockStructure::Rectangular,
                    -1.0,
                    par,
                );
            }
            // Store both triangles so lookups may use either.
            for c in 0..m {
                for i in c + 1..m {
                    z_jj[(c, i)] = z_jj[(i, c)];
                }
            }
        }

        InverseSubset {
            analysis: Arc::clone(&self.analysis),
            values: z,
        }
    }

    /// Diagonal of `A⁻¹` (original ordering).
    pub fn inverse_diagonal(&self) -> Vec<f64> {
        self.inverse_subset().diagonal()
    }

    /// The full dense inverse (`dim` solves; for small problems and tests).
    pub fn inverse(&self) -> Result<nalgebra::DMatrix<f64>> {
        self.solve_many(&nalgebra::DMatrix::identity(self.dim(), self.dim()))
    }

    /// Dimension of the system.
    pub fn dim(&self) -> usize {
        self.analysis.dim
    }

    /// Number of stored entries of the Cholesky factor `L` (a measure of the
    /// fill-in produced by the ordering).
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len()
    }
}

/// Position of `target` in the sorted slice `hay` by exponential then binary
/// search (fast when `target` is near the front).
fn gallop(hay: &[usize], target: usize) -> Option<usize> {
    let mut hi = 1;
    while hi < hay.len() && hay[hi - 1] < target {
        hi *= 2;
    }
    let lo = hi / 2;
    let hi = hi.min(hay.len());
    hay[lo..hi].binary_search(&target).ok().map(|p| lo + p)
}

/// Entries of `A⁻¹` on the pattern of the supernodal factor `L + L'`.
#[derive(Debug, Clone)]
pub struct InverseSubset {
    analysis: Arc<CholeskyAnalysis>,
    /// Values in the supernodal layout of `L` (the `J x J` blocks are stored
    /// in full, the `R x J` blocks below them).
    values: Vec<f64>,
}

impl InverseSubset {
    /// Dimension of `A`.
    pub fn dim(&self) -> usize {
        self.analysis.dim
    }

    /// Number of stored entries (lower triangle of the supernodal pattern).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `A⁻¹[i, j]` (original ordering), or `None` if the entry is outside the
    /// stored pattern.
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        let a = &*self.analysis;
        let (pi, pj) = (a.perm_inv[i], a.perm_inv[j]);
        let (row, col) = if pi >= pj { (pi, pj) } else { (pj, pi) };
        let sn = a.supernodal();
        let s = a.col_supernode[col];
        let (f, l) = (sn.supernode_begin()[s], sn.supernode_end()[s]);
        let pattern = sn.supernode(s).pattern();
        let rows = (l - f) + pattern.len();
        let block = &self.values[sn.col_ptr_for_val()[s]..sn.col_ptr_for_val()[s + 1]];
        let r = if row < l {
            row - f
        } else {
            (l - f) + pattern.binary_search(&row).ok()?
        };
        Some(block[r + (col - f) * rows])
    }

    /// Diagonal of `A⁻¹` (original ordering).
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim())
            .map(|i| self.get(i, i).expect("the diagonal is always stored"))
            .collect()
    }

    /// The stored entries as a symmetric sparse matrix (both triangles,
    /// original ordering).
    pub fn to_sparse(&self) -> SparseMat {
        let a = &*self.analysis;
        let sn = a.supernodal();
        let (begin, end) = (sn.supernode_begin(), sn.supernode_end());
        // original index of each permuted position
        let mut fwd = vec![0; a.dim];
        for (orig, &p) in a.perm_inv.iter().enumerate() {
            fwd[p] = orig;
        }
        let mut tri = TriMat::with_capacity((a.dim, a.dim), 2 * self.values.len());
        for s in 0..sn.n_supernodes() {
            let (f, l) = (begin[s], end[s]);
            let pattern = sn.supernode(s).pattern();
            let rows = (l - f) + pattern.len();
            let block = &self.values[sn.col_ptr_for_val()[s]..sn.col_ptr_for_val()[s + 1]];
            for c in f..l {
                let col_off = (c - f) * rows;
                let mut push = |r: usize, v: f64| {
                    tri.add_triplet(fwd[r], fwd[c], v);
                    if r != c {
                        tri.add_triplet(fwd[c], fwd[r], v);
                    }
                };
                for r in c..l {
                    push(r, block[col_off + (r - f)]);
                }
                for (b, &r) in pattern.iter().enumerate() {
                    push(r, block[col_off + (l - f) + b]);
                }
            }
        }
        tri.to_csc()
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
        let z = solver.inverse_subset().to_sparse();
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

    /// SPD matrix `W'W + D` from a random sparse design (irregular patterns
    /// with large supernodes after ordering).
    fn random_spd(n: usize, rows: usize, per_row: usize, seed: u64) -> SparseMat {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut dense = nalgebra::DMatrix::<f64>::zeros(n, n);
        for _ in 0..rows {
            let cols: Vec<usize> = (0..per_row).map(|_| rng.gen_range(0..n)).collect();
            let vals: Vec<f64> = (0..per_row).map(|_| rng.gen_range(-1.0..1.0)).collect();
            for (a, &i) in cols.iter().enumerate() {
                for (b, &j) in cols.iter().enumerate() {
                    dense[(i, j)] += vals[a] * vals[b];
                }
            }
        }
        let mut tri = TriMat::new((n, n));
        for j in 0..n {
            for i in 0..n {
                let v = dense[(i, j)] + if i == j { 0.5 + (i % 3) as f64 } else { 0.0 };
                if v != 0.0 {
                    tri.add_triplet(i, j, v);
                }
            }
        }
        tri.to_csc()
    }

    fn assert_subset_matches_dense(a: &SparseMat) {
        let n = a.rows();
        let solver = SparseCholeskySolver::new(a).unwrap();
        let a_inv = dense_inverse(a);
        let subset = solver.inverse_subset();
        let z = subset.to_sparse();
        assert!(z.nnz() >= a.nnz());
        for (v, (i, j)) in z.iter() {
            assert!(
                (v - a_inv[(i, j)]).abs() < 1e-9 * a_inv[(i, j)].abs().max(1.0),
                "Z[{i},{j}] = {v}, dense = {}",
                a_inv[(i, j)]
            );
            assert_eq!(subset.get(i, j), Some(*v));
        }
        for (_, (i, j)) in a.iter() {
            let v = subset.get(i, j).expect("pattern of A is stored");
            assert!((v - a_inv[(i, j)]).abs() < 1e-9 * a_inv[(i, j)].abs().max(1.0));
        }
        for (i, d) in subset.diagonal().iter().enumerate() {
            assert!((d - a_inv[(i, i)]).abs() < 1e-9 * a_inv[(i, i)].abs().max(1.0));
        }
        let mut d = nalgebra::DMatrix::zeros(n, n);
        for (v, (i, j)) in a.iter() {
            d[(i, j)] = *v;
        }
        let expected: f64 = 2.0
            * d.cholesky()
                .unwrap()
                .l()
                .diagonal()
                .iter()
                .map(|x| x.ln())
                .sum::<f64>();
        assert!((solver.log_determinant() - expected).abs() < 1e-8 * expected.abs().max(1.0));
    }

    #[test]
    fn selected_inversion_matches_dense_on_varied_patterns() {
        // sparse, moderately dense and nearly dense patterns
        for (n, rows, per_row, seed) in [
            (60, 40, 3, 1),
            (80, 120, 5, 2),
            (50, 200, 8, 3),
            (120, 90, 4, 4),
        ] {
            assert_subset_matches_dense(&random_spd(n, rows, per_row, seed));
        }
        assert_subset_matches_dense(&irregular_spd(150));
        assert_subset_matches_dense(&build_larger_test_matrix());
    }

    #[test]
    fn analysis_is_reused_for_new_values() {
        let a = random_spd(70, 90, 4, 7);
        let analysis = CholeskyAnalysis::new(&a).unwrap();
        // same pattern, different values
        let b = a.map(|v| v * 1.7);
        let from_analysis = analysis.factorize(&b).unwrap();
        assert!(Arc::ptr_eq(from_analysis.analysis(), &analysis));
        let fresh = SparseCholeskySolver::new(&b).unwrap();
        assert!((from_analysis.log_determinant() - fresh.log_determinant()).abs() < 1e-9);
        let d1 = from_analysis.inverse_diagonal();
        let d2 = fresh.inverse_diagonal();
        for (x, y) in d1.iter().zip(&d2) {
            assert!((x - y).abs() < 1e-12);
        }
        // a different pattern is analysed afresh
        let c = random_spd(70, 50, 3, 8);
        let other = analysis.factorize(&c).unwrap();
        assert!(!Arc::ptr_eq(other.analysis(), &analysis));
        assert_subset_matches_dense(&c);
    }

    #[test]
    fn solve_many_matches_single_solves() {
        let a = random_spd(40, 60, 4, 9);
        let solver = SparseCholeskySolver::new(&a).unwrap();
        let b = nalgebra::DMatrix::from_fn(40, 3, |i, j| (i * (j + 1)) as f64 * 0.1 - 1.0);
        let x = solver.solve_many(&b).unwrap();
        for j in 0..3 {
            let col: Vec<f64> = b.column(j).iter().copied().collect();
            let single = solver.solve(&col).unwrap();
            for i in 0..40 {
                assert!((x[(i, j)] - single[i]).abs() < 1e-12);
            }
        }
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
