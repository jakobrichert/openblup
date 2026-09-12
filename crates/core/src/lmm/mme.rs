use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use sprs::CsMat;

use crate::error::{LmmError, Result};
use crate::matrix::sparse::{xt_y, TripletBuilder};
use crate::matrix::sparse_cholesky::SparseCholeskySolver;
use crate::types::SparseMat;

/// The inverse of the MME coefficient matrix, as needed by REML.
///
/// * `Dense` holds the full `C⁻¹` (used by the general engine for structured
///   models, whose `C` is dense anyway).
/// * `Sparse` holds the entries of `C⁻¹` on the pattern of the sparse
///   Cholesky factor (Takahashi inverse subset), the factorization itself
///   (for `C⁻¹ v` products) and the `p` columns of `C⁻¹` belonging to the
///   fixed effects. This is what the scalar-residual engines use, and it is
///   what makes animal models with many thousands of equations tractable.
#[derive(Debug, Clone)]
pub enum MmeInverse {
    /// Full dense inverse.
    Dense(DMatrix<f64>),
    /// Sparse inverse subset with the factorization and the fixed-effect columns.
    Sparse {
        /// `C⁻¹` on the pattern of `L + L'` (symmetric, both triangles).
        subset: SparseMat,
        /// Cholesky factorization of `C` (for solves).
        solver: Arc<SparseCholeskySolver>,
        /// `C⁻¹[:, 0..p]` (dim x p).
        fixed_cols: DMatrix<f64>,
    },
}

impl MmeInverse {
    /// Dimension of the MME.
    pub fn dim(&self) -> usize {
        match self {
            MmeInverse::Dense(m) => m.nrows(),
            MmeInverse::Sparse { subset, .. } => subset.rows(),
        }
    }

    /// `C⁻¹[i, j]`. For the sparse variant the entry must lie in the stored
    /// pattern (which contains the pattern of `C`, in particular every
    /// relationship-matrix entry and the whole fixed-effects block); other
    /// entries return 0.
    pub fn entry(&self, i: usize, j: usize) -> f64 {
        match self {
            MmeInverse::Dense(m) => m[(i, j)],
            MmeInverse::Sparse {
                subset, fixed_cols, ..
            } => {
                let p = fixed_cols.ncols();
                if j < p {
                    fixed_cols[(i, j)]
                } else if i < p {
                    fixed_cols[(j, i)]
                } else {
                    subset.get(i, j).copied().unwrap_or(0.0)
                }
            }
        }
    }

    /// Diagonal of `C⁻¹` (prediction error variances / squared SEs).
    pub fn diagonal(&self) -> Vec<f64> {
        match self {
            MmeInverse::Dense(m) => (0..m.nrows()).map(|i| m[(i, i)]).collect(),
            MmeInverse::Sparse { subset, .. } => (0..subset.rows())
                .map(|i| subset.get(i, i).copied().unwrap_or(0.0))
                .collect(),
        }
    }

    /// `C⁻¹ v`.
    pub fn apply(&self, v: &[f64]) -> Result<Vec<f64>> {
        match self {
            MmeInverse::Dense(m) => Ok((m * DVector::from_column_slice(v)).as_slice().to_vec()),
            MmeInverse::Sparse { solver, .. } => solver.solve(v),
        }
    }

    /// The `p` columns of `C⁻¹` belonging to the fixed effects (dim x p).
    pub fn fixed_columns(&self, p: usize) -> DMatrix<f64> {
        match self {
            MmeInverse::Dense(m) => m.columns(0, p).into_owned(),
            MmeInverse::Sparse { fixed_cols, .. } => fixed_cols.clone(),
        }
    }

    /// The fixed-effects block `C⁻¹[0..p, 0..p]` (covariance of the BLUEs).
    pub fn fixed_block(&self, p: usize) -> DMatrix<f64> {
        match self {
            MmeInverse::Dense(m) => m.view((0, 0), (p, p)).into_owned(),
            MmeInverse::Sparse { fixed_cols, .. } => fixed_cols.rows(0, p).into_owned(),
        }
    }

    /// `Σ_{ij} B_{ij} C⁻¹[off + j, off + i]` for a sparse block `B` located
    /// at `off` on the diagonal of the MME (e.g. `tr(K⁻¹ C^{kk})`).
    pub fn trace_block(&self, b: &SparseMat, off: usize) -> f64 {
        b.iter()
            .map(|(v, (i, j))| v * self.entry(off + j, off + i))
            .sum()
    }

    /// The dense inverse, if this is the dense variant.
    pub fn as_dense(&self) -> Option<&DMatrix<f64>> {
        match self {
            MmeInverse::Dense(m) => Some(m),
            MmeInverse::Sparse { .. } => None,
        }
    }

    /// Full dense inverse (for the sparse variant this performs `dim` solves;
    /// intended for tests and small problems).
    pub fn to_dense(&self) -> Result<DMatrix<f64>> {
        match self {
            MmeInverse::Dense(m) => Ok(m.clone()),
            MmeInverse::Sparse { solver, .. } => {
                let n = solver.dim();
                let mut out = DMatrix::zeros(n, n);
                let mut e = vec![0.0; n];
                for j in 0..n {
                    e[j] = 1.0;
                    let col = solver.solve(&e)?;
                    e[j] = 0.0;
                    for i in 0..n {
                        out[(i, j)] = col[i];
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Henderson's Mixed Model Equations with a sparse coefficient matrix.
///
/// Same system as [`MixedModelEquations`] but `C` is assembled as a sparse
/// matrix (`W'W / σ²_e` plus the sparse `G⁻¹` blocks) and solved with the
/// sparse Cholesky solver; the inverse is obtained as a Takahashi subset.
/// This is the path taken for models with an IID residual, i.e. the usual
/// animal / genomic / plant-trial models.
#[derive(Debug)]
pub struct SparseMixedModelEquations {
    /// The coefficient matrix C (symmetric, both triangles stored, CSC).
    pub coeff_matrix: SparseMat,
    /// The right-hand side vector.
    pub rhs: Vec<f64>,
    /// Number of fixed effect parameters.
    pub n_fixed: usize,
    /// Number of random effect levels per random term.
    pub n_random: Vec<usize>,
    /// Total dimension of the system.
    pub dim: usize,
}

impl SparseMixedModelEquations {
    /// Assemble the MME for `R = σ²_e I` (`r_inv_scale = 1/σ²_e`) from the
    /// sparse design matrices and the sparse `G⁻¹` blocks (already divided by
    /// their variance).
    pub fn assemble(
        x: &CsMat<f64>,
        z_blocks: &[CsMat<f64>],
        y: &[f64],
        r_inv_scale: f64,
        g_inv_blocks: &[CsMat<f64>],
    ) -> Self {
        let n = y.len();
        let p = x.cols();
        let q_vec: Vec<usize> = z_blocks.iter().map(|z| z.cols()).collect();
        let dim = p + q_vec.iter().sum::<usize>();

        // Row access to W = [X Z_1 ... Z_r]
        let x_csr = x.to_csr();
        let z_csr: Vec<CsMat<f64>> = z_blocks.iter().map(|z| z.to_csr()).collect();
        let mut offsets = Vec::with_capacity(z_blocks.len());
        let mut off = p;
        for q in &q_vec {
            offsets.push(off);
            off += q;
        }

        // W'W / σ²_e = Σ_i w_i w_i' / σ²_e, one observation at a time
        let mut builder = TripletBuilder::new(dim, dim);
        let mut row: Vec<(usize, f64)> = Vec::new();
        for i in 0..n {
            row.clear();
            if let Some(r) = x_csr.outer_view(i) {
                for (c, v) in r.iter() {
                    row.push((c, *v));
                }
            }
            for (k, z) in z_csr.iter().enumerate() {
                if let Some(r) = z.outer_view(i) {
                    for (c, v) in r.iter() {
                        row.push((offsets[k] + c, *v));
                    }
                }
            }
            for (a, (ca, va)) in row.iter().enumerate() {
                builder.add(*ca, *ca, va * va * r_inv_scale);
                for (cb, vb) in row.iter().skip(a + 1) {
                    builder.add_symmetric(*ca, *cb, va * vb * r_inv_scale);
                }
            }
        }
        // + G⁻¹ blocks
        for (k, ginv) in g_inv_blocks.iter().enumerate() {
            for (v, (i, j)) in ginv.iter() {
                builder.add(offsets[k] + i, offsets[k] + j, *v);
            }
        }
        let coeff_matrix = builder.to_csc();

        // RHS: W'y / σ²_e
        let mut rhs = vec![0.0; dim];
        for (j, v) in xt_y(x, y).iter().enumerate() {
            rhs[j] = v * r_inv_scale;
        }
        for (k, z) in z_blocks.iter().enumerate() {
            for (j, v) in xt_y(z, y).iter().enumerate() {
                rhs[offsets[k] + j] = v * r_inv_scale;
            }
        }

        Self {
            coeff_matrix,
            rhs,
            n_fixed: p,
            n_random: q_vec,
            dim,
        }
    }

    /// Solve the MME with the sparse Cholesky solver and compute the inverse
    /// subset, the fixed-effect columns of `C⁻¹` and `log|C|`.
    pub fn solve(&self) -> Result<MmeSolution> {
        let solver = SparseCholeskySolver::new(&self.coeff_matrix)?;
        let solution = solver.solve(&self.rhs)?;
        let log_det_c = solver.log_determinant();
        let subset = solver.inverse_subset();

        let p = self.n_fixed;
        let mut fixed_cols = DMatrix::zeros(self.dim, p);
        let mut e = vec![0.0; self.dim];
        for j in 0..p {
            e[j] = 1.0;
            let col = solver.solve(&e)?;
            e[j] = 0.0;
            for i in 0..self.dim {
                fixed_cols[(i, j)] = col[i];
            }
        }

        let c_inv_diag: Vec<f64> = (0..self.dim)
            .map(|i| subset.get(i, i).copied().unwrap_or(0.0))
            .collect();

        let fixed_effects = solution[..p].to_vec();
        let mut random_effects = Vec::new();
        let mut offset = p;
        for &q in &self.n_random {
            random_effects.push(solution[offset..offset + q].to_vec());
            offset += q;
        }

        Ok(MmeSolution {
            solution,
            fixed_effects,
            random_effects,
            log_det_c,
            c_inv_diag,
            c_inv: Some(MmeInverse::Sparse {
                subset,
                solver: Arc::new(solver),
                fixed_cols,
            }),
        })
    }
}

/// Henderson's Mixed Model Equations.
///
/// ```text
/// [X'R⁻¹X       X'R⁻¹Z          ] [b]   [X'R⁻¹y]
/// [Z'R⁻¹X       Z'R⁻¹Z + G⁻¹   ] [u] = [Z'R⁻¹y]
/// ```
///
/// The coefficient matrix C is symmetric positive definite.
#[derive(Debug)]
pub struct MixedModelEquations {
    /// The full coefficient matrix C (symmetric, stored as dense for Phase 1).
    pub coeff_matrix: nalgebra::DMatrix<f64>,
    /// The right-hand side vector.
    pub rhs: Vec<f64>,
    /// Number of fixed effect parameters.
    pub n_fixed: usize,
    /// Number of random effect levels per random term.
    pub n_random: Vec<usize>,
    /// Total dimension of the system.
    pub dim: usize,
}

impl MixedModelEquations {
    /// Assemble the MME from model components and current variance parameters.
    ///
    /// For Phase 1, R is assumed to be sigma_e^2 * I, so R^{-1} = (1/sigma_e^2) * I.
    /// This simplifies the assembly considerably.
    ///
    /// # Arguments
    /// - `x`: Fixed effects design matrix (n x p)
    /// - `z_blocks`: Random effects design matrices, one per random term
    /// - `y`: Response vector (length n)
    /// - `r_inv_scale`: 1/sigma_e^2 (scalar for identity residual structure)
    /// - `g_inv_blocks`: G^{-1} blocks for each random term (q_i x q_i sparse matrices)
    pub fn assemble(
        x: &CsMat<f64>,
        z_blocks: &[CsMat<f64>],
        y: &[f64],
        r_inv_scale: f64,
        g_inv_blocks: &[CsMat<f64>],
    ) -> Self {
        let n = y.len();
        let p = x.cols();
        let q_vec: Vec<usize> = z_blocks.iter().map(|z| z.cols()).collect();
        let q_total: usize = q_vec.iter().sum();
        let dim = p + q_total;

        // Build the coefficient matrix as dense (Phase 1).
        // For large problems, this will be replaced with sparse assembly + sparse Cholesky.
        let mut c = nalgebra::DMatrix::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        // --- X'R⁻¹X block (top-left, p x p) ---
        let c_xtx = compute_xtx_scaled(x, r_inv_scale, n);
        for i in 0..p {
            for j in 0..p {
                c[(i, j)] = c_xtx[(i, j)];
            }
        }

        // --- X'R⁻¹Z blocks (top-right, p x q_i) and Z'R⁻¹X (bottom-left) ---
        let mut col_offset = p;
        for z in z_blocks {
            let xtz = compute_xtz_scaled(x, z, r_inv_scale, n);
            for i in 0..p {
                for j in 0..z.cols() {
                    c[(i, col_offset + j)] = xtz[(i, j)];
                    c[(col_offset + j, i)] = xtz[(i, j)]; // symmetric
                }
            }
            col_offset += z.cols();
        }

        // --- Z'R⁻¹Z + G⁻¹ blocks (bottom-right) ---
        let mut row_offset = p;
        for (k, z) in z_blocks.iter().enumerate() {
            // Z'R⁻¹Z block
            let ztz = compute_xtx_scaled(z, r_inv_scale, n);
            for i in 0..z.cols() {
                for j in 0..z.cols() {
                    c[(row_offset + i, row_offset + j)] = ztz[(i, j)];
                }
            }

            // Add G⁻¹ for this random term
            if k < g_inv_blocks.len() {
                let ginv = &g_inv_blocks[k];
                for (val, (i, j)) in ginv.iter() {
                    c[(row_offset + i, row_offset + j)] += val;
                }
            }

            // Cross-terms between different random terms: Z_k'R⁻¹Z_l
            let mut col_off2 = p;
            for (l, z2) in z_blocks.iter().enumerate() {
                if l != k && l > k {
                    let cross = compute_xtz_scaled(z, z2, r_inv_scale, n);
                    for i in 0..z.cols() {
                        for j in 0..z2.cols() {
                            c[(row_offset + i, col_off2 + j)] = cross[(i, j)];
                            c[(col_off2 + j, row_offset + i)] = cross[(i, j)];
                        }
                    }
                }
                col_off2 += z2.cols();
            }

            row_offset += z.cols();
        }

        // --- RHS: [X'R⁻¹y; Z_1'R⁻¹y; Z_2'R⁻¹y; ...] ---
        let xty: Vec<f64> = xt_y(x, y).iter().map(|v| v * r_inv_scale).collect();
        rhs[..p].copy_from_slice(&xty);

        let mut rhs_offset = p;
        for z in z_blocks {
            let zty: Vec<f64> = xt_y(z, y).iter().map(|v| v * r_inv_scale).collect();
            rhs[rhs_offset..rhs_offset + z.cols()].copy_from_slice(&zty);
            rhs_offset += z.cols();
        }

        Self {
            coeff_matrix: c,
            rhs,
            n_fixed: p,
            n_random: q_vec,
            dim,
        }
    }

    /// Solve the MME system C * sol = rhs using dense Cholesky.
    pub fn solve(&self) -> Result<MmeSolution> {
        let chol = self
            .coeff_matrix
            .clone()
            .cholesky()
            .ok_or(crate::error::LmmError::NotPositiveDefinite)?;

        let rhs_vec = nalgebra::DVector::from_column_slice(&self.rhs);
        let sol = chol.solve(&rhs_vec);
        let sol_vec: Vec<f64> = sol.as_slice().to_vec();

        // Extract fixed effects
        let fixed_effects = sol_vec[..self.n_fixed].to_vec();

        // Extract random effects per term
        let mut random_effects = Vec::new();
        let mut offset = self.n_fixed;
        for &q in &self.n_random {
            random_effects.push(sol_vec[offset..offset + q].to_vec());
            offset += q;
        }

        // Compute log|C| = 2 * sum(log(diag(L)))
        let l = chol.l();
        let log_det_c = 2.0 * (0..self.dim).map(|i| l[(i, i)].ln()).sum::<f64>();

        // Full dense C^{-1}
        let c_inv = chol.inverse();
        let c_inv_diag: Vec<f64> = (0..self.dim).map(|i| c_inv[(i, i)]).collect();

        Ok(MmeSolution {
            solution: sol_vec,
            fixed_effects,
            random_effects,
            log_det_c,
            c_inv_diag,
            c_inv: Some(MmeInverse::Dense(c_inv)),
        })
    }
}

/// Solution of the Mixed Model Equations.
pub struct MmeSolution {
    /// Full solution vector [b; u1; u2; ...].
    pub solution: Vec<f64>,
    /// Fixed effects (BLUE): b-hat.
    pub fixed_effects: Vec<f64>,
    /// Random effects (BLUP): u-hat, one vec per random term.
    pub random_effects: Vec<Vec<f64>>,
    /// Log-determinant of the coefficient matrix: log|C|.
    pub log_det_c: f64,
    /// Diagonal of C^{-1} (for standard errors and trace computations).
    pub c_inv_diag: Vec<f64>,
    /// `C^{-1}` (dense, or a sparse inverse subset with the factorization);
    /// None if not computed.
    pub c_inv: Option<MmeInverse>,
}

impl MmeSolution {
    /// The inverse, or an error if it was not computed.
    pub fn inverse(&self) -> Result<&MmeInverse> {
        self.c_inv
            .as_ref()
            .ok_or_else(|| LmmError::CholeskyFailed("C^{-1} not available".into()))
    }
}

/// Compute X'X scaled by a scalar, efficiently via column iteration.
fn compute_xtx_scaled(x: &CsMat<f64>, scale: f64, _n: usize) -> nalgebra::DMatrix<f64> {
    let p = x.cols();
    let mut result = nalgebra::DMatrix::zeros(p, p);

    // Ensure CSC format
    let x_csc = if x.is_csc() { x.clone() } else { x.to_csc() };

    // Compute X'X using outer views
    for j in 0..p {
        if let Some(col_j) = x_csc.outer_view(j) {
            for i in j..p {
                if let Some(col_i) = x_csc.outer_view(i) {
                    let dot: f64 = col_j.dot(&col_i);
                    result[(i, j)] = dot * scale;
                    if i != j {
                        result[(j, i)] = dot * scale;
                    }
                }
            }
        }
    }

    result
}

impl MixedModelEquations {
    /// Assemble the MME with a full (possibly non-identity) R⁻¹ matrix.
    ///
    /// This generalizes the scalar R⁻¹ assembly for structured residuals.
    /// R⁻¹ is an n × n sparse matrix.
    pub fn assemble_structured_r(
        x: &CsMat<f64>,
        z_blocks: &[CsMat<f64>],
        y: &[f64],
        r_inv: &CsMat<f64>,
        g_inv_blocks: &[CsMat<f64>],
    ) -> Self {
        let n = y.len();
        let p = x.cols();
        let q_vec: Vec<usize> = z_blocks.iter().map(|z| z.cols()).collect();
        let q_total: usize = q_vec.iter().sum();
        let dim = p + q_total;

        let mut c = nalgebra::DMatrix::zeros(dim, dim);
        let mut rhs = vec![0.0; dim];

        // Convert to dense for R⁻¹ products (Phase 1 approach)
        let r_inv_dense = sparse_to_dense(r_inv, n);

        // X'R⁻¹X
        let x_dense = sparse_cols_to_dense(x, n);
        let r_inv_x = &r_inv_dense * &x_dense;
        let xtrinvx = x_dense.transpose() * &r_inv_x;
        for i in 0..p {
            for j in 0..p {
                c[(i, j)] = xtrinvx[(i, j)];
            }
        }

        // X'R⁻¹Z and Z'R⁻¹X blocks
        let mut col_offset = p;
        let mut z_denses = Vec::new();
        for z in z_blocks {
            let z_dense = sparse_cols_to_dense(z, n);
            let r_inv_z = &r_inv_dense * &z_dense;
            let xtrinvz = x_dense.transpose() * &r_inv_z;
            for i in 0..p {
                for j in 0..z.cols() {
                    c[(i, col_offset + j)] = xtrinvz[(i, j)];
                    c[(col_offset + j, i)] = xtrinvz[(i, j)];
                }
            }
            z_denses.push(z_dense);
            col_offset += z.cols();
        }

        // Z'R⁻¹Z + G⁻¹ blocks
        let mut row_offset = p;
        for (k, z_dense) in z_denses.iter().enumerate() {
            let r_inv_z = &r_inv_dense * z_dense;
            let ztrinvz = z_dense.transpose() * &r_inv_z;
            for i in 0..z_blocks[k].cols() {
                for j in 0..z_blocks[k].cols() {
                    c[(row_offset + i, row_offset + j)] = ztrinvz[(i, j)];
                }
            }

            if k < g_inv_blocks.len() {
                let ginv = &g_inv_blocks[k];
                for (val, (i, j)) in ginv.iter() {
                    c[(row_offset + i, row_offset + j)] += val;
                }
            }

            // Cross-terms
            let mut col_off2 = p;
            for (l, z_dense2) in z_denses.iter().enumerate() {
                if l > k {
                    let cross = z_dense.transpose() * &r_inv_dense * z_dense2;
                    for i in 0..z_blocks[k].cols() {
                        for j in 0..z_blocks[l].cols() {
                            c[(row_offset + i, col_off2 + j)] = cross[(i, j)];
                            c[(col_off2 + j, row_offset + i)] = cross[(i, j)];
                        }
                    }
                }
                col_off2 += z_blocks[l].cols();
            }

            row_offset += z_blocks[k].cols();
        }

        // RHS: X'R⁻¹y, Z'R⁻¹y
        let y_vec = nalgebra::DVector::from_column_slice(y);
        let r_inv_y = &r_inv_dense * &y_vec;
        let xtrinvy = x_dense.transpose() * &r_inv_y;
        for i in 0..p {
            rhs[i] = xtrinvy[i];
        }
        let mut rhs_offset = p;
        for z_dense in &z_denses {
            let ztrinvy = z_dense.transpose() * &r_inv_y;
            for j in 0..ztrinvy.nrows() {
                rhs[rhs_offset + j] = ztrinvy[j];
            }
            rhs_offset += ztrinvy.nrows();
        }

        Self {
            coeff_matrix: c,
            rhs,
            n_fixed: p,
            n_random: q_vec,
            dim,
        }
    }
}

/// Convert sparse matrix to dense nalgebra matrix.
fn sparse_to_dense(s: &CsMat<f64>, n: usize) -> nalgebra::DMatrix<f64> {
    let mut d = nalgebra::DMatrix::zeros(n, n);
    for (&val, (i, j)) in s.iter() {
        d[(i, j)] = val;
    }
    d
}

/// Convert sparse design matrix columns to dense.
fn sparse_cols_to_dense(s: &CsMat<f64>, n: usize) -> nalgebra::DMatrix<f64> {
    let p = s.cols();
    let mut d = nalgebra::DMatrix::zeros(n, p);
    for (&val, (i, j)) in s.iter() {
        d[(i, j)] = val;
    }
    d
}

/// Compute X'Z scaled by a scalar.
fn compute_xtz_scaled(
    x: &CsMat<f64>,
    z: &CsMat<f64>,
    scale: f64,
    _n: usize,
) -> nalgebra::DMatrix<f64> {
    let p = x.cols();
    let q = z.cols();
    let mut result = nalgebra::DMatrix::zeros(p, q);

    let x_csc = if x.is_csc() { x.clone() } else { x.to_csc() };
    let z_csc = if z.is_csc() { z.clone() } else { z.to_csc() };

    for i in 0..p {
        if let Some(col_x) = x_csc.outer_view(i) {
            for j in 0..q {
                if let Some(col_z) = z_csc.outer_view(j) {
                    result[(i, j)] = col_x.dot(&col_z) * scale;
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::sparse_diagonal;
    use approx::assert_relative_eq;

    #[test]
    fn test_mme_simple_intercept_only() {
        // Simple model: y = mu + e
        // X = column of ones (3x1), no random effects
        // MME: X'X * b = X'y => 3*mu = sum(y)
        let mut tri = sprs::TriMat::new((3, 1));
        tri.add_triplet(0, 0, 1.0);
        tri.add_triplet(1, 0, 1.0);
        tri.add_triplet(2, 0, 1.0);
        let x = tri.to_csc();

        let y = vec![5.0, 3.0, 7.0];
        let sigma_e2 = 1.0;

        let mme = MixedModelEquations::assemble(&x, &[], &y, 1.0 / sigma_e2, &[]);

        assert_eq!(mme.dim, 1);
        assert_relative_eq!(mme.coeff_matrix[(0, 0)], 3.0, epsilon = 1e-10);
        assert_relative_eq!(mme.rhs[0], 15.0, epsilon = 1e-10);

        let sol = mme.solve().unwrap();
        assert_relative_eq!(sol.fixed_effects[0], 5.0, epsilon = 1e-10); // mean
    }

    #[test]
    fn test_mme_one_fixed_one_random() {
        // y = mu + u + e, where u ~ N(0, sigma_u^2 * I)
        // 4 observations, 2 random levels
        // X = [1; 1; 1; 1] (intercept)
        // Z = [[1,0]; [1,0]; [0,1]; [0,1]]
        let mut x_tri = sprs::TriMat::new((4, 1));
        for i in 0..4 {
            x_tri.add_triplet(i, 0, 1.0);
        }
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((4, 2));
        z_tri.add_triplet(0, 0, 1.0);
        z_tri.add_triplet(1, 0, 1.0);
        z_tri.add_triplet(2, 1, 1.0);
        z_tri.add_triplet(3, 1, 1.0);
        let z = z_tri.to_csc();

        let y = vec![10.0, 12.0, 6.0, 8.0];
        let sigma_e2 = 2.0;
        let sigma_u2 = 4.0;

        // G^{-1} = (1/sigma_u^2) * I
        let ginv = sparse_diagonal(&[1.0 / sigma_u2; 2]);

        let mme = MixedModelEquations::assemble(&x, &[z], &y, 1.0 / sigma_e2, &[ginv]);

        assert_eq!(mme.dim, 3); // 1 fixed + 2 random
        assert_eq!(mme.n_fixed, 1);
        assert_eq!(mme.n_random, vec![2]);

        let sol = mme.solve().unwrap();
        // mu should be close to the overall mean = 9
        // u1 should be positive (group 1 mean = 11 > 9)
        // u2 should be negative (group 2 mean = 7 < 9)
        assert!(sol.fixed_effects[0] > 8.0 && sol.fixed_effects[0] < 10.0);
        assert!(sol.random_effects[0][0] > 0.0); // u1 positive
        assert!(sol.random_effects[0][1] < 0.0); // u2 negative
                                                 // BLUP shrinkage: |u1| + |u2| should be less than 2 (the true difference is 2)
        assert!(sol.random_effects[0][0].abs() < 2.0);
    }

    #[test]
    fn sparse_and_dense_mme_agree() {
        // 2 fixed, one random term with 3 levels and a non-diagonal G⁻¹,
        // one more random term with 2 IID levels.
        let mut x_tri = sprs::TriMat::new((6, 2));
        for i in 0..6 {
            x_tri.add_triplet(i, 0, 1.0);
            if i % 2 == 1 {
                x_tri.add_triplet(i, 1, 1.0);
            }
        }
        let x = x_tri.to_csc();
        let mut z1_tri = sprs::TriMat::new((6, 3));
        for i in 0..6 {
            z1_tri.add_triplet(i, i % 3, 1.0);
        }
        let z1 = z1_tri.to_csc();
        let mut z2_tri = sprs::TriMat::new((6, 2));
        for i in 0..6 {
            z2_tri.add_triplet(i, i / 3, 1.0);
        }
        let z2 = z2_tri.to_csc();
        let y = vec![1.0, 2.5, 3.0, 4.2, 5.1, 6.3];
        let mut g_tri = sprs::TriMat::new((3, 3));
        g_tri.add_triplet(0, 0, 2.0);
        g_tri.add_triplet(1, 1, 2.5);
        g_tri.add_triplet(2, 2, 2.0);
        g_tri.add_triplet(0, 1, -0.5);
        g_tri.add_triplet(1, 0, -0.5);
        g_tri.add_triplet(1, 2, -0.7);
        g_tri.add_triplet(2, 1, -0.7);
        let g1 = g_tri.to_csc();
        let g2 = sparse_diagonal(&[0.8, 0.8]);

        let dense = MixedModelEquations::assemble(
            &x,
            &[z1.clone(), z2.clone()],
            &y,
            0.5,
            &[g1.clone(), g2.clone()],
        );
        let sparse = SparseMixedModelEquations::assemble(&x, &[z1, z2], &y, 0.5, &[g1, g2]);
        assert_eq!(sparse.dim, dense.dim);
        for (v, (i, j)) in sparse.coeff_matrix.iter() {
            assert_relative_eq!(*v, dense.coeff_matrix[(i, j)], epsilon = 1e-12);
        }
        for i in 0..dense.dim {
            for j in 0..dense.dim {
                if dense.coeff_matrix[(i, j)] != 0.0 {
                    assert!(sparse.coeff_matrix.get(i, j).is_some());
                }
            }
            assert_relative_eq!(sparse.rhs[i], dense.rhs[i], epsilon = 1e-12);
        }

        let ds = dense.solve().unwrap();
        let ss = sparse.solve().unwrap();
        for i in 0..dense.dim {
            assert_relative_eq!(ss.solution[i], ds.solution[i], epsilon = 1e-10);
            assert_relative_eq!(ss.c_inv_diag[i], ds.c_inv_diag[i], epsilon = 1e-10);
        }
        assert_relative_eq!(ss.log_det_c, ds.log_det_c, epsilon = 1e-10);

        let di = ds.inverse().unwrap();
        let si = ss.inverse().unwrap();
        let full = si.to_dense().unwrap();
        for i in 0..dense.dim {
            for j in 0..dense.dim {
                assert_relative_eq!(full[(i, j)], di.entry(i, j), epsilon = 1e-10);
            }
        }
        // entries on the pattern of C and the whole fixed block/columns
        for (_, (i, j)) in sparse.coeff_matrix.iter() {
            assert_relative_eq!(si.entry(i, j), di.entry(i, j), epsilon = 1e-10);
        }
        let fb = si.fixed_block(2);
        let fc = si.fixed_columns(2);
        for i in 0..dense.dim {
            for j in 0..2 {
                assert_relative_eq!(fc[(i, j)], di.entry(i, j), epsilon = 1e-10);
                if i < 2 {
                    assert_relative_eq!(fb[(i, j)], di.entry(i, j), epsilon = 1e-10);
                }
            }
        }
        let v: Vec<f64> = (0..dense.dim).map(|i| i as f64 * 0.3 - 1.0).collect();
        let a = si.apply(&v).unwrap();
        let b = di.apply(&v).unwrap();
        for i in 0..dense.dim {
            assert_relative_eq!(a[i], b[i], epsilon = 1e-10);
        }
        // tr(G⁻¹ C^{kk}) through the block helper
        let g1b = {
            let mut t = sprs::TriMat::new((3, 3));
            t.add_triplet(0, 1, -0.5);
            t.add_triplet(1, 0, -0.5);
            t.add_triplet(1, 1, 2.5);
            t.to_csc()
        };
        assert_relative_eq!(
            si.trace_block(&g1b, 2),
            di.trace_block(&g1b, 2),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mme_coefficient_matrix_symmetry() {
        let mut x_tri = sprs::TriMat::new((4, 2));
        x_tri.add_triplet(0, 0, 1.0);
        x_tri.add_triplet(1, 0, 1.0);
        x_tri.add_triplet(2, 1, 1.0);
        x_tri.add_triplet(3, 1, 1.0);
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((4, 3));
        z_tri.add_triplet(0, 0, 1.0);
        z_tri.add_triplet(1, 1, 1.0);
        z_tri.add_triplet(2, 2, 1.0);
        z_tri.add_triplet(3, 0, 1.0);
        let z = z_tri.to_csc();

        let y = vec![1.0, 2.0, 3.0, 4.0];
        let ginv = sparse_diagonal(&[0.5; 3]);

        let mme = MixedModelEquations::assemble(&x, &[z], &y, 1.0, &[ginv]);

        // Check symmetry
        for i in 0..mme.dim {
            for j in 0..mme.dim {
                assert_relative_eq!(
                    mme.coeff_matrix[(i, j)],
                    mme.coeff_matrix[(j, i)],
                    epsilon = 1e-10
                );
            }
        }
    }
}
