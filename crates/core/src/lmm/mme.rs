use sprs::{CsMat, TriMat};

use crate::error::Result;
use crate::matrix::sparse::{spmv, xt_y};

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
        // X'X scaled by r_inv_scale
        for (val, (row, col)) in x.iter() {
            for (val2, (row2, col2)) in x.iter() {
                if row == row2 {
                    c[(col, col2)] += val * val2 * r_inv_scale;
                }
            }
        }

        // More efficient: compute column by column
        // Actually, let's use a more efficient approach
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

        // Get C^{-1} for standard errors (only the diagonal for now)
        let c_inv = chol.inverse();
        let c_inv_diag: Vec<f64> = (0..self.dim).map(|i| c_inv[(i, i)]).collect();

        Ok(MmeSolution {
            solution: sol_vec,
            fixed_effects,
            random_effects,
            log_det_c,
            c_inv_diag,
            c_inv: Some(c_inv),
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
    /// Full C^{-1} (stored for AI-REML computations; None if not computed).
    pub c_inv: Option<nalgebra::DMatrix<f64>>,
}

/// Compute X'X scaled by a scalar, efficiently via column iteration.
fn compute_xtx_scaled(x: &CsMat<f64>, scale: f64, _n: usize) -> nalgebra::DMatrix<f64> {
    let p = x.cols();
    let mut result = nalgebra::DMatrix::zeros(p, p);

    // Ensure CSC format
    let x_csc = if x.is_csc() {
        x.clone()
    } else {
        x.to_csc()
    };

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
    use approx::assert_relative_eq;
    use crate::matrix::sparse::{sparse_identity, sparse_diagonal};

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
        let ginv = sparse_diagonal(&vec![1.0 / sigma_u2; 2]);

        let mme = MixedModelEquations::assemble(
            &x,
            &[z],
            &y,
            1.0 / sigma_e2,
            &[ginv],
        );

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
        let ginv = sparse_diagonal(&vec![0.5; 3]);

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
