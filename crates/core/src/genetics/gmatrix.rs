use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::error::{LmmError, Result};

/// Threshold (number of individuals) above which parallel computation is used
/// for the ZZ' matrix product.
const PARALLEL_THRESHOLD: usize = 50;

/// Compute the genomic relationship matrix (G) using VanRaden Method 1 (2008).
///
/// ```text
/// G = ZZ' / (2 * sum_j p_j * (1 - p_j))
/// ```
///
/// where Z is the centered marker matrix: `Z[i,j] = M[i,j] - 2*p_j`, M is
/// the raw marker matrix coded as 0/1/2 (number of copies of the reference
/// allele), and `p_j` is the allele frequency for marker j.
///
/// # Arguments
///
/// * `markers` - An `n_individuals x n_markers` matrix with 0/1/2 coding.
/// * `allele_freqs` - Optional slice of allele frequencies (length `n_markers`).
///   If `None`, frequencies are estimated from the marker data as column means
///   divided by 2.
///
/// # Returns
///
/// A dense symmetric G-matrix of dimension `n_individuals x n_individuals`.
///
/// # Errors
///
/// * `LmmError::DimensionMismatch` if the provided allele frequency vector
///   length does not match the number of markers.
/// * `LmmError::Data` if the scaling factor is effectively zero (all markers
///   are monomorphic).
pub fn compute_g_matrix(
    markers: &DMatrix<f64>,
    allele_freqs: Option<&[f64]>,
) -> Result<DMatrix<f64>> {
    let n = markers.nrows();
    let m = markers.ncols();

    if n == 0 || m == 0 {
        return Err(LmmError::Data(
            "Marker matrix must have at least one individual and one marker".into(),
        ));
    }

    // Compute or validate allele frequencies.
    let freqs: Vec<f64> = match allele_freqs {
        Some(f) => {
            if f.len() != m {
                return Err(LmmError::DimensionMismatch {
                    expected: m,
                    got: f.len(),
                    context: "allele frequency vector length vs number of markers".into(),
                });
            }
            f.to_vec()
        }
        None => (0..m)
            .map(|j| markers.column(j).sum() / (2.0 * n as f64))
            .collect(),
    };

    // Center markers: Z = M - 2p (subtract 2*p_j from each column j).
    let mut z = markers.clone();
    for j in 0..m {
        let twop = 2.0 * freqs[j];
        for i in 0..n {
            z[(i, j)] -= twop;
        }
    }

    // Scaling factor: 2 * sum_j p_j * (1 - p_j).
    let scale: f64 = 2.0 * freqs.iter().map(|&p| p * (1.0 - p)).sum::<f64>();
    if scale < 1e-10 {
        return Err(LmmError::Data("All markers are monomorphic".into()));
    }

    // G = ZZ' / scale.
    // For large matrices, use rayon to parallelise the row-wise computation.
    let g = if n >= PARALLEL_THRESHOLD {
        compute_zzt_parallel(&z, scale)
    } else {
        (&z * z.transpose()) / scale
    };

    Ok(g)
}

/// Compute ZZ'/scale using rayon parallelism over rows.
fn compute_zzt_parallel(z: &DMatrix<f64>, scale: f64) -> DMatrix<f64> {
    let n = z.nrows();
    let m = z.ncols();

    // Collect rows as slices for efficient parallel access.
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; m];
            for j in 0..m {
                row[j] = z[(i, j)];
            }
            row
        })
        .collect();

    // Compute upper triangle (including diagonal) in parallel.
    let upper: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let rows_ref = &rows;
            (i..n)
                .map(move |j| {
                    let dot: f64 = rows_ref[i]
                        .iter()
                        .zip(rows_ref[j].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    (i, j, dot / scale)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut g = DMatrix::zeros(n, n);
    for (i, j, val) in upper {
        g[(i, j)] = val;
        if i != j {
            g[(j, i)] = val;
        }
    }
    g
}

/// Blend the genomic relationship matrix G with the pedigree-based A22 matrix
/// to improve numerical conditioning and ensure positive definiteness.
///
/// ```text
/// G_blend = (1 - w) * G + w * A22
/// ```
///
/// A typical blending weight is `w = 0.05` (Misztal et al. 2010).
///
/// # Arguments
///
/// * `g` - The genomic relationship matrix (n_genotyped x n_genotyped).
/// * `a22` - The pedigree relationship matrix for genotyped animals only
///   (n_genotyped x n_genotyped).
/// * `weight` - The blending weight `w` (must be in [0, 1]).
///
/// # Errors
///
/// * `LmmError::DimensionMismatch` if G and A22 have different dimensions.
/// * `LmmError::InvalidParameter` if weight is outside [0, 1].
pub fn blend_g_matrix(
    g: &DMatrix<f64>,
    a22: &DMatrix<f64>,
    weight: f64,
) -> Result<DMatrix<f64>> {
    if g.nrows() != a22.nrows() || g.ncols() != a22.ncols() {
        return Err(LmmError::DimensionMismatch {
            expected: g.nrows(),
            got: a22.nrows(),
            context: "G and A22 dimensions must match for blending".into(),
        });
    }
    if !(0.0..=1.0).contains(&weight) {
        return Err(LmmError::InvalidParameter(format!(
            "Blending weight must be in [0, 1], got {}",
            weight
        )));
    }

    Ok(g * (1.0 - weight) + a22 * weight)
}

/// Compute the inverse of the G-matrix using dense Cholesky decomposition.
///
/// G must be symmetric positive definite (e.g. after blending with A22).
///
/// # Errors
///
/// Returns `LmmError::NotPositiveDefinite` if the Cholesky factorisation
/// fails.
pub fn invert_g_matrix(g: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let chol = g
        .clone()
        .cholesky()
        .ok_or(LmmError::NotPositiveDefinite)?;
    Ok(chol.inverse())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert approximate equality of two f64 values.
    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{}: expected {}, got {} (diff = {})",
            msg,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    /// Build a simple 3-individual, 4-marker matrix.
    ///
    /// ```text
    ///       m1  m2  m3  m4
    /// ind1:  2   0   1   0
    /// ind2:  1   1   0   2
    /// ind3:  0   2   1   1
    /// ```
    fn simple_marker_matrix() -> DMatrix<f64> {
        DMatrix::from_row_slice(3, 4, &[
            2.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 2.0,
            0.0, 2.0, 1.0, 1.0,
        ])
    }

    #[test]
    fn test_g_matrix_computed_allele_freqs() {
        let m = simple_marker_matrix();
        let g = compute_g_matrix(&m, None).unwrap();

        assert_eq!(g.nrows(), 3);
        assert_eq!(g.ncols(), 3);

        // Verify symmetry.
        for i in 0..3 {
            for j in 0..3 {
                assert_approx(
                    g[(i, j)],
                    g[(j, i)],
                    1e-12,
                    &format!("G[{},{}] vs G[{},{}]", i, j, j, i),
                );
            }
        }
    }

    #[test]
    fn test_g_matrix_with_provided_allele_freqs() {
        let m = simple_marker_matrix();
        // Allele freqs from data: p = column_sum / (2*n)
        // m1: (2+1+0)/6 = 0.5
        // m2: (0+1+2)/6 = 0.5
        // m3: (1+0+1)/6 = 1/3
        // m4: (0+2+1)/6 = 0.5
        let freqs = vec![0.5, 0.5, 1.0 / 3.0, 0.5];

        let g_auto = compute_g_matrix(&m, None).unwrap();
        let g_manual = compute_g_matrix(&m, Some(&freqs)).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_approx(
                    g_auto[(i, j)],
                    g_manual[(i, j)],
                    1e-12,
                    &format!("G_auto[{},{}] vs G_manual[{},{}]", i, j, i, j),
                );
            }
        }
    }

    #[test]
    fn test_g_matrix_known_values() {
        // Hand-compute G for the simple marker matrix.
        //
        // p = [0.5, 0.5, 1/3, 0.5]
        // 2p = [1.0, 1.0, 2/3, 1.0]
        //
        // Z = M - 2p:
        //   [ 1.0  -1.0   1/3  -1.0 ]
        //   [ 0.0   0.0  -2/3   1.0 ]
        //   [-1.0   1.0   1/3   0.0 ]
        //
        // scale = 2 * sum(p*(1-p))
        //       = 2 * (0.25 + 0.25 + 2/9 + 0.25)
        //       = 2 * (0.75 + 2/9)
        //       = 2 * (6.75/9 + 2/9)
        //       = 2 * (8.75/9)
        //       = 35/18
        //
        // ZZ':
        //   Row 0 . Row 0 = 1 + 1 + 1/9 + 1 = 3 + 1/9 = 28/9
        //   Row 0 . Row 1 = 0 + 0 - 2/9 - 1 = -11/9
        //   Row 0 . Row 2 = -1 - 1 + 1/9 + 0 = -2 + 1/9 = -17/9
        //   Row 1 . Row 1 = 0 + 0 + 4/9 + 1 = 13/9
        //   Row 1 . Row 2 = 0 + 0 - 2/9 + 0 = -2/9
        //   Row 2 . Row 2 = 1 + 1 + 1/9 + 0 = 2 + 1/9 = 19/9
        //
        // G = ZZ' / (35/18) = ZZ' * 18/35

        let m = simple_marker_matrix();
        let g = compute_g_matrix(&m, None).unwrap();

        let s = 18.0 / 35.0;
        assert_approx(g[(0, 0)], 28.0 / 9.0 * s, 1e-10, "G[0,0]");
        assert_approx(g[(0, 1)], -11.0 / 9.0 * s, 1e-10, "G[0,1]");
        assert_approx(g[(0, 2)], -17.0 / 9.0 * s, 1e-10, "G[0,2]");
        assert_approx(g[(1, 1)], 13.0 / 9.0 * s, 1e-10, "G[1,1]");
        assert_approx(g[(1, 2)], -2.0 / 9.0 * s, 1e-10, "G[1,2]");
        assert_approx(g[(2, 2)], 19.0 / 9.0 * s, 1e-10, "G[2,2]");
    }

    #[test]
    fn test_g_matrix_diagonal_approx_1_for_non_inbred() {
        // For a larger random-ish marker matrix where individuals are
        // relatively unrelated, diagonal of G should be approximately 1.
        // We use a structured matrix to keep the test deterministic.
        //
        // With a diverse set of markers and estimated allele freqs,
        // the average diagonal element of G should be close to 1.
        let m = DMatrix::from_row_slice(4, 6, &[
            2.0, 0.0, 1.0, 0.0, 1.0, 2.0,
            0.0, 2.0, 1.0, 2.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 2.0, 1.0,
            1.0, 1.0, 2.0, 1.0, 1.0, 1.0,
        ]);
        let g = compute_g_matrix(&m, None).unwrap();

        // Average diagonal should be near 1.0 (within reason for 4 individuals).
        let avg_diag: f64 = (0..4).map(|i| g[(i, i)]).sum::<f64>() / 4.0;
        assert!(
            (avg_diag - 1.0).abs() < 0.5,
            "Average diagonal {} should be near 1.0",
            avg_diag
        );
    }

    #[test]
    fn test_g_matrix_is_symmetric() {
        let m = simple_marker_matrix();
        let g = compute_g_matrix(&m, None).unwrap();

        for i in 0..g.nrows() {
            for j in 0..g.ncols() {
                assert_approx(
                    g[(i, j)],
                    g[(j, i)],
                    1e-14,
                    &format!("Symmetry G[{},{}]", i, j),
                );
            }
        }
    }

    #[test]
    fn test_monomorphic_markers_error() {
        // All markers fixed at 0 (p=0) or fixed at 2 (p=1) -> monomorphic.
        // p*(1-p) = 0 for each marker, so scaling factor is zero.
        let m = DMatrix::from_row_slice(3, 2, &[
            0.0, 2.0,
            0.0, 2.0,
            0.0, 2.0,
        ]);
        let result = compute_g_matrix(&m, None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("monomorphic"),
            "Expected monomorphic error, got: {}",
            msg
        );
    }

    #[test]
    fn test_allele_freq_dimension_mismatch() {
        let m = simple_marker_matrix();
        let bad_freqs = vec![0.5, 0.5]; // only 2 instead of 4
        let result = compute_g_matrix(&m, Some(&bad_freqs));
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_g_matrix_basic() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.1]);
        let a22 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        let blended = blend_g_matrix(&g, &a22, 0.05).unwrap();

        // G_blend = 0.95*G + 0.05*A22
        assert_approx(blended[(0, 0)], 0.95 * 1.0 + 0.05 * 1.0, 1e-12, "blend[0,0]");
        assert_approx(blended[(0, 1)], 0.95 * 0.3 + 0.05 * 0.0, 1e-12, "blend[0,1]");
        assert_approx(blended[(1, 0)], 0.95 * 0.3 + 0.05 * 0.0, 1e-12, "blend[1,0]");
        assert_approx(blended[(1, 1)], 0.95 * 1.1 + 0.05 * 1.0, 1e-12, "blend[1,1]");
    }

    #[test]
    fn test_blend_g_matrix_weight_zero() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.1]);
        let a22 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let blended = blend_g_matrix(&g, &a22, 0.0).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_approx(blended[(i, j)], g[(i, j)], 1e-14, "w=0");
            }
        }
    }

    #[test]
    fn test_blend_g_matrix_weight_one() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.1]);
        let a22 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let blended = blend_g_matrix(&g, &a22, 1.0).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                assert_approx(blended[(i, j)], a22[(i, j)], 1e-14, "w=1");
            }
        }
    }

    #[test]
    fn test_blend_g_matrix_dimension_mismatch() {
        let g = DMatrix::zeros(3, 3);
        let a22 = DMatrix::zeros(2, 2);
        let result = blend_g_matrix(&g, &a22, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_blend_g_matrix_invalid_weight() {
        let g = DMatrix::zeros(2, 2);
        let a22 = DMatrix::zeros(2, 2);
        assert!(blend_g_matrix(&g, &a22, -0.1).is_err());
        assert!(blend_g_matrix(&g, &a22, 1.1).is_err());
    }

    #[test]
    fn test_invert_g_matrix() {
        // Use a known SPD matrix.
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let g_inv = invert_g_matrix(&g).unwrap();

        // G * G_inv should be identity.
        let product = &g * &g_inv;
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_approx(
                    product[(i, j)],
                    expected,
                    1e-10,
                    &format!("(G*Ginv)[{},{}]", i, j),
                );
            }
        }
    }

    #[test]
    fn test_invert_g_matrix_not_positive_definite() {
        // Singular matrix should fail.
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let result = invert_g_matrix(&g);
        assert!(result.is_err());
    }
}
