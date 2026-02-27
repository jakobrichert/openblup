use nalgebra::DMatrix;
use sprs::TriMat;

use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use super::pedigree::Pedigree;

/// Compute the full numerator relationship matrix (A) from a pedigree using
/// the tabular method.
///
/// The recurrence is:
///
/// ```text
/// A[i, i] = 1 + 0.5 * A[sire_i, dam_i]     (= 1 + F_i)
/// A[i, j] = 0.5 * (A[j, sire_i] + A[j, dam_i])   for j < i
/// A[j, i] = A[i, j]                                (symmetry)
/// ```
///
/// Unknown parents contribute 0 to the relationship sum.
///
/// The pedigree **must** be topologically sorted (parents before offspring)
/// before calling this function.
///
/// # Returns
///
/// A dense symmetric matrix of dimension `n x n`.
///
/// # Errors
///
/// Returns an error if the pedigree is not sorted.
pub fn compute_a_matrix(ped: &Pedigree) -> Result<DMatrix<f64>> {
    if !ped.is_sorted() && ped.n_animals() > 0 {
        return Err(LmmError::Pedigree(
            "Pedigree must be topologically sorted before computing A matrix. \
             Call sort_pedigree() first."
                .to_string(),
        ));
    }

    let n = ped.n_animals();
    let mut a = DMatrix::zeros(n, n);

    for i in 0..n {
        let sire = ped.sire(i);
        let dam = ped.dam(i);

        // Diagonal: A[i,i] = 1 + 0.5 * A[sire, dam].
        let f_i = match (sire, dam) {
            (Some(s), Some(d)) => 0.5 * a[(s, d)],
            _ => 0.0,
        };
        a[(i, i)] = 1.0 + f_i;

        // Off-diagonals for j < i.
        for j in 0..i {
            let val = match (sire, dam) {
                (Some(s), Some(d)) => 0.5 * (a[(j, s)] + a[(j, d)]),
                (Some(s), None) => 0.5 * a[(j, s)],
                (None, Some(d)) => 0.5 * a[(j, d)],
                (None, None) => 0.0,
            };
            a[(i, j)] = val;
            a[(j, i)] = val;
        }
    }

    Ok(a)
}

/// Extract the A22 submatrix from the full A matrix for a subset of animals
/// identified by their pedigree indices.
///
/// # Arguments
///
/// * `a` - The full numerator relationship matrix (n_total x n_total).
/// * `genotyped_indices` - Sorted indices into the pedigree of the genotyped
///   animals.
///
/// # Returns
///
/// A dense matrix of dimension `n_genotyped x n_genotyped`.
pub fn extract_a22(a: &DMatrix<f64>, genotyped_indices: &[usize]) -> DMatrix<f64> {
    let ng = genotyped_indices.len();
    let mut a22 = DMatrix::zeros(ng, ng);
    for (gi, &pi) in genotyped_indices.iter().enumerate() {
        for (gj, &pj) in genotyped_indices.iter().enumerate() {
            a22[(gi, gj)] = a[(pi, pj)];
        }
    }
    a22
}

/// Compute the single-step H-inverse matrix (Legarra et al. 2009, Aguilar et
/// al. 2010).
///
/// ```text
/// H^{-1} = A^{-1} + [ 0            0            ]
///                    [ 0   G^{-1} - A22^{-1}     ]
/// ```
///
/// The genomic adjustment `G^{-1} - A22^{-1}` is added to the rows and columns
/// of H^{-1} corresponding to genotyped animals.
///
/// # Arguments
///
/// * `ped` - The pedigree (must be sorted).
/// * `a_inv` - The full pedigree-based A-inverse (sparse, n_total x n_total).
/// * `g_inv` - The genomic relationship inverse (dense, n_genotyped x n_genotyped).
/// * `genotyped_ids` - Slice of animal ID strings identifying genotyped animals.
///   These must exist in the pedigree.
///
/// # Returns
///
/// A sparse H-inverse matrix in CSC format (n_total x n_total).
///
/// # Errors
///
/// * `LmmError::Pedigree` if any genotyped ID is not found in the pedigree.
/// * `LmmError::Pedigree` if the pedigree is not sorted.
/// * `LmmError::NotPositiveDefinite` if A22 cannot be inverted.
/// * `LmmError::DimensionMismatch` if G-inverse dimensions do not match the
///   number of genotyped animals.
pub fn compute_h_inverse(
    ped: &Pedigree,
    a_inv: &SparseMat,
    g_inv: &DMatrix<f64>,
    genotyped_ids: &[&str],
) -> Result<SparseMat> {
    if !ped.is_sorted() && ped.n_animals() > 0 {
        return Err(LmmError::Pedigree(
            "Pedigree must be topologically sorted before computing H-inverse. \
             Call sort_pedigree() first."
                .to_string(),
        ));
    }

    let n_total = ped.n_animals();
    let n_geno = genotyped_ids.len();

    if g_inv.nrows() != n_geno || g_inv.ncols() != n_geno {
        return Err(LmmError::DimensionMismatch {
            expected: n_geno,
            got: g_inv.nrows(),
            context: "G-inverse dimensions vs number of genotyped animals".into(),
        });
    }

    // Map genotyped IDs to pedigree indices.
    let genotyped_indices: Vec<usize> = genotyped_ids
        .iter()
        .map(|id| {
            ped.animal_index(id).ok_or_else(|| {
                LmmError::Pedigree(format!(
                    "Genotyped animal '{}' not found in pedigree",
                    id
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Compute A matrix and extract A22.
    let a_full = compute_a_matrix(ped)?;
    let a22 = extract_a22(&a_full, &genotyped_indices);

    // Invert A22.
    let a22_inv = a22
        .clone()
        .cholesky()
        .ok_or(LmmError::NotPositiveDefinite)?
        .inverse();

    // Compute the genomic adjustment: delta = G^{-1} - A22^{-1}.
    let delta = g_inv - &a22_inv;

    // Start with A-inverse in triplet form, then add delta entries.
    let mut tri = TriMat::new((n_total, n_total));

    // Copy A-inverse entries into the triplet matrix.
    for (val, (row, col)) in a_inv.iter() {
        tri.add_triplet(row, col, *val);
    }

    // Add delta to the genotyped animal block.
    for (gi, &pi) in genotyped_indices.iter().enumerate() {
        for (gj, &pj) in genotyped_indices.iter().enumerate() {
            let d = delta[(gi, gj)];
            if d.abs() > 1e-15 {
                tri.add_triplet(pi, pj, d);
            }
        }
    }

    Ok(tri.to_csc())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetics::ainverse::compute_a_inverse;

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

    /// Build a simple 3-animal sorted pedigree:
    ///   1: founder, 2: founder, 3: sire=1, dam=2
    fn simple_3_animal_pedigree() -> Pedigree {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            (
                "3".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();
        ped
    }

    /// Build the Mrode 5-animal pedigree:
    ///   1: founder, 2: founder, 3: sire=1, 4: sire=1 dam=2, 5: sire=3 dam=2
    fn mrode_5_animal_pedigree() -> Pedigree {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), None),
            (
                "4".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "5".to_string(),
                Some("3".to_string()),
                Some("2".to_string()),
            ),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();
        ped
    }

    // -----------------------------------------------------------------------
    // A-matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_a_matrix_3_animals() {
        let ped = simple_3_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();

        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);

        // Expected A for (1: founder, 2: founder, 3: child of 1 and 2):
        //   [1.0  0.0  0.5]
        //   [0.0  1.0  0.5]
        //   [0.5  0.5  1.0]
        assert_approx(a[(0, 0)], 1.0, 1e-12, "A[0,0]");
        assert_approx(a[(1, 1)], 1.0, 1e-12, "A[1,1]");
        assert_approx(a[(2, 2)], 1.0, 1e-12, "A[2,2]");
        assert_approx(a[(0, 1)], 0.0, 1e-12, "A[0,1]");
        assert_approx(a[(0, 2)], 0.5, 1e-12, "A[0,2]");
        assert_approx(a[(1, 2)], 0.5, 1e-12, "A[1,2]");

        // Symmetry.
        assert_approx(a[(2, 0)], 0.5, 1e-12, "A[2,0]");
        assert_approx(a[(2, 1)], 0.5, 1e-12, "A[2,1]");
    }

    #[test]
    fn test_a_matrix_5_animals() {
        let ped = mrode_5_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();

        assert_eq!(a.nrows(), 5);

        // Known A matrix (tabular method):
        //
        //   Pedigree: 1(founder), 2(founder), 3(s=1, d=?), 4(s=1, d=2), 5(s=3, d=2)
        //
        //         1     2     3     4     5
        //   1  [1.000 0.000 0.500 0.500 0.250]
        //   2  [0.000 1.000 0.000 0.500 0.500]
        //   3  [0.500 0.000 1.000 0.250 0.500]
        //   4  [0.500 0.500 0.250 1.000 0.375]
        //   5  [0.250 0.500 0.500 0.375 1.000]

        // Animal 3: sire=1, dam=unknown.
        //   A[3,3] = 1.0 (no inbreeding since dam unknown).
        //   A[0,2] = 0.5 * A[0,0] = 0.5  (sire only)
        //   A[1,2] = 0.5 * A[1,0] = 0.0
        assert_approx(a[(0, 0)], 1.0, 1e-12, "A[0,0]");
        assert_approx(a[(1, 1)], 1.0, 1e-12, "A[1,1]");
        assert_approx(a[(2, 2)], 1.0, 1e-12, "A[2,2]");
        assert_approx(a[(3, 3)], 1.0, 1e-12, "A[3,3]");
        assert_approx(a[(4, 4)], 1.0, 1e-12, "A[4,4]");

        assert_approx(a[(0, 1)], 0.0, 1e-12, "A[0,1]");
        assert_approx(a[(0, 2)], 0.5, 1e-12, "A[0,2]");
        assert_approx(a[(0, 3)], 0.5, 1e-12, "A[0,3]");
        assert_approx(a[(0, 4)], 0.25, 1e-12, "A[0,4]");

        assert_approx(a[(1, 2)], 0.0, 1e-12, "A[1,2]");
        assert_approx(a[(1, 3)], 0.5, 1e-12, "A[1,3]");
        assert_approx(a[(1, 4)], 0.5, 1e-12, "A[1,4]");

        assert_approx(a[(2, 3)], 0.25, 1e-12, "A[2,3]");
        assert_approx(a[(2, 4)], 0.5, 1e-12, "A[2,4]");

        assert_approx(a[(3, 4)], 0.375, 1e-12, "A[3,4]");

        // Symmetry check.
        for i in 0..5 {
            for j in 0..5 {
                assert_approx(
                    a[(i, j)],
                    a[(j, i)],
                    1e-14,
                    &format!("Symmetry A[{},{}]", i, j),
                );
            }
        }
    }

    #[test]
    fn test_a_matrix_unsorted_errors() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), Some("1".to_string()), None),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        let result = compute_a_matrix(&ped);
        assert!(result.is_err());
    }

    #[test]
    fn test_a_matrix_inverse_consistency() {
        // A * A^{-1} should be approximately I.
        let ped = simple_3_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();
        let a_inv_sparse = compute_a_inverse(&ped).unwrap();

        // Convert sparse A_inv to dense.
        let n = a.nrows();
        let mut a_inv = DMatrix::zeros(n, n);
        for (val, (row, col)) in a_inv_sparse.iter() {
            a_inv[(row, col)] += *val;
        }

        let product = &a * &a_inv;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_approx(
                    product[(i, j)],
                    expected,
                    1e-10,
                    &format!("(A * A_inv)[{},{}]", i, j),
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // A22 extraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_a22_all_animals() {
        let ped = simple_3_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();
        let a22 = extract_a22(&a, &[0, 1, 2]);

        // Should be identical to A.
        for i in 0..3 {
            for j in 0..3 {
                assert_approx(
                    a22[(i, j)],
                    a[(i, j)],
                    1e-14,
                    &format!("A22[{},{}]", i, j),
                );
            }
        }
    }

    #[test]
    fn test_extract_a22_subset() {
        let ped = mrode_5_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();

        // Extract A22 for genotyped animals 4 and 5 (indices 3, 4).
        let a22 = extract_a22(&a, &[3, 4]);

        assert_eq!(a22.nrows(), 2);
        assert_eq!(a22.ncols(), 2);

        // A22 should be:
        //   [A[3,3]  A[3,4]] = [1.000  0.375]
        //   [A[4,3]  A[4,4]]   [0.375  1.000]
        assert_approx(a22[(0, 0)], 1.0, 1e-12, "A22[0,0]");
        assert_approx(a22[(0, 1)], 0.375, 1e-12, "A22[0,1]");
        assert_approx(a22[(1, 0)], 0.375, 1e-12, "A22[1,0]");
        assert_approx(a22[(1, 1)], 1.0, 1e-12, "A22[1,1]");
    }

    #[test]
    fn test_extract_a22_single_animal() {
        let ped = simple_3_animal_pedigree();
        let a = compute_a_matrix(&ped).unwrap();

        let a22 = extract_a22(&a, &[2]); // just animal 3
        assert_eq!(a22.nrows(), 1);
        assert_approx(a22[(0, 0)], 1.0, 1e-12, "A22 single");
    }

    // -----------------------------------------------------------------------
    // H-inverse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_h_inverse_simple() {
        // Simple 3-animal pedigree: 1, 2 founders; 3 = child of 1 and 2.
        // Suppose animal 3 is genotyped.
        let ped = simple_3_animal_pedigree();
        let a_inv = compute_a_inverse(&ped).unwrap();

        // For a single genotyped animal, G_inv is 1x1.
        // Just use an identity-like G_inv for the test.
        let g_inv = DMatrix::from_element(1, 1, 1.5);

        let h_inv = compute_h_inverse(&ped, &a_inv, &g_inv, &["3"]).unwrap();

        assert_eq!(h_inv.rows(), 3);
        assert_eq!(h_inv.cols(), 3);

        // Convert to dense for checking.
        let n = 3;
        let mut h_dense = DMatrix::zeros(n, n);
        for (val, (row, col)) in h_inv.iter() {
            h_dense[(row, col)] += *val;
        }

        // Compute expected values:
        // A22 for animal 3 (index 2) is 1x1: [[1.0]]
        // A22_inv = [[1.0]]
        // delta = G_inv - A22_inv = [[1.5 - 1.0]] = [[0.5]]
        //
        // H_inv = A_inv + delta at (2,2).
        // A_inv[2,2] = 2.0 from Henderson's rules.
        // H_inv[2,2] = 2.0 + 0.5 = 2.5.
        // Other entries unchanged from A_inv.

        // Convert sparse A_inv to dense.
        let mut a_inv_dense = DMatrix::zeros(n, n);
        for (val, (row, col)) in a_inv.iter() {
            a_inv_dense[(row, col)] += *val;
        }

        // Check that non-genotyped entries are unchanged.
        assert_approx(h_dense[(0, 0)], a_inv_dense[(0, 0)], 1e-10, "H[0,0]");
        assert_approx(h_dense[(0, 1)], a_inv_dense[(0, 1)], 1e-10, "H[0,1]");
        assert_approx(h_dense[(1, 1)], a_inv_dense[(1, 1)], 1e-10, "H[1,1]");
        assert_approx(h_dense[(0, 2)], a_inv_dense[(0, 2)], 1e-10, "H[0,2]");
        assert_approx(h_dense[(1, 2)], a_inv_dense[(1, 2)], 1e-10, "H[1,2]");

        // Check the genotyped animal entry.
        assert_approx(h_dense[(2, 2)], 2.5, 1e-10, "H[2,2] = A_inv[2,2] + delta");
    }

    #[test]
    fn test_h_inverse_two_genotyped() {
        // 5-animal pedigree, animals 4 and 5 are genotyped.
        let ped = mrode_5_animal_pedigree();
        let a_inv = compute_a_inverse(&ped).unwrap();

        // Fabricate a G_inv for 2 genotyped animals.
        // Use a known SPD matrix.
        let g_inv = DMatrix::from_row_slice(2, 2, &[2.5, -0.5, -0.5, 2.5]);

        let h_inv = compute_h_inverse(&ped, &a_inv, &g_inv, &["4", "5"]).unwrap();

        assert_eq!(h_inv.rows(), 5);
        assert_eq!(h_inv.cols(), 5);

        // Convert to dense.
        let n = 5;
        let mut h_dense = DMatrix::zeros(n, n);
        for (val, (row, col)) in h_inv.iter() {
            h_dense[(row, col)] += *val;
        }

        // Compute expected delta:
        // A for 5-animal pedigree at indices (3,4):
        //   A22 = [A[3,3] A[3,4]; A[4,3] A[4,4]] = [1.0 0.375; 0.375 1.0]
        // A22_inv via Cholesky or formula for 2x2:
        //   det(A22) = 1.0*1.0 - 0.375^2 = 1 - 0.140625 = 0.859375
        //   A22_inv = [1.0/det  -0.375/det; -0.375/det  1.0/det]
        //           = [1.163636...  -0.436363...; -0.436363...  1.163636...]
        //
        // delta = G_inv - A22_inv
        //
        // H_inv entries at (3,3), (3,4), (4,3), (4,4) get delta added.
        // All other entries should equal A_inv.

        let a = compute_a_matrix(&ped).unwrap();
        let a22 = extract_a22(&a, &[3, 4]);
        let a22_inv = a22.clone().cholesky().unwrap().inverse();
        let delta = &g_inv - &a22_inv;

        let mut a_inv_dense = DMatrix::zeros(n, n);
        for (val, (row, col)) in a_inv.iter() {
            a_inv_dense[(row, col)] += *val;
        }

        // Non-genotyped block should be unchanged.
        for i in 0..3 {
            for j in 0..3 {
                assert_approx(
                    h_dense[(i, j)],
                    a_inv_dense[(i, j)],
                    1e-10,
                    &format!("H[{},{}] non-geno", i, j),
                );
            }
        }

        // Genotyped block should have delta added.
        let geno_idx = [3, 4];
        for (gi, &pi) in geno_idx.iter().enumerate() {
            for (gj, &pj) in geno_idx.iter().enumerate() {
                let expected = a_inv_dense[(pi, pj)] + delta[(gi, gj)];
                assert_approx(
                    h_dense[(pi, pj)],
                    expected,
                    1e-10,
                    &format!("H[{},{}] geno block", pi, pj),
                );
            }
        }
    }

    #[test]
    fn test_h_inverse_unknown_genotyped_id() {
        let ped = simple_3_animal_pedigree();
        let a_inv = compute_a_inverse(&ped).unwrap();
        let g_inv = DMatrix::from_element(1, 1, 1.0);

        let result = compute_h_inverse(&ped, &a_inv, &g_inv, &["999"]);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not found"), "Error was: {}", msg);
    }

    #[test]
    fn test_h_inverse_dimension_mismatch() {
        let ped = simple_3_animal_pedigree();
        let a_inv = compute_a_inverse(&ped).unwrap();
        // G_inv is 2x2 but only 1 genotyped animal.
        let g_inv = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        let result = compute_h_inverse(&ped, &a_inv, &g_inv, &["3"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_h_inverse_symmetry() {
        let ped = mrode_5_animal_pedigree();
        let a_inv = compute_a_inverse(&ped).unwrap();
        let g_inv = DMatrix::from_row_slice(2, 2, &[2.0, -0.3, -0.3, 2.0]);

        let h_inv = compute_h_inverse(&ped, &a_inv, &g_inv, &["4", "5"]).unwrap();

        let n = 5;
        let mut h_dense = DMatrix::zeros(n, n);
        for (val, (row, col)) in h_inv.iter() {
            h_dense[(row, col)] += *val;
        }

        for i in 0..n {
            for j in 0..n {
                assert_approx(
                    h_dense[(i, j)],
                    h_dense[(j, i)],
                    1e-10,
                    &format!("H symmetry [{},{}]", i, j),
                );
            }
        }
    }
}
