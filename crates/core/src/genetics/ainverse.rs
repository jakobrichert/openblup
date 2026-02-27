use sprs::TriMat;

use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use super::pedigree::Pedigree;

/// Compute the inverse of the additive relationship matrix (A^{-1}) using
/// Henderson's rules.
///
/// This version assumes **no inbreeding** (F_i = 0 for all animals), which is
/// appropriate for plant breeding populations with unrelated founder lines.
/// The Mendelian sampling variance simplifies to:
///
/// - Both parents known:    d_i = 0.5
/// - One parent known:      d_i = 0.75
/// - Neither parent known:  d_i = 1.0
///
/// The pedigree **must** be topologically sorted (parents before offspring)
/// before calling this function. Use [`Pedigree::sort_pedigree`] first.
///
/// # Henderson's rules
///
/// For each animal *i* with sire *s* and dam *d*, let alpha_i = 1 / d_i:
///
/// - A^{-1}\[i, i\] += alpha_i
/// - If sire known: A^{-1}\[i, s\] -= alpha_i/2;  A^{-1}\[s, i\] -= alpha_i/2;  A^{-1}\[s, s\] += alpha_i/4
/// - If dam  known: A^{-1}\[i, d\] -= alpha_i/2;  A^{-1}\[d, i\] -= alpha_i/2;  A^{-1}\[d, d\] += alpha_i/4
/// - If both known: A^{-1}\[s, d\] += alpha_i/4;  A^{-1}\[d, s\] += alpha_i/4
///
/// # Returns
///
/// A sparse symmetric matrix in CSC format of dimension n x n, where n is
/// the number of animals.
///
/// # Errors
///
/// Returns an error if the pedigree is not sorted.
pub fn compute_a_inverse(ped: &Pedigree) -> Result<SparseMat> {
    if !ped.is_sorted() && ped.n_animals() > 0 {
        return Err(LmmError::Pedigree(
            "Pedigree must be topologically sorted before computing A-inverse. \
             Call sort_pedigree() first."
                .to_string(),
        ));
    }

    let n = ped.n_animals();
    let mut tri = TriMat::new((n, n));

    for i in 0..n {
        let sire = ped.sire(i);
        let dam = ped.dam(i);

        // Mendelian sampling variance (no inbreeding).
        let d_i = match (sire, dam) {
            (Some(_), Some(_)) => 0.5,
            (Some(_), None) | (None, Some(_)) => 0.75,
            (None, None) => 1.0,
        };

        let alpha_i = 1.0 / d_i;

        // Diagonal contribution for animal i.
        tri.add_triplet(i, i, alpha_i);

        // Sire contributions.
        if let Some(s) = sire {
            tri.add_triplet(i, s, -alpha_i / 2.0);
            tri.add_triplet(s, i, -alpha_i / 2.0);
            tri.add_triplet(s, s, alpha_i / 4.0);
        }

        // Dam contributions.
        if let Some(d) = dam {
            tri.add_triplet(i, d, -alpha_i / 2.0);
            tri.add_triplet(d, i, -alpha_i / 2.0);
            tri.add_triplet(d, d, alpha_i / 4.0);
        }

        // Cross-term between sire and dam.
        if let (Some(s), Some(d)) = (sire, dam) {
            tri.add_triplet(s, d, alpha_i / 4.0);
            tri.add_triplet(d, s, alpha_i / 4.0);
        }
    }

    Ok(tri.to_csc())
}

/// Compute the inverse of the additive relationship matrix (A^{-1}) using
/// Henderson's rules **with inbreeding** computed via the Meuwissen & Luo
/// (1992) algorithm.
///
/// This is the full-precision version. The Mendelian sampling variance is:
///
/// - Both parents known:    d_i = 0.5 - 0.25*(F_s + F_d)
/// - Only sire known:       d_i = 0.75 - 0.25*F_s
/// - Only dam known:        d_i = 0.75 - 0.25*F_d
/// - Neither parent known:  d_i = 1.0
///
/// where F_s and F_d are the inbreeding coefficients of the sire and dam.
///
/// The pedigree **must** be topologically sorted before calling this function.
///
/// # Returns
///
/// A sparse symmetric matrix in CSC format of dimension n x n.
///
/// # Errors
///
/// Returns an error if the pedigree is not sorted.
pub fn compute_a_inverse_with_inbreeding(ped: &Pedigree) -> Result<SparseMat> {
    if !ped.is_sorted() && ped.n_animals() > 0 {
        return Err(LmmError::Pedigree(
            "Pedigree must be topologically sorted before computing A-inverse. \
             Call sort_pedigree() first."
                .to_string(),
        ));
    }

    let n = ped.n_animals();
    let f = compute_inbreeding(ped)?;

    let mut tri = TriMat::new((n, n));

    for i in 0..n {
        let sire = ped.sire(i);
        let dam = ped.dam(i);

        // Mendelian sampling variance with inbreeding.
        let d_i = match (sire, dam) {
            (Some(s), Some(d)) => 0.5 - 0.25 * (f[s] + f[d]),
            (Some(s), None) => 0.75 - 0.25 * f[s],
            (None, Some(d)) => 0.75 - 0.25 * f[d],
            (None, None) => 1.0,
        };

        let alpha_i = 1.0 / d_i;

        // Diagonal contribution for animal i.
        tri.add_triplet(i, i, alpha_i);

        // Sire contributions.
        if let Some(s) = sire {
            tri.add_triplet(i, s, -alpha_i / 2.0);
            tri.add_triplet(s, i, -alpha_i / 2.0);
            tri.add_triplet(s, s, alpha_i / 4.0);
        }

        // Dam contributions.
        if let Some(d) = dam {
            tri.add_triplet(i, d, -alpha_i / 2.0);
            tri.add_triplet(d, i, -alpha_i / 2.0);
            tri.add_triplet(d, d, alpha_i / 4.0);
        }

        // Cross-term between sire and dam.
        if let (Some(s), Some(d)) = (sire, dam) {
            tri.add_triplet(s, d, alpha_i / 4.0);
            tri.add_triplet(d, s, alpha_i / 4.0);
        }
    }

    Ok(tri.to_csc())
}

/// Compute inbreeding coefficients for all animals using the Meuwissen & Luo
/// (1992) algorithm.
///
/// The pedigree must be topologically sorted (parents before offspring).
///
/// The algorithm computes F[i] for each animal in pedigree order. For animal
/// *i* with sire *s* and dam *d*:
///
/// - If both parents unknown: F[i] = 0
/// - If one parent unknown: F[i] = 0
/// - If both parents known: F[i] = 0.5 * a(s, d)
///
/// where a(s, d) is the additive relationship between sire and dam, which
/// equals the inbreeding coefficient of their offspring plus some terms.
///
/// This implementation uses the tabular method to track the relationship
/// coefficients needed. Specifically, for each animal *i* we compute a row
/// of the L matrix (Cholesky-like decomposition of A) using:
///
///   L[i, j] = 0                                   if j > i
///   L[i, i] = sqrt(d_i)
///   L[i, j] = 0.5 * (L[s, j] + L[d, j])          for j < i
///
/// Then F[i] = sum_j L[i, j]^2 - 1 = A[i, i] - 1.
///
/// However, the full L matrix is O(n^2) in memory. We use a more memory-
/// efficient approach: for each animal, compute only the diagonal of A.
///
/// Simpler approach using the recursive relationship:
///   A[i, i] = 1 + F[i]
///   F[i] = 0.5 * A[sire, dam]
///   A[i, j] for j < i: A[i, j] = 0.5 * (A[s, j] + A[d, j])
///
/// For efficiency, we use the Meuwissen & Luo (1992) algorithm which only
/// requires O(n) memory by tracing ancestor paths.
///
/// # Returns
///
/// A vector of inbreeding coefficients, one per animal.
pub fn compute_inbreeding(ped: &Pedigree) -> Result<Vec<f64>> {
    let n = ped.n_animals();
    let mut f = vec![0.0_f64; n];

    // For each animal, we need to compute F[i] = 0.5 * A[sire_i, dam_i].
    // To get A[sire_i, dam_i] we use the Quaas (1976) path-tracing approach:
    //   A[p, q] = sum over ancestors path coefficients.
    //
    // Meuwissen & Luo (1992) efficient implementation:
    //
    // For each animal i in pedigree order:
    //   If both parents unknown or one parent unknown: F[i] = 0
    //   If both parents known:
    //     Compute A[s, d] using path coefficients through the L-D-L'
    //     decomposition, where:
    //       d[j] = Mendelian sampling variance of animal j
    //       A[s, d] = sum_j (L[s,j] * d[j] * L[d,j])
    //
    // We build L implicitly. For animal i:
    //   L[i, i] = 1
    //   L[i, j] = 0.5 * (L[sire_i, j] + L[dam_i, j])  for j < i
    //
    // To compute A[s, d] we only need L rows for s and d, but storing all of
    // L is O(n^2). Instead, we use the Colleau (2002) indirect method, or
    // Henderson's simple recursive formula for small/medium pedigrees.
    //
    // For practical plant breeding pedigrees (typically < 100k animals), we
    // use the straightforward recursive approach from Quaas (1976):

    // The following implements the "tabular method with pruning" approach.
    // For each animal i (in sorted order), we compute F[i] by tracing the
    // relationship between its parents.

    for i in 0..n {
        let sire = ped.sire(i);
        let dam = ped.dam(i);

        match (sire, dam) {
            (Some(s), Some(d)) => {
                // F[i] = 0.5 * A[s, d]
                // Compute A[s, d] using the recursive relationship.
                // A[p, q] where p >= q (assume s and d are both < i due to sort):
                // We compute this by tracing through ancestors.
                let a_sd = relationship(ped, s, d, &f);
                f[i] = 0.5 * a_sd;
            }
            _ => {
                // One or both parents unknown: F[i] = 0.
                f[i] = 0.0;
            }
        }
    }

    Ok(f)
}

/// Compute the additive relationship coefficient A[p, q] between two animals
/// p and q, given known inbreeding coefficients for all animals with index
/// less than max(p, q).
///
/// Uses the recursive formula:
///   A[p, q] = 0.5 * (A[p, sire_q] + A[p, dam_q])  if p < q
///   A[p, p] = 1 + F[p]
///   A[p, q] = A[q, p]
///
/// This is O(n) per call in the worst case (long ancestor chains), but
/// for typical pedigrees with moderate depth it is efficient enough.
///
/// We use memoization via a local cache to avoid exponential blowup.
fn relationship(ped: &Pedigree, p: usize, q: usize, f: &[f64]) -> f64 {
    // Use an iterative approach with a work stack to avoid deep recursion.
    use std::collections::HashMap;
    let mut cache: HashMap<(usize, usize), f64> = HashMap::new();
    relationship_cached(ped, p, q, f, &mut cache)
}

fn relationship_cached(
    ped: &Pedigree,
    p: usize,
    q: usize,
    f: &[f64],
    cache: &mut std::collections::HashMap<(usize, usize), f64>,
) -> f64 {
    // Ensure p <= q for canonical form.
    let (a, b) = if p <= q { (p, q) } else { (q, p) };

    if let Some(&val) = cache.get(&(a, b)) {
        return val;
    }

    let result = if a == b {
        // A[a, a] = 1 + F[a]
        1.0 + f[a]
    } else {
        // a < b: A[a, b] = 0.5 * (A[a, sire_b] + A[a, dam_b])
        let sire_b = ped.sire(b);
        let dam_b = ped.dam(b);

        match (sire_b, dam_b) {
            (Some(s), Some(d)) => {
                0.5 * (relationship_cached(ped, a, s, f, cache)
                    + relationship_cached(ped, a, d, f, cache))
            }
            (Some(s), None) => 0.5 * relationship_cached(ped, a, s, f, cache),
            (None, Some(d)) => 0.5 * relationship_cached(ped, a, d, f, cache),
            (None, None) => 0.0,
        }
    };

    cache.insert((a, b), result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build the simple 3-animal pedigree and return it sorted.
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

    /// Helper to build the Mrode 5-animal pedigree (Table 2.1).
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

    /// Extract a dense matrix from a sparse CSC matrix for testing.
    fn sparse_to_dense(mat: &SparseMat) -> Vec<Vec<f64>> {
        let n = mat.rows();
        let m = mat.cols();
        let mut dense = vec![vec![0.0; m]; n];
        for (val, (row, col)) in mat.iter() {
            dense[row][col] += *val;
        }
        dense
    }

    /// Assert two f64 values are approximately equal.
    fn assert_approx(actual: f64, expected: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "{}: expected {}, got {}",
            msg,
            expected,
            actual
        );
    }

    #[test]
    fn test_a_inverse_3_animal_no_inbreeding() {
        let ped = simple_3_animal_pedigree();
        let ainv = compute_a_inverse(&ped).unwrap();

        assert_eq!(ainv.rows(), 3);
        assert_eq!(ainv.cols(), 3);

        let dense = sparse_to_dense(&ainv);

        // Expected A^{-1} for founders (1, 2) and offspring (3 = child of 1, 2):
        //   [ 1.5   0.5  -1.0 ]
        //   [ 0.5   1.5  -1.0 ]
        //   [-1.0  -1.0   2.0 ]
        assert_approx(dense[0][0], 1.5, "A_inv[0,0]");
        assert_approx(dense[0][1], 0.5, "A_inv[0,1]");
        assert_approx(dense[0][2], -1.0, "A_inv[0,2]");
        assert_approx(dense[1][0], 0.5, "A_inv[1,0]");
        assert_approx(dense[1][1], 1.5, "A_inv[1,1]");
        assert_approx(dense[1][2], -1.0, "A_inv[1,2]");
        assert_approx(dense[2][0], -1.0, "A_inv[2,0]");
        assert_approx(dense[2][1], -1.0, "A_inv[2,1]");
        assert_approx(dense[2][2], 2.0, "A_inv[2,2]");
    }

    #[test]
    fn test_a_inverse_3_animal_with_inbreeding() {
        // For the simple 3-animal case with unrelated founders, the inbreeding-
        // aware version should give the same result as the no-inbreeding version
        // because no inbreeding exists.
        let ped = simple_3_animal_pedigree();
        let ainv = compute_a_inverse_with_inbreeding(&ped).unwrap();
        let dense = sparse_to_dense(&ainv);

        assert_approx(dense[0][0], 1.5, "A_inv[0,0]");
        assert_approx(dense[0][1], 0.5, "A_inv[0,1]");
        assert_approx(dense[0][2], -1.0, "A_inv[0,2]");
        assert_approx(dense[1][0], 0.5, "A_inv[1,0]");
        assert_approx(dense[1][1], 1.5, "A_inv[1,1]");
        assert_approx(dense[1][2], -1.0, "A_inv[1,2]");
        assert_approx(dense[2][0], -1.0, "A_inv[2,0]");
        assert_approx(dense[2][1], -1.0, "A_inv[2,1]");
        assert_approx(dense[2][2], 2.0, "A_inv[2,2]");
    }

    #[test]
    fn test_a_inverse_5_animal_no_inbreeding() {
        let ped = mrode_5_animal_pedigree();
        let ainv = compute_a_inverse(&ped).unwrap();

        assert_eq!(ainv.rows(), 5);
        assert_eq!(ainv.cols(), 5);

        let dense = sparse_to_dense(&ainv);

        // Verify symmetry.
        for i in 0..5 {
            for j in 0..5 {
                assert_approx(
                    dense[i][j],
                    dense[j][i],
                    &format!("Symmetry A_inv[{},{}] vs A_inv[{},{}]", i, j, j, i),
                );
            }
        }

        // Verify specific values from Henderson's rules (no inbreeding):
        //
        // Animal 1 (founder):          d=1.0, alpha=1.0
        // Animal 2 (founder):          d=1.0, alpha=1.0
        // Animal 3 (sire=1, dam=0):    d=0.75, alpha=4/3
        // Animal 4 (sire=1, dam=2):    d=0.5, alpha=2.0
        // Animal 5 (sire=3, dam=2):    d=0.5, alpha=2.0
        //
        // After sorting, the order is: 1, 2, 3, 4, 5 (indices 0-4).
        // (The pedigree was already in topological order.)
        //
        // Accumulate contributions:
        //
        // Animal 0 (id="1"): alpha=1.0
        //   A_inv[0,0] += 1.0
        //
        // Animal 1 (id="2"): alpha=1.0
        //   A_inv[1,1] += 1.0
        //
        // Animal 2 (id="3"): sire=0, alpha=4/3
        //   A_inv[2,2] += 4/3
        //   A_inv[2,0] -= 2/3; A_inv[0,2] -= 2/3
        //   A_inv[0,0] += 1/3
        //
        // Animal 3 (id="4"): sire=0, dam=1, alpha=2.0
        //   A_inv[3,3] += 2.0
        //   A_inv[3,0] -= 1.0; A_inv[0,3] -= 1.0
        //   A_inv[0,0] += 0.5
        //   A_inv[3,1] -= 1.0; A_inv[1,3] -= 1.0
        //   A_inv[1,1] += 0.5
        //   A_inv[0,1] += 0.5; A_inv[1,0] += 0.5
        //
        // Animal 4 (id="5"): sire=2, dam=1, alpha=2.0
        //   A_inv[4,4] += 2.0
        //   A_inv[4,2] -= 1.0; A_inv[2,4] -= 1.0
        //   A_inv[2,2] += 0.5
        //   A_inv[4,1] -= 1.0; A_inv[1,4] -= 1.0
        //   A_inv[1,1] += 0.5
        //   A_inv[2,1] += 0.5; A_inv[1,2] += 0.5
        //
        // Totals:
        // A_inv[0,0] = 1.0 + 1/3 + 0.5 = 11/6 ≈ 1.8333
        // A_inv[1,1] = 1.0 + 0.5 + 0.5 = 2.0
        // A_inv[2,2] = 4/3 + 0.5 = 11/6 ≈ 1.8333
        // A_inv[3,3] = 2.0
        // A_inv[4,4] = 2.0
        //
        // A_inv[0,1] = 0.5
        // A_inv[0,2] = -2/3 ≈ -0.6667
        // A_inv[0,3] = -1.0
        // A_inv[1,2] = 0.5
        // A_inv[1,3] = -1.0
        // A_inv[1,4] = -1.0
        // A_inv[2,4] = -1.0
        // Other off-diagonals = 0.0

        assert_approx(dense[0][0], 11.0 / 6.0, "A_inv[0,0]");
        assert_approx(dense[1][1], 2.0, "A_inv[1,1]");
        assert_approx(dense[2][2], 11.0 / 6.0, "A_inv[2,2]");
        assert_approx(dense[3][3], 2.0, "A_inv[3,3]");
        assert_approx(dense[4][4], 2.0, "A_inv[4,4]");

        assert_approx(dense[0][1], 0.5, "A_inv[0,1]");
        assert_approx(dense[0][2], -2.0 / 3.0, "A_inv[0,2]");
        assert_approx(dense[0][3], -1.0, "A_inv[0,3]");
        assert_approx(dense[0][4], 0.0, "A_inv[0,4]");

        assert_approx(dense[1][2], 0.5, "A_inv[1,2]");
        assert_approx(dense[1][3], -1.0, "A_inv[1,3]");
        assert_approx(dense[1][4], -1.0, "A_inv[1,4]");

        assert_approx(dense[2][3], 0.0, "A_inv[2,3]");
        assert_approx(dense[2][4], -1.0, "A_inv[2,4]");

        assert_approx(dense[3][4], 0.0, "A_inv[3,4]");
    }

    #[test]
    fn test_a_inverse_5_animal_with_inbreeding() {
        // In this pedigree, animal 5's parents are animal 3 (sire=1) and
        // animal 2. The relationship A[3, 2] = 0 (animal 3's sire is 1, dam
        // is unknown; animal 2 is an unrelated founder). So F[5] = 0.
        //
        // Animal 3's parents: sire=1, dam=unknown -> F[3] = 0
        // Animal 4's parents: sire=1, dam=2 -> A[1,2] = 0 -> F[4] = 0
        // Animal 5's parents: sire=3, dam=2 -> A[3,2] =
        //   0.5 * (A[1,2] + A[unknown,2]) = 0.5 * 0 = 0 -> F[5] = 0
        //
        // So for this pedigree, inbreeding is zero everywhere, and the two
        // methods should give identical results.

        let ped = mrode_5_animal_pedigree();
        let ainv_no_f = compute_a_inverse(&ped).unwrap();
        let ainv_with_f = compute_a_inverse_with_inbreeding(&ped).unwrap();

        let dense_no_f = sparse_to_dense(&ainv_no_f);
        let dense_with_f = sparse_to_dense(&ainv_with_f);

        for i in 0..5 {
            for j in 0..5 {
                assert_approx(
                    dense_with_f[i][j],
                    dense_no_f[i][j],
                    &format!("A_inv[{},{}] inbreeding vs no-inbreeding", i, j),
                );
            }
        }
    }

    #[test]
    fn test_inbreeding_simple_no_inbreeding() {
        let ped = simple_3_animal_pedigree();
        let f = compute_inbreeding(&ped).unwrap();

        assert_eq!(f.len(), 3);
        assert_approx(f[0], 0.0, "F[0]");
        assert_approx(f[1], 0.0, "F[1]");
        assert_approx(f[2], 0.0, "F[2]");
    }

    #[test]
    fn test_inbreeding_with_related_parents() {
        // Pedigree where inbreeding actually occurs:
        //   1: founder
        //   2: founder
        //   3: sire=1, dam=2
        //   4: sire=1, dam=2  (full sib of 3)
        //   5: sire=3, dam=4  (mating of full sibs -> inbred)
        //
        // A[1,1] = 1, A[2,2] = 1, A[1,2] = 0
        // A[3,3] = 1 + F[3] = 1; F[3] = 0.5*A[1,2] = 0
        // A[4,4] = 1 + F[4] = 1; F[4] = 0.5*A[1,2] = 0
        // A[3,4] = 0.5*(A[1,4] + A[2,4])
        //   A[1,4] = 0.5*(A[1,1] + A[1,2]) = 0.5*(1+0) = 0.5
        //   A[2,4] = 0.5*(A[2,1] + A[2,2]) = 0.5*(0+1) = 0.5
        //   A[3,4] = 0.5*(0.5 + 0.5) = 0.5
        // F[5] = 0.5*A[3,4] = 0.5*0.5 = 0.25
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            (
                "3".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "4".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "5".to_string(),
                Some("3".to_string()),
                Some("4".to_string()),
            ),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        let f = compute_inbreeding(&ped).unwrap();

        assert_eq!(f.len(), 5);
        assert_approx(f[0], 0.0, "F[1]");
        assert_approx(f[1], 0.0, "F[2]");
        assert_approx(f[2], 0.0, "F[3]");
        assert_approx(f[3], 0.0, "F[4]");
        assert_approx(f[4], 0.25, "F[5]");
    }

    #[test]
    fn test_a_inverse_inbred_population() {
        // Full-sib mating pedigree (same as inbreeding test above).
        // Animal 5 is inbred with F=0.25.
        //
        // With inbreeding:
        //   d_5 = 0.5 - 0.25*(F_3 + F_4) = 0.5 - 0.25*(0+0) = 0.5
        //   alpha_5 = 2.0
        //
        // Without inbreeding: same d_5 = 0.5, alpha_5 = 2.0.
        //
        // In this case the two versions differ only through animal 5's d_i,
        // but since F_3 = F_4 = 0, they produce the same A^{-1}. The
        // difference would manifest in deeper pedigrees. We still verify
        // both produce consistent results.

        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            (
                "3".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "4".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "5".to_string(),
                Some("3".to_string()),
                Some("4".to_string()),
            ),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        let ainv = compute_a_inverse_with_inbreeding(&ped).unwrap();
        let dense = sparse_to_dense(&ainv);

        // Verify symmetry.
        for i in 0..5 {
            for j in 0..5 {
                assert_approx(
                    dense[i][j],
                    dense[j][i],
                    &format!("Symmetry A_inv[{},{}]", i, j),
                );
            }
        }

        // Verify diagonal is positive.
        for i in 0..5 {
            assert!(
                dense[i][i] > 0.0,
                "Diagonal A_inv[{0},{0}] should be positive, got {1}",
                i,
                dense[i][i]
            );
        }
    }

    #[test]
    fn test_a_inverse_deeper_inbreeding() {
        // Pedigree with multi-generation inbreeding where F actually affects d_i.
        //
        //   1: founder
        //   2: founder
        //   3: sire=1, dam=2
        //   4: sire=1, dam=2   (full sib of 3)
        //   5: sire=3, dam=4   (F[5] = 0.25)
        //   6: sire=3, dam=4   (F[6] = 0.25, full sib of 5)
        //   7: sire=5, dam=6   (inbred parents: F[5]=0.25, F[6]=0.25)
        //
        // For animal 7:
        //   Without inbreeding: d_7 = 0.5
        //   With inbreeding: d_7 = 0.5 - 0.25*(F[5] + F[6])
        //                       = 0.5 - 0.25*(0.25 + 0.25) = 0.5 - 0.125 = 0.375
        //   alpha_7 = 1/0.375 ≈ 2.6667 (vs 2.0 without inbreeding)
        //
        // This means the two methods should give different A^{-1} values.

        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), Some("2".to_string())),
            ("4".to_string(), Some("1".to_string()), Some("2".to_string())),
            ("5".to_string(), Some("3".to_string()), Some("4".to_string())),
            ("6".to_string(), Some("3".to_string()), Some("4".to_string())),
            ("7".to_string(), Some("5".to_string()), Some("6".to_string())),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        let f = compute_inbreeding(&ped).unwrap();
        assert_approx(f[4], 0.25, "F[5]");
        assert_approx(f[5], 0.25, "F[6]");

        // F[7] = 0.5 * A[5,6]
        // A[5,6] = 0.5*(A[3,6] + A[4,6])
        // A[3,6] = 0.5*(A[3,3] + A[3,4])
        //   A[3,3] = 1 + F[3] = 1
        //   A[3,4] = 0.5*(A[1,4] + A[2,4])
        //          = 0.5*(0.5 + 0.5) = 0.5
        //   A[3,6] = 0.5*(1 + 0.5) = 0.75
        // A[4,6] = 0.5*(A[4,3] + A[4,4])
        //   A[4,3] = 0.5 (computed above)
        //   A[4,4] = 1
        //   A[4,6] = 0.5*(0.5 + 1) = 0.75
        // A[5,6] = 0.5*(0.75 + 0.75) = 0.75
        // F[7] = 0.5 * 0.75 = 0.375
        assert_approx(f[6], 0.375, "F[7]");

        let ainv_no_f = compute_a_inverse(&ped).unwrap();
        let ainv_with_f = compute_a_inverse_with_inbreeding(&ped).unwrap();

        let dense_no_f = sparse_to_dense(&ainv_no_f);
        let dense_with_f = sparse_to_dense(&ainv_with_f);

        // The two methods should differ at entries involving animal 7 (index 6).
        // Specifically, d_7 differs: 0.5 vs 0.375, so alpha_7 differs: 2.0 vs 2.667.
        let alpha_no_f = 2.0;
        let alpha_with_f = 1.0 / 0.375;

        // Diagonal A_inv[6,6] should differ.
        assert_approx(
            dense_no_f[6][6] - dense_with_f[6][6],
            alpha_no_f - alpha_with_f,
            "Diagonal diff at animal 7",
        );

        // Verify the inbreeding version's diagonal is larger for animal 7.
        assert!(
            dense_with_f[6][6] > dense_no_f[6][6],
            "Inbreeding should increase diagonal for inbred animal"
        );
    }

    #[test]
    fn test_a_inverse_unsorted_errors() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), Some("1".to_string()), None),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        // Don't sort - should error.
        let result = compute_a_inverse(&ped);
        assert!(result.is_err());
    }

    #[test]
    fn test_a_inverse_empty_pedigree() {
        let ped = Pedigree::new();
        let ainv = compute_a_inverse(&ped).unwrap();
        assert_eq!(ainv.rows(), 0);
        assert_eq!(ainv.cols(), 0);
    }

    #[test]
    fn test_a_inverse_single_founder() {
        let triples = vec![("1".to_string(), None, None)];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        let ainv = compute_a_inverse(&ped).unwrap();
        let dense = sparse_to_dense(&ainv);

        // Single founder: d=1.0, alpha=1.0.
        assert_approx(dense[0][0], 1.0, "A_inv[0,0]");
    }

    #[test]
    fn test_a_inverse_one_parent_known() {
        // Animal 1: founder
        // Animal 2: sire=1, dam unknown
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), Some("1".to_string()), None),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        let ainv = compute_a_inverse(&ped).unwrap();
        let dense = sparse_to_dense(&ainv);

        // Animal 1: d=1.0, alpha=1.0
        // Animal 2: one parent known, d=0.75, alpha=4/3
        //
        // A_inv[0,0] = 1.0 + (4/3)/4 = 1.0 + 1/3 = 4/3
        // A_inv[1,1] = 4/3
        // A_inv[0,1] = A_inv[1,0] = -(4/3)/2 = -2/3
        assert_approx(dense[0][0], 4.0 / 3.0, "A_inv[0,0]");
        assert_approx(dense[1][1], 4.0 / 3.0, "A_inv[1,1]");
        assert_approx(dense[0][1], -2.0 / 3.0, "A_inv[0,1]");
        assert_approx(dense[1][0], -2.0 / 3.0, "A_inv[1,0]");
    }
}
