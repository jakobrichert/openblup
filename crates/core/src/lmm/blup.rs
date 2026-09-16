//! BLUP post-processing utilities: reliability, accuracy and ranking.
//!
//! BLUPs and BLUEs themselves are extracted from the MME solution in the
//! REML engines; this module provides the derived quantities breeders
//! usually report alongside them.

use crate::lmm::result::NamedEffect;

/// Compute reliability (accuracy²) of BLUPs.
///
/// Reliability = 1 - PEV / sigma²_a, where PEV (prediction error variance)
/// is the diagonal of the MME inverse for the effect and sigma²_a is the
/// additive genetic variance. Values are clamped to `[0, 1]`.
///
/// `c_inv_diag` is the diagonal of C⁻¹ for the *whole* MME; `block_start`
/// is the offset of the first level of the random term.
pub fn compute_reliability(
    c_inv_diag: &[f64],
    block_start: usize,
    n_levels: usize,
    sigma2_a: f64,
) -> Vec<f64> {
    (0..n_levels)
        .map(|i| {
            let pev = c_inv_diag[block_start + i];
            (1.0 - pev / sigma2_a).clamp(0.0, 1.0)
        })
        .collect()
}

/// Compute accuracy of BLUPs (square root of reliability).
pub fn compute_accuracy(
    c_inv_diag: &[f64],
    block_start: usize,
    n_levels: usize,
    sigma2_a: f64,
) -> Vec<f64> {
    compute_reliability(c_inv_diag, block_start, n_levels, sigma2_a)
        .iter()
        .map(|r| r.sqrt())
        .collect()
}

/// Reliability of a single BLUP from its standard error:
/// `1 - se² / sigma²_a`, clamped to `[0, 1]`.
pub fn reliability_from_se(se: f64, sigma2_a: f64) -> f64 {
    if sigma2_a <= 0.0 {
        return 0.0;
    }
    (1.0 - se * se / sigma2_a).clamp(0.0, 1.0)
}

/// Rank effects by their estimates (descending). Each entry is the original
/// index of the effect together with a reference to it.
pub fn rank_effects(effects: &[NamedEffect]) -> Vec<(usize, &NamedEffect)> {
    let mut indexed: Vec<(usize, &NamedEffect)> = effects.iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.estimate
            .partial_cmp(&a.1.estimate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reliability_and_accuracy() {
        // Two fixed effects then two random levels: PEV = 2 and 8, sigma2 = 10.
        let c_inv_diag = [0.5, 0.5, 2.0, 8.0];
        let rel = compute_reliability(&c_inv_diag, 2, 2, 10.0);
        assert!((rel[0] - 0.8).abs() < 1e-12);
        assert!((rel[1] - 0.2).abs() < 1e-12);
        let acc = compute_accuracy(&c_inv_diag, 2, 2, 10.0);
        assert!((acc[0] - 0.8f64.sqrt()).abs() < 1e-12);
        // PEV larger than the variance clamps to zero.
        assert_eq!(compute_reliability(&[20.0], 0, 1, 10.0), vec![0.0]);
        assert_eq!(reliability_from_se(2.0, 10.0), 0.6);
        assert_eq!(reliability_from_se(2.0, 0.0), 0.0);
    }

    #[test]
    fn test_rank_effects() {
        let effects = vec![
            NamedEffect {
                term: "g".into(),
                level: "a".into(),
                estimate: 1.0,
                se: 0.1,
            },
            NamedEffect {
                term: "g".into(),
                level: "b".into(),
                estimate: 3.0,
                se: 0.1,
            },
            NamedEffect {
                term: "g".into(),
                level: "c".into(),
                estimate: 2.0,
                se: 0.1,
            },
        ];
        let ranked = rank_effects(&effects);
        let order: Vec<usize> = ranked.iter().map(|(i, _)| *i).collect();
        assert_eq!(order, vec![1, 2, 0]);
    }
}
