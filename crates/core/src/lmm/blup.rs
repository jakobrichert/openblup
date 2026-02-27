// BLUP/BLUE extraction utilities.
// In Phase 1, BLUPs and BLUEs are extracted directly from the MME solution
// in reml.rs and mme.rs. This module provides additional utilities.

use crate::error::Result;
use crate::lmm::result::NamedEffect;

/// Compute reliability (accuracy^2) of BLUPs.
///
/// Reliability = 1 - PEV / sigma^2_a
/// where PEV (Prediction Error Variance) = C^{-1}_{ii} (diagonal of MME inverse)
/// and sigma^2_a is the additive genetic variance.
pub fn compute_reliability(
    c_inv_diag: &[f64],
    block_start: usize,
    n_levels: usize,
    sigma2_a: f64,
) -> Vec<f64> {
    (0..n_levels)
        .map(|i| {
            let pev = c_inv_diag[block_start + i];
            (1.0 - pev / sigma2_a).max(0.0)
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

/// Rank effects by their estimates (descending).
pub fn rank_effects(effects: &[NamedEffect]) -> Vec<(usize, &NamedEffect)> {
    let mut indexed: Vec<(usize, &NamedEffect)> = effects.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.estimate.partial_cmp(&a.1.estimate).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}
