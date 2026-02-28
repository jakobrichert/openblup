use nalgebra::DMatrix;
use crate::pedigree::SortedPedigree;

/// Henderson's rules for A⁻¹ (WASM-compatible, dense output).
pub fn compute_a_inverse(ped: &SortedPedigree) -> DMatrix<f64> {
    let n = ped.ids.len();
    let mut a_inv = DMatrix::zeros(n, n);

    for i in 0..n {
        let s = ped.sire_idx[i];
        let d = ped.dam_idx[i];

        let alpha = match (s, d) {
            (Some(_), Some(_)) => 2.0,
            (Some(_), None) | (None, Some(_)) => 4.0 / 3.0,
            (None, None) => 1.0,
        };

        a_inv[(i, i)] += alpha;

        if let Some(si) = s {
            a_inv[(i, si)] -= alpha / 2.0;
            a_inv[(si, i)] -= alpha / 2.0;
            a_inv[(si, si)] += alpha / 4.0;
        }
        if let Some(di) = d {
            a_inv[(i, di)] -= alpha / 2.0;
            a_inv[(di, i)] -= alpha / 2.0;
            a_inv[(di, di)] += alpha / 4.0;
        }
        if let (Some(si), Some(di)) = (s, d) {
            a_inv[(si, di)] += alpha / 4.0;
            a_inv[(di, si)] += alpha / 4.0;
        }
    }

    a_inv
}
