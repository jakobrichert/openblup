use crate::pedigree::SortedPedigree;
use nalgebra::DMatrix;

/// Henderson's rules for A⁻¹ with inbreeding (Meuwissen & Luo 1992 style
/// Mendelian sampling variances), dense output for WASM.
pub fn compute_a_inverse(ped: &SortedPedigree) -> DMatrix<f64> {
    let n = ped.ids.len();
    let mut a_inv = DMatrix::zeros(n, n);
    let f = &ped.inbreeding;

    for i in 0..n {
        let s = ped.sire_idx[i];
        let d = ped.dam_idx[i];

        // Mendelian sampling variance d_i; alpha = 1/d_i.
        let d_i = match (s, d) {
            (Some(s), Some(d)) => 0.5 - 0.25 * (f[s] + f[d]),
            (Some(s), None) => 0.75 - 0.25 * f[s],
            (None, Some(d)) => 0.75 - 0.25 * f[d],
            (None, None) => 1.0,
        };
        let alpha = 1.0 / d_i;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedigree::WasmPedigree;

    #[test]
    fn test_a_inverse_is_inverse_of_tabular_a() {
        // Pedigree with an inbred animal; A⁻¹ * A must be the identity.
        let mut ped = WasmPedigree::new();
        ped.add_animal("1", "0", "0");
        ped.add_animal("2", "0", "0");
        ped.add_animal("3", "1", "2");
        ped.add_animal("4", "1", "2");
        ped.add_animal("5", "3", "4");
        ped.add_animal("6", "5", "3");
        let sorted = ped.sort().unwrap();
        let a_inv = compute_a_inverse(&sorted);

        // Tabular A.
        let n = sorted.ids.len();
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            let (s, d) = (sorted.sire_idx[i], sorted.dam_idx[i]);
            for j in 0..i {
                let v = 0.5
                    * (s.map(|s| a[(s, j)]).unwrap_or(0.0) + d.map(|d| a[(d, j)]).unwrap_or(0.0));
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
            a[(i, i)] = 1.0
                + 0.5
                    * match (s, d) {
                        (Some(s), Some(d)) => a[(s, d)],
                        _ => 0.0,
                    };
        }
        let prod = &a_inv * &a;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[(i, j)] - expected).abs() < 1e-10,
                    "A_inv*A[{},{}] = {}",
                    i,
                    j,
                    prod[(i, j)]
                );
            }
        }
    }
}
