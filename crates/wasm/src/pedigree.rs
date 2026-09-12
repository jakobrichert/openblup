use std::collections::{HashMap, HashSet, VecDeque};

/// Lightweight pedigree for WASM (no file I/O).
#[derive(Default)]
pub struct WasmPedigree {
    pub animals: Vec<(String, String, String)>, // (id, sire, dam)
}

/// A topologically sorted pedigree with parent indices.
pub struct SortedPedigree {
    pub ids: Vec<String>,
    pub sire_idx: Vec<Option<usize>>,
    pub dam_idx: Vec<Option<usize>>,
    /// Inbreeding coefficient of each animal (tabular method).
    pub inbreeding: Vec<f64>,
}

fn unknown(s: &str) -> bool {
    s == "0" || s.is_empty() || s.eq_ignore_ascii_case("na")
}

impl WasmPedigree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_animal(&mut self, id: &str, sire: &str, dam: &str) {
        self.animals
            .push((id.to_string(), sire.to_string(), dam.to_string()));
    }

    /// Topological sort of the pedigree (parents before offspring).
    ///
    /// Parents that are referenced but never listed as animals are added as
    /// base animals with unknown parents. Returns an error on duplicate IDs
    /// or cycles.
    pub fn sort(&self) -> Result<SortedPedigree, String> {
        let mut seen: HashSet<&str> = HashSet::new();
        for (id, _, _) in &self.animals {
            if !seen.insert(id.as_str()) {
                return Err(format!("Duplicate animal ID '{}'", id));
            }
            if unknown(id) {
                return Err("Animal ID must not be empty, '0' or 'NA'".into());
            }
        }

        // Full list of animals including implicit base parents, in input order.
        let mut all: Vec<(String, String, String)> = Vec::new();
        let mut listed: HashSet<&str> = HashSet::new();
        for (id, s, d) in &self.animals {
            for parent in [s, d] {
                if !unknown(parent) && !seen.contains(parent.as_str()) && listed.insert(parent) {
                    all.push((parent.clone(), String::new(), String::new()));
                }
            }
            all.push((id.clone(), s.clone(), d.clone()));
        }

        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        for (id, sire, dam) in &all {
            let mut deg = 0;
            if !unknown(sire) {
                children.entry(sire.as_str()).or_default().push(id);
                deg += 1;
            }
            if !unknown(dam) {
                children.entry(dam.as_str()).or_default().push(id);
                deg += 1;
            }
            in_degree.insert(id, deg);
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (id, _, _) in &all {
            if *in_degree.get(id.as_str()).unwrap_or(&0) == 0 {
                queue.push_back(id);
            }
        }

        let mut sorted_ids: Vec<String> = Vec::new();
        while let Some(id) = queue.pop_front() {
            sorted_ids.push(id.to_string());
            if let Some(ch) = children.get(id) {
                for &child in ch {
                    let deg = in_degree.get_mut(child).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }

        if sorted_ids.len() != all.len() {
            return Err("Pedigree contains a cycle (an animal is its own ancestor)".into());
        }

        // Build index map
        let idx_map: HashMap<&str, usize> = sorted_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let pedigree_map: HashMap<&str, (&str, &str)> = all
            .iter()
            .map(|(id, s, d)| (id.as_str(), (s.as_str(), d.as_str())))
            .collect();

        let mut sire_idx = Vec::new();
        let mut dam_idx = Vec::new();
        for id in &sorted_ids {
            let (sire, dam) = pedigree_map[id.as_str()];
            sire_idx.push(if unknown(sire) {
                None
            } else {
                idx_map.get(sire).copied()
            });
            dam_idx.push(if unknown(dam) {
                None
            } else {
                idx_map.get(dam).copied()
            });
        }

        let inbreeding = compute_inbreeding(&sire_idx, &dam_idx);

        Ok(SortedPedigree {
            ids: sorted_ids,
            sire_idx,
            dam_idx,
            inbreeding,
        })
    }
}

/// Inbreeding coefficients via the tabular method (dense, O(n²) memory —
/// fine for the browser-sized pedigrees this crate targets).
fn compute_inbreeding(sire_idx: &[Option<usize>], dam_idx: &[Option<usize>]) -> Vec<f64> {
    let n = sire_idx.len();
    let mut a = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let (s, d) = (sire_idx[i], dam_idx[i]);
        for j in 0..i {
            let a_sj = s.map(|s| a[s][j]).unwrap_or(0.0);
            let a_dj = d.map(|d| a[d][j]).unwrap_or(0.0);
            let v = 0.5 * (a_sj + a_dj);
            a[i][j] = v;
            a[j][i] = v;
        }
        let a_sd = match (s, d) {
            (Some(s), Some(d)) => a[s][d],
            _ => 0.0,
        };
        a[i][i] = 1.0 + 0.5 * a_sd;
    }
    (0..n).map(|i| a[i][i] - 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_adds_missing_parents_and_orders() {
        let mut ped = WasmPedigree::new();
        ped.add_animal("C", "A", "B"); // A and B never listed
        let sorted = ped.sort().unwrap();
        assert_eq!(sorted.ids.len(), 3);
        assert_eq!(sorted.ids[2], "C");
        assert_eq!(
            sorted.sire_idx[2],
            Some(sorted.ids.iter().position(|x| x == "A").unwrap())
        );
        assert_eq!(sorted.inbreeding, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_inbreeding_full_sib_mating() {
        let mut ped = WasmPedigree::new();
        ped.add_animal("1", "0", "0");
        ped.add_animal("2", "0", "0");
        ped.add_animal("3", "1", "2");
        ped.add_animal("4", "1", "2");
        ped.add_animal("5", "3", "4");
        let sorted = ped.sort().unwrap();
        let f5 = sorted.inbreeding[sorted.ids.iter().position(|x| x == "5").unwrap()];
        assert!((f5 - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_duplicate_id_errors() {
        let mut ped = WasmPedigree::new();
        ped.add_animal("1", "0", "0");
        ped.add_animal("1", "0", "0");
        assert!(ped.sort().is_err());
    }
}
