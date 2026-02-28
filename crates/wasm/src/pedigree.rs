/// Lightweight pedigree for WASM (no file I/O).
pub struct WasmPedigree {
    pub animals: Vec<(String, String, String)>, // (id, sire, dam)
}

pub struct SortedPedigree {
    pub ids: Vec<String>,
    pub sire_idx: Vec<Option<usize>>,
    pub dam_idx: Vec<Option<usize>>,
}

impl WasmPedigree {
    pub fn new() -> Self {
        Self { animals: Vec::new() }
    }

    pub fn add_animal(&mut self, id: &str, sire: &str, dam: &str) {
        self.animals.push((id.to_string(), sire.to_string(), dam.to_string()));
    }

    /// Topological sort of the pedigree.
    pub fn sort(&self) -> SortedPedigree {
        use std::collections::{HashMap, HashSet, VecDeque};

        let unknown = |s: &str| s == "0" || s.is_empty() || s.eq_ignore_ascii_case("na");

        let all_ids: HashSet<&str> = self.animals.iter().map(|(id, _, _)| id.as_str()).collect();
        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        for (id, sire, dam) in &self.animals {
            let mut deg = 0;
            if !unknown(sire) && all_ids.contains(sire.as_str()) {
                children.entry(sire.as_str()).or_default().push(id);
                deg += 1;
            }
            if !unknown(dam) && all_ids.contains(dam.as_str()) {
                children.entry(dam.as_str()).or_default().push(id);
                deg += 1;
            }
            in_degree.insert(id, deg);
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (id, _, _) in &self.animals {
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

        // Build index map
        let idx_map: HashMap<&str, usize> = sorted_ids.iter().enumerate()
            .map(|(i, id)| (id.as_str(), i)).collect();

        let pedigree_map: HashMap<&str, (&str, &str)> = self.animals.iter()
            .map(|(id, s, d)| (id.as_str(), (s.as_str(), d.as_str())))
            .collect();

        let mut sire_idx = Vec::new();
        let mut dam_idx = Vec::new();
        for id in &sorted_ids {
            if let Some(&(sire, dam)) = pedigree_map.get(id.as_str()) {
                sire_idx.push(if unknown(sire) { None } else { idx_map.get(sire).copied() });
                dam_idx.push(if unknown(dam) { None } else { idx_map.get(dam).copied() });
            } else {
                sire_idx.push(None);
                dam_idx.push(None);
            }
        }

        SortedPedigree { ids: sorted_ids, sire_idx, dam_idx }
    }
}
