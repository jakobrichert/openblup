use std::collections::HashMap;
use std::path::Path;

use crate::error::{LmmError, Result};

/// A single pedigree record: animal with optional sire and dam.
#[derive(Debug, Clone)]
struct PedigreeRecord {
    /// Animal identifier string.
    id: String,
    /// Index of sire in the animals vector, or `None` if unknown.
    sire: Option<usize>,
    /// Index of dam in the animals vector, or `None` if unknown.
    dam: Option<usize>,
}

/// Pedigree representing parent-offspring relationships for additive genetic
/// evaluation.
///
/// Internally, animals are mapped to contiguous 0-based indices. Unknown
/// parents (coded as `"0"` or empty in input) are represented as `None`.
///
/// The pedigree must be sorted so that parents appear before their offspring
/// before computing the A-inverse matrix. Use [`Pedigree::sort_pedigree`] to
/// enforce this ordering.
#[derive(Debug, Clone)]
pub struct Pedigree {
    /// Ordered list of pedigree records.
    records: Vec<PedigreeRecord>,
    /// Mapping from animal ID string to its 0-based index.
    id_to_index: HashMap<String, usize>,
    /// Whether the pedigree has been topologically sorted.
    sorted: bool,
}

impl Pedigree {
    /// Create an empty pedigree.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            id_to_index: HashMap::new(),
            sorted: false,
        }
    }

    /// Number of animals in the pedigree.
    pub fn n_animals(&self) -> usize {
        self.records.len()
    }

    /// Look up the 0-based index of an animal by its ID string.
    pub fn animal_index(&self, id: &str) -> Option<usize> {
        self.id_to_index.get(id).copied()
    }

    /// Look up the ID string of an animal by its 0-based index.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn animal_id(&self, index: usize) -> &str {
        &self.records[index].id
    }

    /// Return the sire index for animal at `index`, or `None` if unknown.
    pub fn sire(&self, index: usize) -> Option<usize> {
        self.records[index].sire
    }

    /// Return the dam index for animal at `index`, or `None` if unknown.
    pub fn dam(&self, index: usize) -> Option<usize> {
        self.records[index].dam
    }

    /// Whether the pedigree has been topologically sorted.
    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    /// Add an animal to the pedigree.
    ///
    /// `sire` and `dam` are optional parent ID strings. If a parent ID is
    /// provided but has not been added to the pedigree yet, the parent is
    /// **not** created automatically. Call [`Pedigree::validate`] after
    /// building to check consistency.
    ///
    /// # Errors
    /// Returns an error if the animal ID already exists.
    pub fn add_animal(
        &mut self,
        id: &str,
        sire: Option<&str>,
        dam: Option<&str>,
    ) -> Result<()> {
        if self.id_to_index.contains_key(id) {
            return Err(LmmError::Pedigree(format!(
                "Duplicate animal ID: '{}'",
                id
            )));
        }

        let index = self.records.len();

        let sire_idx = sire.and_then(|s| self.id_to_index.get(s).copied());
        let dam_idx = dam.and_then(|d| self.id_to_index.get(d).copied());

        self.records.push(PedigreeRecord {
            id: id.to_string(),
            sire: sire_idx,
            dam: dam_idx,
        });
        self.id_to_index.insert(id.to_string(), index);
        self.sorted = false;

        Ok(())
    }

    /// Build a pedigree from (animal, sire, dam) triples.
    ///
    /// Parent values of `None` indicate unknown parents.
    ///
    /// # Errors
    /// Returns an error if duplicate animal IDs are found.
    pub fn from_triples(triples: &[(String, Option<String>, Option<String>)]) -> Result<Self> {
        let mut ped = Self::new();

        // First pass: register all animals so parent lookups can succeed
        // regardless of input order.
        for (id, _, _) in triples {
            if ped.id_to_index.contains_key(id) {
                return Err(LmmError::Pedigree(format!(
                    "Duplicate animal ID: '{}'",
                    id
                )));
            }
            let index = ped.records.len();
            ped.records.push(PedigreeRecord {
                id: id.clone(),
                sire: None,
                dam: None,
            });
            ped.id_to_index.insert(id.clone(), index);
        }

        // Second pass: resolve parent indices.
        for (i, (_, sire, dam)) in triples.iter().enumerate() {
            ped.records[i].sire = sire
                .as_ref()
                .and_then(|s| ped.id_to_index.get(s.as_str()).copied());
            ped.records[i].dam = dam
                .as_ref()
                .and_then(|d| ped.id_to_index.get(d.as_str()).copied());
        }

        Ok(ped)
    }

    /// Read a pedigree from a CSV file.
    ///
    /// Expected columns (header required): `animal`, `sire`, `dam`.
    /// Unknown parents are coded as `"0"`, `""`, or `"NA"`.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, columns are missing, or
    /// duplicate animal IDs are found.
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .trim(csv::Trim::All)
            .from_path(path)?;

        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|h| h.to_lowercase())
            .collect();

        let animal_col = headers
            .iter()
            .position(|h| h == "animal")
            .ok_or_else(|| {
                LmmError::Pedigree("CSV missing 'animal' column".to_string())
            })?;
        let sire_col = headers
            .iter()
            .position(|h| h == "sire")
            .ok_or_else(|| {
                LmmError::Pedigree("CSV missing 'sire' column".to_string())
            })?;
        let dam_col = headers
            .iter()
            .position(|h| h == "dam")
            .ok_or_else(|| {
                LmmError::Pedigree("CSV missing 'dam' column".to_string())
            })?;

        let mut triples = Vec::new();

        for result in reader.records() {
            let record = result?;

            let animal = record
                .get(animal_col)
                .ok_or_else(|| {
                    LmmError::Pedigree("Missing animal field in row".to_string())
                })?
                .to_string();

            let sire_raw = record
                .get(sire_col)
                .ok_or_else(|| {
                    LmmError::Pedigree("Missing sire field in row".to_string())
                })?;

            let dam_raw = record
                .get(dam_col)
                .ok_or_else(|| {
                    LmmError::Pedigree("Missing dam field in row".to_string())
                })?;

            let sire = parse_parent(sire_raw);
            let dam = parse_parent(dam_raw);

            triples.push((animal, sire, dam));
        }

        Self::from_triples(&triples)
    }

    /// Validate the pedigree for consistency.
    ///
    /// Checks:
    /// - All referenced parent IDs exist in the pedigree.
    /// - No animal is its own ancestor (cycle detection).
    ///
    /// Note: this method works on the raw triples. For parent-existence
    /// checking it verifies that every non-None parent index is valid.
    ///
    /// # Errors
    /// Returns an error describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        let n = self.records.len();

        // Check that parent indices are valid.
        for rec in &self.records {
            if let Some(s) = rec.sire {
                if s >= n {
                    return Err(LmmError::Pedigree(format!(
                        "Animal '{}' references sire index {} which is out of range",
                        rec.id, s
                    )));
                }
                if s == self.id_to_index[&rec.id] {
                    return Err(LmmError::Pedigree(format!(
                        "Animal '{}' is listed as its own sire",
                        rec.id
                    )));
                }
            }
            if let Some(d) = rec.dam {
                if d >= n {
                    return Err(LmmError::Pedigree(format!(
                        "Animal '{}' references dam index {} which is out of range",
                        rec.id, d
                    )));
                }
                if d == self.id_to_index[&rec.id] {
                    return Err(LmmError::Pedigree(format!(
                        "Animal '{}' is listed as its own dam",
                        rec.id
                    )));
                }
            }
        }

        // Cycle detection via topological sort (Kahn's algorithm).
        // Build in-degree counts based on parent -> offspring edges.
        // Direction: parent -> child. If we cannot process all nodes,
        // there is a cycle.
        let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0u32; n];

        for (i, rec) in self.records.iter().enumerate() {
            if let Some(s) = rec.sire {
                children_of[s].push(i);
                in_degree[i] += 1;
            }
            if let Some(d) = rec.dam {
                children_of[d].push(i);
                in_degree[i] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .collect();
        let mut visited = 0usize;

        while let Some(node) = queue.pop() {
            visited += 1;
            for &child in &children_of[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push(child);
                }
            }
        }

        if visited != n {
            return Err(LmmError::Pedigree(
                "Pedigree contains a cycle".to_string(),
            ));
        }

        Ok(())
    }

    /// Topologically sort the pedigree so that parents appear before their
    /// offspring.
    ///
    /// This is required before computing the A-inverse matrix. The method
    /// uses Kahn's algorithm and updates all internal indices.
    ///
    /// # Errors
    /// Returns an error if the pedigree contains a cycle.
    pub fn sort_pedigree(&mut self) -> Result<()> {
        let n = self.records.len();
        if n == 0 {
            self.sorted = true;
            return Ok(());
        }

        // Build adjacency: parent -> children, and compute in-degrees.
        let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0u32; n];

        for (i, rec) in self.records.iter().enumerate() {
            if let Some(s) = rec.sire {
                children_of[s].push(i);
                in_degree[i] += 1;
            }
            if let Some(d) = rec.dam {
                children_of[d].push(i);
                in_degree[i] += 1;
            }
        }

        // Kahn's algorithm: use a VecDeque for stable ordering.
        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &child in &children_of[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }

        if order.len() != n {
            return Err(LmmError::Pedigree(
                "Pedigree contains a cycle; cannot sort".to_string(),
            ));
        }

        // Build old->new index mapping.
        let mut old_to_new = vec![0usize; n];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            old_to_new[old_idx] = new_idx;
        }

        // Rebuild records in topological order with remapped indices.
        let old_records = self.records.clone();
        let mut new_records = Vec::with_capacity(n);

        for &old_idx in &order {
            let old_rec = &old_records[old_idx];
            new_records.push(PedigreeRecord {
                id: old_rec.id.clone(),
                sire: old_rec.sire.map(|s| old_to_new[s]),
                dam: old_rec.dam.map(|d| old_to_new[d]),
            });
        }

        self.records = new_records;

        // Rebuild the ID-to-index map.
        self.id_to_index.clear();
        for (i, rec) in self.records.iter().enumerate() {
            self.id_to_index.insert(rec.id.clone(), i);
        }

        self.sorted = true;
        Ok(())
    }
}

impl Default for Pedigree {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a parent string, returning `None` for unknown parents.
///
/// Unknown parents are coded as `"0"`, `""`, `"NA"`, or `"na"`.
fn parse_parent(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "0" || trimmed.eq_ignore_ascii_case("na") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Helper: write CSV content to a temporary file and return the path.
    fn write_temp_csv(content: &str) -> String {
        let dir = std::env::temp_dir();
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("test_pedigree_{}_{}.csv", std::process::id(), id);
        let path = dir.join(file_name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_simple_3_animal_pedigree() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), Some("2".to_string())),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        assert_eq!(ped.n_animals(), 3);

        assert_eq!(ped.animal_index("1"), Some(0));
        assert_eq!(ped.animal_index("2"), Some(1));
        assert_eq!(ped.animal_index("3"), Some(2));

        assert_eq!(ped.sire(2), Some(0));
        assert_eq!(ped.dam(2), Some(1));
        assert_eq!(ped.sire(0), None);
        assert_eq!(ped.dam(0), None);
    }

    #[test]
    fn test_mrode_5_animal_pedigree() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), None),
            ("4".to_string(), Some("1".to_string()), Some("2".to_string())),
            ("5".to_string(), Some("3".to_string()), Some("2".to_string())),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        assert_eq!(ped.n_animals(), 5);

        // Animal 3: sire=1 (index 0), dam=unknown
        assert_eq!(ped.sire(2), Some(0));
        assert_eq!(ped.dam(2), None);

        // Animal 4: sire=1 (index 0), dam=2 (index 1)
        assert_eq!(ped.sire(3), Some(0));
        assert_eq!(ped.dam(3), Some(1));

        // Animal 5: sire=3 (index 2), dam=2 (index 1)
        assert_eq!(ped.sire(4), Some(2));
        assert_eq!(ped.dam(4), Some(1));
    }

    #[test]
    fn test_from_csv_basic() {
        let csv = "animal,sire,dam\n1,0,0\n2,0,0\n3,1,2\n";
        let path = write_temp_csv(csv);
        let ped = Pedigree::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(ped.n_animals(), 3);
        assert_eq!(ped.sire(2), Some(0));
        assert_eq!(ped.dam(2), Some(1));
        assert_eq!(ped.sire(0), None);
    }

    #[test]
    fn test_from_csv_empty_parents() {
        let csv = "animal,sire,dam\nA,,\nB,A,\nC,A,B\n";
        let path = write_temp_csv(csv);
        let ped = Pedigree::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(ped.n_animals(), 3);
        assert_eq!(ped.sire(0), None);
        assert_eq!(ped.dam(0), None);
        assert_eq!(ped.sire(1), Some(0)); // B's sire is A
        assert_eq!(ped.dam(1), None);
        assert_eq!(ped.sire(2), Some(0)); // C's sire is A
        assert_eq!(ped.dam(2), Some(1)); // C's dam is B
    }

    #[test]
    fn test_from_csv_na_parents() {
        let csv = "animal,sire,dam\nX,NA,NA\nY,X,NA\n";
        let path = write_temp_csv(csv);
        let ped = Pedigree::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(ped.n_animals(), 2);
        assert_eq!(ped.sire(0), None);
        assert_eq!(ped.sire(1), Some(0));
        assert_eq!(ped.dam(1), None);
    }

    #[test]
    fn test_topological_sort_already_sorted() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), Some("2".to_string())),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        assert!(ped.is_sorted());
        // Order should stay the same since it was already valid.
        assert_eq!(ped.animal_id(0), "1");
        assert_eq!(ped.animal_id(1), "2");
        assert_eq!(ped.animal_id(2), "3");
    }

    #[test]
    fn test_topological_sort_reorders() {
        // Give offspring before parents.
        let triples = vec![
            ("3".to_string(), Some("1".to_string()), Some("2".to_string())),
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        assert!(ped.is_sorted());

        // After sort, parents must come before offspring.
        let idx_1 = ped.animal_index("1").unwrap();
        let idx_2 = ped.animal_index("2").unwrap();
        let idx_3 = ped.animal_index("3").unwrap();

        assert!(idx_1 < idx_3, "Parent '1' must precede offspring '3'");
        assert!(idx_2 < idx_3, "Parent '2' must precede offspring '3'");

        // Verify parent links are correct after re-indexing.
        assert_eq!(ped.sire(idx_3), Some(idx_1));
        assert_eq!(ped.dam(idx_3), Some(idx_2));
    }

    #[test]
    fn test_topological_sort_deep_chain() {
        // Chain: 4 -> 3 -> 2 -> 1, given in reverse order.
        let triples = vec![
            ("4".to_string(), Some("3".to_string()), None),
            ("3".to_string(), Some("2".to_string()), None),
            ("2".to_string(), Some("1".to_string()), None),
            ("1".to_string(), None, None),
        ];
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();

        // After sort: 1 before 2 before 3 before 4.
        let idx_1 = ped.animal_index("1").unwrap();
        let idx_2 = ped.animal_index("2").unwrap();
        let idx_3 = ped.animal_index("3").unwrap();
        let idx_4 = ped.animal_index("4").unwrap();

        assert!(idx_1 < idx_2);
        assert!(idx_2 < idx_3);
        assert!(idx_3 < idx_4);
    }

    #[test]
    fn test_validate_ok() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
            ("3".to_string(), Some("1".to_string()), Some("2".to_string())),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        assert!(ped.validate().is_ok());
    }

    #[test]
    fn test_validate_detects_self_parent() {
        // Manually construct a pathological case: animal is its own sire.
        let mut ped = Pedigree::new();
        // Insert animal 1.
        ped.records.push(PedigreeRecord {
            id: "1".to_string(),
            sire: Some(0), // points to itself
            dam: None,
        });
        ped.id_to_index.insert("1".to_string(), 0);

        let result = ped.validate();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("own sire"), "Error was: {}", msg);
    }

    #[test]
    fn test_duplicate_animal_id() {
        let triples = vec![
            ("1".to_string(), None, None),
            ("1".to_string(), None, None),
        ];
        let result = Pedigree::from_triples(&triples);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Duplicate"), "Error was: {}", msg);
    }

    #[test]
    fn test_add_animal_incremental() {
        let mut ped = Pedigree::new();
        ped.add_animal("S1", None, None).unwrap();
        ped.add_animal("D1", None, None).unwrap();
        ped.add_animal("O1", Some("S1"), Some("D1")).unwrap();

        assert_eq!(ped.n_animals(), 3);
        let idx = ped.animal_index("O1").unwrap();
        assert_eq!(ped.sire(idx), Some(0));
        assert_eq!(ped.dam(idx), Some(1));
    }

    #[test]
    fn test_add_animal_duplicate_errors() {
        let mut ped = Pedigree::new();
        ped.add_animal("A", None, None).unwrap();
        let result = ped.add_animal("A", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_animal_id_lookup() {
        let triples = vec![
            ("Alpha".to_string(), None, None),
            ("Beta".to_string(), None, None),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        assert_eq!(ped.animal_id(0), "Alpha");
        assert_eq!(ped.animal_id(1), "Beta");
        assert_eq!(ped.animal_index("Alpha"), Some(0));
        assert_eq!(ped.animal_index("Gamma"), None);
    }

    #[test]
    fn test_parse_parent_variants() {
        assert_eq!(parse_parent("0"), None);
        assert_eq!(parse_parent(""), None);
        assert_eq!(parse_parent("  "), None);
        assert_eq!(parse_parent("NA"), None);
        assert_eq!(parse_parent("na"), None);
        assert_eq!(parse_parent("Na"), None);
        assert_eq!(parse_parent("1"), Some("1".to_string()));
        assert_eq!(parse_parent("SireA"), Some("SireA".to_string()));
    }
}
