use indexmap::IndexMap;

/// A categorical (factor) column that maps string levels to integer codes.
///
/// Levels are discovered in order of first appearance and assigned 0-based codes.
/// This is the standard representation for fixed and random effects in mixed models.
#[derive(Debug, Clone)]
pub struct FactorColumn {
    /// Maps level string -> integer code (0-based), ordered by first appearance.
    levels: IndexMap<String, usize>,
    /// The integer codes for each observation.
    codes: Vec<usize>,
}

impl FactorColumn {
    /// Create a new `FactorColumn` from a slice of string values.
    ///
    /// Levels are auto-discovered in order of first appearance and assigned
    /// consecutive 0-based integer codes.
    ///
    /// # Examples
    /// ```
    /// use plant_breeding_lmm_core::data::FactorColumn;
    ///
    /// let col = FactorColumn::new(&["A", "B", "A", "C", "B"]);
    /// assert_eq!(col.n_levels(), 3);
    /// assert_eq!(col.codes(), &[0, 1, 0, 2, 1]);
    /// ```
    pub fn new(values: &[&str]) -> Self {
        let mut levels = IndexMap::new();
        let mut codes = Vec::with_capacity(values.len());

        for &val in values {
            let next_code = levels.len();
            let code = *levels.entry(val.to_string()).or_insert(next_code);
            codes.push(code);
        }

        FactorColumn { levels, codes }
    }

    /// Create a `FactorColumn` from owned strings and a pre-built level map.
    ///
    /// This is used internally (e.g. when coercing integer columns to factors).
    pub(crate) fn from_parts(levels: IndexMap<String, usize>, codes: Vec<usize>) -> Self {
        FactorColumn { levels, codes }
    }

    /// Returns the number of distinct levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns a slice of the integer codes for each observation.
    pub fn codes(&self) -> &[usize] {
        &self.codes
    }

    /// Returns a reference to the ordered level map (level name -> code).
    pub fn levels(&self) -> &IndexMap<String, usize> {
        &self.levels
    }

    /// Returns the level name for a given integer code, or `None` if the code
    /// is out of range.
    pub fn level_name(&self, code: usize) -> Option<&str> {
        self.levels
            .iter()
            .find(|(_, &c)| c == code)
            .map(|(name, _)| name.as_str())
    }

    /// Returns the number of observations (rows).
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if the column has no observations.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_basic() {
        let col = FactorColumn::new(&["A", "B", "A", "C", "B"]);
        assert_eq!(col.n_levels(), 3);
        assert_eq!(col.len(), 5);
        assert_eq!(col.codes(), &[0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_level_order_is_first_appearance() {
        let col = FactorColumn::new(&["C", "A", "B", "A"]);
        let level_names: Vec<&str> = col.levels().keys().map(|s| s.as_str()).collect();
        assert_eq!(level_names, vec!["C", "A", "B"]);
        assert_eq!(col.codes(), &[0, 1, 2, 1]);
    }

    #[test]
    fn test_level_name_lookup() {
        let col = FactorColumn::new(&["X", "Y", "Z"]);
        assert_eq!(col.level_name(0), Some("X"));
        assert_eq!(col.level_name(1), Some("Y"));
        assert_eq!(col.level_name(2), Some("Z"));
        assert_eq!(col.level_name(3), None);
    }

    #[test]
    fn test_single_level() {
        let col = FactorColumn::new(&["only", "only", "only"]);
        assert_eq!(col.n_levels(), 1);
        assert_eq!(col.codes(), &[0, 0, 0]);
        assert_eq!(col.level_name(0), Some("only"));
    }

    #[test]
    fn test_empty() {
        let col = FactorColumn::new(&[]);
        assert_eq!(col.n_levels(), 0);
        assert_eq!(col.len(), 0);
        assert!(col.is_empty());
    }

    #[test]
    fn test_levels_map_values() {
        let col = FactorColumn::new(&["red", "green", "blue", "red"]);
        assert_eq!(col.levels()["red"], 0);
        assert_eq!(col.levels()["green"], 1);
        assert_eq!(col.levels()["blue"], 2);
    }

    #[test]
    fn test_from_parts() {
        let mut levels = IndexMap::new();
        levels.insert("low".to_string(), 0);
        levels.insert("high".to_string(), 1);
        let codes = vec![0, 1, 1, 0];
        let col = FactorColumn::from_parts(levels, codes);
        assert_eq!(col.n_levels(), 2);
        assert_eq!(col.codes(), &[0, 1, 1, 0]);
        assert_eq!(col.level_name(0), Some("low"));
        assert_eq!(col.level_name(1), Some("high"));
    }
}
