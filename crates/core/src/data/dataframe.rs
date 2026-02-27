use indexmap::IndexMap;

use super::factor::FactorColumn;
use crate::error::{LmmError, Result};

/// A single column in a [`DataFrame`], which can hold floating-point numbers,
/// integers, or categorical (factor) data.
#[derive(Debug, Clone)]
pub enum Column {
    /// A column of 64-bit floating-point values.
    Float(Vec<f64>),
    /// A column of 64-bit signed integers.
    Integer(Vec<i64>),
    /// A categorical column with string levels mapped to integer codes.
    Factor(FactorColumn),
}

impl Column {
    /// Returns the number of elements in the column.
    pub fn len(&self) -> usize {
        match self {
            Column::Float(v) => v.len(),
            Column::Integer(v) => v.len(),
            Column::Factor(f) => f.len(),
        }
    }

    /// Returns `true` if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A lightweight columnar DataFrame for mixed-model data.
///
/// Columns are stored in insertion order using an [`IndexMap`]. All columns
/// must have the same number of rows.
#[derive(Debug, Clone)]
pub struct DataFrame {
    /// Ordered map of column name -> column data.
    pub(crate) columns: IndexMap<String, Column>,
    /// Number of rows (observations). Zero when the DataFrame is empty.
    pub(crate) nrows: usize,
}

impl DataFrame {
    /// Create an empty DataFrame with no columns and no rows.
    pub fn new() -> Self {
        DataFrame {
            columns: IndexMap::new(),
            nrows: 0,
        }
    }

    /// Add a floating-point column.
    ///
    /// # Errors
    /// Returns an error if the column length does not match existing rows,
    /// or if a column with the same name already exists.
    pub fn add_float_column(&mut self, name: &str, data: Vec<f64>) -> Result<()> {
        self.validate_and_insert(name, Column::Float(data))
    }

    /// Add an integer column.
    ///
    /// # Errors
    /// Returns an error if the column length does not match existing rows,
    /// or if a column with the same name already exists.
    pub fn add_integer_column(&mut self, name: &str, data: Vec<i64>) -> Result<()> {
        self.validate_and_insert(name, Column::Integer(data))
    }

    /// Add a factor (categorical) column from a slice of string values.
    ///
    /// Levels are auto-discovered in order of first appearance.
    ///
    /// # Errors
    /// Returns an error if the column length does not match existing rows,
    /// or if a column with the same name already exists.
    pub fn add_factor_column(&mut self, name: &str, data: &[&str]) -> Result<()> {
        let factor = FactorColumn::new(data);
        self.validate_and_insert(name, Column::Factor(factor))
    }

    /// Retrieve a column by name.
    ///
    /// # Errors
    /// Returns [`LmmError::ColumnNotFound`] if no column with the given name exists.
    pub fn get_column(&self, name: &str) -> Result<&Column> {
        self.columns
            .get(name)
            .ok_or_else(|| LmmError::ColumnNotFound(name.to_string()))
    }

    /// Retrieve a float column's data as a slice.
    ///
    /// # Errors
    /// Returns an error if the column does not exist or is not a `Float` column.
    pub fn get_float(&self, name: &str) -> Result<&[f64]> {
        match self.get_column(name)? {
            Column::Float(v) => Ok(v.as_slice()),
            _ => Err(LmmError::Data(format!(
                "Column '{}' is not a Float column",
                name
            ))),
        }
    }

    /// Retrieve a factor column reference.
    ///
    /// # Errors
    /// Returns an error if the column does not exist or is not a `Factor` column.
    pub fn get_factor(&self, name: &str) -> Result<&FactorColumn> {
        match self.get_column(name)? {
            Column::Factor(f) => Ok(f),
            _ => Err(LmmError::Data(format!(
                "Column '{}' is not a Factor column",
                name
            ))),
        }
    }

    /// Returns the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Returns a vector of column names in insertion order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Coerce an existing column to a `Factor` column in-place.
    ///
    /// - If the column is already a `Factor`, this is a no-op.
    /// - If the column is `Integer`, each distinct integer value becomes a level
    ///   (sorted numerically, so the level ordering is deterministic).
    /// - If the column is `Float`, returns an error (float-to-factor coercion is
    ///   generally ill-defined for continuous data).
    ///
    /// # Errors
    /// Returns an error if the column does not exist or is a `Float` column.
    pub fn as_factor(&mut self, name: &str) -> Result<()> {
        let col = self
            .columns
            .get(name)
            .ok_or_else(|| LmmError::ColumnNotFound(name.to_string()))?;

        let new_col = match col {
            Column::Factor(_) => return Ok(()),
            Column::Float(_) => {
                return Err(LmmError::Data(format!(
                    "Cannot coerce Float column '{}' to Factor",
                    name
                )));
            }
            Column::Integer(vals) => {
                // Collect unique values and sort them so level order is deterministic.
                let mut unique: Vec<i64> = vals.iter().copied().collect();
                unique.sort_unstable();
                unique.dedup();

                let mut levels = IndexMap::new();
                for (i, &val) in unique.iter().enumerate() {
                    levels.insert(val.to_string(), i);
                }

                let codes: Vec<usize> = vals
                    .iter()
                    .map(|v| levels[&v.to_string()])
                    .collect();

                Column::Factor(FactorColumn::from_parts(levels, codes))
            }
        };

        // Replace the column in the map.
        *self.columns.get_mut(name).unwrap() = new_col;
        Ok(())
    }

    // ---- internal helpers ----

    /// Validate column length and name uniqueness, then insert.
    fn validate_and_insert(&mut self, name: &str, column: Column) -> Result<()> {
        if self.columns.contains_key(name) {
            return Err(LmmError::Data(format!(
                "Column '{}' already exists in DataFrame",
                name
            )));
        }

        let col_len = column.len();

        if self.columns.is_empty() {
            self.nrows = col_len;
        } else if col_len != self.nrows {
            return Err(LmmError::DimensionMismatch {
                expected: self.nrows,
                got: col_len,
                context: format!("adding column '{}'", name),
            });
        }

        self.columns.insert(name.to_string(), column);
        Ok(())
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column("yield", vec![5.2, 3.1, 4.7]).unwrap();
        df.add_integer_column("block", vec![1, 2, 1]).unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G1"]).unwrap();
        df
    }

    #[test]
    fn test_new_is_empty() {
        let df = DataFrame::new();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
        assert!(df.column_names().is_empty());
    }

    #[test]
    fn test_add_columns_and_dimensions() {
        let df = sample_df();
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 3);
        assert_eq!(df.column_names(), vec!["yield", "block", "genotype"]);
    }

    #[test]
    fn test_get_float() {
        let df = sample_df();
        let yields = df.get_float("yield").unwrap();
        assert_eq!(yields, &[5.2, 3.1, 4.7]);
    }

    #[test]
    fn test_get_factor() {
        let df = sample_df();
        let geno = df.get_factor("genotype").unwrap();
        assert_eq!(geno.n_levels(), 2);
        assert_eq!(geno.codes(), &[0, 1, 0]);
    }

    #[test]
    fn test_get_column_not_found() {
        let df = sample_df();
        let err = df.get_column("missing").unwrap_err();
        assert!(matches!(err, LmmError::ColumnNotFound(_)));
    }

    #[test]
    fn test_get_float_wrong_type() {
        let df = sample_df();
        let err = df.get_float("genotype").unwrap_err();
        assert!(matches!(err, LmmError::Data(_)));
    }

    #[test]
    fn test_get_factor_wrong_type() {
        let df = sample_df();
        let err = df.get_factor("yield").unwrap_err();
        assert!(matches!(err, LmmError::Data(_)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut df = DataFrame::new();
        df.add_float_column("a", vec![1.0, 2.0]).unwrap();
        let err = df.add_float_column("b", vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, LmmError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_duplicate_column_name() {
        let mut df = DataFrame::new();
        df.add_float_column("x", vec![1.0]).unwrap();
        let err = df.add_float_column("x", vec![2.0]).unwrap_err();
        assert!(matches!(err, LmmError::Data(_)));
    }

    #[test]
    fn test_as_factor_from_integer() {
        let mut df = DataFrame::new();
        df.add_integer_column("block", vec![3, 1, 2, 1, 3]).unwrap();
        df.as_factor("block").unwrap();

        let factor = df.get_factor("block").unwrap();
        // Levels should be sorted: "1" -> 0, "2" -> 1, "3" -> 2
        assert_eq!(factor.n_levels(), 3);
        assert_eq!(factor.level_name(0), Some("1"));
        assert_eq!(factor.level_name(1), Some("2"));
        assert_eq!(factor.level_name(2), Some("3"));
        assert_eq!(factor.codes(), &[2, 0, 1, 0, 2]);
    }

    #[test]
    fn test_as_factor_already_factor() {
        let mut df = sample_df();
        // Should be a no-op and succeed.
        df.as_factor("genotype").unwrap();
        let geno = df.get_factor("genotype").unwrap();
        assert_eq!(geno.n_levels(), 2);
    }

    #[test]
    fn test_as_factor_float_errors() {
        let mut df = sample_df();
        let err = df.as_factor("yield").unwrap_err();
        assert!(matches!(err, LmmError::Data(_)));
    }

    #[test]
    fn test_as_factor_not_found() {
        let mut df = DataFrame::new();
        let err = df.as_factor("nope").unwrap_err();
        assert!(matches!(err, LmmError::ColumnNotFound(_)));
    }

    #[test]
    fn test_column_len_and_is_empty() {
        let col_f = Column::Float(vec![1.0, 2.0]);
        assert_eq!(col_f.len(), 2);
        assert!(!col_f.is_empty());

        let col_empty = Column::Integer(vec![]);
        assert_eq!(col_empty.len(), 0);
        assert!(col_empty.is_empty());
    }

    #[test]
    fn test_default_trait() {
        let df = DataFrame::default();
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
    }
}
