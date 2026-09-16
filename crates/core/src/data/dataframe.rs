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
    /// - If the column is `Float`, each distinct value becomes a level (sorted
    ///   numerically; integral values are named without a decimal part, so
    ///   `1.0` becomes level `"1"`).
    ///
    /// # Errors
    /// Returns an error if the column does not exist or if a `Float` column
    /// contains missing (NaN) values.
    pub fn as_factor(&mut self, name: &str) -> Result<()> {
        let col = self
            .columns
            .get(name)
            .ok_or_else(|| LmmError::ColumnNotFound(name.to_string()))?;

        let new_col = match col {
            Column::Factor(_) => return Ok(()),
            Column::Float(vals) => Column::Factor(float_to_factor(name, vals)?),
            Column::Integer(vals) => Column::Factor(integer_to_factor(vals)),
        };

        // Replace the column in the map.
        *self.columns.get_mut(name).unwrap() = new_col;
        Ok(())
    }

    /// Borrow a column as a factor, converting numeric columns on the fly.
    ///
    /// `Factor` columns are borrowed as-is; `Integer` and `Float` columns are
    /// converted with the same rules as [`DataFrame::as_factor`] (without
    /// modifying the DataFrame). This is what random-effect terms use, since
    /// a grouping variable is categorical by definition even when it is coded
    /// numerically (e.g. animal IDs).
    ///
    /// # Errors
    /// Returns an error if the column does not exist or if a `Float` column
    /// contains missing (NaN) values.
    pub fn factor_view(&self, name: &str) -> Result<std::borrow::Cow<'_, FactorColumn>> {
        use std::borrow::Cow;
        match self.get_column(name)? {
            Column::Factor(f) => Ok(Cow::Borrowed(f)),
            Column::Integer(vals) => Ok(Cow::Owned(integer_to_factor(vals))),
            Column::Float(vals) => Ok(Cow::Owned(float_to_factor(name, vals)?)),
        }
    }

    /// Return `true` if a float column contains missing (NaN) values.
    ///
    /// Non-float columns never report missing values.
    pub fn has_missing(&self, name: &str) -> Result<bool> {
        Ok(match self.get_column(name)? {
            Column::Float(v) => v.iter().any(|x| x.is_nan()),
            _ => false,
        })
    }

    /// Return a new DataFrame containing only the rows where `keep[i]` is true.
    ///
    /// Factor columns are re-coded so that levels which no longer occur are
    /// dropped.
    ///
    /// # Errors
    /// Returns an error if `keep` does not have one entry per row.
    pub fn filter_rows(&self, keep: &[bool]) -> Result<DataFrame> {
        if keep.len() != self.nrows {
            return Err(LmmError::DimensionMismatch {
                expected: self.nrows,
                got: keep.len(),
                context: "row mask for filter_rows".into(),
            });
        }

        let mut out = DataFrame::new();
        for (name, col) in &self.columns {
            let new_col = match col {
                Column::Float(v) => Column::Float(
                    v.iter()
                        .zip(keep)
                        .filter(|(_, &k)| k)
                        .map(|(x, _)| *x)
                        .collect(),
                ),
                Column::Integer(v) => Column::Integer(
                    v.iter()
                        .zip(keep)
                        .filter(|(_, &k)| k)
                        .map(|(x, _)| *x)
                        .collect(),
                ),
                Column::Factor(f) => Column::Factor(f.subset(keep)),
            };
            out.validate_and_insert(name, new_col)?;
        }
        // An all-false mask on a frame with columns must still report 0 rows.
        if out.columns.is_empty() {
            out.nrows = 0;
        }
        Ok(out)
    }

    /// Return a new DataFrame without the rows where the float column `name`
    /// is missing (NaN).
    ///
    /// This is the usual first step before fitting a model on field data
    /// where some plots have no phenotype.
    pub fn drop_missing(&self, name: &str) -> Result<DataFrame> {
        let values = self.get_float(name)?;
        let keep: Vec<bool> = values.iter().map(|v| !v.is_nan()).collect();
        self.filter_rows(&keep)
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

/// Format a numeric value as a factor level name: integral values are
/// printed without a fractional part (`3.0` -> `"3"`).
fn format_numeric_level(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Convert integer codes to a factor with numerically sorted levels.
fn integer_to_factor(vals: &[i64]) -> FactorColumn {
    let mut unique: Vec<i64> = vals.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let mut levels = IndexMap::new();
    for (i, &val) in unique.iter().enumerate() {
        levels.insert(val.to_string(), i);
    }

    let codes: Vec<usize> = vals.iter().map(|v| levels[&v.to_string()]).collect();
    FactorColumn::from_parts(levels, codes)
}

/// Convert numeric codes (e.g. block = 1, 2, 3 or animal IDs) to a factor
/// with numerically sorted levels named like the original values.
fn float_to_factor(name: &str, vals: &[f64]) -> Result<FactorColumn> {
    if let Some(pos) = vals.iter().position(|v| !v.is_finite()) {
        return Err(LmmError::Data(format!(
            "Cannot use Float column '{}' as a factor: missing value at row {}",
            name,
            pos + 1
        )));
    }
    let mut unique: Vec<f64> = vals.to_vec();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup();

    let mut levels = IndexMap::new();
    for (i, &val) in unique.iter().enumerate() {
        levels.insert(format_numeric_level(val), i);
    }

    let codes: Vec<usize> = vals
        .iter()
        .map(|v| levels[&format_numeric_level(*v)])
        .collect();
    Ok(FactorColumn::from_parts(levels, codes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column("yield", vec![5.2, 3.1, 4.7]).unwrap();
        df.add_integer_column("block", vec![1, 2, 1]).unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G1"])
            .unwrap();
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
    fn test_as_factor_from_float_codes() {
        let mut df = DataFrame::new();
        df.add_float_column("block", vec![3.0, 1.0, 2.0, 1.0, 3.0])
            .unwrap();
        df.as_factor("block").unwrap();

        let factor = df.get_factor("block").unwrap();
        assert_eq!(factor.n_levels(), 3);
        assert_eq!(factor.level_name(0), Some("1"));
        assert_eq!(factor.level_name(1), Some("2"));
        assert_eq!(factor.level_name(2), Some("3"));
        assert_eq!(factor.codes(), &[2, 0, 1, 0, 2]);
    }

    #[test]
    fn test_as_factor_float_with_missing_errors() {
        let mut df = DataFrame::new();
        df.add_float_column("block", vec![1.0, f64::NAN]).unwrap();
        let err = df.as_factor("block").unwrap_err();
        assert!(matches!(err, LmmError::Data(_)));
    }

    #[test]
    fn test_has_missing_and_drop_missing() {
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![1.0, f64::NAN, 3.0, f64::NAN])
            .unwrap();
        df.add_integer_column("block", vec![1, 2, 3, 4]).unwrap();
        df.add_factor_column("g", &["a", "b", "c", "b"]).unwrap();

        assert!(df.has_missing("y").unwrap());
        assert!(!df.has_missing("g").unwrap());

        let clean = df.drop_missing("y").unwrap();
        assert_eq!(clean.nrows(), 2);
        assert_eq!(clean.get_float("y").unwrap(), &[1.0, 3.0]);
        assert!(!clean.has_missing("y").unwrap());
        // Unused factor level "b" disappears.
        let g = clean.get_factor("g").unwrap();
        assert_eq!(g.n_levels(), 2);
        assert_eq!(g.level_name(0), Some("a"));
        assert_eq!(g.level_name(1), Some("c"));
        match clean.get_column("block").unwrap() {
            Column::Integer(v) => assert_eq!(v, &vec![1, 3]),
            _ => panic!("block should stay an Integer column"),
        }
    }

    #[test]
    fn test_factor_view_converts_numeric_columns() {
        let df = sample_df();
        let geno = df.factor_view("genotype").unwrap();
        assert!(matches!(geno, std::borrow::Cow::Borrowed(_)));
        let block = df.factor_view("block").unwrap();
        assert_eq!(block.n_levels(), 2);
        assert_eq!(block.level_names(), vec!["1", "2"]);
        assert_eq!(block.codes(), &[0, 1, 0]);
        let y = df.factor_view("yield").unwrap();
        assert_eq!(y.n_levels(), 3);
        assert_eq!(y.level_names(), vec!["3.1", "4.7", "5.2"]);
        // The DataFrame itself is unchanged.
        assert!(df.get_float("yield").is_ok());
    }

    #[test]
    fn test_filter_rows_wrong_length_errors() {
        let df = sample_df();
        let err = df.filter_rows(&[true, false]).unwrap_err();
        assert!(matches!(err, LmmError::DimensionMismatch { .. }));
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
