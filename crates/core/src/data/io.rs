use std::path::Path;

use crate::error::{LmmError, Result};

use super::dataframe::{Column, DataFrame};
use super::factor::FactorColumn;

impl DataFrame {
    /// Read a CSV file into a DataFrame.
    ///
    /// The first row is treated as a header. Each column is auto-detected:
    /// - If **every** value in the column parses as `f64`, it becomes a `Float` column.
    /// - Otherwise it becomes a `Factor` column (categorical).
    ///
    /// Integer columns are stored as `Float` because CSV numeric detection
    /// uses `f64::parse`; callers can convert to `Integer` or `Factor` as needed
    /// with [`DataFrame::as_factor`].
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened, if the CSV is malformed,
    /// or if rows have inconsistent numbers of fields.
    ///
    /// # Examples
    /// ```no_run
    /// use plant_breeding_lmm_core::data::DataFrame;
    ///
    /// let df = DataFrame::from_csv("trial_data.csv").unwrap();
    /// println!("rows = {}, cols = {}", df.nrows(), df.ncols());
    /// ```
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .trim(csv::Trim::All)
            .from_path(path)?;

        // Collect headers.
        let headers: Vec<String> = reader
            .headers()?
            .iter()
            .map(|h| h.to_string())
            .collect();

        if headers.is_empty() {
            return Ok(DataFrame::new());
        }

        let ncols = headers.len();

        // Read all records into column-oriented vectors of strings.
        let mut string_columns: Vec<Vec<String>> = vec![Vec::new(); ncols];

        for result in reader.records() {
            let record = result?;
            if record.len() != ncols {
                return Err(LmmError::Data(format!(
                    "Row has {} fields but header has {} columns",
                    record.len(),
                    ncols
                )));
            }
            for (i, field) in record.iter().enumerate() {
                string_columns[i].push(field.to_string());
            }
        }

        // Determine the number of rows.
        let nrows = if ncols > 0 {
            string_columns[0].len()
        } else {
            0
        };

        if nrows == 0 {
            return Ok(DataFrame::new());
        }

        // Auto-detect column types and build the DataFrame.
        let mut df = DataFrame::new();

        for (col_idx, header) in headers.iter().enumerate() {
            let raw = &string_columns[col_idx];

            // Try to parse every value as f64.
            let float_values: std::result::Result<Vec<f64>, _> =
                raw.iter().map(|s| s.parse::<f64>()).collect();

            match float_values {
                Ok(values) => {
                    df.add_float_column(header, values)?;
                }
                Err(_) => {
                    // Fall back to factor column.
                    let str_refs: Vec<&str> = raw.iter().map(|s| s.as_str()).collect();
                    let factor = FactorColumn::new(&str_refs);
                    // We need to insert via the internal path since add_factor_column
                    // takes &[&str] and rebuilds the factor; reuse the one we built.
                    df.insert_column(header, Column::Factor(factor))?;
                }
            }
        }

        Ok(df)
    }

    /// Internal helper: insert a pre-built column (used by CSV reader).
    fn insert_column(&mut self, name: &str, column: Column) -> Result<()> {
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
        let file_name = format!("test_plant_lmm_{}_{}.csv", std::process::id(), id);
        let path = dir.join(file_name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_from_csv_basic() {
        let csv = "genotype,block,yield\nG1,1,5.2\nG2,2,3.1\nG1,1,4.7\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 3);

        // "genotype" is non-numeric -> Factor
        let geno = df.get_factor("genotype").unwrap();
        assert_eq!(geno.n_levels(), 2);
        assert_eq!(geno.codes(), &[0, 1, 0]);

        // "block" is all-numeric -> Float (stored as f64)
        let block = df.get_float("block").unwrap();
        assert_eq!(block, &[1.0, 2.0, 1.0]);

        // "yield" is all-numeric -> Float
        let yields = df.get_float("yield").unwrap();
        assert_eq!(yields, &[5.2, 3.1, 4.7]);
    }

    #[test]
    fn test_from_csv_all_string_columns() {
        let csv = "color,shape\nred,circle\nblue,square\nred,triangle\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);

        let color = df.get_factor("color").unwrap();
        assert_eq!(color.n_levels(), 2);
        assert_eq!(color.level_name(0), Some("red"));
        assert_eq!(color.level_name(1), Some("blue"));

        let shape = df.get_factor("shape").unwrap();
        assert_eq!(shape.n_levels(), 3);
    }

    #[test]
    fn test_from_csv_empty_body() {
        let csv = "a,b,c\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // Header only, no data rows.
        assert_eq!(df.nrows(), 0);
        assert_eq!(df.ncols(), 0);
    }

    #[test]
    fn test_from_csv_mixed_makes_factor() {
        // A column with mixed numeric and string values should become Factor.
        let csv = "id,value\n1,10.5\n2,NA\n3,7.3\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        // "id" is all numeric -> Float
        assert!(df.get_float("id").is_ok());

        // "value" has "NA" -> not all f64 -> Factor
        let val = df.get_factor("value").unwrap();
        assert_eq!(val.n_levels(), 3); // "10.5", "NA", "7.3"
    }

    #[test]
    fn test_from_csv_file_not_found() {
        let result = DataFrame::from_csv("/nonexistent/path/data.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_csv_single_column() {
        let csv = "trait\n1.1\n2.2\n3.3\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 1);
        assert_eq!(df.get_float("trait").unwrap(), &[1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_from_csv_whitespace_trimmed() {
        let csv = "name , score\n  Alice , 95 \n  Bob , 88 \n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(df.column_names(), vec!["name", "score"]);
        let name = df.get_factor("name").unwrap();
        assert_eq!(name.level_name(0), Some("Alice"));
        assert_eq!(name.level_name(1), Some("Bob"));
        assert_eq!(df.get_float("score").unwrap(), &[95.0, 88.0]);
    }

    #[test]
    fn test_from_csv_negative_and_scientific() {
        let csv = "x,y\n-1.5,3e2\n0,1.2e-3\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let x = df.get_float("x").unwrap();
        assert_eq!(x, &[-1.5, 0.0]);
        let y = df.get_float("y").unwrap();
        assert!((y[0] - 300.0).abs() < 1e-10);
        assert!((y[1] - 0.0012).abs() < 1e-10);
    }

    #[test]
    fn test_from_csv_column_names_order() {
        let csv = "z,a,m\n1,2,3\n";
        let path = write_temp_csv(csv);
        let df = DataFrame::from_csv(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(df.column_names(), vec!["z", "a", "m"]);
    }
}
