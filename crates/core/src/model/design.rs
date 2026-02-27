use sprs::TriMat;

use crate::data::{DataFrame, FactorColumn};
use crate::error::{LmmError, Result};
use crate::types::SparseMat;

/// Build a fixed-effects design matrix (X) from the DataFrame.
///
/// Fixed effects terms are parsed from a formula string like "mu + rep + block".
/// - "mu" (or "intercept") adds an intercept column (all ones).
/// - Other terms refer to factor columns, which are expanded into dummy variables
///   (one column per level).
///
/// Returns a sparse matrix of dimension (nobs x p) where p is the total number
/// of fixed-effect columns.
pub fn build_fixed_design(
    df: &DataFrame,
    terms: &[FixedTerm],
) -> Result<(SparseMat, Vec<FixedEffectLabel>)> {
    let n = df.nrows();
    if n == 0 {
        return Err(LmmError::Data("DataFrame has no observations".into()));
    }

    // First pass: compute total number of columns and collect labels
    let mut total_cols = 0;
    let mut labels = Vec::new();

    for term in terms {
        match term {
            FixedTerm::Intercept => {
                labels.push(FixedEffectLabel {
                    term: "mu".to_string(),
                    level: "intercept".to_string(),
                });
                total_cols += 1;
            }
            FixedTerm::Factor(col_name) => {
                let factor = df.get_factor(col_name)?;
                for (level_name, _) in factor.levels().iter() {
                    labels.push(FixedEffectLabel {
                        term: col_name.clone(),
                        level: level_name.clone(),
                    });
                }
                total_cols += factor.n_levels();
            }
            FixedTerm::Covariate(col_name) => {
                // Validate column exists
                df.get_float(col_name)?;
                labels.push(FixedEffectLabel {
                    term: col_name.clone(),
                    level: "covariate".to_string(),
                });
                total_cols += 1;
            }
        }
    }

    // Second pass: build the TriMat with known dimensions
    let mut tri = TriMat::new((n, total_cols));
    let mut col_offset = 0;

    for term in terms {
        match term {
            FixedTerm::Intercept => {
                for i in 0..n {
                    tri.add_triplet(i, col_offset, 1.0);
                }
                col_offset += 1;
            }
            FixedTerm::Factor(col_name) => {
                let factor = df.get_factor(col_name)?;
                for (i, &code) in factor.codes().iter().enumerate() {
                    tri.add_triplet(i, col_offset + code, 1.0);
                }
                col_offset += factor.n_levels();
            }
            FixedTerm::Covariate(col_name) => {
                let values = df.get_float(col_name)?;
                for (i, &val) in values.iter().enumerate() {
                    if val != 0.0 {
                        tri.add_triplet(i, col_offset, val);
                    }
                }
                col_offset += 1;
            }
        }
    }

    Ok((tri.to_csc(), labels))
}

/// Build a random-effects design matrix (Z) for a single random term.
///
/// Z is an incidence matrix mapping observations to factor levels.
/// For observation i with factor code k, Z[i, k] = 1.
///
/// Returns a sparse matrix of dimension (nobs x q) where q is the number of levels.
pub fn build_random_design(df: &DataFrame, column: &str) -> Result<(SparseMat, Vec<String>)> {
    let n = df.nrows();
    let factor = df.get_factor(column)?;
    let q = factor.n_levels();

    let mut tri = TriMat::new((n, q));
    for (i, &code) in factor.codes().iter().enumerate() {
        tri.add_triplet(i, code, 1.0);
    }

    let level_names: Vec<String> = factor
        .levels()
        .keys()
        .cloned()
        .collect();

    Ok((tri.to_csc(), level_names))
}

/// Build the block-diagonal Z matrix from multiple random terms.
/// Z = [Z1 | Z2 | ... | Zk] (horizontal concatenation).
pub fn build_combined_random_design(z_blocks: &[SparseMat], n: usize) -> SparseMat {
    let total_cols: usize = z_blocks.iter().map(|z| z.cols()).sum();
    let mut tri = TriMat::new((n, total_cols));
    let mut col_offset = 0;

    for z in z_blocks {
        for (val, (row, col)) in z.iter() {
            tri.add_triplet(row, col_offset + col, *val);
        }
        col_offset += z.cols();
    }

    tri.to_csc()
}

/// Parse a formula string like "mu + rep + block" into fixed terms.
pub fn parse_fixed_formula(formula: &str, df: &DataFrame) -> Result<Vec<FixedTerm>> {
    let mut terms = Vec::new();

    for part in formula.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        match part.to_lowercase().as_str() {
            "mu" | "intercept" | "1" => {
                terms.push(FixedTerm::Intercept);
            }
            _ => {
                // Check if it's a factor or float column
                match df.get_column(part)? {
                    crate::data::Column::Factor(_) => {
                        terms.push(FixedTerm::Factor(part.to_string()));
                    }
                    crate::data::Column::Float(_) => {
                        terms.push(FixedTerm::Covariate(part.to_string()));
                    }
                    crate::data::Column::Integer(_) => {
                        // Treat integers as factors for fixed effects
                        terms.push(FixedTerm::Factor(part.to_string()));
                    }
                }
            }
        }
    }

    Ok(terms)
}

/// A single fixed-effect term in the model.
#[derive(Debug, Clone)]
pub enum FixedTerm {
    /// An intercept (column of ones).
    Intercept,
    /// A factor column (expanded to dummy variables).
    Factor(String),
    /// A continuous covariate.
    Covariate(String),
}

/// Label for a single column in the fixed-effects design matrix.
#[derive(Debug, Clone)]
pub struct FixedEffectLabel {
    pub term: String,
    pub level: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFrame;
    use crate::matrix::sparse::spmv;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column("yield", vec![5.0, 3.0, 7.0, 4.0]).unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G1", "G3"]).unwrap();
        df.add_factor_column("rep", &["R1", "R2", "R1", "R2"]).unwrap();
        df
    }

    #[test]
    fn test_build_fixed_intercept() {
        let df = sample_df();
        let terms = vec![FixedTerm::Intercept];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        assert_eq!(x.rows(), 4);
        assert_eq!(x.cols(), 1);
        // All ones
        let result = spmv(&x, &[1.0]);
        assert_eq!(result, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn test_build_fixed_factor() {
        let df = sample_df();
        let terms = vec![FixedTerm::Factor("rep".to_string())];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        assert_eq!(x.rows(), 4);
        assert_eq!(x.cols(), 2); // R1, R2
        assert_eq!(labels.len(), 2);

        // Row 0 (R1): [1, 0]
        // Row 1 (R2): [0, 1]
        // Row 2 (R1): [1, 0]
        // Row 3 (R2): [0, 1]
        let result = spmv(&x, &[1.0, 0.0]);
        assert_eq!(result, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_build_fixed_intercept_and_factor() {
        let df = sample_df();
        let terms = vec![
            FixedTerm::Intercept,
            FixedTerm::Factor("rep".to_string()),
        ];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        assert_eq!(x.rows(), 4);
        assert_eq!(x.cols(), 3); // intercept + 2 rep levels
        assert_eq!(labels.len(), 3);
    }

    #[test]
    fn test_build_random_design() {
        let df = sample_df();
        let (z, levels) = build_random_design(&df, "genotype").unwrap();
        assert_eq!(z.rows(), 4);
        assert_eq!(z.cols(), 3); // G1, G2, G3
        assert_eq!(levels, vec!["G1", "G2", "G3"]);

        // Row 0 (G1): [1, 0, 0]
        // Row 1 (G2): [0, 1, 0]
        // Row 2 (G1): [1, 0, 0]
        // Row 3 (G3): [0, 0, 1]
        let result = spmv(&z, &[10.0, 20.0, 30.0]);
        assert_eq!(result, vec![10.0, 20.0, 10.0, 30.0]);
    }

    #[test]
    fn test_combined_random_design() {
        let df = sample_df();
        let (z1, _) = build_random_design(&df, "genotype").unwrap();
        let (z2, _) = build_random_design(&df, "rep").unwrap();
        let z = build_combined_random_design(&[z1, z2], 4);
        assert_eq!(z.rows(), 4);
        assert_eq!(z.cols(), 5); // 3 genotype + 2 rep levels
    }

    #[test]
    fn test_parse_fixed_formula() {
        let df = sample_df();
        let terms = parse_fixed_formula("mu + rep", &df).unwrap();
        assert_eq!(terms.len(), 2);
        assert!(matches!(&terms[0], FixedTerm::Intercept));
        assert!(matches!(&terms[1], FixedTerm::Factor(name) if name == "rep"));
    }

    #[test]
    fn test_build_fixed_covariate() {
        let df = sample_df();
        let terms = vec![FixedTerm::Covariate("yield".to_string())];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        assert_eq!(x.rows(), 4);
        assert_eq!(x.cols(), 1);
        let result = spmv(&x, &[2.0]);
        assert_eq!(result, vec![10.0, 6.0, 14.0, 8.0]);
        assert_eq!(labels.len(), 1);
    }
}
