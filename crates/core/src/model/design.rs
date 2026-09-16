use std::collections::HashMap;

use sprs::TriMat;

use crate::data::{DataFrame, FactorColumn};
use crate::error::{LmmError, Result};
use crate::types::SparseMat;

/// Build a fixed-effects design matrix (X) from the DataFrame.
///
/// Fixed effects terms are parsed from a formula string like "mu + rep + block".
/// - "mu" (or "intercept") adds an intercept column (all ones).
/// - Factor columns are expanded into indicator (dummy) columns.
/// - Float columns are included as covariates (a single column of values).
///
/// Factors are coded with *treatment contrasts*, the same convention as R's
/// `model.matrix`, so that X has full column rank:
///
/// - When the model has an intercept, the first level of every factor is the
///   reference level and is dropped; the remaining coefficients are contrasts
///   against that reference.
/// - Without an intercept, the first factor is coded with one column per level
///   (its levels absorb the mean) and every subsequent factor drops its first
///   level.
///
/// Returns a sparse matrix of dimension (nobs x p) where p is the total number
/// of fixed-effect columns, together with one label per column.
pub fn build_fixed_design(
    df: &DataFrame,
    terms: &[FixedTerm],
) -> Result<(SparseMat, Vec<FixedEffectLabel>)> {
    let n = df.nrows();
    if n == 0 {
        return Err(LmmError::Data("DataFrame has no observations".into()));
    }

    let has_intercept = terms.iter().any(|t| matches!(t, FixedTerm::Intercept));

    // First pass: decide which factors drop their reference level, count the
    // total number of columns and collect labels.
    let mut mean_absorbed = has_intercept;
    let mut drop_first: Vec<bool> = Vec::with_capacity(terms.len());
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
                drop_first.push(false);
            }
            FixedTerm::Factor(col_name) => {
                let factor = df.factor_view(col_name)?;
                if factor.n_levels() == 0 {
                    return Err(LmmError::ModelSpec(format!(
                        "Fixed factor '{}' has no levels",
                        col_name
                    )));
                }
                let drop = mean_absorbed;
                mean_absorbed = true;
                let start = usize::from(drop);
                for (level_name, &code) in factor.levels().iter() {
                    if code < start {
                        continue;
                    }
                    labels.push(FixedEffectLabel {
                        term: col_name.clone(),
                        level: level_name.clone(),
                    });
                }
                total_cols += factor.n_levels() - start;
                drop_first.push(drop);
            }
            FixedTerm::Covariate(col_name) => {
                // Validate column exists
                df.get_float(col_name)?;
                labels.push(FixedEffectLabel {
                    term: col_name.clone(),
                    level: "covariate".to_string(),
                });
                total_cols += 1;
                drop_first.push(false);
            }
        }
    }

    if total_cols == 0 {
        return Err(LmmError::ModelSpec(
            "Fixed-effects design has no columns; add an intercept ('mu') or a factor".into(),
        ));
    }

    // Second pass: build the TriMat with known dimensions
    let mut tri = TriMat::new((n, total_cols));
    let mut col_offset = 0;

    for (term, &drop) in terms.iter().zip(drop_first.iter()) {
        match term {
            FixedTerm::Intercept => {
                for i in 0..n {
                    tri.add_triplet(i, col_offset, 1.0);
                }
                col_offset += 1;
            }
            FixedTerm::Factor(col_name) => {
                let factor = df.factor_view(col_name)?;
                let start = usize::from(drop);
                for (i, &code) in factor.codes().iter().enumerate() {
                    if code >= start {
                        tri.add_triplet(i, col_offset + code - start, 1.0);
                    }
                }
                col_offset += factor.n_levels() - start;
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
/// For observation i with factor code k, Z[i, k] = 1. Numeric columns are
/// treated as factors (see [`DataFrame::factor_view`]).
///
/// Returns a sparse matrix of dimension (nobs x q) where q is the number of levels.
pub fn build_random_design(df: &DataFrame, column: &str) -> Result<(SparseMat, Vec<String>)> {
    let n = df.nrows();
    let factor = df.factor_view(column)?;
    let q = factor.n_levels();

    let mut tri = TriMat::new((n, q));
    for (i, &code) in factor.codes().iter().enumerate() {
        tri.add_triplet(i, code, 1.0);
    }

    let level_names: Vec<String> = factor.levels().keys().cloned().collect();

    Ok((tri.to_csc(), level_names))
}

/// Build a random-effects design matrix (Z) whose columns follow an
/// externally supplied level ordering.
///
/// This is required whenever the random term carries a relationship matrix
/// (for example a pedigree A⁻¹ or a genomic G⁻¹) whose row/column order is
/// defined outside the data: column `j` of Z corresponds to `levels[j]`.
/// Levels without any observation (e.g. ancestors without records) simply
/// produce an empty column, which is exactly what an animal model needs.
///
/// # Errors
/// Returns an error if `levels` contains duplicates or if an observation's
/// level is not present in `levels`.
pub fn build_random_design_with_levels(
    df: &DataFrame,
    column: &str,
    levels: &[String],
) -> Result<(SparseMat, Vec<String>)> {
    let n = df.nrows();
    let factor = df.factor_view(column)?;

    let mut index: HashMap<&str, usize> = HashMap::with_capacity(levels.len());
    for (j, level) in levels.iter().enumerate() {
        if index.insert(level.as_str(), j).is_some() {
            return Err(LmmError::ModelSpec(format!(
                "Duplicate level '{}' in the level list supplied for random term '{}'",
                level, column
            )));
        }
    }

    // Map each factor code to a column of Z (None if the level is unknown).
    let mut code_to_col: Vec<Option<usize>> = vec![None; factor.n_levels()];
    for (name, &code) in factor.levels().iter() {
        code_to_col[code] = index.get(name.as_str()).copied();
    }

    let mut tri = TriMat::new((n, levels.len()));
    for (i, &code) in factor.codes().iter().enumerate() {
        match code_to_col[code] {
            Some(col) => tri.add_triplet(i, col, 1.0),
            None => {
                let name = factor.level_name(code).unwrap_or("?");
                return Err(LmmError::ModelSpec(format!(
                    "Level '{}' of random term '{}' (row {}) is not among the supplied \
                     levels (e.g. the animal is missing from the pedigree)",
                    name,
                    column,
                    i + 1
                )));
            }
        }
    }

    Ok((tri.to_csc(), levels.to_vec()))
}

/// Build the design matrix of an interaction random term `a:b` whose levels
/// are all combinations of `levels_a` and `levels_b`, ordered `a`-major
/// (column index `ia * levels_b.len() + ib`). This is the ordering of a
/// Kronecker covariance `Sigma_a ⊗ Sigma_b`.
///
/// Returns the sparse incidence matrix and the combined level labels
/// `"a_level:b_level"`. Every observed combination must be covered by the
/// supplied levels; combinations without observations get empty columns.
pub fn build_random_design_interaction(
    df: &DataFrame,
    col_a: &str,
    levels_a: &[String],
    col_b: &str,
    levels_b: &[String],
) -> Result<(SparseMat, Vec<String>)> {
    let n = df.nrows();
    let fa = df.factor_view(col_a)?;
    let fb = df.factor_view(col_b)?;

    let index_of = |levels: &[String], col: &str| -> Result<HashMap<String, usize>> {
        let mut index = HashMap::with_capacity(levels.len());
        for (j, level) in levels.iter().enumerate() {
            if index.insert(level.clone(), j).is_some() {
                return Err(LmmError::ModelSpec(format!(
                    "Duplicate level '{}' in the level list for '{}'",
                    level, col
                )));
            }
        }
        Ok(index)
    };
    let index_a = index_of(levels_a, col_a)?;
    let index_b = index_of(levels_b, col_b)?;

    let map_codes =
        |f: &FactorColumn, index: &HashMap<String, usize>, col: &str| -> Result<Vec<usize>> {
            f.level_names()
                .iter()
                .map(|name| {
                    index.get(*name).copied().ok_or_else(|| {
                        LmmError::ModelSpec(format!(
                            "Level '{}' of '{}' is not among the supplied levels",
                            name, col
                        ))
                    })
                })
                .collect()
        };
    let code_a = map_codes(&fa, &index_a, col_a)?;
    let code_b = map_codes(&fb, &index_b, col_b)?;

    let qb = levels_b.len();
    let mut tri = TriMat::new((n, levels_a.len() * qb));
    for i in 0..n {
        let ia = code_a[fa.codes()[i]];
        let ib = code_b[fb.codes()[i]];
        tri.add_triplet(i, ia * qb + ib, 1.0);
    }

    let mut labels = Vec::with_capacity(levels_a.len() * qb);
    for la in levels_a {
        for lb in levels_b {
            labels.push(format!("{}:{}", la, lb));
        }
    }
    Ok((tri.to_csc(), labels))
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
    /// A continuous covariate (single column of values).
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
        df.add_float_column("yield", vec![5.0, 3.0, 7.0, 4.0])
            .unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G1", "G3"])
            .unwrap();
        df.add_factor_column("rep", &["R1", "R2", "R1", "R2"])
            .unwrap();
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
        assert_eq!(x.cols(), 2); // R1, R2 (no intercept: full coding)
        assert_eq!(labels.len(), 2);

        // Row 0 (R1): [1, 0]
        // Row 1 (R2): [0, 1]
        // Row 2 (R1): [1, 0]
        // Row 3 (R2): [0, 1]
        let result = spmv(&x, &[1.0, 0.0]);
        assert_eq!(result, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_build_fixed_intercept_and_factor_uses_treatment_contrasts() {
        let df = sample_df();
        let terms = vec![FixedTerm::Intercept, FixedTerm::Factor("rep".to_string())];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        assert_eq!(x.rows(), 4);
        // intercept + (2 rep levels - 1 reference level)
        assert_eq!(x.cols(), 2);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].term, "mu");
        assert_eq!(labels[1].term, "rep");
        assert_eq!(labels[1].level, "R2"); // R1 is the reference

        // Column 1 is the R2 indicator: rows 1 and 3
        let result = spmv(&x, &[0.0, 1.0]);
        assert_eq!(result, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_build_fixed_two_factors_without_intercept_is_full_rank() {
        let df = sample_df();
        let terms = vec![
            FixedTerm::Factor("rep".to_string()),
            FixedTerm::Factor("genotype".to_string()),
        ];
        let (x, labels) = build_fixed_design(&df, &terms).unwrap();
        // rep: 2 columns (absorbs the mean); genotype: 3 - 1 = 2 columns
        assert_eq!(x.cols(), 4);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[2].level, "G2");
        assert_eq!(labels[3].level, "G3");

        // X'X must be non-singular (full column rank).
        let xtx = crate::matrix::sparse::xtx_dense(&x);
        assert!(xtx.cholesky().is_some(), "X'X should be positive definite");
    }

    #[test]
    fn test_build_fixed_no_columns_errors() {
        let df = sample_df();
        let result = build_fixed_design(&df, &[]);
        assert!(result.is_err());
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
    fn test_build_random_design_from_numeric_column() {
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        df.add_float_column("animal", vec![10.0, 12.0, 10.0, 11.0])
            .unwrap();
        let (z, levels) = build_random_design(&df, "animal").unwrap();
        assert_eq!(z.cols(), 3);
        assert_eq!(levels, vec!["10", "11", "12"]);
        let result = spmv(&z, &[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![1.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_build_random_design_with_levels() {
        let df = sample_df();
        // External ordering with an extra level that has no records (like a
        // base animal in a pedigree).
        let levels: Vec<String> = ["G3", "G0", "G1", "G2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (z, out_levels) = build_random_design_with_levels(&df, "genotype", &levels).unwrap();
        assert_eq!(z.rows(), 4);
        assert_eq!(z.cols(), 4);
        assert_eq!(out_levels, levels);

        // Column order follows `levels`: G3=0, G0=1, G1=2, G2=3
        let result = spmv(&z, &[30.0, 0.0, 10.0, 20.0]);
        assert_eq!(result, vec![10.0, 20.0, 10.0, 30.0]);

        // The G0 column is empty.
        let col_sum = spmv(&z, &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(col_sum, vec![0.0; 4]);
    }

    #[test]
    fn test_build_random_design_with_levels_missing_level_errors() {
        let df = sample_df();
        let levels: Vec<String> = ["G1", "G2"].iter().map(|s| s.to_string()).collect();
        let err = build_random_design_with_levels(&df, "genotype", &levels).unwrap_err();
        assert!(matches!(err, LmmError::ModelSpec(_)));
        assert!(err.to_string().contains("G3"));
    }

    #[test]
    fn test_build_random_design_with_levels_duplicate_errors() {
        let df = sample_df();
        let levels: Vec<String> = ["G1", "G2", "G3", "G1"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let err = build_random_design_with_levels(&df, "genotype", &levels).unwrap_err();
        assert!(matches!(err, LmmError::ModelSpec(_)));
    }

    #[test]
    fn test_build_random_design_interaction() {
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        df.add_factor_column("env", &["E1", "E1", "E2", "E2"])
            .unwrap();
        df.add_factor_column("geno", &["G2", "G1", "G2", "G1"])
            .unwrap();
        let envs: Vec<String> = vec!["E1".into(), "E2".into()];
        let genos: Vec<String> = vec!["G1".into(), "G2".into(), "G3".into()];
        let (z, labels) =
            build_random_design_interaction(&df, "env", &envs, "geno", &genos).unwrap();
        assert_eq!(z.cols(), 6);
        assert_eq!(labels[0], "E1:G1");
        assert_eq!(labels[5], "E2:G3");
        // obs 0: E1:G2 -> column 0*3+1 = 1; obs 3: E2:G1 -> column 1*3+0 = 3
        let result = spmv(&z, &[0.0, 10.0, 0.0, 20.0, 0.0, 0.0]);
        assert_eq!(result, vec![10.0, 0.0, 0.0, 20.0]);
        // missing level errors
        let bad: Vec<String> = vec!["G1".into()];
        assert!(build_random_design_interaction(&df, "env", &envs, "geno", &bad).is_err());
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
