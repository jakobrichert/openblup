use crate::data::DataFrame;
use crate::error::{LmmError, Result};
use crate::genetics::{compute_a_inverse_with_inbreeding, Pedigree};
use crate::types::SparseMat;
use crate::variance::VarStruct;

use super::design::{
    build_combined_random_design, build_fixed_design, build_random_design,
    build_random_design_with_levels, parse_fixed_formula, FixedEffectLabel, FixedTerm,
};

/// A fully specified mixed model, ready for fitting.
pub struct MixedModel {
    /// Number of observations.
    pub n_obs: usize,
    /// Response vector (y).
    pub y: Vec<f64>,
    /// Fixed effects design matrix (X).
    pub x: SparseMat,
    /// Labels for fixed effect columns.
    pub fixed_labels: Vec<FixedEffectLabel>,
    /// Random effects design matrices, one per random term.
    pub z_blocks: Vec<SparseMat>,
    /// Combined random effects design matrix [Z1 | Z2 | ...].
    pub z_combined: SparseMat,
    /// Level names for each random term.
    pub random_level_names: Vec<Vec<String>>,
    /// Variance structures for each random term.
    pub random_var_structs: Vec<Box<dyn VarStruct>>,
    /// Relationship matrix inverses for each random term (None = use I).
    pub ginv_matrices: Vec<Option<SparseMat>>,
    /// Random term names (column names).
    pub random_term_names: Vec<String>,
    /// Residual variance structure.
    pub residual_var_struct: Box<dyn VarStruct>,
    /// REML configuration.
    pub max_iter: usize,
    pub convergence_tol: f64,
}

/// Builder for constructing a [`MixedModel`].
///
/// ```no_run
/// use plant_breeding_lmm_core::data::DataFrame;
/// use plant_breeding_lmm_core::genetics::Pedigree;
/// use plant_breeding_lmm_core::model::MixedModelBuilder;
/// use plant_breeding_lmm_core::variance::Identity;
///
/// # fn main() -> plant_breeding_lmm_core::Result<()> {
/// let df = DataFrame::from_csv("trial.csv")?;
/// let ped = Pedigree::from_csv("pedigree.csv")?;
///
/// let mut model = MixedModelBuilder::new()
///     .data(&df)
///     .response("yield")
///     .fixed("mu + rep")
///     .random_pedigree("animal", Identity::new(1.0), &ped)
///     .build()?;
/// let result = model.fit_reml()?;
/// println!("{}", result.summary());
/// # Ok(())
/// # }
/// ```
pub struct MixedModelBuilder<'a> {
    data: Option<&'a DataFrame>,
    response: Option<String>,
    fixed_formula: Option<String>,
    fixed_terms: Vec<FixedTerm>,
    random_terms: Vec<RandomTermSpec<'a>>,
    residual_structure: Option<Box<dyn VarStruct>>,
    max_iter: usize,
    convergence_tol: f64,
}

/// Where the levels (columns of Z) of a random term come from.
enum RandomLevels<'a> {
    /// Levels are the distinct values observed in the data column.
    FromData,
    /// Levels are supplied explicitly (must cover every observed value).
    Explicit(Vec<String>),
    /// Levels are the animals of a pedigree, in sorted pedigree order, and
    /// the term uses the pedigree A⁻¹ as its relationship matrix inverse.
    Pedigree(&'a Pedigree),
}

struct RandomTermSpec<'a> {
    column: String,
    variance_structure: Box<dyn VarStruct>,
    ginv: Option<SparseMat>,
    levels: RandomLevels<'a>,
}

impl<'a> MixedModelBuilder<'a> {
    /// Create a new builder with sensible defaults.
    pub fn new() -> Self {
        Self {
            data: None,
            response: None,
            fixed_formula: None,
            fixed_terms: Vec::new(),
            random_terms: Vec::new(),
            residual_structure: None,
            max_iter: 50,
            convergence_tol: 1e-6,
        }
    }

    /// Set the data source.
    pub fn data(mut self, df: &'a DataFrame) -> Self {
        self.data = Some(df);
        self
    }

    /// Set the response variable (column name).
    pub fn response(mut self, col: &str) -> Self {
        self.response = Some(col.to_string());
        self
    }

    /// Set the fixed effects formula (e.g., "mu + rep + block").
    ///
    /// Factors are coded with treatment contrasts (see
    /// [`build_fixed_design`]), so `"mu + rep"` estimates an intercept plus
    /// the contrasts of every rep level against the first one.
    pub fn fixed(mut self, formula: &str) -> Self {
        self.fixed_formula = Some(formula.to_string());
        self
    }

    /// Set the fixed effects as explicit terms instead of a formula string.
    pub fn fixed_terms(mut self, terms: Vec<FixedTerm>) -> Self {
        self.fixed_terms = terms;
        self
    }

    /// Add a random effect term whose levels are the distinct values found
    /// in the data column.
    ///
    /// - `column`: the grouping factor column name
    /// - `vs`: the variance structure for this random term
    /// - `ginv`: optional relationship matrix inverse (e.g., A⁻¹ or G⁻¹). Its
    ///   row/column order must match the order of first appearance of the
    ///   levels in the data; use [`random_with_levels`](Self::random_with_levels)
    ///   or [`random_pedigree`](Self::random_pedigree) when the matrix has its
    ///   own ordering.
    pub fn random(
        mut self,
        column: &str,
        vs: impl VarStruct + 'static,
        ginv: Option<SparseMat>,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: column.to_string(),
            variance_structure: Box::new(vs),
            ginv,
            levels: RandomLevels::FromData,
        });
        self
    }

    /// Add a random effect term with an explicit level ordering.
    ///
    /// Column `j` of Z (and row/column `j` of `ginv`) corresponds to
    /// `levels[j]`. Levels that never occur in the data are allowed and get
    /// an empty column, which is how ancestors without records enter an
    /// animal model. Every value observed in the data column must appear in
    /// `levels`.
    pub fn random_with_levels(
        mut self,
        column: &str,
        vs: impl VarStruct + 'static,
        ginv: Option<SparseMat>,
        levels: Vec<String>,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: column.to_string(),
            variance_structure: Box::new(vs),
            ginv,
            levels: RandomLevels::Explicit(levels),
        });
        self
    }

    /// Add a pedigree-based random effect (animal model).
    ///
    /// The levels of the term are all animals of `pedigree` in topologically
    /// sorted order, and the relationship matrix inverse is A⁻¹ computed with
    /// Henderson's rules including inbreeding (Meuwissen & Luo 1992). The
    /// pedigree is sorted on the fly if needed. Every value in `column` must
    /// be an animal of the pedigree.
    pub fn random_pedigree(
        mut self,
        column: &str,
        vs: impl VarStruct + 'static,
        pedigree: &'a Pedigree,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: column.to_string(),
            variance_structure: Box::new(vs),
            ginv: None,
            levels: RandomLevels::Pedigree(pedigree),
        });
        self
    }

    /// Set the residual variance structure.
    /// If not called, defaults to Identity (homogeneous residual variance).
    pub fn residual(mut self, vs: impl VarStruct + 'static) -> Self {
        self.residual_structure = Some(Box::new(vs));
        self
    }

    /// Set maximum REML iterations (default: 50).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance (default: 1e-6).
    pub fn convergence(mut self, tol: f64) -> Self {
        self.convergence_tol = tol;
        self
    }

    /// Build the model. Validates all inputs and constructs design matrices.
    pub fn build(self) -> Result<MixedModel> {
        let df = self
            .data
            .ok_or_else(|| LmmError::ModelSpec("No data provided".into()))?;

        let response_col = self
            .response
            .ok_or_else(|| LmmError::ModelSpec("No response variable specified".into()))?;

        let n = df.nrows();
        if n == 0 {
            return Err(LmmError::ModelSpec("DataFrame is empty".into()));
        }

        // Get response vector
        let y = df.get_float(&response_col)?.to_vec();
        if let Some(pos) = y.iter().position(|v| !v.is_finite()) {
            return Err(LmmError::Data(format!(
                "Response '{}' has a missing or non-finite value at row {}; \
                 remove such rows first (see DataFrame::drop_missing)",
                response_col,
                pos + 1
            )));
        }

        // Build fixed effects design matrix
        let fixed_terms = if let Some(ref formula) = self.fixed_formula {
            parse_fixed_formula(formula, df)?
        } else if !self.fixed_terms.is_empty() {
            self.fixed_terms
        } else {
            // Default: intercept only
            vec![FixedTerm::Intercept]
        };

        let (x, fixed_labels) = build_fixed_design(df, &fixed_terms)?;

        // Build random effects design matrices
        let mut z_blocks = Vec::new();
        let mut random_level_names = Vec::new();
        let mut random_var_structs = Vec::new();
        let mut ginv_matrices = Vec::new();
        let mut random_term_names = Vec::new();

        for rt in self.random_terms {
            let (z, levels, ginv) = match rt.levels {
                RandomLevels::FromData => {
                    let (z, levels) = build_random_design(df, &rt.column)?;
                    (z, levels, rt.ginv)
                }
                RandomLevels::Explicit(levels) => {
                    let (z, levels) = build_random_design_with_levels(df, &rt.column, &levels)?;
                    (z, levels, rt.ginv)
                }
                RandomLevels::Pedigree(ped) => {
                    let sorted;
                    let ped_ref: &Pedigree = if ped.is_sorted() {
                        ped
                    } else {
                        let mut p = ped.clone();
                        p.sort_pedigree()?;
                        sorted = p;
                        &sorted
                    };
                    let ids: Vec<String> = (0..ped_ref.n_animals())
                        .map(|i| ped_ref.animal_id(i).to_string())
                        .collect();
                    let a_inv = compute_a_inverse_with_inbreeding(ped_ref)?;
                    let (z, levels) = build_random_design_with_levels(df, &rt.column, &ids)?;
                    (z, levels, Some(a_inv))
                }
            };

            // Validate ginv dimensions if provided
            if let Some(ref ginv) = ginv {
                let q = levels.len();
                if ginv.rows() != q || ginv.cols() != q {
                    return Err(LmmError::DimensionMismatch {
                        expected: q,
                        got: ginv.rows(),
                        context: format!(
                            "G-inverse for '{}' should be {}x{} but is {}x{}",
                            rt.column,
                            q,
                            q,
                            ginv.rows(),
                            ginv.cols()
                        ),
                    });
                }
            }

            z_blocks.push(z);
            random_level_names.push(levels);
            random_var_structs.push(rt.variance_structure);
            ginv_matrices.push(ginv);
            random_term_names.push(rt.column);
        }

        let z_combined = build_combined_random_design(&z_blocks, n);

        // Residual structure defaults to Identity
        let residual_var_struct = self
            .residual_structure
            .unwrap_or_else(|| Box::new(crate::variance::Identity::default()));

        Ok(MixedModel {
            n_obs: n,
            y,
            x,
            fixed_labels,
            z_blocks,
            z_combined,
            random_level_names,
            random_var_structs,
            ginv_matrices,
            random_term_names,
            residual_var_struct,
            max_iter: self.max_iter,
            convergence_tol: self.convergence_tol,
        })
    }
}

impl<'a> Default for MixedModelBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedModel {
    /// Convenience method: fit the model using AI-REML.
    pub fn fit_reml(&mut self) -> Result<crate::lmm::FitResult> {
        let reml = crate::lmm::AiReml::new(self.max_iter, self.convergence_tol);
        reml.fit(self)
    }

    /// Convenience method: fit the model using EM-REML (slower but very robust).
    pub fn fit_em_reml(&mut self) -> Result<crate::lmm::FitResult> {
        let reml = crate::lmm::EmReml::new(self.max_iter, self.convergence_tol);
        reml.fit(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFrame;
    use crate::variance::Identity;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column("yield", vec![5.0, 3.0, 7.0, 4.0, 6.0, 2.0])
            .unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G3", "G1", "G2", "G3"])
            .unwrap();
        df.add_factor_column("rep", &["R1", "R1", "R1", "R2", "R2", "R2"])
            .unwrap();
        df
    }

    #[test]
    fn test_builder_basic() {
        let df = sample_df();
        let model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("mu + rep")
            .random("genotype", Identity::new(1.0), None)
            .build()
            .unwrap();

        assert_eq!(model.n_obs, 6);
        assert_eq!(model.y.len(), 6);
        assert_eq!(model.x.rows(), 6);
        assert_eq!(model.x.cols(), 2); // intercept + (2 rep levels - reference)
        assert_eq!(model.z_blocks.len(), 1);
        assert_eq!(model.z_blocks[0].cols(), 3); // 3 genotypes
        assert_eq!(model.random_term_names, vec!["genotype"]);
    }

    #[test]
    fn test_builder_intercept_plus_factor_fits() {
        // Regression test: "mu + rep" used to produce a rank-deficient X and a
        // singular coefficient matrix.
        let df = sample_df();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("mu + rep")
            .random("genotype", Identity::new(1.0), None)
            .build()
            .unwrap();
        let result = model.fit_reml().unwrap();
        assert_eq!(result.n_fixed_params, 2);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_builder_no_data_errors() {
        let result = MixedModelBuilder::new()
            .response("yield")
            .fixed("mu")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_no_response_errors() {
        let df = sample_df();
        let result = MixedModelBuilder::new().data(&df).fixed("mu").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_response_errors() {
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![1.0, f64::NAN, 3.0]).unwrap();
        df.add_factor_column("g", &["a", "b", "a"]).unwrap();
        let err = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .random("g", Identity::new(1.0), None)
            .build()
            .err()
            .expect("build should fail");
        assert!(err.to_string().contains("row 2"));
    }

    #[test]
    fn test_builder_default_intercept() {
        let df = sample_df();
        let model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .random("genotype", Identity::new(1.0), None)
            .build()
            .unwrap();

        // Should default to intercept only
        assert_eq!(model.x.cols(), 1);
    }

    #[test]
    fn test_builder_ginv_dimension_check() {
        use crate::matrix::sparse::sparse_identity;

        let df = sample_df();
        // Wrong dimension: 2x2 instead of 3x3
        let wrong_ginv = sparse_identity(2);
        let result = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("mu")
            .random("genotype", Identity::new(1.0), Some(wrong_ginv))
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_random_with_levels() {
        let df = sample_df();
        let levels: Vec<String> = ["G3", "G2", "G1", "G0"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let model = MixedModelBuilder::new()
            .data(&df)
            .response("yield")
            .fixed("rep")
            .random_with_levels("genotype", Identity::new(1.0), None, levels.clone())
            .build()
            .unwrap();
        assert_eq!(model.z_blocks[0].cols(), 4);
        assert_eq!(model.random_level_names[0], levels);
    }

    #[test]
    fn test_builder_random_pedigree_orders_levels_like_pedigree() {
        // Data only has records on animals 3 and 4; animals 1 and 2 are
        // parents without records and must still get columns.
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![4.5, 2.9, 3.9]).unwrap();
        df.add_factor_column("animal", &["4", "3", "4"]).unwrap();

        let triples = vec![
            (
                "3".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            (
                "4".to_string(),
                Some("1".to_string()),
                Some("2".to_string()),
            ),
            ("1".to_string(), None, None),
            ("2".to_string(), None, None),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap(); // deliberately unsorted

        let model = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .fixed("mu")
            .random_pedigree("animal", Identity::new(1.0), &ped)
            .build()
            .unwrap();

        assert_eq!(model.z_blocks[0].cols(), 4);
        let levels = &model.random_level_names[0];
        assert_eq!(levels.len(), 4);
        // Parents must come before offspring in the level order.
        let pos = |id: &str| levels.iter().position(|l| l == id).unwrap();
        assert!(pos("1") < pos("3"));
        assert!(pos("2") < pos("3"));
        assert!(pos("1") < pos("4"));
        let ginv = model.ginv_matrices[0].as_ref().unwrap();
        assert_eq!(ginv.rows(), 4);
        assert_eq!(ginv.cols(), 4);
    }

    #[test]
    fn test_builder_random_pedigree_unknown_animal_errors() {
        let mut df = DataFrame::new();
        df.add_float_column("y", vec![4.5, 2.9]).unwrap();
        df.add_factor_column("animal", &["4", "99"]).unwrap();
        let triples = vec![
            ("1".to_string(), None, None),
            ("4".to_string(), Some("1".to_string()), None),
        ];
        let ped = Pedigree::from_triples(&triples).unwrap();
        let err = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .random_pedigree("animal", Identity::new(1.0), &ped)
            .build()
            .err()
            .expect("build should fail");
        assert!(err.to_string().contains("99"));
    }
}
