use crate::data::DataFrame;
use crate::error::{LmmError, Result};
use crate::types::SparseMat;
use crate::variance::VarStruct;

use super::design::{
    build_combined_random_design, build_fixed_design, build_random_design, parse_fixed_formula,
    FixedEffectLabel, FixedTerm,
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
pub struct MixedModelBuilder<'a> {
    data: Option<&'a DataFrame>,
    response: Option<String>,
    fixed_formula: Option<String>,
    fixed_terms: Vec<FixedTerm>,
    random_terms: Vec<RandomTermSpec>,
    residual_structure: Option<Box<dyn VarStruct>>,
    max_iter: usize,
    convergence_tol: f64,
}

struct RandomTermSpec {
    column: String,
    variance_structure: Box<dyn VarStruct>,
    ginv: Option<SparseMat>,
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
    pub fn fixed(mut self, formula: &str) -> Self {
        self.fixed_formula = Some(formula.to_string());
        self
    }

    /// Add a random effect term.
    ///
    /// - `column`: the grouping factor column name
    /// - `vs`: the variance structure for this random term
    /// - `ginv`: optional relationship matrix inverse (e.g., A^{-1} for pedigree BLUP)
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
            let (z, levels) = build_random_design(df, &rt.column)?;

            // Validate ginv dimensions if provided
            if let Some(ref ginv) = rt.ginv {
                let q = levels.len();
                if ginv.rows() != q || ginv.cols() != q {
                    return Err(LmmError::DimensionMismatch {
                        expected: q,
                        got: ginv.rows(),
                        context: format!(
                            "G-inverse for '{}' should be {}x{} but is {}x{}",
                            rt.column, q, q, ginv.rows(), ginv.cols()
                        ),
                    });
                }
            }

            z_blocks.push(z);
            random_level_names.push(levels);
            random_var_structs.push(rt.variance_structure);
            ginv_matrices.push(rt.ginv);
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
        assert_eq!(model.x.cols(), 3); // intercept + 2 rep levels
        assert_eq!(model.z_blocks.len(), 1);
        assert_eq!(model.z_blocks[0].cols(), 3); // 3 genotypes
        assert_eq!(model.random_term_names, vec!["genotype"]);
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
}
