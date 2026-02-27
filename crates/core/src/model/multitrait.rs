use nalgebra::DMatrix;

use crate::data::DataFrame;
use crate::error::{LmmError, Result};
use crate::types::SparseMat;

use super::design::{
    build_fixed_design, build_random_design, parse_fixed_formula, FixedEffectLabel, FixedTerm,
};

/// A multi-trait mixed model with Kronecker-structured covariance.
///
/// For `t` traits, each measured on `n` observations:
///
/// ```text
/// y_stacked = [y_1; y_2; ...; y_t]      (nt x 1)
/// X_mt = I_t kron X                      (nt x tp)
/// Z_mt = I_t kron Z                      (nt x tq)
/// G_mt = G0 kron K                       (tq x tq)
/// R_mt = R0 kron I_n                     (nt x nt)
/// ```
///
/// where G0 is the t x t trait genetic covariance matrix and R0 is the
/// t x t trait residual covariance matrix.
pub struct MultiTraitModel {
    /// Number of traits.
    pub n_traits: usize,
    /// Number of observations per trait (assumed equal across traits).
    pub n_obs: usize,
    /// Stacked response vector [y1; y2; ...; yt] of length n_traits * n_obs.
    pub y: Vec<f64>,
    /// Single-trait fixed-effects design matrix X (n x p).
    pub x_single: SparseMat,
    /// Single-trait random-effects design matrices, one per random term.
    pub z_single_blocks: Vec<SparseMat>,
    /// Current estimate of G0 (trait genetic covariance, t x t).
    pub g0: DMatrix<f64>,
    /// Current estimate of R0 (trait residual covariance, t x t).
    pub r0: DMatrix<f64>,
    /// K^{-1} matrices for each random term (None means identity).
    pub kinv_matrices: Vec<Option<SparseMat>>,
    /// Trait names.
    pub trait_names: Vec<String>,
    /// Random term names (column names).
    pub random_term_names: Vec<String>,
    /// Fixed effect labels (from the single-trait design).
    pub fixed_labels: Vec<FixedEffectLabel>,
    /// Random level names per term.
    pub random_level_names: Vec<Vec<String>>,
    /// Maximum REML iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

/// Builder for constructing a [`MultiTraitModel`].
pub struct MultiTraitModelBuilder<'a> {
    data: Option<&'a DataFrame>,
    trait_names: Vec<String>,
    fixed_formula: Option<String>,
    random_terms: Vec<MultiTraitRandomSpec>,
    g0: Option<DMatrix<f64>>,
    r0: Option<DMatrix<f64>>,
    max_iter: usize,
    tol: f64,
}

struct MultiTraitRandomSpec {
    column: String,
    kinv: Option<SparseMat>,
}

impl<'a> MultiTraitModelBuilder<'a> {
    /// Create a new builder with sensible defaults.
    pub fn new() -> Self {
        Self {
            data: None,
            trait_names: Vec::new(),
            fixed_formula: None,
            random_terms: Vec::new(),
            g0: None,
            r0: None,
            max_iter: 100,
            tol: 1e-6,
        }
    }

    /// Set the data source.
    pub fn data(mut self, df: &'a DataFrame) -> Self {
        self.data = Some(df);
        self
    }

    /// Set the response trait columns. Each trait is a separate float column
    /// in the DataFrame. All must have the same number of observations.
    pub fn traits(mut self, trait_names: &[&str]) -> Self {
        self.trait_names = trait_names.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set the fixed effects formula (e.g., "mu + rep").
    /// The same fixed-effects structure is used for every trait.
    pub fn fixed(mut self, formula: &str) -> Self {
        self.fixed_formula = Some(formula.to_string());
        self
    }

    /// Add a random effect term, shared across all traits.
    ///
    /// - `column`: the grouping factor column name
    /// - `kinv`: optional relationship matrix inverse (e.g., A^{-1})
    pub fn random(mut self, column: &str, kinv: Option<SparseMat>) -> Self {
        self.random_terms.push(MultiTraitRandomSpec {
            column: column.to_string(),
            kinv,
        });
        self
    }

    /// Set the initial trait genetic covariance matrix G0 (t x t).
    /// Must be symmetric positive definite with dimension = number of traits.
    pub fn g0(mut self, g0: DMatrix<f64>) -> Self {
        self.g0 = Some(g0);
        self
    }

    /// Set the initial trait residual covariance matrix R0 (t x t).
    /// Must be symmetric positive definite with dimension = number of traits.
    pub fn r0(mut self, r0: DMatrix<f64>) -> Self {
        self.r0 = Some(r0);
        self
    }

    /// Set maximum REML iterations (default: 100).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance (default: 1e-6).
    pub fn convergence(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Build the multi-trait model. Validates inputs and constructs design matrices.
    pub fn build(self) -> Result<MultiTraitModel> {
        let df = self
            .data
            .ok_or_else(|| LmmError::ModelSpec("No data provided".into()))?;

        let n_traits = self.trait_names.len();
        if n_traits < 2 {
            return Err(LmmError::ModelSpec(
                "Multi-trait model requires at least 2 traits".into(),
            ));
        }

        let n = df.nrows();
        if n == 0 {
            return Err(LmmError::ModelSpec("DataFrame is empty".into()));
        }

        // Collect response vectors for each trait, then stack them
        let mut y = Vec::with_capacity(n * n_traits);
        for trait_name in &self.trait_names {
            let trait_y = df.get_float(trait_name)?;
            if trait_y.len() != n {
                return Err(LmmError::DimensionMismatch {
                    expected: n,
                    got: trait_y.len(),
                    context: format!("Trait '{}' length", trait_name),
                });
            }
            y.extend_from_slice(trait_y);
        }

        // Build single-trait fixed effects design matrix
        let fixed_terms = if let Some(ref formula) = self.fixed_formula {
            parse_fixed_formula(formula, df)?
        } else {
            vec![FixedTerm::Intercept]
        };
        let (x_single, fixed_labels) = build_fixed_design(df, &fixed_terms)?;

        // Build single-trait random effects design matrices
        let mut z_single_blocks = Vec::new();
        let mut random_level_names = Vec::new();
        let mut kinv_matrices = Vec::new();
        let mut random_term_names = Vec::new();

        for rt in self.random_terms {
            let (z, levels) = build_random_design(df, &rt.column)?;

            // Validate kinv dimensions
            if let Some(ref kinv) = rt.kinv {
                let q = levels.len();
                if kinv.rows() != q || kinv.cols() != q {
                    return Err(LmmError::DimensionMismatch {
                        expected: q,
                        got: kinv.rows(),
                        context: format!(
                            "K-inverse for '{}' should be {}x{} but is {}x{}",
                            rt.column,
                            q,
                            q,
                            kinv.rows(),
                            kinv.cols()
                        ),
                    });
                }
            }

            z_single_blocks.push(z);
            random_level_names.push(levels);
            kinv_matrices.push(rt.kinv);
            random_term_names.push(rt.column);
        }

        if z_single_blocks.is_empty() {
            return Err(LmmError::ModelSpec(
                "Multi-trait model requires at least one random term".into(),
            ));
        }

        // Initialize G0 and R0
        // Compute phenotypic variance from the stacked response to use as a
        // scale for default starting values.
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        let y_var: f64 =
            y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (y.len() - 1).max(1) as f64;
        let default_var = (y_var / 2.0).max(0.01);

        let g0 = if let Some(g0) = self.g0 {
            if g0.nrows() != n_traits || g0.ncols() != n_traits {
                return Err(LmmError::DimensionMismatch {
                    expected: n_traits,
                    got: g0.nrows(),
                    context: "G0 dimensions must match number of traits".into(),
                });
            }
            g0
        } else {
            // Default: diagonal with half the phenotypic variance
            DMatrix::from_diagonal(&nalgebra::DVector::from_element(n_traits, default_var))
        };

        let r0 = if let Some(r0) = self.r0 {
            if r0.nrows() != n_traits || r0.ncols() != n_traits {
                return Err(LmmError::DimensionMismatch {
                    expected: n_traits,
                    got: r0.nrows(),
                    context: "R0 dimensions must match number of traits".into(),
                });
            }
            r0
        } else {
            DMatrix::from_diagonal(&nalgebra::DVector::from_element(n_traits, default_var))
        };

        Ok(MultiTraitModel {
            n_traits,
            n_obs: n,
            y,
            x_single,
            z_single_blocks,
            g0,
            r0,
            kinv_matrices,
            trait_names: self.trait_names,
            random_term_names,
            fixed_labels,
            random_level_names,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

impl<'a> Default for MultiTraitModelBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiTraitModel {
    /// Convenience: fit with multi-trait REML.
    pub fn fit_reml(&mut self) -> Result<crate::lmm::MultiTraitFitResult> {
        let engine = crate::lmm::MultiTraitReml::new(self.max_iter, self.tol);
        engine.fit(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataFrame;

    fn sample_mt_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_float_column("yield", vec![10.0, 8.0, 6.0, 12.0, 10.0, 8.0])
            .unwrap();
        df.add_float_column("height", vec![50.0, 45.0, 40.0, 55.0, 48.0, 42.0])
            .unwrap();
        df.add_factor_column("genotype", &["G1", "G2", "G3", "G1", "G2", "G3"])
            .unwrap();
        df.add_factor_column("rep", &["R1", "R1", "R1", "R2", "R2", "R2"])
            .unwrap();
        df
    }

    #[test]
    fn test_mt_builder_basic() {
        let df = sample_mt_df();
        let model = MultiTraitModelBuilder::new()
            .data(&df)
            .traits(&["yield", "height"])
            .fixed("rep")
            .random("genotype", None)
            .build()
            .unwrap();

        assert_eq!(model.n_traits, 2);
        assert_eq!(model.n_obs, 6);
        assert_eq!(model.y.len(), 12); // 2 traits * 6 obs
        assert_eq!(model.x_single.rows(), 6);
        assert_eq!(model.z_single_blocks.len(), 1);
        assert_eq!(model.z_single_blocks[0].cols(), 3);
        assert_eq!(model.g0.nrows(), 2);
        assert_eq!(model.r0.nrows(), 2);
    }

    #[test]
    fn test_mt_builder_needs_two_traits() {
        let df = sample_mt_df();
        let result = MultiTraitModelBuilder::new()
            .data(&df)
            .traits(&["yield"])
            .fixed("rep")
            .random("genotype", None)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_mt_builder_no_data_errors() {
        let result = MultiTraitModelBuilder::new()
            .traits(&["yield", "height"])
            .fixed("rep")
            .random("genotype", None)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_mt_builder_custom_g0_r0() {
        let df = sample_mt_df();
        let g0 = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let r0 = DMatrix::from_row_slice(2, 2, &[2.0, 0.5, 0.5, 1.5]);
        let model = MultiTraitModelBuilder::new()
            .data(&df)
            .traits(&["yield", "height"])
            .fixed("rep")
            .random("genotype", None)
            .g0(g0.clone())
            .r0(r0.clone())
            .build()
            .unwrap();

        assert_eq!(model.g0, g0);
        assert_eq!(model.r0, r0);
    }

    #[test]
    fn test_mt_builder_g0_wrong_dim() {
        let df = sample_mt_df();
        let g0 = DMatrix::from_row_slice(3, 3, &[1.0; 9]);
        let result = MultiTraitModelBuilder::new()
            .data(&df)
            .traits(&["yield", "height"])
            .fixed("rep")
            .random("genotype", None)
            .g0(g0)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_mt_stacked_response() {
        let df = sample_mt_df();
        let model = MultiTraitModelBuilder::new()
            .data(&df)
            .traits(&["yield", "height"])
            .fixed("rep")
            .random("genotype", None)
            .build()
            .unwrap();

        // First n_obs entries should be yield, next n_obs should be height
        assert_eq!(&model.y[0..6], &[10.0, 8.0, 6.0, 12.0, 10.0, 8.0]);
        assert_eq!(&model.y[6..12], &[50.0, 45.0, 40.0, 55.0, 48.0, 42.0]);
    }
}
