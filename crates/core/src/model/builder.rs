use std::collections::HashSet;

use crate::data::DataFrame;
use crate::error::{LmmError, Result};
use crate::genetics::{compute_a_inverse_with_inbreeding, Pedigree};
use crate::types::SparseMat;
use crate::variance::{Known, KroneckerStruct, StructureSpec, VarStruct};

use super::design::{
    build_combined_random_design, build_fixed_design, build_random_design,
    build_random_design_interaction, build_random_design_with_levels, parse_fixed_formula,
    FixedEffectLabel, FixedTerm,
};

/// Mapping of observations onto the cells of a `rows x cols` grid for a
/// separable (Kronecker) residual structure.
#[derive(Debug, Clone)]
pub struct ResidualGrid {
    /// Number of grid cells (`n_rows * n_cols`), the dimension of the
    /// residual variance structure.
    pub n_cells: usize,
    /// Cell index (`row * n_cols + col`) of each observation.
    pub cell_index: Vec<usize>,
    /// Row factor levels (grid order).
    pub row_levels: Vec<String>,
    /// Column factor levels (grid order).
    pub col_levels: Vec<String>,
}

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
    /// Grid layout of a separable residual structure, if any. When set, the
    /// residual structure is defined on `n_cells` cells and the observations
    /// use the cells listed in `cell_index` (missing plots are allowed).
    pub residual_grid: Option<ResidualGrid>,
    /// REML configuration.
    pub max_iter: usize,
    pub convergence_tol: f64,
}

impl MixedModel {
    /// k-fold cross-validation of the prediction accuracy of a model with a
    /// single scaled-identity / relationship-matrix random term (the genomic
    /// or pedigree prediction setting), e.g. `y = Xb + Zu`, `u ~ N(0, σ² K)`.
    ///
    /// Each fold refits the variance components on the training plots with
    /// EM-REML and predicts the held-out plots.
    pub fn cross_validate(
        &self,
        n_folds: usize,
        seed: u64,
    ) -> Result<crate::diagnostics::CrossValResult> {
        if self.z_blocks.len() != 1 || self.needs_general_engine() {
            return Err(LmmError::ModelSpec(
                "cross_validate supports models with exactly one random term (scaled identity \
                 or relationship matrix) and an IID residual"
                    .into(),
            ));
        }
        if n_folds < 2 {
            return Err(LmmError::ModelSpec("n_folds must be at least 2".into()));
        }
        crate::diagnostics::CrossValidator::new(n_folds)
            .seed(seed)
            .max_iter(self.max_iter.max(100))
            .tolerance(self.convergence_tol)
            .run(
                &self.y,
                &self.x,
                &self.z_blocks[0],
                self.ginv_matrices[0].as_ref(),
            )
    }

    /// Whether the model needs the general REML engine: any random term
    /// with a structure other than a single scaled identity/relationship
    /// matrix, or a non-IID residual.
    pub fn needs_general_engine(&self) -> bool {
        let simple = |vs: &dyn VarStruct| vs.name() == "Identity" && vs.n_params() == 1;
        !self.random_var_structs.iter().all(|vs| simple(vs.as_ref()))
            || !simple(self.residual_var_struct.as_ref())
            || self.residual_grid.is_some()
    }
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
    residual_structure: Option<VarSpec>,
    residual_grid: Option<ResidualGridSpec>,
    max_iter: usize,
    convergence_tol: f64,
}

/// A variance structure that is either ready or resolved once the number of
/// levels is known.
enum VarSpec {
    Ready(Box<dyn VarStruct>),
    Deferred(StructureSpec),
}

impl VarSpec {
    fn resolve(self, dim: usize) -> Result<Box<dyn VarStruct>> {
        match self {
            VarSpec::Ready(vs) => Ok(vs),
            VarSpec::Deferred(spec) => spec.instantiate(dim),
        }
    }
}

/// A separable residual `Sigma_row(theta) ⊗ Sigma_col(phi)` over a grid.
struct ResidualGridSpec {
    row_col: String,
    row_vs: VarSpec,
    col_col: String,
    col_vs: VarSpec,
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

/// The second factor of an interaction term.
enum InnerFactor<'a> {
    /// Independent levels (`Known::identity`).
    Independent,
    /// A parameterised structure (e.g. `AR1::correlation` for columns).
    Structured(VarSpec),
    /// Pedigree relationship matrix (`Known(A⁻¹)`), levels in pedigree order.
    Pedigree(&'a Pedigree),
}

struct RandomTermSpec<'a> {
    column: String,
    variance_structure: VarSpec,
    ginv: Option<SparseMat>,
    levels: RandomLevels<'a>,
    /// For interaction terms `column:inner`: the inner factor.
    interaction: Option<(String, InnerFactor<'a>)>,
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
            residual_grid: None,
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
            variance_structure: VarSpec::Ready(Box::new(vs)),
            ginv,
            levels: RandomLevels::FromData,
            interaction: None,
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
            variance_structure: VarSpec::Ready(Box::new(vs)),
            ginv,
            levels: RandomLevels::Explicit(levels),
            interaction: None,
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
            variance_structure: VarSpec::Ready(Box::new(vs)),
            ginv: None,
            levels: RandomLevels::Pedigree(pedigree),
            interaction: None,
        });
        self
    }

    /// Add an interaction random term `outer:inner` with a separable
    /// covariance `Sigma_outer(theta) ⊗ I`, e.g. genotype-by-environment
    /// effects with a factor analytic or diagonal structure across
    /// environments: `.random_interaction("env", fa1(n_env), "genotype")`.
    ///
    /// The levels of both factors are taken from the data; `outer_vs` must be
    /// defined for the number of outer levels.
    pub fn random_interaction(
        mut self,
        outer: &str,
        outer_vs: impl VarStruct + 'static,
        inner: &str,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: outer.to_string(),
            variance_structure: VarSpec::Ready(Box::new(outer_vs)),
            ginv: None,
            levels: RandomLevels::FromData,
            interaction: Some((inner.to_string(), InnerFactor::Independent)),
        });
        self
    }

    /// Add an interaction random term with structures on both factors,
    /// `Sigma_outer(theta) ⊗ Sigma_inner(phi)`, e.g. a separable spatial
    /// random effect `.random_interaction_with("row", AR1::new(1.0, 0.5), "col", AR1::correlation(0.5))`.
    /// Use a correlation-only structure on one side so that the scale is
    /// identifiable.
    pub fn random_interaction_with(
        mut self,
        outer: &str,
        outer_vs: impl VarStruct + 'static,
        inner: &str,
        inner_vs: impl VarStruct + 'static,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: outer.to_string(),
            variance_structure: VarSpec::Ready(Box::new(outer_vs)),
            ginv: None,
            levels: RandomLevels::FromData,
            interaction: Some((
                inner.to_string(),
                InnerFactor::Structured(VarSpec::Ready(Box::new(inner_vs))),
            )),
        });
        self
    }

    /// Add an interaction random term `outer:animal` with covariance
    /// `Sigma_outer(theta) ⊗ A`, where `A` is the pedigree relationship
    /// matrix (multi-environment or multi-trait-like animal models). The
    /// inner levels are all animals of the pedigree in sorted order.
    pub fn random_interaction_pedigree(
        mut self,
        outer: &str,
        outer_vs: impl VarStruct + 'static,
        inner: &str,
        pedigree: &'a Pedigree,
    ) -> Self {
        self.random_terms.push(RandomTermSpec {
            column: outer.to_string(),
            variance_structure: VarSpec::Ready(Box::new(outer_vs)),
            ginv: None,
            levels: RandomLevels::FromData,
            interaction: Some((inner.to_string(), InnerFactor::Pedigree(pedigree))),
        });
        self
    }

    /// Set the residual variance structure.
    /// If not called, defaults to Identity (homogeneous residual variance).
    ///
    /// A structure with an intrinsic dimension (Diagonal, Unstructured,
    /// FactorAnalytic) must match the number of observations; AR1 applies
    /// to the observations in data order.
    pub fn residual(mut self, vs: impl VarStruct + 'static) -> Self {
        self.residual_structure = Some(VarSpec::Ready(Box::new(vs)));
        self.residual_grid = None;
        self
    }

    /// Set a separable residual structure `Sigma_row(theta) ⊗ Sigma_col(phi)`
    /// over the grid defined by two factor columns (typically field rows and
    /// columns): `.residual_interaction("row", AR1::new(1.0, 0.5), "col", AR1::correlation(0.5))`.
    ///
    /// Every observation must occupy a distinct `(row, col)` cell; cells
    /// without an observation (missing plots) are allowed. Use a
    /// correlation-only structure on one side so that the residual variance
    /// is identifiable.
    pub fn residual_interaction(
        mut self,
        row: &str,
        row_vs: impl VarStruct + 'static,
        col: &str,
        col_vs: impl VarStruct + 'static,
    ) -> Self {
        self.residual_structure = None;
        self.residual_grid = Some(ResidualGridSpec {
            row_col: row.to_string(),
            row_vs: VarSpec::Ready(Box::new(row_vs)),
            col_col: col.to_string(),
            col_vs: VarSpec::Ready(Box::new(col_vs)),
        });
        self
    }

    // ---- specification-based variants (used by the CLI and Python) ----

    /// Add a random term with a structure given as a [`StructureSpec`]
    /// (e.g. parsed from `"ar1(0.3)"`); the dimension is resolved at build
    /// time. An optional pedigree turns an identity spec into an animal-model
    /// term (see [`random_pedigree`](Self::random_pedigree)).
    pub fn random_spec(
        mut self,
        column: &str,
        spec: StructureSpec,
        pedigree: Option<&'a Pedigree>,
    ) -> Self {
        let levels = match pedigree {
            Some(ped) => RandomLevels::Pedigree(ped),
            None => RandomLevels::FromData,
        };
        self.random_terms.push(RandomTermSpec {
            column: column.to_string(),
            variance_structure: VarSpec::Deferred(spec),
            ginv: None,
            levels,
            interaction: None,
        });
        self
    }

    /// Add an interaction term `outer:inner` from specifications. `inner_spec`
    /// `None` means independent inner levels; a pedigree makes the inner
    /// factor an animal factor with `A` as its covariance (its spec is then
    /// ignored).
    pub fn random_interaction_spec(
        mut self,
        outer: &str,
        outer_spec: StructureSpec,
        inner: &str,
        inner_spec: Option<StructureSpec>,
        pedigree: Option<&'a Pedigree>,
    ) -> Self {
        let inner_factor = match (pedigree, inner_spec) {
            (Some(ped), _) => InnerFactor::Pedigree(ped),
            (None, Some(spec)) if !spec.is_identity() && spec != StructureSpec::Known => {
                InnerFactor::Structured(VarSpec::Deferred(spec))
            }
            _ => InnerFactor::Independent,
        };
        self.random_terms.push(RandomTermSpec {
            column: outer.to_string(),
            variance_structure: VarSpec::Deferred(outer_spec),
            ginv: None,
            levels: RandomLevels::FromData,
            interaction: Some((inner.to_string(), inner_factor)),
        });
        self
    }

    /// Set the residual structure from a specification (applied to the
    /// observations in data order).
    pub fn residual_spec(mut self, spec: StructureSpec) -> Self {
        self.residual_structure = Some(VarSpec::Deferred(spec));
        self.residual_grid = None;
        self
    }

    /// Set a separable residual over a `row x col` grid from specifications,
    /// e.g. `("row", ar1, "col", ar1c)`.
    pub fn residual_interaction_spec(
        mut self,
        row: &str,
        row_spec: StructureSpec,
        col: &str,
        col_spec: StructureSpec,
    ) -> Self {
        self.residual_structure = None;
        self.residual_grid = Some(ResidualGridSpec {
            row_col: row.to_string(),
            row_vs: VarSpec::Deferred(row_spec),
            col_col: col.to_string(),
            col_vs: VarSpec::Deferred(col_spec),
        });
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
            if let Some((inner_col, inner)) = rt.interaction {
                let outer_levels: Vec<String> = df
                    .factor_view(&rt.column)?
                    .level_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                let (inner_levels, inner_vs): (Vec<String>, Box<dyn VarStruct>) = match inner {
                    InnerFactor::Independent => {
                        let levels: Vec<String> = df
                            .factor_view(&inner_col)?
                            .level_names()
                            .iter()
                            .map(|s| s.to_string())
                            .collect();
                        let q = levels.len();
                        (levels, Box::new(Known::identity(q)))
                    }
                    InnerFactor::Structured(vs) => {
                        let levels: Vec<String> = df
                            .factor_view(&inner_col)?
                            .level_names()
                            .iter()
                            .map(|s| s.to_string())
                            .collect();
                        let q = levels.len();
                        (levels, vs.resolve(q)?)
                    }
                    InnerFactor::Pedigree(ped) => {
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
                        (ids, Box::new(Known::from_inverse(a_inv)?))
                    }
                };
                let (z, labels) = build_random_design_interaction(
                    df,
                    &rt.column,
                    &outer_levels,
                    &inner_col,
                    &inner_levels,
                )?;
                let outer_vs = rt.variance_structure.resolve(outer_levels.len())?;
                let vs = KroneckerStruct::new(
                    outer_vs,
                    outer_levels.len(),
                    inner_vs,
                    inner_levels.len(),
                    &rt.column,
                    &inner_col,
                )?;
                z_blocks.push(z);
                random_level_names.push(labels);
                random_var_structs.push(Box::new(vs) as Box<dyn VarStruct>);
                ginv_matrices.push(None);
                random_term_names.push(format!("{}:{}", rt.column, inner_col));
                continue;
            }

            let q_term = match &rt.levels {
                RandomLevels::FromData => df.factor_view(&rt.column)?.n_levels(),
                RandomLevels::Explicit(levels) => levels.len(),
                RandomLevels::Pedigree(ped) => ped.n_animals(),
            };
            let variance_structure = rt.variance_structure.resolve(q_term)?;
            if let Some(d) = variance_structure.fixed_dim() {
                if d != q_term {
                    return Err(LmmError::DimensionMismatch {
                        expected: q_term,
                        got: d,
                        context: format!(
                            "{} structure for '{}' is defined for {} levels but the term has {}",
                            variance_structure.name(),
                            rt.column,
                            d,
                            q_term
                        ),
                    });
                }
            }
            if variance_structure.name() != "Identity"
                && (rt.ginv.is_some() || matches!(rt.levels, RandomLevels::Pedigree(_)))
            {
                return Err(LmmError::ModelSpec(format!(
                    "Random term '{}': a relationship matrix can only be combined with an \
                     Identity structure; use random_interaction_pedigree for structured terms",
                    rt.column
                )));
            }

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
            random_var_structs.push(variance_structure);
            ginv_matrices.push(ginv);
            random_term_names.push(rt.column);
        }

        let z_combined = build_combined_random_design(&z_blocks, n);

        // Residual structure defaults to Identity
        let (residual_var_struct, residual_grid): (Box<dyn VarStruct>, Option<ResidualGrid>) =
            match (self.residual_structure, self.residual_grid) {
                (_, Some(spec)) => {
                    let rows = df.factor_view(&spec.row_col)?;
                    let cols = df.factor_view(&spec.col_col)?;
                    let row_levels: Vec<String> =
                        rows.level_names().iter().map(|s| s.to_string()).collect();
                    let col_levels: Vec<String> =
                        cols.level_names().iter().map(|s| s.to_string()).collect();
                    let n_cols = col_levels.len();
                    let mut seen = HashSet::with_capacity(n);
                    let mut cell_index = Vec::with_capacity(n);
                    for i in 0..n {
                        let cell = rows.codes()[i] * n_cols + cols.codes()[i];
                        if !seen.insert(cell) {
                            return Err(LmmError::ModelSpec(format!(
                                "Residual grid: observation {} shares cell ({}, {}) with another \
                                 observation; each (row, col) may occur only once",
                                i + 1,
                                row_levels[rows.codes()[i]],
                                col_levels[cols.codes()[i]]
                            )));
                        }
                        cell_index.push(cell);
                    }
                    let row_vs = spec.row_vs.resolve(row_levels.len())?;
                    let col_vs = spec.col_vs.resolve(n_cols)?;
                    let vs = KroneckerStruct::new(
                        row_vs,
                        row_levels.len(),
                        col_vs,
                        n_cols,
                        &spec.row_col,
                        &spec.col_col,
                    )?;
                    let grid = ResidualGrid {
                        n_cells: row_levels.len() * n_cols,
                        cell_index,
                        row_levels,
                        col_levels,
                    };
                    (Box::new(vs), Some(grid))
                }
                (Some(vs), None) => {
                    let vs = vs.resolve(n)?;
                    if let Some(d) = vs.fixed_dim() {
                        if d != n {
                            return Err(LmmError::DimensionMismatch {
                                expected: n,
                                got: d,
                                context: format!(
                                    "residual {} structure is defined for {} observations but there are {}",
                                    vs.name(),
                                    d,
                                    n
                                ),
                            });
                        }
                    }
                    (vs, None)
                }
                (None, None) => (Box::new(crate::variance::Identity::default()), None),
            };

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
            residual_grid,
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
