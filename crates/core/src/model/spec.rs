//! Text model specifications shared by the CLI and OpenBLUP Studio.
//!
//! A random or residual term is written `factor[:structure]` or as an
//! interaction `factor[:structure]*factor[:structure]`, with the structures
//! of [`StructureSpec`]:
//!
//! | Term | Meaning |
//! |------|---------|
//! | `genotype` | IID genotype effects |
//! | `animal` + pedigree | animal model with the pedigree A-inverse |
//! | `env:fa1*genotype` | FA1 genotype-by-environment interaction |
//! | `row:ar1*col:ar1c` | separable AR1 x AR1 (as a residual: spatial residual) |
//! | `trait:us*animal` | multi-trait animal model in long format |
//!
//! [`FitSpec::prepare`] turns such a specification and a data frame into a
//! [`MixedModel`], applying the same data preparation as the CLI.

use serde::Deserialize;

use crate::data::DataFrame;
use crate::error::{LmmError, Result};
use crate::genetics::Pedigree;
use crate::variance::StructureSpec;

use super::{MixedModel, MixedModelBuilder};

/// One factor of a term specification: `name[:structure]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorSpec {
    /// Column name.
    pub name: String,
    /// Variance structure (scaled identity when not given).
    pub structure: StructureSpec,
    /// Whether the structure was written out explicitly.
    pub explicit_structure: bool,
}

impl FactorSpec {
    /// Parse `name[:structure]`.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(LmmError::ModelSpec(
                "Empty factor in term specification".into(),
            ));
        }
        match s.split_once(':') {
            Some((name, structure)) => Ok(FactorSpec {
                name: name.trim().to_string(),
                structure: StructureSpec::parse(structure).map_err(|e| {
                    LmmError::ModelSpec(format!("Invalid structure in '{}': {}", s, e))
                })?,
                explicit_structure: true,
            }),
            None => Ok(FactorSpec {
                name: s.to_string(),
                structure: StructureSpec::Identity { sigma2: 1.0 },
                explicit_structure: false,
            }),
        }
    }
}

/// A random or residual term: one factor or an interaction of two.
#[derive(Debug, Clone, PartialEq)]
pub struct TermSpec {
    /// The (first) factor.
    pub outer: FactorSpec,
    /// The second factor of an `outer*inner` interaction.
    pub inner: Option<FactorSpec>,
}

impl TermSpec {
    /// Parse `factor[:struct]` or `factor[:struct]*factor[:struct]`.
    ///
    /// ```
    /// use plant_breeding_lmm_core::model::TermSpec;
    ///
    /// let t = TermSpec::parse("env:fa1*genotype").unwrap();
    /// assert_eq!(t.label(), "env:genotype");
    /// assert_eq!(t.columns(), vec!["env", "genotype"]);
    /// assert!(TermSpec::parse("a*b*c").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('*').collect();
        match parts.as_slice() {
            [single] => Ok(TermSpec {
                outer: FactorSpec::parse(single)?,
                inner: None,
            }),
            [outer, inner] => Ok(TermSpec {
                outer: FactorSpec::parse(outer)?,
                inner: Some(FactorSpec::parse(inner)?),
            }),
            _ => Err(LmmError::ModelSpec(format!(
                "Term '{}' has more than two factors; only `a*b` interactions are supported",
                s
            ))),
        }
    }

    /// The factors' column names.
    pub fn columns(&self) -> Vec<&str> {
        let mut v = vec![self.outer.name.as_str()];
        if let Some(inner) = &self.inner {
            v.push(inner.name.as_str());
        }
        v
    }

    /// The term's name in the results (`a` or `a:b`).
    pub fn label(&self) -> String {
        match &self.inner {
            Some(inner) => format!("{}:{}", self.outer.name, inner.name),
            None => self.outer.name.clone(),
        }
    }
}

fn default_fixed() -> String {
    "mu".to_string()
}

fn default_max_iter() -> usize {
    50
}

fn default_tolerance() -> f64 {
    1e-6
}

/// A complete model specification in the vocabulary of the CLI flags
/// (`--response`, `--fixed`, `--random`, `--residual`, `--factor`,
/// `--pedigree-term`). Deserializable from JSON, e.g.
/// `{"response": "yield", "fixed": "mu + rep", "random": ["genotype"]}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FitSpec {
    /// Response column.
    pub response: String,
    /// Fixed-effects formula such as `"mu + rep"`.
    #[serde(default = "default_fixed")]
    pub fixed: String,
    /// Random terms.
    #[serde(default)]
    pub random: Vec<String>,
    /// Residual term (IID when `None`).
    #[serde(default)]
    pub residual: Option<String>,
    /// Numeric columns to treat as factors.
    #[serde(default)]
    pub factors: Vec<String>,
    /// Factor that carries the pedigree relationship matrix (default: the
    /// first random term's factor, or its inner factor for an interaction).
    #[serde(default)]
    pub pedigree_term: Option<String>,
    /// Maximum number of REML iterations.
    #[serde(default = "default_max_iter")]
    pub max_iter: usize,
    /// Convergence tolerance (relative change in the variance parameters).
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

impl FitSpec {
    /// A specification with the given response and defaults otherwise.
    pub fn new(response: &str) -> Self {
        FitSpec {
            response: response.to_string(),
            fixed: default_fixed(),
            random: Vec::new(),
            residual: None,
            factors: Vec::new(),
            pedigree_term: None,
            max_iter: default_max_iter(),
            tolerance: default_tolerance(),
        }
    }

    /// Prepare the data and build the model.
    ///
    /// Rows with a missing response are dropped, `factors` and every column
    /// used by a random or residual term are converted to factors, and the
    /// pedigree (already validated and sorted) is attached to the pedigree
    /// term. What was done is reported in [`PreparedModel::notes`].
    pub fn prepare(&self, data: &DataFrame, pedigree: Option<&Pedigree>) -> Result<PreparedModel> {
        let mut notes = Vec::new();
        let mut df = data.clone();

        if df.get_float(&self.response).is_err() {
            return Err(LmmError::ModelSpec(format!(
                "Response column '{}' is missing or not numeric (columns: {})",
                self.response,
                df.column_names().join(", ")
            )));
        }
        if df.has_missing(&self.response)? {
            let before = df.nrows();
            df = df.drop_missing(&self.response)?;
            notes.push(format!(
                "Dropped {} rows with missing '{}' ({} remain)",
                before - df.nrows(),
                self.response,
                df.nrows()
            ));
        }
        if df.nrows() == 0 {
            return Err(LmmError::ModelSpec(
                "No observations with a non-missing response".into(),
            ));
        }

        for col in &self.factors {
            df.as_factor(col).map_err(|e| {
                LmmError::ModelSpec(format!("Cannot treat column '{}' as a factor: {}", col, e))
            })?;
        }

        // ---- terms ----
        let random_terms: Vec<TermSpec> = self
            .random
            .iter()
            .map(|s| {
                TermSpec::parse(s)
                    .map_err(|e| LmmError::ModelSpec(format!("Invalid random term '{}': {}", s, e)))
            })
            .collect::<Result<_>>()?;
        // A residual without `*` is a structure over the observations in data
        // order (`ar1`); `row:ar1*col:ar1c` is a structure over a grid.
        let mut residual_structure = None;
        let mut residual_term = None;
        match self.residual.as_deref().map(str::trim) {
            None => {}
            Some(s) if !s.contains('*') => {
                residual_structure = Some(StructureSpec::parse(s).map_err(|e| {
                    LmmError::ModelSpec(format!("Invalid residual structure '{}': {}", s, e))
                })?);
            }
            Some(s) => {
                residual_term = Some(TermSpec::parse(s).map_err(|e| {
                    LmmError::ModelSpec(format!("Invalid residual term '{}': {}", s, e))
                })?);
            }
        }
        for term in random_terms.iter().chain(residual_term.iter()) {
            for col in term.columns() {
                if df.get_column(col).is_err() {
                    return Err(LmmError::ModelSpec(format!(
                        "Column '{}' (term '{}') not found in the data (columns: {})",
                        col,
                        term.label(),
                        df.column_names().join(", ")
                    )));
                }
                if df.get_factor(col).is_err() {
                    df.as_factor(col).map_err(|e| {
                        LmmError::ModelSpec(format!(
                            "Term '{}' must use categorical columns: {}",
                            col, e
                        ))
                    })?;
                    notes.push(format!("Treating numeric column '{}' as a factor", col));
                }
            }
        }

        // ---- pedigree ----
        let pedigree_factor: Option<String> = match (pedigree, &self.pedigree_term) {
            (None, _) => None,
            (Some(_), Some(term)) => {
                let known = random_terms
                    .iter()
                    .any(|t| t.columns().contains(&term.as_str()));
                if !known {
                    return Err(LmmError::ModelSpec(format!(
                        "Pedigree term '{}' is not a factor of any random term",
                        term
                    )));
                }
                Some(term.clone())
            }
            (Some(_), None) => match random_terms.first() {
                Some(t) => Some(
                    t.inner
                        .as_ref()
                        .map(|f| f.name.clone())
                        .unwrap_or_else(|| t.outer.name.clone()),
                ),
                None => {
                    return Err(LmmError::ModelSpec(
                        "A pedigree requires at least one random term".into(),
                    ))
                }
            },
        };

        // ---- model ----
        let mut builder = MixedModelBuilder::new()
            .data(&df)
            .response(&self.response)
            .fixed(&self.fixed)
            .max_iterations(self.max_iter)
            .convergence(self.tolerance);

        for term in &random_terms {
            match &term.inner {
                None => {
                    let uses_ped = pedigree_factor.as_deref() == Some(term.outer.name.as_str());
                    if uses_ped
                        && term.outer.explicit_structure
                        && !term.outer.structure.is_identity()
                    {
                        return Err(LmmError::ModelSpec(format!(
                            "Term '{}': a pedigree can only be combined with the default (idv) \
                             structure; use an interaction such as env:fa1*{} for structured \
                             genetic effects",
                            term.label(),
                            term.outer.name
                        )));
                    }
                    let ped = if uses_ped { pedigree } else { None };
                    if let Some(p) = ped {
                        notes.push(format!(
                            "Using pedigree A-inverse ({0}x{0}, with inbreeding) for random term '{1}'",
                            p.n_animals(),
                            term.outer.name
                        ));
                    }
                    builder =
                        builder.random_spec(&term.outer.name, term.outer.structure.clone(), ped);
                }
                Some(inner) => {
                    if pedigree_factor.as_deref() == Some(term.outer.name.as_str()) {
                        return Err(LmmError::ModelSpec(format!(
                            "Term '{}': the pedigree factor must be the second (inner) factor \
                             of an interaction, e.g. {}:fa1*{}",
                            term.label(),
                            inner.name,
                            term.outer.name
                        )));
                    }
                    let uses_ped = pedigree_factor.as_deref() == Some(inner.name.as_str());
                    let ped = if uses_ped { pedigree } else { None };
                    if let Some(p) = ped {
                        notes.push(format!(
                            "Using pedigree A-inverse ({0}x{0}, with inbreeding) for factor '{1}' of term '{2}'",
                            p.n_animals(),
                            inner.name,
                            term.label()
                        ));
                    }
                    let inner_spec = if inner.explicit_structure {
                        Some(inner.structure.clone())
                    } else {
                        None
                    };
                    builder = builder.random_interaction_spec(
                        &term.outer.name,
                        term.outer.structure.clone(),
                        &inner.name,
                        inner_spec,
                        ped,
                    );
                }
            }
        }

        if let Some(structure) = residual_structure {
            builder = builder.residual_spec(structure);
        }
        if let Some(TermSpec {
            outer,
            inner: Some(inner),
        }) = &residual_term
        {
            builder = builder.residual_interaction_spec(
                &outer.name,
                outer.structure.clone(),
                &inner.name,
                inner.structure.clone(),
            );
        }

        let model = builder.build()?;
        Ok(PreparedModel {
            model,
            data: df,
            random_terms,
            residual_term,
            notes,
        })
    }
}

/// The output of [`FitSpec::prepare`].
pub struct PreparedModel {
    /// The model, ready to fit.
    pub model: MixedModel,
    /// The data the model was built from (missing responses dropped, term
    /// columns converted to factors), row-aligned with the model.
    pub data: DataFrame,
    /// Parsed random terms, in model order.
    pub random_terms: Vec<TermSpec>,
    /// Parsed residual grid term (`row*col`), if any.
    pub residual_term: Option<TermSpec>,
    /// Human-readable notes about the data preparation.
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial() -> DataFrame {
        let csv =
            "genotype,rep,yield\nG1,1,10\nG2,1,8\nG3,1,6\nG1,2,12\nG2,2,10\nG3,2,8\nG1,3,NA\n";
        DataFrame::from_csv_reader(csv.as_bytes()).unwrap()
    }

    #[test]
    fn parses_terms() {
        let t = TermSpec::parse("row:ar1*col:ar1c").unwrap();
        assert_eq!(t.outer.name, "row");
        assert!(t.outer.explicit_structure);
        assert_eq!(t.inner.as_ref().unwrap().name, "col");
        let t = TermSpec::parse(" genotype ").unwrap();
        assert_eq!(t.label(), "genotype");
        assert!(!t.outer.explicit_structure);
        assert!(TermSpec::parse("genotype:nope").is_err());
        assert!(TermSpec::parse("").is_err());
    }

    #[test]
    fn prepare_drops_missing_and_converts_factors() {
        let mut spec = FitSpec::new("yield");
        spec.fixed = "mu + rep".into();
        spec.factors = vec!["rep".into()];
        spec.random = vec!["genotype".into()];
        let prepared = spec.prepare(&trial(), None).unwrap();
        assert_eq!(prepared.data.nrows(), 6);
        assert_eq!(prepared.model.y.len(), 6);
        assert!(prepared.notes[0].contains("Dropped 1 rows"));
        assert!(prepared.data.get_factor("rep").is_ok());
    }

    #[test]
    fn prepare_reports_bad_columns() {
        let mut spec = FitSpec::new("yield");
        spec.random = vec!["missing".into()];
        let err = spec.prepare(&trial(), None).err().unwrap().to_string();
        assert!(err.contains("'missing'"), "{}", err);

        let err = FitSpec::new("genotype")
            .prepare(&trial(), None)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("not numeric"), "{}", err);
    }

    #[test]
    fn residual_without_interaction_is_a_data_order_structure() {
        let mut spec = FitSpec::new("yield");
        spec.random = vec!["genotype".into()];
        spec.residual = Some("ar1".into());
        let prepared = spec.prepare(&trial(), None).unwrap();
        assert!(prepared.residual_term.is_none());
        assert_eq!(prepared.model.residual_var_struct.n_params(), 2);

        spec.residual = Some("nope".into());
        let err = spec.prepare(&trial(), None).err().unwrap().to_string();
        assert!(err.contains("Invalid residual structure 'nope'"), "{}", err);
    }

    #[test]
    fn deserializes_with_defaults() {
        let spec: FitSpec =
            serde_json::from_str(r#"{"response": "yield", "random": ["genotype"]}"#).unwrap();
        assert_eq!(spec.fixed, "mu");
        assert_eq!(spec.max_iter, 50);
        assert_eq!(spec.residual, None);
    }
}
