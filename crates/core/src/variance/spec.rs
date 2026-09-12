//! Textual variance-structure specifications shared by the CLI and the
//! Python bindings.
//!
//! A [`StructureSpec`] is parsed from a short string and instantiated once
//! the number of levels of the term is known:
//!
//! | Spec | Structure |
//! |------|-----------|
//! | `idv`, `id`, `identity` | [`Identity`] (one variance) |
//! | `ar1`, `ar1(rho)` | [`AR1`] with variance and autocorrelation |
//! | `ar1c`, `ar1c(rho)`, `corar1(rho)` | [`AR1::correlation`] (unit variance, `rho` only) |
//! | `diag`, `diag(v)` | [`Diagonal`] with one variance per level |
//! | `us` | [`Unstructured`] (Cholesky parameterised) |
//! | `fa1`, `fa2`, ... `faK` | [`FactorAnalytic`] with `K` factors |
//! | `fixed`, `known` | [`Known::identity`] (no parameters) |

use crate::error::{LmmError, Result};

use super::{Diagonal, FactorAnalytic, Identity, Known, Unstructured, VarStruct, AR1};

/// A variance structure specification whose dimension is resolved later.
#[derive(Debug, Clone, PartialEq)]
pub enum StructureSpec {
    /// Scaled identity, one variance parameter.
    Identity { sigma2: f64 },
    /// First-order autoregressive with variance and correlation.
    Ar1 { rho: f64 },
    /// AR1 correlation only (unit variance).
    Ar1Correlation { rho: f64 },
    /// Heterogeneous variances, one per level.
    Diagonal { sigma2: f64 },
    /// Unstructured covariance.
    Unstructured,
    /// Factor analytic with `k` factors.
    FactorAnalytic { k: usize },
    /// Parameter-free identity.
    Known,
}

impl StructureSpec {
    /// Parse a specification such as `"ar1"`, `"ar1(0.3)"`, `"diag"` or `"fa2"`
    /// (case-insensitive).
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim().to_lowercase();
        let (name, arg) = match s.find('(') {
            Some(i) => {
                if !s.ends_with(')') {
                    return Err(LmmError::ModelSpec(format!(
                        "Malformed structure spec '{}': missing ')'",
                        spec
                    )));
                }
                (
                    s[..i].trim().to_string(),
                    Some(s[i + 1..s.len() - 1].trim().to_string()),
                )
            }
            None => (s.clone(), None),
        };
        let number = |what: &str| -> Result<Option<f64>> {
            match &arg {
                None => Ok(None),
                Some(a) => a.parse::<f64>().map(Some).map_err(|_| {
                    LmmError::ModelSpec(format!(
                        "Structure spec '{}': '{}' is not a valid {}",
                        spec, a, what
                    ))
                }),
            }
        };
        let no_arg = || -> Result<()> {
            if arg.is_some() {
                Err(LmmError::ModelSpec(format!(
                    "Structure spec '{}' does not take an argument",
                    spec
                )))
            } else {
                Ok(())
            }
        };
        let check_rho = |rho: f64| -> Result<f64> {
            if rho.abs() >= 1.0 {
                Err(LmmError::ModelSpec(format!(
                    "Structure spec '{}': rho must satisfy |rho| < 1",
                    spec
                )))
            } else {
                Ok(rho)
            }
        };
        match name.as_str() {
            "idv" | "id" | "identity" => Ok(StructureSpec::Identity {
                sigma2: number("variance")?.unwrap_or(1.0),
            }),
            "ar1" => Ok(StructureSpec::Ar1 {
                rho: check_rho(number("correlation")?.unwrap_or(0.5))?,
            }),
            "ar1c" | "corar1" | "ar1corr" => Ok(StructureSpec::Ar1Correlation {
                rho: check_rho(number("correlation")?.unwrap_or(0.5))?,
            }),
            "diag" | "diagonal" => Ok(StructureSpec::Diagonal {
                sigma2: number("variance")?.unwrap_or(1.0),
            }),
            "us" | "unstructured" => {
                no_arg()?;
                Ok(StructureSpec::Unstructured)
            }
            "fixed" | "known" => {
                no_arg()?;
                Ok(StructureSpec::Known)
            }
            _ => {
                if let Some(k) = name.strip_prefix("fa") {
                    no_arg()?;
                    let k: usize = k.parse().map_err(|_| {
                        LmmError::ModelSpec(format!(
                            "Unknown variance structure '{}' (expected fa1, fa2, ...)",
                            spec
                        ))
                    })?;
                    if k == 0 {
                        return Err(LmmError::ModelSpec(
                            "Factor analytic structure needs at least one factor".into(),
                        ));
                    }
                    Ok(StructureSpec::FactorAnalytic { k })
                } else {
                    Err(LmmError::ModelSpec(format!(
                        "Unknown variance structure '{}' (expected idv, ar1, ar1c, diag, us, fa1, fa2, ... or fixed)",
                        spec
                    )))
                }
            }
        }
    }

    /// Instantiate the structure for a term with `dim` levels.
    pub fn instantiate(&self, dim: usize) -> Result<Box<dyn VarStruct>> {
        Ok(match self {
            StructureSpec::Identity { sigma2 } => Box::new(Identity::new(*sigma2)),
            StructureSpec::Ar1 { rho } => Box::new(AR1::new(1.0, *rho)),
            StructureSpec::Ar1Correlation { rho } => Box::new(AR1::correlation(*rho)),
            StructureSpec::Diagonal { sigma2 } => Box::new(Diagonal::new(vec![*sigma2; dim])),
            StructureSpec::Unstructured => Box::new(Unstructured::default_start(dim)),
            StructureSpec::FactorAnalytic { k } => {
                if *k > dim {
                    return Err(LmmError::ModelSpec(format!(
                        "fa{} needs at least {} levels, the term has {}",
                        k, k, dim
                    )));
                }
                Box::new(FactorAnalytic::new(dim, *k))
            }
            StructureSpec::Known => Box::new(Known::identity(dim)),
        })
    }

    /// Whether this spec is the plain scaled identity.
    pub fn is_identity(&self) -> bool {
        matches!(self, StructureSpec::Identity { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_specs() {
        assert_eq!(
            StructureSpec::parse("idv").unwrap(),
            StructureSpec::Identity { sigma2: 1.0 }
        );
        assert_eq!(
            StructureSpec::parse("AR1").unwrap(),
            StructureSpec::Ar1 { rho: 0.5 }
        );
        assert_eq!(
            StructureSpec::parse("ar1(0.3)").unwrap(),
            StructureSpec::Ar1 { rho: 0.3 }
        );
        assert_eq!(
            StructureSpec::parse("ar1c(-0.2)").unwrap(),
            StructureSpec::Ar1Correlation { rho: -0.2 }
        );
        assert_eq!(
            StructureSpec::parse("diag").unwrap(),
            StructureSpec::Diagonal { sigma2: 1.0 }
        );
        assert_eq!(
            StructureSpec::parse("us").unwrap(),
            StructureSpec::Unstructured
        );
        assert_eq!(
            StructureSpec::parse("fa2").unwrap(),
            StructureSpec::FactorAnalytic { k: 2 }
        );
        assert_eq!(StructureSpec::parse("known").unwrap(), StructureSpec::Known);
        for bad in [
            "ar1(1.2)", "ar1(x)", "fa0", "fa", "us(2)", "nope", "ar1(0.5",
        ] {
            assert!(StructureSpec::parse(bad).is_err(), "{} should fail", bad);
        }
    }

    #[test]
    fn test_instantiate() {
        let s = StructureSpec::parse("diag")
            .unwrap()
            .instantiate(4)
            .unwrap();
        assert_eq!(s.n_params(), 4);
        let s = StructureSpec::parse("fa1").unwrap().instantiate(3).unwrap();
        assert_eq!(s.n_params(), 6);
        assert!(StructureSpec::parse("fa3").unwrap().instantiate(2).is_err());
        let s = StructureSpec::parse("ar1c(0.4)")
            .unwrap()
            .instantiate(10)
            .unwrap();
        assert_eq!(s.params(), vec![0.4]);
        assert_eq!(
            StructureSpec::parse("known")
                .unwrap()
                .instantiate(5)
                .unwrap()
                .n_params(),
            0
        );
    }
}
