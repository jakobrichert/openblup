//! WebAssembly build of the full OpenBLUP engine for **OpenBLUP Studio**, the
//! browser interface in `studio/`.
//!
//! Unlike the small, dependency-free `openblup-wasm` crate, this crate links
//! the complete `plant-breeding-lmm-core` engine (AI-REML, the general
//! variance-structure engine, sparse MME, Wald tests, diagnostics and
//! cross-validation). Models are described with the same term vocabulary as
//! the CLI ([`FitSpec`]), so the Studio can show the equivalent command line.
//!
//! The JavaScript API takes and returns JSON strings:
//!
//! - [`inspect`]: column summary and preview of a CSV file
//! - [`fit`]: fit a model and return everything the Studio plots
//!
//! The same functions are available natively as [`inspect_value`] and
//! [`fit_value`] for testing.

use serde::Deserialize;
use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

use plant_breeding_lmm_core as core;

use core::data::{Column, DataFrame};
use core::diagnostics::{wald_tests, CrossValResult, WaldTest};
use core::genetics::{compute_inbreeding, Pedigree};
use core::lmm::{AiReml, EmReml, FitResult};
use core::model::{FitSpec, PreparedModel, TermSpec};
use core::types::SparseMat;
use core::variance::StructureSpec;

/// Version of the engine.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Summarise a CSV file: `{n_rows, columns: [...], preview: [[...]]}`.
#[wasm_bindgen]
pub fn inspect(csv: &str) -> Result<String, JsError> {
    inspect_value(csv)
        .map(|v| v.to_string())
        .map_err(|e| JsError::new(&e))
}

/// Fit a model described by a JSON [`FitRequest`] and return the results as
/// JSON.
#[wasm_bindgen]
pub fn fit(request: &str) -> Result<String, JsError> {
    let request: FitRequest =
        serde_json::from_str(request).map_err(|e| JsError::new(&format!("Bad request: {}", e)))?;
    fit_value(&request)
        .map(|v| v.to_string())
        .map_err(|e| JsError::new(&e))
}

/// REML algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Algorithm {
    /// Average Information REML.
    #[default]
    Ai,
    /// EM-REML (identity structures only).
    Em,
}

/// Denominator degrees of freedom for the Wald F-tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ddf {
    /// n - rank(X).
    #[default]
    Containment,
    /// Satterthwaite (falls back to containment).
    Satterthwaite,
    /// Kenward-Roger (falls back to Satterthwaite, then containment).
    KenwardRoger,
}

impl Ddf {
    fn name(self) -> &'static str {
        match self {
            Ddf::Containment => "containment",
            Ddf::Satterthwaite => "satterthwaite",
            Ddf::KenwardRoger => "kenward-roger",
        }
    }
}

/// A fit request from the Studio.
#[derive(Debug, Clone, Deserialize)]
pub struct FitRequest {
    /// The data as CSV text.
    pub data: String,
    /// Optional pedigree as CSV text (`animal,sire,dam`).
    #[serde(default)]
    pub pedigree: Option<String>,
    /// The model.
    pub spec: FitSpec,
    /// REML algorithm.
    #[serde(default)]
    pub algorithm: Algorithm,
    /// Denominator df method.
    #[serde(default)]
    pub ddf: Ddf,
    /// Number of cross-validation folds (none when absent).
    #[serde(default)]
    pub cv_folds: Option<usize>,
    /// Seed for the cross-validation folds.
    #[serde(default)]
    pub cv_seed: u64,
}

/// Native form of [`inspect`].
pub fn inspect_value(csv: &str) -> Result<Value, String> {
    let df = DataFrame::from_csv_reader(csv.as_bytes()).map_err(|e| e.to_string())?;
    let n = df.nrows();
    let preview_rows = n.min(8);
    let mut columns = Vec::new();
    let mut preview: Vec<Vec<Value>> = vec![Vec::new(); preview_rows];
    for name in df.column_names() {
        let col = df.get_column(name).map_err(|e| e.to_string())?;
        match col {
            Column::Float(v) => {
                let present: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
                let integer_like = present.iter().all(|x| x.fract() == 0.0);
                let mut distinct: Vec<f64> = present.clone();
                distinct.sort_by(f64::total_cmp);
                distinct.dedup();
                columns.push(json!({
                    "name": name,
                    "kind": "numeric",
                    "n_missing": n - present.len(),
                    "n_distinct": distinct.len(),
                    "integer_like": integer_like,
                    "min": distinct.first(),
                    "max": distinct.last(),
                }));
                for (i, row) in preview.iter_mut().enumerate() {
                    row.push(json!(v[i]));
                }
            }
            Column::Integer(v) => {
                let mut distinct = v.clone();
                distinct.sort_unstable();
                distinct.dedup();
                columns.push(json!({
                    "name": name,
                    "kind": "numeric",
                    "n_missing": 0,
                    "n_distinct": distinct.len(),
                    "integer_like": true,
                    "min": distinct.first(),
                    "max": distinct.last(),
                }));
                for (i, row) in preview.iter_mut().enumerate() {
                    row.push(json!(v[i]));
                }
            }
            Column::Factor(f) => {
                let levels = f.level_names();
                columns.push(json!({
                    "name": name,
                    "kind": "factor",
                    "n_missing": 0,
                    "n_distinct": levels.len(),
                    "levels": levels.iter().take(12).collect::<Vec<_>>(),
                }));
                for (i, row) in preview.iter_mut().enumerate() {
                    row.push(json!(f.level_name(f.codes()[i])));
                }
            }
        }
    }
    Ok(json!({ "n_rows": n, "columns": columns, "preview": preview }))
}

/// Native form of [`fit`].
pub fn fit_value(request: &FitRequest) -> Result<Value, String> {
    if request.cv_folds.is_some_and(|k| k < 2) {
        return Err("Cross-validation needs at least 2 folds".into());
    }
    let df = DataFrame::from_csv_reader(request.data.as_bytes())
        .map_err(|e| format!("Could not read the data: {}", e))?;
    let pedigree = match &request.pedigree {
        Some(csv) => Some(load_pedigree(csv)?),
        None => None,
    };
    if pedigree.is_some() && request.spec.random.is_empty() {
        return Err("A pedigree needs at least one random term".into());
    }

    let prepared = request
        .spec
        .prepare(&df, pedigree.as_ref())
        .map_err(|e| e.to_string())?;
    let engine = if prepared.model.needs_general_engine() {
        "general"
    } else {
        "sparse"
    };
    let n_levels: Vec<usize> = prepared
        .model
        .random_level_names
        .iter()
        .map(|l| l.len())
        .collect();
    let PreparedModel {
        mut model,
        data,
        random_terms,
        residual_term,
        notes,
    } = prepared;

    let result = match request.algorithm {
        Algorithm::Ai => AiReml::new(request.spec.max_iter, request.spec.tolerance)
            .fit(&mut model)
            .map_err(|e| format!("AI-REML failed: {}", e))?,
        Algorithm::Em => EmReml::new(request.spec.max_iter, request.spec.tolerance)
            .fit(&mut model)
            .map_err(|e| format!("EM-REML failed: {}", e))?,
    };

    let mut warnings = Vec::new();
    if !result.converged {
        warnings.push(format!(
            "REML did not converge in {} iterations; results may be unreliable.",
            result.n_iterations
        ));
    }

    let (tests, ddf_used) = wald_tests_for(&result, request.ddf);
    if ddf_used != request.ddf {
        warnings.push(format!(
            "{} df are not available for this fit (they need an AI-REML fit with no variance \
             parameter on the boundary; Kenward-Roger also needs scaled-identity terms and an \
             IID residual). Reporting {} df.",
            request.ddf.name(),
            ddf_used.name()
        ));
    }

    let cv = match request.cv_folds {
        Some(k) => match model.cross_validate(k, request.cv_seed) {
            Ok(cv) => Some(cv),
            Err(core::LmmError::ModelSpec(msg)) => {
                warnings.push(format!("Cross-validation skipped: {}.", msg));
                None
            }
            Err(e) => return Err(format!("Cross-validation failed: {}", e)),
        },
        None => None,
    };

    let diagnostics = result.residual_diagnostics(&model).map(|d| {
        json!({
            "fitted": d.fitted,
            "standardized": d.standardized,
            "marginal": d.marginal,
            "leverage": d.leverage,
            "cooks_distance": d.cooks_distance,
        })
    });

    // Per-observation labels for every column a term uses (and the response),
    // so the Studio can group residuals by environment, draw the field, ...
    let mut labels = serde_json::Map::new();
    for term in random_terms.iter().chain(residual_term.iter()) {
        for col in term.columns() {
            if labels.contains_key(col) {
                continue;
            }
            if let Ok(f) = data.get_factor(col) {
                let names: Vec<&str> = f
                    .codes()
                    .iter()
                    .map(|&c| f.level_name(c).unwrap_or(""))
                    .collect();
                labels.insert(col.to_string(), json!(names));
            }
        }
    }
    let fitted: Vec<f64> = model
        .y
        .iter()
        .zip(&result.residuals)
        .map(|(y, e)| y - e)
        .collect();

    let grid = model
        .residual_grid
        .as_ref()
        .zip(residual_term.as_ref())
        .map(|(g, t)| {
            json!({
                "row": t.outer.name,
                "col": t.inner.as_ref().map(|f| f.name.as_str()),
                "row_levels": g.row_levels,
                "col_levels": g.col_levels,
                "cell_index": g.cell_index,
            })
        });

    let covariances = structured_covariances(&data, &random_terms, residual_term.as_ref(), &result);

    let pedigree_info = match &pedigree {
        Some(ped) => {
            let f = compute_inbreeding(ped).map_err(|e| e.to_string())?;
            let n = f.len().max(1) as f64;
            Some(json!({
                "n_animals": ped.n_animals(),
                "mean_inbreeding": f.iter().sum::<f64>() / n,
                "max_inbreeding": f.iter().cloned().fold(0.0_f64, f64::max),
                "n_inbred": f.iter().filter(|&&x| x > 1e-12).count(),
                "ids": (0..ped.n_animals()).map(|i| ped.animal_id(i)).collect::<Vec<_>>(),
                "inbreeding": f,
            }))
        }
        None => None,
    };

    let terms: Vec<Value> = random_terms
        .iter()
        .enumerate()
        .map(|(i, t)| {
            json!({
                "label": t.label(),
                "columns": t.columns(),
                "outer_structure": structure_name(&t.outer.structure),
                "inner_structure": t.inner.as_ref().map(|f| structure_name(&f.structure)),
                "n_levels": n_levels.get(i),
            })
        })
        .collect();

    Ok(json!({
        "engine": engine,
        "notes": notes,
        "warnings": warnings,
        "model": {
            "response": request.spec.response,
            "n_obs": result.n_obs,
            "n_fixed": result.n_fixed_params,
            "n_variance_params": result.n_variance_params,
            "random_terms": terms,
            "residual": residual_term.as_ref().map(|t| t.label()),
        },
        "fit": result,
        "aic": result.aic(),
        "bic": result.bic(),
        "ddf_method": ddf_used.name(),
        "wald_tests": tests.iter().map(wald_to_json).collect::<Vec<_>>(),
        "observations": {
            "y": model.y,
            "fitted": fitted,
            "labels": labels,
        },
        "diagnostics": diagnostics,
        "grid": grid,
        "covariances": covariances,
        "pedigree": pedigree_info,
        "cv": cv.as_ref().map(cv_to_json),
    }))
}

fn load_pedigree(csv: &str) -> Result<Pedigree, String> {
    let mut ped = Pedigree::from_csv_reader(csv.as_bytes())
        .map_err(|e| format!("Could not read the pedigree: {}", e))?;
    ped.validate()
        .map_err(|e| format!("The pedigree is inconsistent: {}", e))?;
    ped.sort_pedigree()
        .map_err(|e| format!("Could not sort the pedigree: {}", e))?;
    Ok(ped)
}

fn wald_tests_for(result: &FitResult, ddf: Ddf) -> (Vec<WaldTest>, Ddf) {
    if ddf == Ddf::KenwardRoger {
        if let Some(tests) = result.wald_tests_kenward_roger() {
            return (tests, Ddf::KenwardRoger);
        }
    }
    if ddf != Ddf::Containment {
        if let Some(tests) = result.wald_tests_satterthwaite() {
            return (tests, Ddf::Satterthwaite);
        }
    }
    (wald_tests(result), Ddf::Containment)
}

fn wald_to_json(t: &WaldTest) -> Value {
    json!({
        "term": t.term,
        "f_statistic": t.f_statistic,
        "num_df": t.num_df,
        "den_df": t.den_df,
        "p_value": t.p_value,
    })
}

fn cv_to_json(cv: &CrossValResult) -> Value {
    json!({
        "accuracy": cv.accuracy,
        "msep": cv.msep,
        "bias": cv.bias,
        "mae": cv.mae,
        "folds": cv.folds.iter().map(|f| json!({
            "fold": f.fold,
            "n_validation": f.validation_indices.len(),
            "accuracy": f.accuracy,
            "msep": f.msep,
            "predicted": f.predicted,
            "observed": f.observed,
        })).collect::<Vec<_>>(),
    })
}

fn structure_name(s: &StructureSpec) -> String {
    match s {
        StructureSpec::Identity { .. } => "idv".into(),
        StructureSpec::Ar1 { .. } => "ar1".into(),
        StructureSpec::Ar1Correlation { .. } => "ar1c".into(),
        StructureSpec::Diagonal { .. } => "diag".into(),
        StructureSpec::Unstructured => "us".into(),
        StructureSpec::FactorAnalytic { k } => format!("fa{}", k),
        StructureSpec::Known => "fixed".into(),
    }
}

/// Covariance and correlation matrices of the outer factor of every
/// interaction term with a diag / US / FA structure (for example the genetic
/// covariance between environments of `env:fa1*genotype`).
fn structured_covariances(
    data: &DataFrame,
    random_terms: &[TermSpec],
    residual_term: Option<&TermSpec>,
    result: &FitResult,
) -> Vec<Value> {
    let residual = residual_term.map(|t| (t, "residual".to_string()));
    let terms = random_terms.iter().map(|t| (t, t.label())).chain(residual);
    let mut out = Vec::new();
    for (term, component) in terms {
        let spec = &term.outer.structure;
        let k = match spec {
            StructureSpec::Diagonal { .. } | StructureSpec::Unstructured => 0,
            StructureSpec::FactorAnalytic { k } => *k,
            _ => continue,
        };
        if term.inner.is_none() {
            continue;
        }
        let Some(vc) = result
            .variance_components
            .iter()
            .find(|v| v.name == component)
        else {
            continue;
        };
        let Ok(factor) = data.get_factor(&term.outer.name) else {
            continue;
        };
        let levels: Vec<String> = factor.level_names().iter().map(|s| s.to_string()).collect();
        let dim = levels.len();
        let prefix = format!("{}.", term.outer.name);
        let params: Vec<(&str, f64)> = vc
            .parameters
            .iter()
            .filter_map(|(n, v)| n.strip_prefix(&prefix).map(|n| (n, *v)))
            .collect();
        let Ok(mut vs) = spec.instantiate(dim) else {
            continue;
        };
        let values: Vec<f64> = params.iter().map(|(_, v)| *v).collect();
        if values.len() != vs.n_params() || vs.set_params(&values).is_err() {
            continue;
        }
        let cov = dense(&vs.covariance_matrix(dim), dim);
        let corr: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let d = (cov[i][i] * cov[j][j]).sqrt();
                        if d > 0.0 {
                            cov[i][j] / d
                        } else if i == j {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let mut entry = json!({
            "term": component,
            "factor": term.outer.name,
            "structure": structure_name(spec),
            "levels": levels,
            "covariance": cov,
            "correlation": corr,
        });
        if k > 0 {
            let param = |name: String| params.iter().find(|(n, _)| *n == name).map(|(_, v)| *v);
            let loadings: Vec<Vec<Option<f64>>> = (1..=dim)
                .map(|i| {
                    (1..=k)
                        .map(|f| param(format!("lambda_{}_{}", i, f)))
                        .collect()
                })
                .collect();
            let specific: Vec<Option<f64>> =
                (1..=dim).map(|i| param(format!("psi_{}", i))).collect();
            entry["loadings"] = json!(loadings);
            entry["specific"] = json!(specific);
        }
        out.push(entry);
    }
    out
}

fn dense(m: &SparseMat, dim: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; dim]; dim];
    for (v, (i, j)) in m.iter() {
        if i < dim && j < dim {
            out[i][j] = *v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example(name: &str) -> String {
        std::fs::read_to_string(format!(
            "{}/../../examples/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        ))
        .unwrap()
    }

    fn request(data: &str, spec: &str) -> FitRequest {
        serde_json::from_value(json!({
            "data": data,
            "spec": serde_json::from_str::<Value>(spec).unwrap(),
        }))
        .unwrap()
    }

    #[test]
    fn inspect_reports_columns_and_missing_values() {
        let v = inspect_value(&example("field_trial.csv")).unwrap();
        assert_eq!(v["n_rows"], 18);
        let cols = v["columns"].as_array().unwrap();
        let yield_col = cols.iter().find(|c| c["name"] == "yield").unwrap();
        assert_eq!(yield_col["kind"], "numeric");
        assert_eq!(yield_col["n_missing"], 1);
        let rep = cols.iter().find(|c| c["name"] == "rep").unwrap();
        assert_eq!(rep["integer_like"], true);
        let genotype = cols.iter().find(|c| c["name"] == "genotype").unwrap();
        assert_eq!(genotype["kind"], "factor");
        assert_eq!(genotype["n_distinct"], 6);
        assert_eq!(v["preview"].as_array().unwrap().len(), 8);
        assert!(inspect_value("a,b\n1\n").is_err());
    }

    #[test]
    fn fits_a_simple_plant_trial() {
        let mut req = request(
            &example("field_trial.csv"),
            r#"{"response": "yield", "fixed": "mu + rep", "factors": ["rep"], "random": ["genotype"]}"#,
        );
        req.ddf = Ddf::Satterthwaite;
        req.cv_folds = Some(3);
        let v = fit_value(&req).unwrap();
        assert_eq!(v["engine"], "sparse");
        assert_eq!(v["model"]["n_obs"], 17);
        assert_eq!(v["fit"]["converged"], true);
        assert_eq!(v["ddf_method"], "satterthwaite");
        assert_eq!(v["observations"]["y"].as_array().unwrap().len(), 17);
        assert_eq!(
            v["observations"]["labels"]["genotype"]
                .as_array()
                .unwrap()
                .len(),
            17
        );
        assert!(v["diagnostics"]["leverage"].is_array());
        assert_eq!(v["cv"]["folds"].as_array().unwrap().len(), 3);
        assert!(v["grid"].is_null());
        assert!(v["notes"][0].as_str().unwrap().contains("Dropped 1 rows"));
    }

    #[test]
    fn spatial_residual_exposes_the_field_grid() {
        let mut req = request(
            &example("field_trial.csv"),
            r#"{"response": "yield", "fixed": "mu + rep", "factors": ["rep"],
                "random": ["genotype"], "residual": "row:ar1*col:ar1c"}"#,
        );
        req.cv_folds = Some(3);
        let v = fit_value(&req).unwrap();
        assert_eq!(v["engine"], "general");
        assert_eq!(v["grid"]["row_levels"].as_array().unwrap().len(), 3);
        assert_eq!(v["grid"]["col_levels"].as_array().unwrap().len(), 6);
        assert_eq!(v["grid"]["cell_index"].as_array().unwrap().len(), 17);
        assert!(v["diagnostics"].is_null());
        assert!(v["cv"].is_null());
        assert!(v["warnings"][0]
            .as_str()
            .unwrap()
            .contains("Cross-validation skipped"));
    }

    #[test]
    fn factor_analytic_term_reports_genetic_correlations() {
        let req = request(
            &example("met_trial.csv"),
            r#"{"response": "yield", "fixed": "mu + env", "random": ["env:fa1*genotype"]}"#,
        );
        let v = fit_value(&req).unwrap();
        let cov = &v["covariances"][0];
        assert_eq!(cov["term"], "env:genotype");
        assert_eq!(cov["structure"], "fa1");
        assert_eq!(cov["levels"].as_array().unwrap().len(), 4);
        // Var_i = lambda_i^2 + psi_i, and the correlation matrix has a unit diagonal.
        for i in 0..4 {
            let l = cov["loadings"][i][0].as_f64().unwrap();
            let psi = cov["specific"][i].as_f64().unwrap();
            let var = cov["covariance"][i][i].as_f64().unwrap();
            assert!(
                (var - (l * l + psi)).abs() < 1e-8,
                "{} vs {}",
                var,
                l * l + psi
            );
            assert!((cov["correlation"][i][i].as_f64().unwrap() - 1.0).abs() < 1e-12);
        }
        let r01 = cov["correlation"][0][1].as_f64().unwrap();
        assert!(r01 > 0.0 && r01 < 1.0);
    }

    #[test]
    fn animal_model_with_pedigree() {
        let mut req = request(
            &example("animal_records.csv"),
            r#"{"response": "weight", "fixed": "sex", "random": ["animal"]}"#,
        );
        req.pedigree = Some(example("animal_pedigree.csv"));
        req.ddf = Ddf::KenwardRoger;
        let v = fit_value(&req).unwrap();
        assert_eq!(v["pedigree"]["n_animals"], 63);
        assert_eq!(
            v["fit"]["random_effects"][0]["effects"]
                .as_array()
                .unwrap()
                .len(),
            63
        );
        assert_eq!(v["ddf_method"], "kenward-roger");
        let s2a = v["fit"]["variance_components"][0]["parameters"][0][1]
            .as_f64()
            .unwrap();
        assert!((s2a - 36.54).abs() < 0.01, "{}", s2a);
    }

    #[test]
    fn reports_user_errors() {
        let req = request(
            &example("field_trial.csv"),
            r#"{"response": "yield", "random": ["nope"]}"#,
        );
        assert!(fit_value(&req).unwrap_err().contains("'nope'"));
        let mut req = request(&example("field_trial.csv"), r#"{"response": "yield"}"#);
        req.pedigree = Some(example("mrode_pedigree.csv"));
        assert!(fit_value(&req).unwrap_err().contains("random term"));
    }
}
