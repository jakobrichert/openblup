//! OpenBLUP WebAssembly module.
//!
//! Provides browser-friendly breeding value estimation with self-contained
//! dense implementations (no rayon, no faer, no file I/O). The JSON-based
//! functions are exported to JavaScript through `wasm-bindgen`:
//!
//! ```js
//! import init, { computeAInverse, fitMixedModel, computeGMatrix } from "./pkg/openblup_wasm.js";
//! await init();
//! const ainv = JSON.parse(computeAInverse(JSON.stringify(pedigree)));
//! ```
//!
//! Build with `wasm-pack build crates/wasm --target web --out-dir www/pkg`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

mod ainverse;
mod gmatrix;
mod mme;
mod pedigree;
mod reml;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Pedigree entry for JSON input. Unknown parents are `"0"`, `""` or `"NA"`.
#[derive(Deserialize)]
pub struct PedigreeEntry {
    pub animal: String,
    #[serde(default)]
    pub sire: String,
    #[serde(default)]
    pub dam: String,
}

/// Model input for JSON.
#[derive(Deserialize)]
pub struct ModelInput {
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    pub z: Vec<Vec<f64>>,
    #[serde(default)]
    pub ginv: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub max_iter: Option<usize>,
    #[serde(default)]
    pub tolerance: Option<f64>,
}

/// Model output.
#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    pub fixed_effects: Vec<f64>,
    pub random_effects: Vec<f64>,
    pub sigma2_random: f64,
    pub sigma2_residual: f64,
    pub heritability: f64,
    pub log_likelihood: f64,
    pub converged: bool,
    pub n_iterations: usize,
}

/// Compute A-inverse from a pedigree (JSON input/output).
///
/// Input: a JSON array of `{"animal": .., "sire": .., "dam": ..}` objects.
/// Output: `{"dim": n, "animal_ids": [...], "values": [...]}` where `values`
/// is the dense n x n matrix in row-major order following `animal_ids`
/// (sorted so that parents precede offspring).
pub fn compute_a_inverse_json(pedigree_json: &str) -> Result<String, String> {
    let entries: Vec<PedigreeEntry> =
        serde_json::from_str(pedigree_json).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut ped = pedigree::WasmPedigree::new();
    for entry in &entries {
        ped.add_animal(&entry.animal, &entry.sire, &entry.dam);
    }

    let sorted = ped.sort()?;
    let a_inv = ainverse::compute_a_inverse(&sorted);
    let n = a_inv.nrows();

    let a_inv_ref = &a_inv;
    let values: Vec<f64> = (0..n)
        .flat_map(|i| (0..n).map(move |j| a_inv_ref[(i, j)]))
        .collect();

    let result = serde_json::json!({
        "dim": n,
        "values": values,
        "animal_ids": sorted.ids,
        "inbreeding": sorted.inbreeding,
    });

    serde_json::to_string(&result).map_err(|e| format!("JSON serialize error: {}", e))
}

/// Fit a simple mixed model with one random term (JSON input/output).
///
/// Input: `{"y": [...], "x": [[..]], "z": [[..]], "ginv": [[..]] | null,
/// "max_iter": 200, "tolerance": 1e-6}`; `x` and `z` are given row by row.
pub fn fit_mixed_model_json(model_json: &str) -> Result<String, String> {
    let input: ModelInput =
        serde_json::from_str(model_json).map_err(|e| format!("JSON parse error: {}", e))?;

    let n = input.y.len();
    let p = input.x.first().map(|r| r.len()).unwrap_or(0);
    let q = input.z.first().map(|r| r.len()).unwrap_or(0);

    if input.x.len() != n || input.z.len() != n {
        return Err("X and Z must have same number of rows as y".into());
    }
    if input.x.iter().any(|r| r.len() != p) {
        return Err("All rows of X must have the same length".into());
    }
    if input.z.iter().any(|r| r.len() != q) {
        return Err("All rows of Z must have the same length".into());
    }

    let x = nalgebra::DMatrix::from_fn(n, p, |i, j| input.x[i][j]);
    let z = nalgebra::DMatrix::from_fn(n, q, |i, j| input.z[i][j]);

    let ginv = match input.ginv {
        Some(g) => {
            let dim = g.len();
            if dim != q || g.iter().any(|r| r.len() != q) {
                return Err(format!(
                    "ginv must be a {0}x{0} matrix (one row/col per column of Z)",
                    q
                ));
            }
            Some(nalgebra::DMatrix::from_fn(dim, dim, |i, j| g[i][j]))
        }
        None => None,
    };

    let result = reml::fit_em_reml(
        &input.y,
        &x,
        &z,
        ginv.as_ref(),
        input.max_iter.unwrap_or(200),
        input.tolerance.unwrap_or(1e-6),
    )
    .map_err(|e| format!("Fit error: {}", e))?;

    let total = result.sigma2_random + result.sigma2_residual;
    let output = ModelOutput {
        fixed_effects: result.fixed_effects,
        random_effects: result.random_effects,
        sigma2_random: result.sigma2_random,
        sigma2_residual: result.sigma2_residual,
        heritability: if total > 0.0 {
            result.sigma2_random / total
        } else {
            0.0
        },
        log_likelihood: result.log_likelihood,
        converged: result.converged,
        n_iterations: result.n_iterations,
    };

    serde_json::to_string(&output).map_err(|e| format!("JSON serialize error: {}", e))
}

/// Compute G-matrix (VanRaden method 1) from row-major marker data.
pub fn compute_g_matrix_flat(
    markers: &[f64],
    n_individuals: usize,
    n_markers: usize,
) -> Result<Vec<f64>, String> {
    if markers.len() != n_individuals * n_markers {
        return Err("Marker array length must equal n_individuals * n_markers".into());
    }

    let m = nalgebra::DMatrix::from_fn(n_individuals, n_markers, |i, j| markers[i * n_markers + j]);

    let g = gmatrix::compute_g_matrix(&m).map_err(|e| format!("G-matrix error: {}", e))?;

    let g_ref = &g;
    let values: Vec<f64> = (0..n_individuals)
        .flat_map(|i| (0..n_individuals).map(move |j| g_ref[(i, j)]))
        .collect();

    Ok(values)
}

// ---------------------------------------------------------------------------
// JavaScript exports
// ---------------------------------------------------------------------------

/// Library version.
#[wasm_bindgen(js_name = version)]
pub fn version_js() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// `computeAInverse(pedigreeJson) -> resultJson` (see [`compute_a_inverse_json`]).
#[wasm_bindgen(js_name = computeAInverse)]
pub fn compute_a_inverse_js(pedigree_json: &str) -> Result<String, JsError> {
    compute_a_inverse_json(pedigree_json).map_err(|e| JsError::new(&e))
}

/// `fitMixedModel(modelJson) -> resultJson` (see [`fit_mixed_model_json`]).
#[wasm_bindgen(js_name = fitMixedModel)]
pub fn fit_mixed_model_js(model_json: &str) -> Result<String, JsError> {
    fit_mixed_model_json(model_json).map_err(|e| JsError::new(&e))
}

/// `computeGMatrix(markers: Float64Array, nIndividuals, nMarkers) -> Float64Array`
/// with the G-matrix in row-major order (see [`compute_g_matrix_flat`]).
#[wasm_bindgen(js_name = computeGMatrix)]
pub fn compute_g_matrix_js(
    markers: &[f64],
    n_individuals: usize,
    n_markers: usize,
) -> Result<Vec<f64>, JsError> {
    compute_g_matrix_flat(markers, n_individuals, n_markers).map_err(|e| JsError::new(&e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_a_inverse_json() {
        let json = r#"[
            {"animal": "3", "sire": "1", "dam": "2"},
            {"animal": "1", "sire": "0", "dam": "0"},
            {"animal": "2", "sire": "", "dam": "NA"}
        ]"#;
        let result = compute_a_inverse_json(json).unwrap();
        let v: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(v["dim"], 3);
        let ids: Vec<String> = v["animal_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        // Parents before offspring.
        assert_eq!(ids[2], "3");
        let values = v["values"].as_array().unwrap();
        assert_eq!(values.len(), 9);
        // Offspring with both parents known: diagonal 2, parent diagonals 1.5.
        assert_relative_eq!(values[8].as_f64().unwrap(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(values[0].as_f64().unwrap(), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_a_inverse_json_cycle_errors() {
        let json = r#"[
            {"animal": "1", "sire": "2", "dam": "0"},
            {"animal": "2", "sire": "1", "dam": "0"}
        ]"#;
        assert!(compute_a_inverse_json(json).is_err());
    }

    #[test]
    fn test_fit_model_json() {
        let json = r#"{
            "y": [10.0, 12.0, 6.0, 8.0],
            "x": [[1.0], [1.0], [1.0], [1.0]],
            "z": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            "ginv": null
        }"#;
        let result = fit_mixed_model_json(json).unwrap();
        let output: ModelOutput = serde_json::from_str(&result).unwrap();
        assert_eq!(output.fixed_effects.len(), 1);
        assert_eq!(output.random_effects.len(), 2);
        assert!(output.log_likelihood.is_finite());
        assert!(output.random_effects[0] > output.random_effects[1]);
        assert!(output.heritability > 0.0 && output.heritability < 1.0);
    }

    #[test]
    fn test_fit_model_json_bad_ginv_dims() {
        let json = r#"{
            "y": [10.0, 12.0, 6.0, 8.0],
            "x": [[1.0], [1.0], [1.0], [1.0]],
            "z": [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            "ginv": [[1.0]]
        }"#;
        assert!(fit_mixed_model_json(json).is_err());
    }

    #[test]
    fn test_g_matrix_flat() {
        let markers = vec![0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let result = compute_g_matrix_flat(&markers, 3, 3).unwrap();
        assert_eq!(result.len(), 9); // 3×3
                                     // symmetric
        assert_relative_eq!(result[1], result[3], epsilon = 1e-12);
    }

    /// The dense WASM EM-REML must agree with the core engine, including the
    /// tr(K⁻¹ C^{uu}) term for a non-identity relationship matrix.
    #[test]
    fn test_em_reml_matches_core_engine_with_relationship_matrix() {
        use plant_breeding_lmm_core::data::DataFrame;
        use plant_breeding_lmm_core::lmm::EmReml;
        use plant_breeding_lmm_core::model::MixedModelBuilder;
        use plant_breeding_lmm_core::variance::Identity;

        // 3 genotypes x 4 reps with a non-trivial 3x3 K⁻¹.
        let y = vec![
            10.2, 11.1, 9.7, 10.9, // G1
            8.1, 8.9, 7.6, 8.4, // G2
            6.3, 5.9, 6.8, 6.1, // G3
        ];
        let genos = ["G1", "G2", "G3"];
        let geno_col: Vec<&str> = (0..12).map(|i| genos[i / 4]).collect();
        let kinv_rows = [[1.5, -0.5, 0.0], [-0.5, 2.0, -0.5], [0.0, -0.5, 1.5]];

        // Core engine via the builder.
        let mut df = DataFrame::new();
        df.add_float_column("y", y.clone()).unwrap();
        df.add_factor_column("g", &geno_col).unwrap();
        let mut tri = sprs::TriMat::new((3, 3));
        for (i, row) in kinv_rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    tri.add_triplet(i, j, v);
                }
            }
        }
        let levels: Vec<String> = genos.iter().map(|s| s.to_string()).collect();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .fixed("mu")
            .random_with_levels("g", Identity::new(1.0), Some(tri.to_csc()), levels)
            .build()
            .unwrap();
        let core_fit = EmReml::new(500, 1e-10).fit(&mut model).unwrap();
        let core_sigma2_g = core_fit.variance_component("g").unwrap();
        let core_sigma2_e = core_fit.variance_component("residual").unwrap();

        // WASM engine.
        let x = nalgebra::DMatrix::from_element(12, 1, 1.0);
        let z = nalgebra::DMatrix::from_fn(12, 3, |i, j| if i / 4 == j { 1.0 } else { 0.0 });
        let kinv = nalgebra::DMatrix::from_fn(3, 3, |i, j| kinv_rows[i][j]);
        let wasm_fit = reml::fit_em_reml(&y, &x, &z, Some(&kinv), 500, 1e-10).unwrap();

        assert_relative_eq!(wasm_fit.sigma2_random, core_sigma2_g, max_relative = 1e-4);
        assert_relative_eq!(wasm_fit.sigma2_residual, core_sigma2_e, max_relative = 1e-4);
        for (a, b) in wasm_fit
            .random_effects
            .iter()
            .zip(core_fit.random_effects[0].effects.iter())
        {
            assert_relative_eq!(*a, b.estimate, max_relative = 1e-4);
        }
        // Both report the exact REML log-likelihood, including log|K|.
        assert_relative_eq!(
            wasm_fit.log_likelihood,
            core_fit.log_likelihood,
            max_relative = 1e-6
        );
    }
}
