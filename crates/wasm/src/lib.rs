//! OpenBLUP WebAssembly module.
//!
//! Provides browser-friendly breeding value estimation.
//! Self-contained implementations (no rayon, no faer, no file I/O).

mod ainverse;
mod gmatrix;
mod mme;
mod pedigree;
mod reml;

use serde::{Deserialize, Serialize};

/// Pedigree entry for JSON input.
#[derive(Deserialize)]
pub struct PedigreeEntry {
    pub animal: String,
    pub sire: String,
    pub dam: String,
}

/// Model input for JSON.
#[derive(Deserialize)]
pub struct ModelInput {
    pub y: Vec<f64>,
    pub x: Vec<Vec<f64>>,
    pub z: Vec<Vec<f64>>,
    pub ginv: Option<Vec<Vec<f64>>>,
}

/// Model output.
#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    pub fixed_effects: Vec<f64>,
    pub random_effects: Vec<f64>,
    pub sigma2_random: f64,
    pub sigma2_residual: f64,
    pub log_likelihood: f64,
    pub converged: bool,
    pub n_iterations: usize,
}

/// Compute A-inverse from a pedigree (JSON input/output).
pub fn compute_a_inverse_json(pedigree_json: &str) -> Result<String, String> {
    let entries: Vec<PedigreeEntry> =
        serde_json::from_str(pedigree_json).map_err(|e| format!("JSON parse error: {}", e))?;

    let mut ped = pedigree::WasmPedigree::new();
    for entry in &entries {
        ped.add_animal(&entry.animal, &entry.sire, &entry.dam);
    }

    let sorted = ped.sort();
    let a_inv = ainverse::compute_a_inverse(&sorted);
    let n = a_inv.nrows();

    // Return as JSON: {dim: n, values: [...]}
    let a_inv_ref = &a_inv;
    let values: Vec<f64> = (0..n)
        .flat_map(|i| (0..n).map(move |j| a_inv_ref[(i, j)]))
        .collect();

    let result = serde_json::json!({
        "dim": n,
        "values": values,
        "animal_ids": sorted.ids,
    });

    serde_json::to_string(&result).map_err(|e| format!("JSON serialize error: {}", e))
}

/// Fit a simple mixed model (JSON input/output).
pub fn fit_mixed_model_json(model_json: &str) -> Result<String, String> {
    let input: ModelInput =
        serde_json::from_str(model_json).map_err(|e| format!("JSON parse error: {}", e))?;

    let n = input.y.len();
    let p = input.x.first().map(|r| r.len()).unwrap_or(0);
    let q = input.z.first().map(|r| r.len()).unwrap_or(0);

    if input.x.len() != n || input.z.len() != n {
        return Err("X and Z must have same number of rows as y".into());
    }

    let x = nalgebra::DMatrix::from_fn(n, p, |i, j| input.x[i][j]);
    let z = nalgebra::DMatrix::from_fn(n, q, |i, j| input.z[i][j]);

    let ginv = input.ginv.map(|g| {
        let dim = g.len();
        nalgebra::DMatrix::from_fn(dim, dim, |i, j| g[i][j])
    });

    let result = reml::fit_em_reml(&input.y, &x, &z, ginv.as_ref())
        .map_err(|e| format!("Fit error: {}", e))?;

    let output = ModelOutput {
        fixed_effects: result.fixed_effects,
        random_effects: result.random_effects,
        sigma2_random: result.sigma2_random,
        sigma2_residual: result.sigma2_residual,
        log_likelihood: result.log_likelihood,
        converged: result.converged,
        n_iterations: result.n_iterations,
    };

    serde_json::to_string(&output).map_err(|e| format!("JSON serialize error: {}", e))
}

/// Compute G-matrix from marker data.
pub fn compute_g_matrix_flat(
    markers: &[f64],
    n_individuals: usize,
    n_markers: usize,
) -> Result<Vec<f64>, String> {
    if markers.len() != n_individuals * n_markers {
        return Err("Marker array length must equal n_individuals * n_markers".into());
    }

    let m = nalgebra::DMatrix::from_fn(n_individuals, n_markers, |i, j| {
        markers[i * n_markers + j]
    });

    let g = gmatrix::compute_g_matrix(&m).map_err(|e| format!("G-matrix error: {}", e))?;

    let g_ref = &g;
    let values: Vec<f64> = (0..n_individuals)
        .flat_map(|i| (0..n_individuals).map(move |j| g_ref[(i, j)]))
        .collect();

    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_inverse_json() {
        let json = r#"[
            {"animal": "1", "sire": "0", "dam": "0"},
            {"animal": "2", "sire": "0", "dam": "0"},
            {"animal": "3", "sire": "1", "dam": "2"}
        ]"#;
        let result = compute_a_inverse_json(json).unwrap();
        assert!(result.contains("dim"));
        assert!(result.contains("values"));
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
    }

    #[test]
    fn test_g_matrix_flat() {
        let markers = vec![
            0.0, 1.0, 2.0,
            2.0, 1.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let result = compute_g_matrix_flat(&markers, 3, 3).unwrap();
        assert_eq!(result.len(), 9); // 3×3
    }
}
