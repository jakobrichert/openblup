use nalgebra::DMatrix;

/// VanRaden Method 1 G-matrix (WASM-compatible, no rayon).
pub fn compute_g_matrix(markers: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = markers.nrows();
    let m = markers.ncols();

    // Allele frequencies
    let mut freqs = vec![0.0; m];
    for j in 0..m {
        let sum: f64 = (0..n).map(|i| markers[(i, j)]).sum();
        freqs[j] = sum / (2.0 * n as f64);
    }

    // Center
    let mut z = markers.clone();
    for j in 0..m {
        let two_p = 2.0 * freqs[j];
        for i in 0..n {
            z[(i, j)] -= two_p;
        }
    }

    let denom: f64 = freqs.iter().map(|&p| 2.0 * p * (1.0 - p)).sum();
    if denom < 1e-10 {
        return Err("All markers monomorphic".into());
    }

    Ok((&z * z.transpose()) / denom)
}
