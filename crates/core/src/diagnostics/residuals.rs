use crate::types::SparseMat;

/// Comprehensive residual diagnostics for a fitted mixed model.
#[derive(Debug, Clone)]
pub struct ResidualDiagnostics {
    /// Conditional residuals: e_c = y - Xb̂ - Zû
    pub conditional: Vec<f64>,
    /// Marginal residuals: e_m = y - Xb̂
    pub marginal: Vec<f64>,
    /// Standardized conditional residuals: e_c / (σ_e * √(1 - h_ii))
    pub standardized: Vec<f64>,
    /// Studentized residuals (internally): e_c / (σ_e * √(1 - h_ii))
    pub studentized: Vec<f64>,
    /// Hat matrix diagonal (leverage): h_ii
    pub leverage: Vec<f64>,
    /// Cook's distance: D_i
    pub cooks_distance: Vec<f64>,
    /// Fitted values: ŷ = Xb̂ + Zû
    pub fitted: Vec<f64>,
}

/// Compute residual diagnostics from model components.
///
/// # Arguments
/// - `y`: response vector (n)
/// - `x`: fixed effects design matrix (n × p)
/// - `z`: combined random effects design matrix (n × q)
/// - `fixed`: estimated fixed effects b̂ (p)
/// - `random`: estimated random effects û (q)
/// - `c_inv`: full C⁻¹ matrix ((p+q) × (p+q))
/// - `sigma2_e`: residual variance
pub fn compute_diagnostics(
    y: &[f64],
    x: &SparseMat,
    z: &SparseMat,
    fixed: &[f64],
    random: &[f64],
    c_inv: &nalgebra::DMatrix<f64>,
    sigma2_e: f64,
) -> ResidualDiagnostics {
    let n = y.len();
    let p = x.cols();
    let q = z.cols();

    // Fitted values: ŷ = Xb̂ + Zû
    let xb = crate::matrix::sparse::spmv(x, fixed);
    let zu = crate::matrix::sparse::spmv(z, random);
    let fitted: Vec<f64> = (0..n).map(|i| xb[i] + zu[i]).collect();

    // Conditional residuals: e_c = y - ŷ
    let conditional: Vec<f64> = (0..n).map(|i| y[i] - fitted[i]).collect();

    // Marginal residuals: e_m = y - Xb̂
    let marginal: Vec<f64> = (0..n).map(|i| y[i] - xb[i]).collect();

    // Leverage: h_ii = (1/σ²_e) * w_i' C⁻¹ w_i where w_i = [x_i; z_i]
    let mut leverage = vec![0.0; n];
    for i in 0..n {
        // Construct w_i = [x_i; z_i] (row i of [X, Z])
        let mut w = vec![0.0; p + q];
        // Extract row i from X
        for (val, (r, c)) in x.iter() {
            if r == i {
                w[c] = *val;
            }
        }
        // Extract row i from Z
        for (val, (r, c)) in z.iter() {
            if r == i {
                w[p + c] = *val;
            }
        }

        // h_ii = (1/σ²_e) * w' C⁻¹ w
        let mut h = 0.0;
        for a in 0..(p + q) {
            for b in 0..(p + q) {
                h += w[a] * c_inv[(a, b)] * w[b];
            }
        }
        leverage[i] = (h / sigma2_e).clamp(0.0, 1.0);
    }

    // Standardized residuals: e_c / (σ_e * √(1 - h_ii))
    let sigma_e = sigma2_e.sqrt();
    let standardized: Vec<f64> = (0..n)
        .map(|i| {
            let denom = sigma_e * (1.0 - leverage[i]).max(1e-10).sqrt();
            conditional[i] / denom
        })
        .collect();

    // Studentized = standardized (internal studentization)
    let studentized = standardized.clone();

    // Cook's distance: D_i = (e_i² * h_ii) / (p * σ²_e * (1 - h_ii)²)
    let cooks_distance: Vec<f64> = (0..n)
        .map(|i| {
            let h = leverage[i];
            let one_minus_h = (1.0 - h).max(1e-10);
            (conditional[i].powi(2) * h) / (p as f64 * sigma2_e * one_minus_h.powi(2))
        })
        .collect();

    ResidualDiagnostics {
        conditional,
        marginal,
        standardized,
        studentized,
        leverage,
        cooks_distance,
        fitted,
    }
}

/// Generate a summary of diagnostic statistics.
pub fn diagnostics_summary(diag: &ResidualDiagnostics) -> String {
    let n = diag.conditional.len();
    let mut s = String::new();

    s.push_str("=== Residual Diagnostics ===\n\n");

    // Conditional residuals
    let (min_r, max_r, mean_r) = summary_stats(&diag.conditional);
    s.push_str(&format!(
        "Conditional residuals:\n  Min: {:.4}  Max: {:.4}  Mean: {:.6}\n\n",
        min_r, max_r, mean_r
    ));

    // Standardized residuals
    let (min_s, max_s, _) = summary_stats(&diag.standardized);
    s.push_str(&format!(
        "Standardized residuals:\n  Min: {:.4}  Max: {:.4}\n\n",
        min_s, max_s
    ));

    // Leverage
    let (min_h, max_h, mean_h) = summary_stats(&diag.leverage);
    s.push_str(&format!(
        "Leverage (h_ii):\n  Min: {:.4}  Max: {:.4}  Mean: {:.4}\n",
        min_h, max_h, mean_h
    ));

    // Flag high-leverage observations
    let p_approx = mean_h * n as f64;
    let threshold = 2.0 * p_approx / n as f64;
    let high_lev: Vec<usize> = diag
        .leverage
        .iter()
        .enumerate()
        .filter(|(_, &h)| h > threshold)
        .map(|(i, _)| i)
        .collect();
    if !high_lev.is_empty() {
        s.push_str(&format!(
            "  High leverage (>{:.4}): {:?}\n",
            threshold, high_lev
        ));
    }
    s.push('\n');

    // Cook's distance
    let (_, max_d, _) = summary_stats(&diag.cooks_distance);
    let cook_threshold = 4.0 / n as f64;
    let influential: Vec<usize> = diag
        .cooks_distance
        .iter()
        .enumerate()
        .filter(|(_, &d)| d > cook_threshold)
        .map(|(i, _)| i)
        .collect();
    s.push_str(&format!(
        "Cook's distance:\n  Max: {:.4}\n",
        max_d
    ));
    if !influential.is_empty() {
        s.push_str(&format!(
            "  Influential (>{:.4}): {:?}\n",
            cook_threshold, influential
        ));
    }

    s
}

fn summary_stats(v: &[f64]) -> (f64, f64, f64) {
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (min, max, mean)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn simple_model() -> (Vec<f64>, SparseMat, SparseMat, Vec<f64>, Vec<f64>, nalgebra::DMatrix<f64>, f64) {
        // 4 obs, 1 fixed (intercept), 2 random levels
        let y = vec![10.0, 12.0, 6.0, 8.0];

        let mut x_tri = sprs::TriMat::new((4, 1));
        for i in 0..4 { x_tri.add_triplet(i, 0, 1.0); }
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((4, 2));
        z_tri.add_triplet(0, 0, 1.0);
        z_tri.add_triplet(1, 0, 1.0);
        z_tri.add_triplet(2, 1, 1.0);
        z_tri.add_triplet(3, 1, 1.0);
        let z = z_tri.to_csc();

        let fixed = vec![9.0];
        let random = vec![2.0, -2.0];
        let sigma2_e = 1.0;

        // Simple C⁻¹ (3×3 identity scaled)
        let c_inv = nalgebra::DMatrix::from_diagonal_element(3, 3, 0.5);

        (y, x, z, fixed, random, c_inv, sigma2_e)
    }

    #[test]
    fn test_fitted_plus_residual_equals_y() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);

        for i in 0..y.len() {
            assert_relative_eq!(diag.fitted[i] + diag.conditional[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_leverage_bounds() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);

        for &h in &diag.leverage {
            assert!(h >= 0.0, "leverage must be non-negative, got {}", h);
            assert!(h <= 1.0, "leverage must be <= 1, got {}", h);
        }
    }

    #[test]
    fn test_cooks_distance_non_negative() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);

        for &d in &diag.cooks_distance {
            assert!(d >= 0.0, "Cook's D must be non-negative, got {}", d);
        }
    }

    #[test]
    fn test_conditional_residuals() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);

        // y = [10, 12, 6, 8], fitted = [9+2, 9+2, 9-2, 9-2] = [11, 11, 7, 7]
        assert_relative_eq!(diag.conditional[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(diag.conditional[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(diag.conditional[2], -1.0, epsilon = 1e-10);
        assert_relative_eq!(diag.conditional[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_marginal_residuals() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);

        // marginal = y - Xb = y - 9
        assert_relative_eq!(diag.marginal[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(diag.marginal[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(diag.marginal[2], -3.0, epsilon = 1e-10);
        assert_relative_eq!(diag.marginal[3], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_summary_output() {
        let (y, x, z, fixed, random, c_inv, sigma2_e) = simple_model();
        let diag = compute_diagnostics(&y, &x, &z, &fixed, &random, &c_inv, sigma2_e);
        let summary = diagnostics_summary(&diag);
        assert!(summary.contains("Residual Diagnostics"));
        assert!(summary.contains("Leverage"));
        assert!(summary.contains("Cook's distance"));
    }
}
