use nalgebra::DMatrix;

pub struct RemlResult {
    pub fixed_effects: Vec<f64>,
    pub random_effects: Vec<f64>,
    pub sigma2_random: f64,
    pub sigma2_residual: f64,
    pub log_likelihood: f64,
    pub converged: bool,
    pub n_iterations: usize,
}

/// Simple EM-REML for WASM (dense only, suitable for small problems).
pub fn fit_em_reml(
    y: &[f64],
    x: &DMatrix<f64>,
    z: &DMatrix<f64>,
    ginv: Option<&DMatrix<f64>>,
) -> Result<RemlResult, String> {
    let n = y.len();
    let p = x.ncols();
    let q = z.ncols();

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let init_var = (y_var / 2.0).max(0.01);

    let mut sigma2_u = init_var;
    let mut sigma2_e = init_var;

    let default_ginv = DMatrix::identity(q, q);
    let k_inv = ginv.unwrap_or(&default_ginv);

    let max_iter = 50;
    let tol = 1e-6;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        let g_inv = k_inv / sigma2_u;
        let r_inv_scale = 1.0 / sigma2_e;

        let sol = crate::mme::solve_mme(x, z, y, r_inv_scale, &g_inv)
            .map_err(|e| format!("MME solve failed: {}", e))?;

        let old_sigma2_u = sigma2_u;
        let old_sigma2_e = sigma2_e;

        // EM update for σ²_e: y'ê / (n - p)
        let xb = x * nalgebra::DVector::from_column_slice(&sol.fixed_effects);
        let zu = z * nalgebra::DVector::from_column_slice(&sol.random_effects);
        let mut y_e = 0.0;
        for i in 0..n {
            y_e += y[i] * (y[i] - xb[i] - zu[i]);
        }
        sigma2_e = (y_e / (n - p) as f64).max(1e-10);

        // EM update for σ²_u
        let u = &sol.random_effects;
        let u_kinv_u: f64 = {
            let u_vec = nalgebra::DVector::from_column_slice(u);
            let kinv_u = k_inv * &u_vec;
            u_vec.dot(&kinv_u)
        };
        let trace_cinv_uu: f64 = (0..q).map(|i| sol.c_inv_diag[p + i]).sum();
        sigma2_u = ((u_kinv_u + trace_cinv_uu) / q as f64).max(1e-10);

        let change = ((sigma2_u - old_sigma2_u).powi(2) + (sigma2_e - old_sigma2_e).powi(2)).sqrt()
            / (old_sigma2_u.powi(2) + old_sigma2_e.powi(2)).sqrt().max(1e-10);

        n_iter = iter + 1;
        if iter > 0 && change < tol {
            converged = true;
            break;
        }
    }

    // Final solve
    let g_inv = k_inv / sigma2_u;
    let sol = crate::mme::solve_mme(x, z, y, 1.0 / sigma2_e, &g_inv)
        .map_err(|e| format!("Final MME solve failed: {}", e))?;

    Ok(RemlResult {
        fixed_effects: sol.fixed_effects,
        random_effects: sol.random_effects,
        sigma2_random: sigma2_u,
        sigma2_residual: sigma2_e,
        log_likelihood: 0.0, // Simplified
        converged,
        n_iterations: n_iter,
    })
}
