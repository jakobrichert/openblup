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

/// REML log-likelihood (up to the usual constant) for the single-random-term
/// model with R = σ²_e I and G = σ²_u K:
///
/// ```text
/// -2 logL = (n - p) log 2π + n log σ²_e + q log σ²_u - log|K⁻¹| + log|C| + y'Py
/// ```
fn reml_log_likelihood(
    n: usize,
    p: usize,
    q: usize,
    sigma2_e: f64,
    sigma2_u: f64,
    log_det_k_inv: f64,
    log_det_c: f64,
    y_p_y: f64,
) -> f64 {
    let log_2_pi = (2.0 * std::f64::consts::PI).ln();
    -0.5 * ((n - p) as f64 * log_2_pi + n as f64 * sigma2_e.ln() + q as f64 * sigma2_u.ln()
        - log_det_k_inv
        + log_det_c
        + y_p_y)
}

/// EM-REML for a single random term (dense, suitable for small problems).
///
/// `ginv` is the inverse of the relationship matrix K (e.g. A⁻¹); `None`
/// means K = I. The EM updates are (Mrode 2005):
///
/// ```text
/// σ²_e = y'ê / (n - p)
/// σ²_u = (û'K⁻¹û + tr(K⁻¹ C^{uu})) / q
/// ```
///
/// where `C^{uu}` is the random-effects block of C⁻¹.
pub fn fit_em_reml(
    y: &[f64],
    x: &DMatrix<f64>,
    z: &DMatrix<f64>,
    ginv: Option<&DMatrix<f64>>,
    max_iter: usize,
    tol: f64,
) -> Result<RemlResult, String> {
    let n = y.len();
    let p = x.ncols();
    let q = z.ncols();

    if n == 0 {
        return Err("y must not be empty".into());
    }
    if q == 0 {
        return Err("Z must have at least one column (random level)".into());
    }
    if n <= p {
        return Err(format!(
            "Need more observations ({}) than fixed effects ({}) for REML",
            n, p
        ));
    }

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
    let init_var = (y_var / 2.0).max(0.01);

    let mut sigma2_u = init_var;
    let mut sigma2_e = init_var;

    let default_ginv = DMatrix::identity(q, q);
    let k_inv = ginv.unwrap_or(&default_ginv);
    if k_inv.nrows() != q || k_inv.ncols() != q {
        return Err(format!(
            "ginv must be {}x{} to match the columns of Z, got {}x{}",
            q,
            q,
            k_inv.nrows(),
            k_inv.ncols()
        ));
    }
    // log|K⁻¹| (0 for the identity).
    let log_det_k_inv = if ginv.is_some() {
        let chol = k_inv
            .clone()
            .cholesky()
            .ok_or("ginv must be symmetric positive definite")?;
        let l = chol.l();
        2.0 * (0..q).map(|i| l[(i, i)].ln()).sum::<f64>()
    } else {
        0.0
    };

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        let g_inv = k_inv / sigma2_u;
        let r_inv_scale = 1.0 / sigma2_e;

        let sol = crate::mme::solve_mme(x, z, y, r_inv_scale, &g_inv)
            .map_err(|e| format!("MME solve failed: {}", e))?;

        let old_sigma2_u = sigma2_u;
        let old_sigma2_e = sigma2_e;

        // EM update for σ²_e: y'ê / (n - p) where ê = y - Xb - Zu
        let xb = x * nalgebra::DVector::from_column_slice(&sol.fixed_effects);
        let zu = z * nalgebra::DVector::from_column_slice(&sol.random_effects);
        let y_e: f64 = (0..n).map(|i| y[i] * (y[i] - xb[i] - zu[i])).sum();
        sigma2_e = (y_e / (n - p) as f64).max(1e-10);

        // EM update for σ²_u: (u'K⁻¹u + tr(K⁻¹ C^{uu})) / q
        let u_vec = nalgebra::DVector::from_column_slice(&sol.random_effects);
        let u_kinv_u = u_vec.dot(&(k_inv * &u_vec));
        let c_uu = sol.c_inv.view((p, p), (q, q));
        let trace_kinv_cuu: f64 = (0..q)
            .map(|i| (0..q).map(|j| k_inv[(i, j)] * c_uu[(j, i)]).sum::<f64>())
            .sum();
        sigma2_u = ((u_kinv_u + trace_kinv_cuu) / q as f64).max(1e-10);

        let change = ((sigma2_u - old_sigma2_u).powi(2) + (sigma2_e - old_sigma2_e).powi(2)).sqrt()
            / (old_sigma2_u.powi(2) + old_sigma2_e.powi(2))
                .sqrt()
                .max(1e-10);

        n_iter = iter + 1;
        if iter > 0 && change < tol {
            converged = true;
            break;
        }
    }

    // Final solve at the converged parameters
    let g_inv = k_inv / sigma2_u;
    let sol = crate::mme::solve_mme(x, z, y, 1.0 / sigma2_e, &g_inv)
        .map_err(|e| format!("Final MME solve failed: {}", e))?;

    let log_likelihood = reml_log_likelihood(
        n,
        p,
        q,
        sigma2_e,
        sigma2_u,
        log_det_k_inv,
        sol.log_det_c,
        sol.y_p_y,
    );

    Ok(RemlResult {
        fixed_effects: sol.fixed_effects,
        random_effects: sol.random_effects,
        sigma2_random: sigma2_u,
        sigma2_residual: sigma2_e,
        log_likelihood,
        converged,
        n_iterations: n_iter,
    })
}
