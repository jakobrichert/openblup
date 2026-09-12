use nalgebra::{DMatrix, DVector};

/// Dense MME assembly and solve (WASM-compatible).
pub struct MmeSolution {
    pub fixed_effects: Vec<f64>,
    pub random_effects: Vec<f64>,
    /// log|C| of the coefficient matrix.
    pub log_det_c: f64,
    /// Full inverse of the coefficient matrix (p+q x p+q).
    pub c_inv: DMatrix<f64>,
    /// y'Py = y'R⁻¹y - sol'rhs, the REML quadratic form.
    pub y_p_y: f64,
}

/// Assemble and solve Henderson's mixed model equations with R = σ²_e I
/// (`r_inv_scale = 1/σ²_e`) and a full G⁻¹ (`g_inv`, q x q).
pub fn solve_mme(
    x: &DMatrix<f64>,
    z: &DMatrix<f64>,
    y: &[f64],
    r_inv_scale: f64,
    g_inv: &DMatrix<f64>,
) -> Result<MmeSolution, String> {
    let p = x.ncols();
    let q = z.ncols();
    let dim = p + q;

    if x.nrows() != y.len() || z.nrows() != y.len() {
        return Err("X and Z must have one row per observation".into());
    }
    if g_inv.nrows() != q || g_inv.ncols() != q {
        return Err(format!(
            "G-inverse must be {}x{} (one row/col per random level), got {}x{}",
            q,
            q,
            g_inv.nrows(),
            g_inv.ncols()
        ));
    }

    let y_vec = DVector::from_column_slice(y);

    // Build coefficient matrix
    let mut c = DMatrix::zeros(dim, dim);

    // X'R⁻¹X
    let xtx = x.transpose() * x * r_inv_scale;
    c.view_mut((0, 0), (p, p)).copy_from(&xtx);

    // X'R⁻¹Z and Z'R⁻¹X
    let xtz = x.transpose() * z * r_inv_scale;
    c.view_mut((0, p), (p, q)).copy_from(&xtz);
    c.view_mut((p, 0), (q, p)).copy_from(&xtz.transpose());

    // Z'R⁻¹Z + G⁻¹
    let ztz = z.transpose() * z * r_inv_scale + g_inv;
    c.view_mut((p, p), (q, q)).copy_from(&ztz);

    // RHS
    let mut rhs = DVector::zeros(dim);
    let xty = x.transpose() * &y_vec * r_inv_scale;
    let zty = z.transpose() * &y_vec * r_inv_scale;
    rhs.rows_mut(0, p).copy_from(&xty);
    rhs.rows_mut(p, q).copy_from(&zty);

    let chol = c
        .clone()
        .cholesky()
        .ok_or("Coefficient matrix is not positive definite (rank-deficient X?)")?;
    let sol = chol.solve(&rhs);

    let l = chol.l();
    let log_det_c = 2.0 * (0..dim).map(|i| l[(i, i)].ln()).sum::<f64>();
    let c_inv = chol.inverse();

    let y_r_inv_y = y_vec.dot(&y_vec) * r_inv_scale;
    let y_p_y = y_r_inv_y - sol.dot(&rhs);

    Ok(MmeSolution {
        fixed_effects: sol.as_slice()[..p].to_vec(),
        random_effects: sol.as_slice()[p..].to_vec(),
        log_det_c,
        c_inv,
        y_p_y,
    })
}
