use nalgebra::{DMatrix, DVector};

/// Dense MME assembly and solve (WASM-compatible).
pub struct MmeSolution {
    pub fixed_effects: Vec<f64>,
    pub random_effects: Vec<f64>,
    pub log_det_c: f64,
    pub c_inv_diag: Vec<f64>,
}

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

    let y_vec = DVector::from_column_slice(y);

    // Build coefficient matrix
    let mut c = DMatrix::zeros(dim, dim);

    // X'R⁻¹X
    let xtx = x.transpose() * x * r_inv_scale;
    for i in 0..p { for j in 0..p { c[(i, j)] = xtx[(i, j)]; } }

    // X'R⁻¹Z and Z'R⁻¹X
    let xtz = x.transpose() * z * r_inv_scale;
    for i in 0..p { for j in 0..q { c[(i, p+j)] = xtz[(i, j)]; c[(p+j, i)] = xtz[(i, j)]; } }

    // Z'R⁻¹Z + G⁻¹
    let ztz = z.transpose() * z * r_inv_scale;
    for i in 0..q { for j in 0..q { c[(p+i, p+j)] = ztz[(i, j)] + g_inv[(i, j)]; } }

    // RHS
    let mut rhs = DVector::zeros(dim);
    let xty = x.transpose() * &y_vec * r_inv_scale;
    let zty = z.transpose() * &y_vec * r_inv_scale;
    for i in 0..p { rhs[i] = xty[i]; }
    for i in 0..q { rhs[p + i] = zty[i]; }

    let chol = c.clone().cholesky().ok_or("Matrix not positive definite")?;
    let sol = chol.solve(&rhs);

    let l = chol.l();
    let log_det_c = 2.0 * (0..dim).map(|i| l[(i, i)].ln()).sum::<f64>();
    let c_inv = chol.inverse();
    let c_inv_diag: Vec<f64> = (0..dim).map(|i| c_inv[(i, i)]).collect();

    Ok(MmeSolution {
        fixed_effects: sol.as_slice()[..p].to_vec(),
        random_effects: sol.as_slice()[p..].to_vec(),
        log_det_c,
        c_inv_diag,
    })
}
