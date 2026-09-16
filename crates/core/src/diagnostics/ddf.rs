use nalgebra::{DMatrix, DVector};

use crate::lmm::MmeInverse;
use crate::matrix::sparse::{spmv, xt_y};
use crate::matrix::sparse_cholesky::SparseCholeskySolver;
use crate::types::SparseMat;

use crate::lmm::FitResult;

use super::wald::{f_distribution_sf, WaldTest};

/// Method for computing denominator degrees of freedom in Wald F-tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DdfMethod {
    /// Containment method: den_df = n - rank(X).
    /// Simple and conservative, exact for balanced designs.
    Containment,
    /// Satterthwaite approximation using the variance-covariance matrix of
    /// fixed effects and the inverse Average Information matrix.
    /// More accurate for unbalanced designs.
    Satterthwaite,
    /// Kenward-Roger (1997): bias-adjusted covariance of the fixed effects
    /// and a scaled F-statistic with matched denominator df.
    KenwardRoger,
}

/// Satterthwaite denominator degrees of freedom for Wald tests of fixed
/// effects (Giesbrecht & Burns 1985; Fai & Cornelius 1996 for multi-df
/// terms).
///
/// The calculator works from
///
/// * `Φ = C⁻¹_{bb}`, the covariance matrix of the fixed effects,
/// * `∂Φ/∂θ_k` for every variance parameter `θ_k`, and
/// * the inverse of the average information matrix, `AI⁻¹ ≈ Var(θ̂)`.
///
/// For a contrast `l'β`, `ν = 2 (l'Φl)² / (g' AI⁻¹ g)` with
/// `g_k = l' (∂Φ/∂θ_k) l`. The derivatives are
/// `∂Φ/∂θ_k = −C⁻¹_{b·} (∂C/∂θ_k) C⁻¹_{·b}`; the REML engines compute them
/// at convergence from the columns of `C⁻¹` belonging to the fixed effects,
/// so no dense `C⁻¹` is needed (see
/// [`fixed_cov_derivatives_scaled_identity`]).
///
/// With the additional Kenward-Roger terms `P_i` and `Q_ij` (see
/// [`KenwardRogerTerms`]) the calculator also provides the Kenward-Roger
/// (1997) test: the bias-adjusted covariance `Φ_A` and the scaled
/// F-statistic with its denominator df.
pub struct DdfCalculator {
    /// `Φ = C⁻¹_{bb}` (p x p).
    phi: DMatrix<f64>,
    /// `∂Φ/∂θ_k`, one p x p matrix per variance parameter.
    dphi: Vec<DMatrix<f64>>,
    /// Inverse of the Average Information matrix (n_vc x n_vc).
    ai_inv: DMatrix<f64>,
    /// Number of fixed effect parameters (p).
    n_fixed: usize,
    /// Number of observations.
    n_obs: usize,
}

impl DdfCalculator {
    /// Create a new `DdfCalculator`.
    ///
    /// # Arguments
    ///
    /// * `phi` - Covariance matrix of the fixed effects (p x p).
    /// * `dphi` - `∂Φ/∂θ_k` for each variance parameter (p x p each).
    /// * `ai_matrix` - The Average Information matrix at convergence.
    /// * `n_obs` - Number of observations.
    ///
    /// # Returns
    ///
    /// `None` if the AI matrix is not positive definite or the number of
    /// derivatives does not match its dimension.
    pub fn new(
        phi: DMatrix<f64>,
        dphi: Vec<DMatrix<f64>>,
        ai_matrix: DMatrix<f64>,
        n_obs: usize,
    ) -> Option<Self> {
        let n_fixed = phi.nrows();
        if phi.ncols() != n_fixed
            || dphi.len() != ai_matrix.nrows()
            || dphi
                .iter()
                .any(|d| d.nrows() != n_fixed || d.ncols() != n_fixed)
        {
            return None;
        }
        let ai_inv = ai_matrix.cholesky()?.inverse();
        Some(Self {
            phi,
            dphi,
            ai_inv,
            n_fixed,
            n_obs,
        })
    }

    /// The fixed-effects covariance matrix `Φ = C^{-1}_{bb}` (p x p).
    pub fn phi(&self) -> &DMatrix<f64> {
        &self.phi
    }

    /// Compute the Satterthwaite denominator degrees of freedom for a single
    /// contrast `l'beta`.
    ///
    /// ```text
    /// nu = 2 * (l' Phi l)^2 / sum_{i,j} g_i * AI_inv_{ij} * g_j
    /// ```
    ///
    /// where `g_k = l' (dPhi/dtheta_k) l`.
    ///
    /// # Arguments
    ///
    /// * `contrast` - The contrast vector `l` of length `p` (number of fixed effects).
    ///
    /// # Returns
    ///
    /// The Satterthwaite degrees of freedom, clamped to `[1, n - p]`.
    pub fn satterthwaite_ddf(&self, contrast: &[f64]) -> f64 {
        let p = self.n_fixed;
        assert_eq!(
            contrast.len(),
            p,
            "Contrast vector length ({}) must equal number of fixed effects ({})",
            contrast.len(),
            p
        );

        let n_params = self.dphi.len();
        let l_phi_l = quad_form(contrast, &self.phi);
        if l_phi_l <= 0.0 {
            return 1.0;
        }

        let g: Vec<f64> = (0..n_params)
            .map(|k| quad_form(contrast, &self.dphi[k]))
            .collect();

        let mut denom = 0.0;
        for i in 0..n_params {
            for j in 0..n_params {
                denom += g[i] * self.ai_inv[(i, j)] * g[j];
            }
        }

        let max_df = (self.n_obs - self.n_fixed) as f64;
        if denom <= 0.0 {
            return max_df;
        }

        let nu = 2.0 * l_phi_l * l_phi_l / denom;
        nu.max(1.0).min(max_df)
    }

    /// Compute the generalized Satterthwaite denominator degrees of freedom
    /// for a multi-parameter Wald F-test.
    ///
    /// For a contrast matrix `L` (q x p) testing q parameters simultaneously,
    /// we use the method of Fai and Cornelius (1996), which estimates a single
    /// denominator df by matching moments of the approximate F distribution.
    ///
    /// The effective ddf is computed as a combination of per-eigenvalue
    /// Satterthwaite ddf values from the spectral decomposition of the
    /// variance-covariance matrix of the contrast, `L Phi L'`:
    /// ```text
    /// nu = 2 * E / (E - q)  where E = sum_i nu_i / (nu_i - 2)  for nu_i > 2
    /// ```
    ///
    /// # Arguments
    ///
    /// * `contrast_matrix` - Rows of the contrast matrix `L`, each of length p.
    pub fn satterthwaite_ddf_multi(&self, contrast_matrix: &[Vec<f64>]) -> f64 {
        let q = contrast_matrix.len();
        if q == 0 {
            return 1.0;
        }
        if q == 1 {
            return self.satterthwaite_ddf(&contrast_matrix[0]);
        }

        let p = self.n_fixed;
        let phi = &self.phi;

        // L * Phi * L' (q x q)
        let mut l_phi_lt = DMatrix::zeros(q, q);
        for i in 0..q {
            for j in 0..q {
                let mut val = 0.0;
                for r in 0..p {
                    for s in 0..p {
                        val += contrast_matrix[i][r] * phi[(r, s)] * contrast_matrix[j][s];
                    }
                }
                l_phi_lt[(i, j)] = val;
            }
        }

        let eigen = l_phi_lt.symmetric_eigen();
        let eigenvalues = &eigen.eigenvalues;
        let eigenvectors = &eigen.eigenvectors;

        let mut nu_values = Vec::with_capacity(q);
        for i in 0..q {
            if eigenvalues[i] <= 1e-14 {
                continue;
            }
            // Transformed contrast: l_i = eigenvector_i' * L
            let mut l_transformed = vec![0.0; p];
            for j in 0..p {
                let mut val = 0.0;
                for kk in 0..q {
                    val += eigenvectors[(kk, i)] * contrast_matrix[kk][j];
                }
                l_transformed[j] = val;
            }
            nu_values.push(self.satterthwaite_ddf(&l_transformed));
        }

        if nu_values.is_empty() {
            return 1.0;
        }

        let q_eff = nu_values.len() as f64;
        let e_sum: f64 = nu_values
            .iter()
            .map(|&nu_i| {
                if nu_i > 2.0 {
                    nu_i / (nu_i - 2.0)
                } else {
                    // The expectation does not exist for nu_i <= 2; a large
                    // contribution pushes the combined df down.
                    100.0
                }
            })
            .sum();

        let combined_nu = if e_sum > q_eff {
            2.0 * e_sum / (e_sum - q_eff)
        } else {
            2.0
        };

        let max_df = (self.n_obs - self.n_fixed) as f64;
        combined_nu.max(1.0).min(max_df)
    }

    /// Containment degrees of freedom: `n - rank(X)`.
    pub fn containment_ddf(&self) -> f64 {
        (self.n_obs - self.n_fixed) as f64
    }

    /// Kenward-Roger (1997) test of `L β = 0` for a contrast matrix `L`
    /// (rows of length p) with the estimates `beta`.
    ///
    /// Returns the scaled statistic `λ / ℓ · (Lβ̂)' (L Φ_A L')⁻¹ (Lβ̂)` built
    /// on the bias-adjusted covariance
    ///
    /// ```text
    /// Φ_A = Φ + 2 Φ [ Σ_ij W_ij (Q_ij − P_i Φ P_j) ] Φ,   W = AI⁻¹,
    /// ```
    ///
    /// with the denominator df `m` and scale `λ` from Kenward & Roger's
    /// moment matching (their `A1`, `A2`, `B`, `g`, `c1..c3`, `E*`, `V*`,
    /// `ρ`). For balanced designs this reproduces the ANOVA F-test and its
    /// degrees of freedom.
    ///
    /// `None` if `L Φ L'` is singular.
    pub fn kenward_roger(
        &self,
        kr: &KenwardRogerTerms,
        contrast_matrix: &[Vec<f64>],
        beta: &[f64],
    ) -> Option<KenwardRogerTest> {
        let p = self.n_fixed;
        let ell = contrast_matrix.len();
        let n_par = self.dphi.len();
        if ell == 0 || kr.p.len() != n_par || kr.q.len() != n_par || beta.len() != p {
            return None;
        }
        let phi = &self.phi;
        let w = &self.ai_inv;

        // Φ_A
        let mut adj = DMatrix::zeros(p, p);
        for i in 0..n_par {
            for j in 0..n_par {
                let pipj = &kr.p[i] * phi * &kr.p[j];
                adj += (&kr.q[i][j] - pipj) * w[(i, j)];
            }
        }
        let phi_a = phi + 2.0 * (phi * adj * phi);

        // Θ = L' (L Φ L')⁻¹ L
        let l = DMatrix::from_fn(ell, p, |r, c| contrast_matrix[r][c]);
        let lpl = &l * phi * l.transpose();
        let lpl_inv = lpl.try_inverse()?;
        let theta = l.transpose() * &lpl_inv * &l;

        // A1, A2
        let m: Vec<DMatrix<f64>> = (0..n_par).map(|i| &theta * phi * &kr.p[i] * phi).collect();
        let (mut a1, mut a2) = (0.0, 0.0);
        for i in 0..n_par {
            let tr_i = m[i].trace();
            for j in 0..n_par {
                a1 += w[(i, j)] * tr_i * m[j].trace();
                a2 += w[(i, j)] * (&m[i] * &m[j]).trace();
            }
        }

        let ell_f = ell as f64;
        let max_df = (self.n_obs - self.n_fixed) as f64;
        let (den_df, lambda) = if a2.abs() < 1e-12 {
            // No variance-parameter uncertainty reaches the contrast: the
            // Wald test is exact with the containment df and no scaling.
            (max_df, 1.0)
        } else {
            let b = (a1 + 6.0 * a2) / (2.0 * ell_f);
            let g = ((ell_f + 1.0) * a1 - (ell_f + 4.0) * a2) / ((ell_f + 2.0) * a2);
            let denom = 3.0 * ell_f + 2.0 * (1.0 - g);
            let c1 = g / denom;
            let c2 = (ell_f - g) / denom;
            let c3 = (ell_f + 2.0 - g) / denom;
            let e_star = 1.0 / (1.0 - a2 / ell_f);
            let v_star =
                (2.0 / ell_f) * (1.0 + c1 * b) / ((1.0 - c2 * b) * (1.0 - c2 * b) * (1.0 - c3 * b));
            let rho = v_star / (2.0 * e_star * e_star);
            let m_df = 4.0 + (ell_f + 2.0) / (ell_f * rho - 1.0);
            let lambda = m_df / (e_star * (m_df - 2.0));
            if !m_df.is_finite() || !lambda.is_finite() || m_df <= 2.0 || lambda <= 0.0 {
                (max_df, 1.0)
            } else {
                (m_df.max(1.0).min(max_df), lambda)
            }
        };

        // Scaled F on Φ_A
        let lb = &l * DVector::from_column_slice(beta);
        let lpal = &l * &phi_a * l.transpose();
        let lpal_inv = lpal.try_inverse()?;
        let f_statistic = lambda / ell_f * (lb.transpose() * lpal_inv * &lb)[(0, 0)];
        Some(KenwardRogerTest {
            f_statistic: f_statistic.max(0.0),
            num_df: ell,
            den_df,
            lambda,
        })
    }
}

/// Result of a Kenward-Roger test of one term.
#[derive(Debug, Clone, Copy)]
pub struct KenwardRogerTest {
    /// Scaled F-statistic (`λ F`).
    pub f_statistic: f64,
    /// Numerator degrees of freedom (number of contrasts).
    pub num_df: usize,
    /// Denominator degrees of freedom `m`.
    pub den_df: f64,
    /// Scale factor `λ` applied to the Wald F.
    pub lambda: f64,
}

/// The matrices Kenward & Roger (1997) need beyond `Φ` and `AI⁻¹`:
///
/// ```text
/// P_i  = X' (∂V⁻¹/∂θ_i) X               = −(V⁻¹X)' (∂V/∂θ_i) (V⁻¹X)
/// Q_ij = X' (∂V⁻¹/∂θ_i) V (∂V⁻¹/∂θ_j) X = (∂V/∂θ_i V⁻¹X)' V⁻¹ (∂V/∂θ_j V⁻¹X)
/// ```
///
/// (p x p each, `V = ZGZ' + R`). They satisfy `∂Φ/∂θ_i = −Φ P_i Φ`.
#[derive(Debug, Clone)]
pub struct KenwardRogerTerms {
    /// `P_i`, one per variance parameter.
    pub p: Vec<DMatrix<f64>>,
    /// `Q_ij`, indexed `[i][j]`.
    pub q: Vec<Vec<DMatrix<f64>>>,
}

/// Kenward-Roger terms for a model with scaled variances `[σ²_1, ..., σ²_r,
/// σ²_e]`, `u_k ~ N(0, σ²_k K_k)` and an IID residual, computed through the
/// random-effects block of the MME, `C_zz = Z'Z/σ²_e + blockdiag(K_k⁻¹/σ²_k)`
/// (sparse):
///
/// ```text
/// V⁻¹ M          = (M − Z C_zz⁻¹ Z'M/σ²_e) / σ²_e
/// ∂V/∂σ²_k V⁻¹X  = Z_k [C_zz⁻¹ Z'X/σ²_e]_k / σ²_k
/// ∂V/∂σ²_e V⁻¹X  = V⁻¹X
/// ```
///
/// so neither `K_k` nor a dense `V` is ever formed. `ginv_scaled[k]` is
/// `K_k⁻¹/σ²_k` exactly as used to assemble the MME.
pub fn kenward_roger_terms_scaled_identity(
    x: &SparseMat,
    z_blocks: &[SparseMat],
    ginv_scaled: &[SparseMat],
    variance_params: &[f64],
) -> crate::error::Result<KenwardRogerTerms> {
    let n = x.rows();
    let p = x.cols();
    let n_terms = z_blocks.len();
    let sigma2_e = variance_params[n_terms];
    let q_vec: Vec<usize> = z_blocks.iter().map(|z| z.cols()).collect();
    let mut offsets = Vec::with_capacity(n_terms);
    let mut off = 0;
    for q in &q_vec {
        offsets.push(off);
        off += q;
    }
    let q_total = off;

    // C_zz: the MME without fixed effects.
    let empty_x: SparseMat = sprs::TriMat::new((n, 0)).to_csc();
    let czz = crate::lmm::SparseMixedModelEquations::assemble(
        &empty_x,
        z_blocks,
        &vec![0.0; n],
        1.0 / sigma2_e,
        ginv_scaled,
    );
    let solver = SparseCholeskySolver::new(&czz.coeff_matrix)?;

    // V⁻¹ M for a dense n x p matrix; also returns S = C_zz⁻¹ Z'M/σ²_e.
    let v_inv = |m: &DMatrix<f64>| -> crate::error::Result<(DMatrix<f64>, DMatrix<f64>)> {
        let mut out = DMatrix::zeros(n, p);
        let mut s_all = DMatrix::zeros(q_total, p);
        for j in 0..p {
            let col: Vec<f64> = (0..n).map(|i| m[(i, j)]).collect();
            let mut rhs = vec![0.0; q_total];
            for (k, z) in z_blocks.iter().enumerate() {
                for (r, v) in xt_y(z, &col).iter().enumerate() {
                    rhs[offsets[k] + r] = v / sigma2_e;
                }
            }
            let s = solver.solve(&rhs)?;
            let mut zs = vec![0.0; n];
            for (k, z) in z_blocks.iter().enumerate() {
                let sk = &s[offsets[k]..offsets[k] + q_vec[k]];
                for (i, v) in spmv(z, sk).iter().enumerate() {
                    zs[i] += v;
                }
            }
            for i in 0..n {
                out[(i, j)] = (col[i] - zs[i]) / sigma2_e;
            }
            for r in 0..q_total {
                s_all[(r, j)] = s[r];
            }
        }
        Ok((out, s_all))
    };

    let mut x_dense = DMatrix::zeros(n, p);
    for (v, (i, j)) in x.iter() {
        x_dense[(i, j)] = *v;
    }
    let (vx, s_x) = v_inv(&x_dense)?;

    // T_i V⁻¹X for every parameter
    let n_par = n_terms + 1;
    let mut tvx: Vec<DMatrix<f64>> = Vec::with_capacity(n_par);
    for (k, z) in z_blocks.iter().enumerate() {
        let mut t = DMatrix::zeros(n, p);
        for j in 0..p {
            let sk: Vec<f64> = (0..q_vec[k]).map(|r| s_x[(offsets[k] + r, j)]).collect();
            for (i, v) in spmv(z, &sk).iter().enumerate() {
                t[(i, j)] = v / variance_params[k];
            }
        }
        tvx.push(t);
    }
    tvx.push(vx.clone());

    let u: Vec<DMatrix<f64>> = tvx
        .iter()
        .map(|t| v_inv(t).map(|(vt, _)| vt))
        .collect::<crate::error::Result<_>>()?;

    let p_mats: Vec<DMatrix<f64>> = tvx.iter().map(|t| -(vx.transpose() * t)).collect();
    let q_mats: Vec<Vec<DMatrix<f64>>> = (0..n_par)
        .map(|i| (0..n_par).map(|j| tvx[i].transpose() * &u[j]).collect())
        .collect();
    Ok(KenwardRogerTerms {
        p: p_mats,
        q: q_mats,
    })
}

/// `∂Φ/∂θ` for a model whose variance parameters are the scaled variances
/// `[σ²_1, ..., σ²_r, σ²_e]` of random terms `u_k ~ N(0, σ²_k K_k)` and an
/// IID residual.
///
/// With `F = C⁻¹_{·b}` (the `p` columns of `C⁻¹` belonging to the fixed
/// effects) and `F_k` its rows for random term `k`:
///
/// ```text
/// ∂Φ/∂σ²_k = F_k' K_k⁻¹ F_k / σ⁴_k
/// ∂Φ/∂σ²_e = (W F)' (W F) / σ⁴_e,   W = [X Z_1 ... Z_r]
/// ```
///
/// `kinv[k]` is the relationship-matrix inverse `K_k⁻¹` of term `k` (`None`
/// for `K = I`).
pub fn fixed_cov_derivatives_scaled_identity(
    c_inv: &MmeInverse,
    x: &SparseMat,
    z_blocks: &[SparseMat],
    kinv: &[Option<&SparseMat>],
    variance_params: &[f64],
) -> Vec<DMatrix<f64>> {
    let p = x.cols();
    let n = x.rows();
    let f = c_inv.fixed_columns(p);
    let n_terms = z_blocks.len();
    let mut out = Vec::with_capacity(n_terms + 1);

    let mut offset = p;
    for (k, z) in z_blocks.iter().enumerate() {
        let q = z.cols();
        let f_k = f.rows(offset, q).into_owned();
        let kinv_f = match kinv.get(k).copied().flatten() {
            Some(kinv_k) => sparse_times_dense(kinv_k, &f_k),
            None => f_k.clone(),
        };
        let s4 = variance_params[k] * variance_params[k];
        out.push(f_k.transpose() * kinv_f / s4);
        offset += q;
    }

    // W F (n x p)
    let mut wf = DMatrix::zeros(n, p);
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| f[(i, j)]).collect();
        let mut acc = spmv(x, &col);
        let mut off = p;
        for z in z_blocks {
            let q = z.cols();
            let col_k: Vec<f64> = (0..q).map(|i| f[(off + i, j)]).collect();
            for (a, v) in spmv(z, &col_k).iter().enumerate() {
                acc[a] += v;
            }
            off += q;
        }
        for i in 0..n {
            wf[(i, j)] = acc[i];
        }
    }
    let s4e = variance_params[n_terms] * variance_params[n_terms];
    out.push(wf.transpose() * &wf / s4e);
    out
}

/// `A B` for sparse `A` and dense `B`.
fn sparse_times_dense(a: &SparseMat, b: &DMatrix<f64>) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(a.rows(), b.ncols());
    for (v, (i, k)) in a.iter() {
        for j in 0..b.ncols() {
            out[(i, j)] += v * b[(k, j)];
        }
    }
    out
}

/// Compute the quadratic form `x' A x` for vector x and matrix A.
fn quad_form(x: &[f64], a: &DMatrix<f64>) -> f64 {
    let n = x.len();
    let mut result = 0.0;
    for i in 0..n {
        for j in 0..n {
            result += x[i] * a[(i, j)] * x[j];
        }
    }
    result
}

/// Compute Wald F-tests using Satterthwaite denominator degrees of freedom.
///
/// Uses the fixed-effects covariance matrix and its derivatives stored in
/// the [`FitResult`] together with the Average Information matrix at
/// convergence.
///
/// # Returns
///
/// A vector of `WaldTest` results with Satterthwaite ddf, or `None` if the
/// `DdfCalculator` could not be constructed (singular AI matrix, or the
/// derivatives are not available).
pub fn wald_tests_satterthwaite(
    result: &FitResult,
    ai_matrix: &DMatrix<f64>,
) -> Option<Vec<WaldTest>> {
    if result.fixed_effects.is_empty() {
        return Some(Vec::new());
    }

    let n_fixed = result.n_fixed_params;
    let phi = rows_to_matrix(&result.fixed_cov, n_fixed)?;
    let dphi: Vec<DMatrix<f64>> = result
        .fixed_cov_derivatives
        .iter()
        .map(|m| rows_to_matrix(m, n_fixed))
        .collect::<Option<_>>()?;

    let calc = DdfCalculator::new(phi.clone(), dphi, ai_matrix.clone(), result.n_obs)?;

    // Group fixed effects by term name (same logic as in wald_tests)
    let mut term_order: Vec<String> = Vec::new();
    let mut term_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, ef) in result.fixed_effects.iter().enumerate() {
        if !term_indices.contains_key(&ef.term) {
            term_order.push(ef.term.clone());
        }
        term_indices.entry(ef.term.clone()).or_default().push(i);
    }

    let mut tests = Vec::new();

    for term in &term_order {
        let indices = &term_indices[term];
        let num_df = indices.len();

        if num_df == 0 {
            continue;
        }

        if num_df == 1 {
            // Single-parameter term: F = (beta / SE)^2
            let idx = indices[0];
            let ef = &result.fixed_effects[idx];
            let se = ef.se;

            let mut contrast = vec![0.0; n_fixed];
            contrast[idx] = 1.0;
            let den_df = calc.satterthwaite_ddf(&contrast);

            if se > 0.0 {
                let t = ef.estimate / se;
                let f_stat = t * t;
                let p_value = f_distribution_sf(f_stat, 1.0, den_df);
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: f_stat,
                    num_df: 1,
                    den_df,
                    p_value,
                });
            } else {
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: 0.0,
                    num_df: 1,
                    den_df,
                    p_value: 1.0,
                });
            }
        } else {
            // Multi-parameter term: build contrast matrix
            let contrast_matrix: Vec<Vec<f64>> = indices
                .iter()
                .map(|&idx| {
                    let mut row = vec![0.0; n_fixed];
                    row[idx] = 1.0;
                    row
                })
                .collect();

            let den_df = calc.satterthwaite_ddf_multi(&contrast_matrix);

            // General Wald F using the covariance block of Phi.
            let beta: Vec<f64> = indices
                .iter()
                .map(|&idx| result.fixed_effects[idx].estimate)
                .collect();
            let k = indices.len();
            let cov = DMatrix::from_fn(k, k, |a, b| phi[(indices[a], indices[b])]);
            let (f_stat, rank) = super::wald::wald_f_general(&beta, &cov);
            let num_df = rank.max(1);

            let p_value = if rank == 0 {
                1.0
            } else {
                f_distribution_sf(f_stat, num_df as f64, den_df)
            };

            tests.push(WaldTest {
                term: term.clone(),
                f_statistic: f_stat,
                num_df,
                den_df,
                p_value,
            });
        }
    }

    Some(tests)
}

/// Wald tests of the fixed-effect terms with the Kenward-Roger adjustment
/// (bias-adjusted covariance, scaled F and matched denominator df).
///
/// Needs the derivatives of the fixed-effects covariance, the Kenward-Roger
/// terms and the Average Information matrix stored in the [`FitResult`];
/// `None` when any of them is unavailable.
pub fn wald_tests_kenward_roger(
    result: &FitResult,
    ai_matrix: &DMatrix<f64>,
) -> Option<Vec<WaldTest>> {
    if result.fixed_effects.is_empty() {
        return Some(Vec::new());
    }
    let kr = result.kenward_roger_terms.as_ref()?;
    let n_fixed = result.n_fixed_params;
    let phi = rows_to_matrix(&result.fixed_cov, n_fixed)?;
    let dphi: Vec<DMatrix<f64>> = result
        .fixed_cov_derivatives
        .iter()
        .map(|m| rows_to_matrix(m, n_fixed))
        .collect::<Option<_>>()?;
    let calc = DdfCalculator::new(phi, dphi, ai_matrix.clone(), result.n_obs)?;
    let beta: Vec<f64> = result.fixed_effects.iter().map(|e| e.estimate).collect();

    let mut term_order: Vec<String> = Vec::new();
    let mut term_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, ef) in result.fixed_effects.iter().enumerate() {
        if !term_indices.contains_key(&ef.term) {
            term_order.push(ef.term.clone());
        }
        term_indices.entry(ef.term.clone()).or_default().push(i);
    }

    let mut tests = Vec::new();
    for term in &term_order {
        let indices = &term_indices[term];
        let contrast_matrix: Vec<Vec<f64>> = indices
            .iter()
            .map(|&idx| {
                let mut row = vec![0.0; n_fixed];
                row[idx] = 1.0;
                row
            })
            .collect();
        let test = calc.kenward_roger(kr, &contrast_matrix, &beta)?;
        let p_value = f_distribution_sf(test.f_statistic, test.num_df as f64, test.den_df);
        tests.push(WaldTest {
            term: term.clone(),
            f_statistic: test.f_statistic,
            num_df: test.num_df,
            den_df: test.den_df,
            p_value,
        });
    }
    Some(tests)
}

fn rows_to_matrix(rows: &[Vec<f64>], p: usize) -> Option<DMatrix<f64>> {
    if rows.len() != p || rows.iter().any(|r| r.len() != p) {
        return None;
    }
    Some(DMatrix::from_fn(p, p, |i, j| rows[i][j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmm::{FitResult, MixedModelEquations, NamedEffect, VarianceEstimate};
    use crate::matrix::sparse::sparse_diagonal;
    use approx::assert_relative_eq;

    /// Balanced one-way random model: intercept, `q` groups with
    /// `n_per_group` observations each. Returns (X, Z, y).
    fn one_way_design(q: usize, n_per_group: usize) -> (SparseMat, SparseMat, Vec<f64>) {
        let n = q * n_per_group;
        let mut x = sprs::TriMat::new((n, 1));
        let mut z = sprs::TriMat::new((n, q));
        let mut y = Vec::with_capacity(n);
        for g in 0..q {
            for r in 0..n_per_group {
                let i = g * n_per_group + r;
                x.add_triplet(i, 0, 1.0);
                z.add_triplet(i, g, 1.0);
                y.push(10.0 + g as f64 - 0.3 * r as f64);
            }
        }
        (x.to_csc(), z.to_csc(), y)
    }

    /// Dense `C⁻¹` of the one-way model at the given variances.
    fn one_way_inverse(q: usize, n_per_group: usize, sigma2_u: f64, sigma2_e: f64) -> MmeInverse {
        let (x, z, y) = one_way_design(q, n_per_group);
        let ginv = sparse_diagonal(&vec![1.0 / sigma2_u; q]);
        let mme = MixedModelEquations::assemble(&x, &[z], &y, 1.0 / sigma2_e, &[ginv]);
        mme.solve().unwrap().c_inv.unwrap()
    }

    fn one_way_calculator(
        q: usize,
        n_per_group: usize,
        sigma2_u: f64,
        sigma2_e: f64,
        ai: DMatrix<f64>,
    ) -> Option<DdfCalculator> {
        let (x, z, _) = one_way_design(q, n_per_group);
        let c_inv = one_way_inverse(q, n_per_group, sigma2_u, sigma2_e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[sigma2_u, sigma2_e],
        );
        DdfCalculator::new(c_inv.fixed_block(1), dphi, ai, q * n_per_group)
    }

    /// Helper to build an AI matrix for testing.
    fn make_simple_ai(sigma2_u: f64, sigma2_e: f64, q: usize, n: usize) -> DMatrix<f64> {
        let mut ai = DMatrix::zeros(2, 2);
        ai[(0, 0)] = (q as f64) / (2.0 * sigma2_u * sigma2_u);
        ai[(1, 1)] = (n as f64) / (2.0 * sigma2_e * sigma2_e);
        ai[(0, 1)] = 0.01;
        ai[(1, 0)] = 0.01;
        ai
    }

    #[test]
    fn test_ddf_calculator_creation() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        assert!(one_way_calculator(q, n_per, s2u, s2e, ai).is_some());
    }

    #[test]
    fn test_ddf_calculator_singular_ai() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = DMatrix::zeros(2, 2);
        assert!(one_way_calculator(q, n_per, s2u, s2e, ai).is_none());
    }

    #[test]
    fn derivatives_match_numerical_differentiation() {
        let (q, n_per, s2u, s2e) = (6, 4, 1.7, 0.9);
        let (x, z, y) = one_way_design(q, n_per);
        let phi_at = |s2u: f64, s2e: f64| {
            let ginv = sparse_diagonal(&vec![1.0 / s2u; q]);
            let mme =
                MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
            mme.solve().unwrap().c_inv.unwrap().fixed_block(1)[(0, 0)]
        };
        let c_inv = one_way_inverse(q, n_per, s2u, s2e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[s2u, s2e],
        );
        let h = 1e-5;
        let num_u = (phi_at(s2u + h, s2e) - phi_at(s2u - h, s2e)) / (2.0 * h);
        let num_e = (phi_at(s2u, s2e + h) - phi_at(s2u, s2e - h)) / (2.0 * h);
        assert_relative_eq!(dphi[0][(0, 0)], num_u, epsilon = 1e-7, max_relative = 1e-5);
        assert_relative_eq!(dphi[1][(0, 0)], num_e, epsilon = 1e-7, max_relative = 1e-5);
    }

    #[test]
    fn derivatives_with_relationship_matrix_match_numerical_differentiation() {
        // Two related "animals" per group: K⁻¹ is not the identity.
        let q = 4;
        let n_per = 3;
        let (x, z, y) = one_way_design(q, n_per);
        let mut kinv_tri = sprs::TriMat::new((q, q));
        for i in 0..q {
            kinv_tri.add_triplet(i, i, 1.5);
        }
        kinv_tri.add_triplet(0, 1, -0.5);
        kinv_tri.add_triplet(1, 0, -0.5);
        kinv_tri.add_triplet(2, 3, -0.4);
        kinv_tri.add_triplet(3, 2, -0.4);
        let kinv = kinv_tri.to_csc();
        let (s2u, s2e) = (2.0, 1.3);
        let phi_at = |s2u: f64, s2e: f64| {
            let ginv = kinv.map(|v| v / s2u);
            let mme =
                MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
            mme.solve().unwrap().c_inv.unwrap().fixed_block(1)[(0, 0)]
        };
        let ginv = kinv.map(|v| v / s2u);
        let mme =
            MixedModelEquations::assemble(&x, std::slice::from_ref(&z), &y, 1.0 / s2e, &[ginv]);
        let c_inv = mme.solve().unwrap().c_inv.unwrap();
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[Some(&kinv)],
            &[s2u, s2e],
        );
        let h = 1e-5;
        let num_u = (phi_at(s2u + h, s2e) - phi_at(s2u - h, s2e)) / (2.0 * h);
        let num_e = (phi_at(s2u, s2e + h) - phi_at(s2u, s2e - h)) / (2.0 * h);
        assert_relative_eq!(dphi[0][(0, 0)], num_u, epsilon = 1e-7, max_relative = 1e-5);
        assert_relative_eq!(dphi[1][(0, 0)], num_e, epsilon = 1e-7, max_relative = 1e-5);
    }

    #[test]
    fn test_satterthwaite_ddf_single_contrast() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let n_obs = q * n_per;
        let ai = make_simple_ai(s2u, s2e, q, n_obs);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu > 0.0, "Satterthwaite ddf should be positive, got {}", nu);
        assert!(nu <= (n_obs - 1) as f64);
        // The intercept of a one-way random model is estimated from the q
        // group means: its df are close to q - 1 when the group variance
        // dominates the residual.
        assert!(nu < 20.0, "ddf {} should reflect the between-group df", nu);
    }

    #[test]
    fn test_satterthwaite_ddf_bounded() {
        let (q, n_per, s2e, s2u) = (3, 5, 1.0, 0.001);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu >= 1.0, "ddf should be at least 1, got {}", nu);
        assert!(nu <= (q * n_per - 1) as f64, "ddf should be at most n - p");
    }

    #[test]
    fn test_containment_ddf() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        assert_relative_eq!(calc.containment_ddf(), (q * n_per - 1) as f64);
    }

    #[test]
    fn test_satterthwaite_multi_df() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        // A repeated contrast is rank 1: one eigenvalue is zero and the
        // multi-df value equals the single-contrast value.
        let nu_multi = calc.satterthwaite_ddf_multi(&[vec![1.0], vec![1.0]]);
        assert!(nu_multi >= 1.0);
        assert!(nu_multi <= (q * n_per - 1) as f64);
        assert_relative_eq!(nu_multi, calc.satterthwaite_ddf(&[1.0]), epsilon = 1e-9);
    }

    #[test]
    fn test_phi_extraction() {
        let (q, n_per, s2e, s2u) = (4, 5, 2.0, 3.0);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let expected = one_way_inverse(q, n_per, s2u, s2e).entry(0, 0);
        assert_eq!(calc.phi().nrows(), 1);
        assert_relative_eq!(calc.phi()[(0, 0)], expected, epsilon = 1e-12);
    }

    fn one_way_fit_result(q: usize, n_per: usize, s2u: f64, s2e: f64) -> FitResult {
        let (x, z, _) = one_way_design(q, n_per);
        let c_inv = one_way_inverse(q, n_per, s2u, s2e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[s2u, s2e],
        );
        let phi = c_inv.fixed_block(1);
        FitResult {
            variance_components: vec![
                VarianceEstimate {
                    name: "group".into(),
                    structure: "Identity".into(),
                    parameters: vec![("sigma2".into(), s2u)],
                    se: vec![0.0],
                    at_boundary: vec![false],
                },
                VarianceEstimate {
                    name: "residual".into(),
                    structure: "Identity".into(),
                    parameters: vec![("sigma2".into(), s2e)],
                    se: vec![0.0],
                    at_boundary: vec![false],
                },
            ],
            fixed_effects: vec![NamedEffect {
                term: "mu".into(),
                level: "intercept".into(),
                estimate: 12.0,
                se: phi[(0, 0)].sqrt(),
            }],
            random_effects: vec![],
            log_likelihood: 0.0,
            n_iterations: 1,
            converged: true,
            history: vec![],
            variance_se: vec![0.0, 0.0],
            residuals: vec![],
            fixed_cov: vec![vec![phi[(0, 0)]]],
            at_boundary: vec![false, false],
            n_obs: q * n_per,
            n_fixed_params: 1,
            n_variance_params: 2,
            c_inv: Some(c_inv),
            ai_matrix: None,
            n_random_per_term: vec![q],
            kenward_roger_terms: None,
            fixed_cov_derivatives: dphi.iter().map(|m| vec![vec![m[(0, 0)]]]).collect(),
        }
    }

    #[test]
    fn test_wald_tests_satterthwaite_basic() {
        let (q, n_per, s2u, s2e) = (5, 10, 3.0, 2.0);
        let result = one_way_fit_result(q, n_per, s2u, s2e);
        let ai = make_simple_ai(s2u, s2e, q, q * n_per);
        let tests = wald_tests_satterthwaite(&result, &ai).expect("Should produce Wald tests");
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);
        assert!(tests[0].den_df >= 1.0 && tests[0].den_df <= (q * n_per - 1) as f64);
        assert!(tests[0].f_statistic > 0.0);
        assert!(tests[0].p_value >= 0.0 && tests[0].p_value <= 1.0);
        // Through the FitResult method the AI matrix is required.
        assert!(result.wald_tests_satterthwaite().is_none());
        let mut with_ai = result.clone();
        with_ai.ai_matrix = Some(ai);
        assert_eq!(with_ai.wald_tests_satterthwaite().unwrap().len(), 1);
    }

    #[test]
    fn test_wald_tests_satterthwaite_empty() {
        let mut result = one_way_fit_result(3, 4, 1.0, 1.0);
        result.fixed_effects.clear();
        let ai = make_simple_ai(1.0, 1.0, 3, 12);
        let tests = wald_tests_satterthwaite(&result, &ai);
        assert!(tests.is_some());
        assert!(tests.unwrap().is_empty());
    }

    #[test]
    fn test_satterthwaite_vs_containment_balanced() {
        let (q, n_per, s2e, s2u) = (5, 10, 2.0, 3.0);
        let n_obs = q * n_per;
        let ai = make_simple_ai(s2u, s2e, q, n_obs);
        let calc = one_way_calculator(q, n_per, s2u, s2e, ai).unwrap();
        let nu = calc.satterthwaite_ddf(&[1.0]);
        assert!(nu <= calc.containment_ddf());
        assert!(nu >= 1.0);
    }

    #[test]
    fn kenward_roger_p_matches_covariance_derivatives() {
        // ∂Φ/∂θ_i = −Φ P_i Φ for the one-way model (identity K).
        let (q, n_per, s2u, s2e) = (6, 4, 1.7, 0.9);
        let (x, z, _) = one_way_design(q, n_per);
        let c_inv = one_way_inverse(q, n_per, s2u, s2e);
        let dphi = fixed_cov_derivatives_scaled_identity(
            &c_inv,
            &x,
            std::slice::from_ref(&z),
            &[None],
            &[s2u, s2e],
        );
        let ginv = sparse_diagonal(&vec![1.0 / s2u; q]);
        let kr =
            kenward_roger_terms_scaled_identity(&x, std::slice::from_ref(&z), &[ginv], &[s2u, s2e])
                .unwrap();
        let phi = c_inv.fixed_block(1);
        for i in 0..2 {
            let from_p = -(&phi * &kr.p[i] * &phi);
            assert_relative_eq!(from_p[(0, 0)], dphi[i][(0, 0)], max_relative = 1e-9);
        }
        // Q_ii is a Gram matrix in the V⁻¹ inner product: non-negative.
        for i in 0..2 {
            assert!(kr.q[i][i][(0, 0)] >= 0.0);
        }
    }

    #[test]
    fn kenward_roger_reproduces_anova_for_balanced_rcbd() {
        use crate::data::DataFrame;
        use crate::model::MixedModelBuilder;
        use crate::variance::Identity;

        // t treatments (fixed) x b blocks (random), one plot per cell.
        let (t, b) = (4usize, 5usize);
        let tau = [0.0, 1.5, -1.0, 0.5];
        let blk = [-3.0, -1.0, 0.0, 1.2, 2.8];
        // Deterministic pseudo-random residuals.
        let mut state: u64 = 12345;
        let mut noise = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0
        };
        let mut y = Vec::new();
        let mut treat = Vec::new();
        let mut block = Vec::new();
        for bi in 0..b {
            for ti in 0..t {
                y.push(10.0 + tau[ti] + blk[bi] + noise());
                treat.push(format!("T{ti}"));
                block.push(format!("B{bi}"));
            }
        }
        let mut df = DataFrame::new();
        df.add_float_column("y", y.clone()).unwrap();
        let treat_ref: Vec<&str> = treat.iter().map(|s| s.as_str()).collect();
        let block_ref: Vec<&str> = block.iter().map(|s| s.as_str()).collect();
        df.add_factor_column("treat", &treat_ref).unwrap();
        df.add_factor_column("block", &block_ref).unwrap();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .fixed("mu + treat")
            .random("block", Identity::new(1.0), None)
            .max_iterations(200)
            .convergence(1e-12)
            .build()
            .unwrap();
        let result = model.fit_reml().unwrap();
        assert!(result.converged);
        assert!(
            !result.at_boundary.iter().any(|f| *f),
            "{:?}",
            result.at_boundary
        );

        // Two-way ANOVA F for treatments.
        let n = (t * b) as f64;
        let grand = y.iter().sum::<f64>() / n;
        let mut ss_t = 0.0;
        for ti in 0..t {
            let m = (0..b).map(|bi| y[bi * t + ti]).sum::<f64>() / b as f64;
            ss_t += b as f64 * (m - grand).powi(2);
        }
        let mut ss_b = 0.0;
        for bi in 0..b {
            let m = (0..t).map(|ti| y[bi * t + ti]).sum::<f64>() / t as f64;
            ss_b += t as f64 * (m - grand).powi(2);
        }
        let ss_tot: f64 = y.iter().map(|v| (v - grand).powi(2)).sum();
        let ss_e = ss_tot - ss_t - ss_b;
        let df_e = ((t - 1) * (b - 1)) as f64;
        let f_anova = (ss_t / (t - 1) as f64) / (ss_e / df_e);

        let kr = result.wald_tests_kenward_roger().expect("KR available");
        let test = kr.iter().find(|w| w.term == "treat").unwrap();
        assert_eq!(test.num_df, t - 1);
        assert_relative_eq!(test.den_df, df_e, epsilon = 1e-6);
        assert_relative_eq!(test.f_statistic, f_anova, max_relative = 1e-6);
        // The residual variance is the ANOVA error mean square.
        assert_relative_eq!(
            result.variance_component("residual").unwrap(),
            ss_e / df_e,
            max_relative = 1e-6
        );
        // Satterthwaite agrees on the balanced design as well.
        let satt = result.wald_tests_satterthwaite().unwrap();
        let st = satt.iter().find(|w| w.term == "treat").unwrap();
        assert_relative_eq!(st.den_df, df_e, epsilon = 1e-4);
    }
}
