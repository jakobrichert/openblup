use crate::error::{LmmError, Result};
use nalgebra::{DMatrix, DVector};

/// Result from fitting an RR-BLUP model.
#[derive(Debug, Clone)]
pub struct RrBlupResult {
    /// Estimated fixed effects b̂.
    pub fixed_effects: Vec<f64>,
    /// Estimated marker effects û.
    pub marker_effects: Vec<f64>,
    /// Genomic breeding values: g = Mû.
    pub breeding_values: Vec<f64>,
    /// Marker variance σ²_u.
    pub sigma2_u: f64,
    /// Residual variance σ²_e.
    pub sigma2_e: f64,
    /// Variance ratio λ = σ²_e / σ²_u.
    pub lambda: f64,
    /// REML log-likelihood at (σ²_u, σ²_e), on the same scale as
    /// [`FitResult::log_likelihood`](crate::lmm::FitResult).
    pub log_likelihood: f64,
    /// Genomic heritability: h² = σ²_g / (σ²_g + σ²_e) with
    /// σ²_g = σ²_u · tr(MM') / n, the average genetic variance of the
    /// individuals (σ²_u · Σ 2p(1-p) for centred 0/1/2 genotypes).
    pub heritability: f64,
    /// Number of REML log-likelihood evaluations (1 for a fixed λ).
    pub n_iterations: usize,
    /// Whether REML converged.
    pub converged: bool,
}

/// Ridge Regression BLUP for estimating individual SNP marker effects.
///
/// Model: y = Xb + Mu + e
/// where M is the centered marker matrix (n × m),
/// u ~ N(0, σ²_u I_m), e ~ N(0, σ²_e I_n).
///
/// The MME: [X'X, X'M; M'X, M'M + λI] [b̂; û] = [X'y; M'y]
/// where λ = σ²_e / σ²_u.
///
/// All computations are done in the smaller of the two spaces: with
/// `n <= m` through the `n × n` matrix `MM' + λI`, otherwise through the
/// `(p + m) × (p + m)` MME, so many thousands of markers are fine as long as
/// the number of individuals is moderate (and vice versa).
///
/// Reference: Meuwissen et al. (2001). Prediction of total genetic value using
/// genome-wide dense marker maps. Genetics, 157, 1819-1829.
#[derive(Debug)]
pub struct RrBlup {
    /// Centered marker matrix M (n × m).
    marker_matrix: DMatrix<f64>,
    /// Fixed effects design matrix X (n × p).
    x_matrix: DMatrix<f64>,
    /// Response vector y.
    y: Vec<f64>,
    /// Allele frequencies for centering.
    allele_freqs: Vec<f64>,
    /// Fitted results.
    result: Option<RrBlupResult>,
}

/// Search range for `ln(λ / scale)`, where `scale` is the average genetic
/// variance per unit σ²_u, i.e. `ln(σ²_e / σ²_g)` from about -10 to 10
/// (h² from 0.99995 down to 0.00005).
const LOG_RATIO_RANGE: f64 = 10.0;
/// Number of grid points for the initial search over `ln(λ / scale)`.
const GRID_POINTS: usize = 101;

/// The space the model is solved in.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Space {
    /// `n × n` system in `K = MM'` (used when `n <= m`).
    Individuals,
    /// `(p + m) × (p + m)` MME in `M'M` (used when `n > m`).
    Markers,
}

impl RrBlup {
    /// Create from raw 0/1/2 coded genotype matrix.
    /// Centers markers: M = genotypes - 2p.
    pub fn new(genotypes: &DMatrix<f64>, x: &DMatrix<f64>, y: &[f64]) -> Self {
        let n = genotypes.nrows();
        let m = genotypes.ncols();
        assert_eq!(n, y.len(), "genotype rows must match y length");
        assert_eq!(n, x.nrows(), "X rows must match y length");

        // Compute allele frequencies
        let mut allele_freqs = vec![0.0; m];
        for j in 0..m {
            let sum: f64 = (0..n).map(|i| genotypes[(i, j)]).sum();
            allele_freqs[j] = sum / (2.0 * n as f64);
        }

        // Center: M = genotypes - 2p
        let mut centered = genotypes.clone();
        for j in 0..m {
            let two_p = 2.0 * allele_freqs[j];
            for i in 0..n {
                centered[(i, j)] -= two_p;
            }
        }

        Self {
            marker_matrix: centered,
            x_matrix: x.clone(),
            y: y.to_vec(),
            allele_freqs,
            result: None,
        }
    }

    /// Create from pre-centered marker matrix.
    pub fn from_centered(m_centered: DMatrix<f64>, x: DMatrix<f64>, y: Vec<f64>) -> Self {
        let m = m_centered.ncols();
        Self {
            marker_matrix: m_centered,
            x_matrix: x,
            y,
            allele_freqs: vec![0.0; m],
            result: None,
        }
    }

    /// Fit the model by REML.
    ///
    /// σ²_u is profiled out of the restricted likelihood, which is then
    /// maximised over λ alone using the eigendecomposition of `SMM'S`
    /// (`S = I - X(X'X)⁻¹X'`) or, equivalently, of `M'SM` (Kang et al. 2008,
    /// EMMA; Endelman 2011, rrBLUP). The result is the exact REML estimate,
    /// not an EM approximation. If the likelihood keeps increasing towards
    /// h² = 0 (or 1), the estimate is the end of the search range, i.e. σ²_u
    /// (or σ²_e) is a tiny fraction of the other variance.
    pub fn fit(&mut self) -> Result<RrBlupResult> {
        let space = self.default_space();
        self.fit_in(space)
    }

    /// Fit with a known λ = σ²_e / σ²_u (skip REML).
    ///
    /// σ²_u (and σ²_e = λσ²_u) are the REML estimates given λ, i.e.
    /// σ²_e = y'ê / (n - p).
    pub fn fit_with_lambda(&mut self, lambda: f64) -> Result<RrBlupResult> {
        let space = self.default_space();
        self.fit_with_lambda_in(space, lambda)
    }

    fn default_space(&self) -> Space {
        if self.marker_matrix.nrows() <= self.marker_matrix.ncols() {
            Space::Individuals
        } else {
            Space::Markers
        }
    }

    fn fit_in(&mut self, space: Space) -> Result<RrBlupResult> {
        let (gram, xtx_chol) = self.prepare(space)?;
        let profile = ProfileLikelihood::new(self, space, &gram, &xtx_chol)?;
        let (lambda, n_evals) = profile.maximize()?;
        let result = self.solve(space, &gram, lambda, n_evals)?;
        self.result = Some(result.clone());
        Ok(result)
    }

    fn fit_with_lambda_in(&mut self, space: Space, lambda: f64) -> Result<RrBlupResult> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(LmmError::InvalidParameter(format!(
                "lambda must be positive and finite, got {}",
                lambda
            )));
        }
        let (gram, _) = self.prepare(space)?;
        let result = self.solve(space, &gram, lambda, 1)?;
        self.result = Some(result.clone());
        Ok(result)
    }

    /// Check the dimensions and form the Gram matrix of the chosen space
    /// (`MM'` or `M'M`) and the Cholesky factor of `X'X`.
    fn prepare(
        &self,
        space: Space,
    ) -> Result<(DMatrix<f64>, nalgebra::Cholesky<f64, nalgebra::Dyn>)> {
        let n = self.y.len();
        let p = self.x_matrix.ncols();
        let m = self.marker_matrix.ncols();
        if m == 0 {
            return Err(LmmError::InvalidParameter(
                "RR-BLUP needs at least one marker".into(),
            ));
        }
        if n <= p {
            return Err(LmmError::InvalidParameter(format!(
                "RR-BLUP needs more observations ({}) than fixed effects ({})",
                n, p
            )));
        }
        if self.y.iter().any(|v| !v.is_finite())
            || self.x_matrix.iter().any(|v| !v.is_finite())
            || self.marker_matrix.iter().any(|v| !v.is_finite())
        {
            return Err(LmmError::Data(
                "RR-BLUP input contains missing or non-finite values".into(),
            ));
        }
        let xtx_chol = (self.x_matrix.transpose() * &self.x_matrix)
            .cholesky()
            .ok_or_else(|| LmmError::SingularMatrix {
                context: "RR-BLUP X'X (fixed effects are not of full column rank)".into(),
            })?;
        let mm = &self.marker_matrix;
        let gram = match space {
            Space::Individuals => mm * mm.transpose(),
            Space::Markers => mm.transpose() * mm,
        };
        Ok((gram, xtx_chol))
    }

    /// BLUE/BLUP, the REML variances given λ and the REML log-likelihood.
    fn solve(
        &self,
        space: Space,
        gram: &DMatrix<f64>,
        lambda: f64,
        n_evals: usize,
    ) -> Result<RrBlupResult> {
        let n = self.y.len();
        let p = self.x_matrix.ncols();
        let m = self.marker_matrix.ncols();
        let x = &self.x_matrix;
        let mm = &self.marker_matrix;
        let y = DVector::from_column_slice(&self.y);
        let log_det = |l: &DMatrix<f64>| 2.0 * l.diagonal().iter().map(|d| d.ln()).sum::<f64>();

        // `logdet` is log|H| + log|X'H⁻¹X| with H = MM' + λI, and
        // `q` = (y - Xb̂)' H⁻¹ (y - Xb̂) (so that y'Py = q / σ²_u).
        let (fixed, markers, logdet, q) = match space {
            Space::Individuals => {
                let mut h = gram.clone();
                for i in 0..n {
                    h[(i, i)] += lambda;
                }
                let h_chol = h.cholesky().ok_or(LmmError::NotPositiveDefinite)?;
                let hinv_x = h_chol.solve(x);
                let hinv_y = h_chol.solve(&y);
                let xhx_chol = (x.transpose() * &hinv_x)
                    .cholesky()
                    .ok_or(LmmError::NotPositiveDefinite)?;
                let b = xhx_chol.solve(&(x.transpose() * &hinv_y));
                let r = &y - x * &b;
                let hinv_r = h_chol.solve(&r);
                let u = mm.transpose() * &hinv_r;
                let logdet = log_det(&h_chol.l()) + log_det(&xhx_chol.l());
                (b, u, logdet, r.dot(&hinv_r))
            }
            Space::Markers => {
                let dim = p + m;
                let xtm = x.transpose() * mm;
                let mut c = DMatrix::zeros(dim, dim);
                c.view_mut((0, 0), (p, p)).copy_from(&(x.transpose() * x));
                c.view_mut((0, p), (p, m)).copy_from(&xtm);
                c.view_mut((p, 0), (m, p)).copy_from(&xtm.transpose());
                c.view_mut((p, p), (m, m)).copy_from(gram);
                for i in 0..m {
                    c[(p + i, p + i)] += lambda;
                }
                let mut rhs = DVector::zeros(dim);
                rhs.rows_mut(0, p).copy_from(&(x.transpose() * &y));
                rhs.rows_mut(p, m).copy_from(&(mm.transpose() * &y));
                let c_chol = c.cholesky().ok_or(LmmError::NotPositiveDefinite)?;
                let sol = c_chol.solve(&rhs);
                let b = sol.rows(0, p).into_owned();
                let u = sol.rows(p, m).into_owned();
                let e = &y - x * &b - mm * &u;
                // log|C| = log|X'X| + log|M'SM + λI|, and
                // log|H| + log|X'H⁻¹X| = log|C| + (n - m - p) log λ;
                // y'H⁻¹(y - Xb̂) = y'ê / λ = ê'ê / λ + û'û.
                let logdet = log_det(&c_chol.l()) + (n as f64 - m as f64 - p as f64) * lambda.ln();
                let q = e.dot(&e) / lambda + u.dot(&u);
                (b, u, logdet, q)
            }
        };

        let n_eff = (n - p) as f64;
        let sigma2_u = q / n_eff;
        let sigma2_e = lambda * sigma2_u;
        let log_likelihood =
            -0.5 * (n_eff * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma2_u.ln()) + logdet);

        let breeding_values = mm * &markers;
        let sigma2_g = sigma2_u * mm.norm_squared() / n as f64;
        let heritability = sigma2_g / (sigma2_g + sigma2_e);

        Ok(RrBlupResult {
            fixed_effects: fixed.as_slice().to_vec(),
            marker_effects: markers.as_slice().to_vec(),
            breeding_values: breeding_values.as_slice().to_vec(),
            sigma2_u,
            sigma2_e,
            lambda,
            log_likelihood,
            heritability,
            n_iterations: n_evals,
            converged: sigma2_u.is_finite() && sigma2_u > 0.0,
        })
    }

    /// Predict breeding values for new individuals.
    pub fn predict(&self, new_genotypes: &DMatrix<f64>) -> Result<Vec<f64>> {
        let result = self
            .result
            .as_ref()
            .ok_or_else(|| LmmError::ModelSpec("Model not fitted".into()))?;
        let m = self.marker_matrix.ncols();
        assert_eq!(new_genotypes.ncols(), m, "marker count mismatch");

        // Center using training allele frequencies
        let mut centered = new_genotypes.clone();
        for j in 0..m {
            let two_p = 2.0 * self.allele_freqs[j];
            for i in 0..centered.nrows() {
                centered[(i, j)] -= two_p;
            }
        }

        let u = nalgebra::DVector::from_column_slice(&result.marker_effects);
        let gebv = &centered * &u;
        Ok(gebv.as_slice().to_vec())
    }

    /// Get marker effects (after fitting).
    pub fn marker_effects(&self) -> Option<&[f64]> {
        self.result.as_ref().map(|r| r.marker_effects.as_slice())
    }
}

/// The REML log-likelihood with σ²_u profiled out, as a function of λ.
///
/// With `ξ_i` the non-zero-space eigenvalues of `SMM'S` (`n - p` of them)
/// and `q(λ) = Σ η_i² / (ξ_i + λ)`, `η` the rotated residuals `Sy`:
///
/// ```text
/// σ²_u(λ) = q(λ) / (n - p)
/// logL(λ) = -1/2 [ (n - p)(log 2π + 1 + log σ²_u(λ)) + log|X'X| + Σ log(ξ_i + λ) ]
/// ```
///
/// In the marker space (`M'SM` with eigenvalues `Λ_j`, `g = Q'M'Sy`):
///
/// ```text
/// Σ_i log(ξ_i + λ) = Σ_j log(Λ_j + λ) + (n - p - m) log λ
/// q(λ) = |Sy|²_⊥ / λ + Σ_{Λ_j > 0} g_j² / (Λ_j (Λ_j + λ))
/// ```
///
/// where `|Sy|²_⊥ = |Sy|² - Σ_{Λ_j > 0} g_j² / Λ_j` is the part of `Sy`
/// outside the column space of `SM`.
struct ProfileLikelihood {
    n_eff: f64,
    /// log|X'X|
    log_det_xtx: f64,
    /// Eigenvalues (non-negative).
    eigenvalues: Vec<f64>,
    /// Squared rotated residuals, one per eigenvalue.
    weights: Vec<f64>,
    /// Marker space only: squared norm of the residual component orthogonal
    /// to `SM`, and the `(n - p - m)` multiplier of `log λ`.
    marker_space: Option<(f64, f64)>,
    /// Average genetic variance per unit σ²_u (centre of the λ search).
    scale: f64,
}

impl ProfileLikelihood {
    fn new(
        model: &RrBlup,
        space: Space,
        gram: &DMatrix<f64>,
        xtx_chol: &nalgebra::Cholesky<f64, nalgebra::Dyn>,
    ) -> Result<Self> {
        let n = model.y.len();
        let p = model.x_matrix.ncols();
        let m = model.marker_matrix.ncols();
        let x = &model.x_matrix;
        let mm = &model.marker_matrix;
        let y = DVector::from_column_slice(&model.y);
        let n_eff = (n - p) as f64;
        let log_det_xtx = 2.0 * xtx_chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

        // Sy = y - X(X'X)⁻¹X'y
        let sy = &y - x * xtx_chol.solve(&(x.transpose() * &y));

        match space {
            Space::Individuals => {
                // SK = K - X(X'X)⁻¹X'K, SKS = SK - SKX(X'X)⁻¹X'
                let sk = gram - x * xtx_chol.solve(&(x.transpose() * gram));
                let sks = &sk - (&sk * x) * xtx_chol.solve(&x.transpose());
                let trace = sks.trace().max(0.0);
                let scale = if trace > 0.0 { trace / n_eff } else { 1.0 };
                // Eigenvectors in the column space of X have eigenvalue 0 in
                // SKS; adding scale·S lifts all others to >= scale, so the
                // n - p largest eigenvalues of SKS + scale·S are ξ_i + scale.
                let proj = DMatrix::identity(n, n) - x * xtx_chol.solve(&x.transpose());
                let mut lifted = sks + proj * scale;
                lifted = (&lifted + lifted.transpose()) * 0.5;
                let eig = lifted.symmetric_eigen();
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
                let keep = &order[..n - p];
                let eigenvalues = keep
                    .iter()
                    .map(|&i| (eig.eigenvalues[i] - scale).max(0.0))
                    .collect();
                let weights = keep
                    .iter()
                    .map(|&i| eig.eigenvectors.column(i).dot(&sy).powi(2))
                    .collect();
                Ok(Self {
                    n_eff,
                    log_det_xtx,
                    eigenvalues,
                    weights,
                    marker_space: None,
                    scale,
                })
            }
            Space::Markers => {
                // M'SM = M'M - M'X(X'X)⁻¹X'M
                let xtm = x.transpose() * mm;
                let mut msm = gram - xtm.transpose() * xtx_chol.solve(&xtm);
                msm = (&msm + msm.transpose()) * 0.5;
                let trace = msm.trace().max(0.0);
                let scale = if trace > 0.0 { trace / n_eff } else { 1.0 };
                let eig = msm.symmetric_eigen();
                let max_eig = eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
                let tol = max_eig * m.max(n) as f64 * f64::EPSILON;
                let mts = mm.transpose() * &sy;
                let mut eigenvalues = Vec::with_capacity(m);
                let mut weights = Vec::with_capacity(m);
                let mut perp = sy.norm_squared();
                for j in 0..m {
                    let lam = eig.eigenvalues[j];
                    if lam > tol {
                        let g = eig.eigenvectors.column(j).dot(&mts);
                        perp -= g * g / lam;
                        eigenvalues.push(lam);
                        weights.push(g * g);
                    } else {
                        eigenvalues.push(0.0);
                        weights.push(0.0);
                    }
                }
                Ok(Self {
                    n_eff,
                    log_det_xtx,
                    eigenvalues,
                    weights,
                    marker_space: Some((perp.max(0.0), n_eff - m as f64)),
                    scale,
                })
            }
        }
    }

    /// Profiled REML log-likelihood at λ (−∞ where it is undefined).
    fn log_likelihood(&self, lambda: f64) -> f64 {
        let mut log_det: f64 = self.eigenvalues.iter().map(|&e| (e + lambda).ln()).sum();
        let q = match self.marker_space {
            None => self
                .eigenvalues
                .iter()
                .zip(&self.weights)
                .map(|(&e, &w)| w / (e + lambda))
                .sum::<f64>(),
            Some((perp, extra)) => {
                log_det += extra * lambda.ln();
                perp / lambda
                    + self
                        .eigenvalues
                        .iter()
                        .zip(&self.weights)
                        .filter(|(&e, _)| e > 0.0)
                        .map(|(&e, &w)| w / (e * (e + lambda)))
                        .sum::<f64>()
            }
        };
        if !(q > 0.0 && q.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let sigma2_u = q / self.n_eff;
        -0.5 * (self.n_eff * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma2_u.ln())
            + self.log_det_xtx
            + log_det)
    }

    /// Maximise over `t = ln(λ / scale)`: a grid search followed by a
    /// golden-section refinement around every local maximum of the grid.
    /// Returns λ and the number of likelihood evaluations.
    fn maximize(&self) -> Result<(f64, usize)> {
        let lambda_at = |t: f64| self.scale * t.exp();
        let step = 2.0 * LOG_RATIO_RANGE / (GRID_POINTS - 1) as f64;
        let grid: Vec<f64> = (0..GRID_POINTS)
            .map(|k| -LOG_RATIO_RANGE + k as f64 * step)
            .collect();
        let values: Vec<f64> = grid
            .iter()
            .map(|&t| self.log_likelihood(lambda_at(t)))
            .collect();
        let mut n_evals = GRID_POINTS;

        let mut best_t = f64::NAN;
        let mut best = f64::NEG_INFINITY;
        for k in 0..GRID_POINTS {
            let left = if k > 0 {
                values[k - 1]
            } else {
                f64::NEG_INFINITY
            };
            let right = if k + 1 < GRID_POINTS {
                values[k + 1]
            } else {
                f64::NEG_INFINITY
            };
            if !(values[k] > f64::NEG_INFINITY && values[k] >= left && values[k] >= right) {
                continue;
            }
            let (t, v) = if k == 0 || k + 1 == GRID_POINTS {
                (grid[k], values[k])
            } else {
                let (t, v, evals) = golden_section_max(
                    |t| self.log_likelihood(lambda_at(t)),
                    grid[k - 1],
                    grid[k + 1],
                );
                n_evals += evals;
                if v >= values[k] {
                    (t, v)
                } else {
                    (grid[k], values[k])
                }
            };
            if v > best {
                best = v;
                best_t = t;
            }
        }
        if !best.is_finite() {
            return Err(LmmError::Data(
                "RR-BLUP REML likelihood is undefined (the fixed effects fit the response exactly?)"
                    .into(),
            ));
        }
        Ok((lambda_at(best_t), n_evals))
    }
}

/// Golden-section search for the maximum of `f` on `[a, b]`.
/// Returns `(argmax, max, evaluations)`.
fn golden_section_max(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> (f64, f64, usize) {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    let mut evals = 2;
    while (b - a) > 1e-10 {
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - ratio * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + ratio * (b - a);
            fd = f(d);
        }
        evals += 1;
    }
    if fc >= fd {
        (c, fc, evals)
    } else {
        (d, fd, evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn simple_data() -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
        // 5 individuals, 10 markers
        let geno = DMatrix::from_row_slice(
            5,
            10,
            &[
                0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0,
                1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 2.0,
            ],
        );
        // Intercept-only X
        let x = DMatrix::from_element(5, 1, 1.0);
        let y = vec![105.0, 98.0, 101.0, 96.0, 110.0];
        (geno, x, y)
    }

    #[test]
    fn test_rrblup_centering() {
        let (geno, x, y) = simple_data();
        let model = RrBlup::new(&geno, &x, &y);
        // Centered matrix should have column means ≈ 0
        for j in 0..10 {
            let col_mean: f64 = (0..5).map(|i| model.marker_matrix[(i, j)]).sum::<f64>() / 5.0;
            assert!(col_mean.abs() < 1e-10, "column {} mean = {}", j, col_mean);
        }
    }

    #[test]
    fn test_rrblup_fit_with_lambda() {
        let (geno, x, y) = simple_data();
        let mut model = RrBlup::new(&geno, &x, &y);
        let result = model.fit_with_lambda(1.0).unwrap();
        assert_eq!(result.marker_effects.len(), 10);
        assert_eq!(result.breeding_values.len(), 5);
        assert_eq!(result.fixed_effects.len(), 1);
    }

    #[test]
    fn test_rrblup_fit_reml() {
        let (geno, x, y) = simple_data();
        let mut model = RrBlup::new(&geno, &x, &y);
        let result = model.fit().unwrap();
        assert!(result.sigma2_u > 0.0);
        assert!(result.sigma2_e > 0.0);
        assert!(result.heritability >= 0.0 && result.heritability <= 1.0);
    }

    #[test]
    fn test_rrblup_breeding_values_length() {
        let (geno, x, y) = simple_data();
        let mut model = RrBlup::new(&geno, &x, &y);
        model.fit_with_lambda(2.0).unwrap();
        assert_eq!(model.marker_effects().unwrap().len(), 10);
    }

    #[test]
    fn test_rrblup_predict() {
        let (geno, x, y) = simple_data();
        let mut model = RrBlup::new(&geno, &x, &y);
        model.fit_with_lambda(1.0).unwrap();

        // Predict for new individuals with same genotypes
        let predictions = model.predict(&geno).unwrap();
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_rrblup_known_effects() {
        // Create data where true effects are known
        let m = DMatrix::from_row_slice(
            4,
            3,
            &[
                -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0,
            ],
        );
        let x = DMatrix::from_element(4, 1, 1.0);
        // True: b=100, u=[2, -1, 1], y = Xb + Mu
        let true_u = nalgebra::DVector::from_column_slice(&[2.0, -1.0, 1.0]);
        let mu = &m * &true_u;
        let y: Vec<f64> = (0..4).map(|i| 100.0 + mu[i]).collect();

        let mut model = RrBlup::from_centered(m, x, y);
        let result = model.fit_with_lambda(0.001).unwrap(); // Small λ = trust data

        // Intercept should be close to 100
        assert_relative_eq!(result.fixed_effects[0], 100.0, epsilon = 1.0);
    }

    #[test]
    fn test_rrblup_from_centered() {
        let m = DMatrix::from_row_slice(3, 2, &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0]);
        let x = DMatrix::from_element(3, 1, 1.0);
        let y = vec![5.0, 4.0, 6.0];
        let mut model = RrBlup::from_centered(m, x, y);
        let result = model.fit_with_lambda(1.0).unwrap();
        assert_eq!(result.marker_effects.len(), 2);
    }

    /// Simulated centred 0/1/2 genotypes, intercept + covariate, y with
    /// known marker variance.
    fn simulated(n: usize, m: usize, seed: u64) -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut normal = || {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let geno = DMatrix::from_fn(n, m, |_, _| ((normal() + 1.0).max(0.0).round()).min(2.0));
        let sigma_u = (1.0 / m as f64).sqrt();
        let u: Vec<f64> = (0..m).map(|_| normal() * sigma_u).collect();
        let x = DMatrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { (i % 7) as f64 });
        let y = (0..n)
            .map(|i| {
                let g: f64 = (0..m).map(|j| geno[(i, j)] * u[j]).sum();
                10.0 + 0.3 * x[(i, 1)] + g + normal()
            })
            .collect();
        (geno, x, y)
    }

    /// Exact REML log-likelihood of y ~ N(Xb, σ²_u MM' + σ²_e I) from the
    /// dense n × n covariance matrix.
    fn dense_reml(model: &RrBlup, sigma2_u: f64, sigma2_e: f64) -> f64 {
        let (x, mm) = (&model.x_matrix, &model.marker_matrix);
        let n = model.y.len();
        let p = x.ncols();
        let y = DVector::from_column_slice(&model.y);
        let v = mm * mm.transpose() * sigma2_u + DMatrix::identity(n, n) * sigma2_e;
        let v_chol = v.cholesky().unwrap();
        let log_det = |l: DMatrix<f64>| 2.0 * l.diagonal().iter().map(|d| d.ln()).sum::<f64>();
        let xvx_chol = (x.transpose() * v_chol.solve(x)).cholesky().unwrap();
        let b = xvx_chol.solve(&(x.transpose() * v_chol.solve(&y)));
        let r = &y - x * b;
        let ypy = r.dot(&v_chol.solve(&r));
        -0.5 * ((n - p) as f64 * (2.0 * std::f64::consts::PI).ln()
            + log_det(v_chol.l())
            + log_det(xvx_chol.l())
            + ypy)
    }

    /// The fit is a stationary point of the exact REML likelihood, reports
    /// that likelihood, and nothing nearby is better.
    fn assert_exact_reml(model: &RrBlup, result: &RrBlupResult) {
        let (su, se) = (result.sigma2_u, result.sigma2_e);
        let logl = dense_reml(model, su, se);
        assert_relative_eq!(result.log_likelihood, logl, epsilon = 1e-7 * logl.abs());
        for (dsu, dse) in [(1e-4, 0.0), (0.0, 1e-4)] {
            let (hu, he) = (su * dsu, se * dse);
            let grad = (dense_reml(model, su + hu, se + he) - dense_reml(model, su - hu, se - he))
                / (2.0 * (hu + he));
            // Relative to the curvature scale of each parameter.
            let scale = if dsu > 0.0 { su } else { se };
            assert!(
                (grad * scale).abs() < 1e-4,
                "REML gradient not zero: d/d{} = {} at su={}, se={}",
                if dsu > 0.0 { "su" } else { "se" },
                grad,
                su,
                se
            );
        }
        for (fu, fe) in [(1.05, 1.0), (0.95, 1.0), (1.0, 1.05), (1.0, 0.95)] {
            assert!(dense_reml(model, su * fu, se * fe) < logl);
        }
    }

    #[test]
    fn test_rrblup_reml_is_exact_with_more_markers_than_individuals() {
        let (geno, x, y) = simulated(60, 150, 1);
        let mut model = RrBlup::new(&geno, &x, &y);
        let result = model.fit().unwrap();
        assert_eq!(model.default_space(), Space::Individuals);
        assert!(result.converged);
        assert!(
            result.sigma2_e > 0.1,
            "sigma2_e collapsed: {}",
            result.sigma2_e
        );
        assert_exact_reml(&model, &result);
    }

    #[test]
    fn test_rrblup_reml_is_exact_with_more_individuals_than_markers() {
        let (geno, x, y) = simulated(150, 40, 2);
        let mut model = RrBlup::new(&geno, &x, &y);
        let result = model.fit().unwrap();
        assert_eq!(model.default_space(), Space::Markers);
        assert!(result.converged);
        assert_exact_reml(&model, &result);
    }

    #[test]
    fn test_rrblup_both_spaces_agree() {
        for (n, m) in [(40, 70), (70, 40)] {
            let (geno, x, y) = simulated(n, m, 3);
            let mut model = RrBlup::new(&geno, &x, &y);
            let a = model.fit_in(Space::Individuals).unwrap();
            let b = model.fit_in(Space::Markers).unwrap();
            assert_relative_eq!(a.lambda, b.lambda, max_relative = 1e-6);
            assert_relative_eq!(a.sigma2_u, b.sigma2_u, max_relative = 1e-6);
            assert_relative_eq!(a.log_likelihood, b.log_likelihood, max_relative = 1e-9);

            let a = model.fit_with_lambda_in(Space::Individuals, 7.5).unwrap();
            let b = model.fit_with_lambda_in(Space::Markers, 7.5).unwrap();
            assert_relative_eq!(a.sigma2_e, b.sigma2_e, max_relative = 1e-9);
            assert_relative_eq!(a.log_likelihood, b.log_likelihood, max_relative = 1e-9);
            for (u, v) in a.fixed_effects.iter().zip(&b.fixed_effects) {
                assert_relative_eq!(u, v, epsilon = 1e-8);
            }
            for (u, v) in a.marker_effects.iter().zip(&b.marker_effects) {
                assert_relative_eq!(u, v, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_rrblup_fixed_lambda_variances_and_likelihood() {
        let (geno, x, y) = simulated(50, 30, 4);
        let mut model = RrBlup::new(&geno, &x, &y);
        let result = model.fit_with_lambda(12.0).unwrap();
        let (n, p) = (50.0, 2.0);
        // σ²_e = y'ê / (n - p)
        let fitted = &model.x_matrix * DVector::from_column_slice(&result.fixed_effects)
            + DVector::from_column_slice(&result.breeding_values);
        let y_e: f64 = (0..50).map(|i| y[i] * (y[i] - fitted[i])).sum();
        assert_relative_eq!(result.sigma2_e, y_e / (n - p), max_relative = 1e-9);
        assert_relative_eq!(
            result.sigma2_e / result.sigma2_u,
            12.0,
            max_relative = 1e-12
        );
        let logl = dense_reml(&model, result.sigma2_u, result.sigma2_e);
        assert_relative_eq!(result.log_likelihood, logl, max_relative = 1e-9);
        // The REML fit is at least as good.
        let reml = model.fit().unwrap();
        assert!(reml.log_likelihood >= logl);
    }

    #[test]
    fn test_rrblup_heritability_uses_marker_variance() {
        let (geno, x, y) = simulated(40, 20, 5);
        let mut model = RrBlup::new(&geno, &x, &y);
        let r = model.fit().unwrap();
        let sigma2_g = r.sigma2_u * model.marker_matrix.norm_squared() / 40.0;
        assert_relative_eq!(
            r.heritability,
            sigma2_g / (sigma2_g + r.sigma2_e),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_rrblup_rejects_bad_input() {
        let (geno, x, mut y) = simple_data();
        assert!(RrBlup::new(&geno, &x, &y).fit_with_lambda(0.0).is_err());
        assert!(RrBlup::new(&geno, &x, &y)
            .fit_with_lambda(f64::NAN)
            .is_err());
        y[2] = f64::NAN;
        assert!(RrBlup::new(&geno, &x, &y).fit().is_err());
    }
}
