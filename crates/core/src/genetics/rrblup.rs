use crate::error::{LmmError, Result};
use nalgebra::DMatrix;

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
    /// REML log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Genomic heritability: h² = σ²_u * m / (σ²_u * m + σ²_e).
    pub heritability: f64,
    /// Number of REML iterations.
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

    /// Fit the model using EM-REML.
    pub fn fit(&mut self) -> Result<RrBlupResult> {
        let n = self.y.len();
        let p = self.x_matrix.ncols();
        let m = self.marker_matrix.ncols();

        // Pre-compute cross products
        let mt = self.marker_matrix.transpose();
        let xt = self.x_matrix.transpose();
        let xtx = &xt * &self.x_matrix;
        let xtm = &xt * &self.marker_matrix;
        let mtm = &mt * &self.marker_matrix;
        let y_vec = nalgebra::DVector::from_column_slice(&self.y);
        let xty = &xt * &y_vec;
        let mty = &mt * &y_vec;

        // Initialize variance components
        let y_var = self.y.iter().map(|&v| v * v).sum::<f64>() / n as f64
            - (self.y.iter().sum::<f64>() / n as f64).powi(2);
        let mut sigma2_u = (y_var * 0.5 / m as f64).max(1e-6);
        let mut sigma2_e = (y_var * 0.5).max(1e-6);

        let max_iter = 100;
        let tol = 1e-6;
        let mut converged = false;
        let mut n_iter = 0;

        for iter in 0..max_iter {
            let lambda = sigma2_e / sigma2_u;

            // Solve MME
            let dim = p + m;
            let mut c = DMatrix::zeros(dim, dim);
            let mut rhs = nalgebra::DVector::zeros(dim);

            // X'X block
            for i in 0..p {
                for j in 0..p {
                    c[(i, j)] = xtx[(i, j)];
                }
            }
            // X'M and M'X blocks
            for i in 0..p {
                for j in 0..m {
                    c[(i, p + j)] = xtm[(i, j)];
                    c[(p + j, i)] = xtm[(i, j)];
                }
            }
            // M'M + λI block
            for i in 0..m {
                for j in 0..m {
                    c[(p + i, p + j)] = mtm[(i, j)];
                }
                c[(p + i, p + i)] += lambda;
            }

            // RHS
            for i in 0..p {
                rhs[i] = xty[i];
            }
            for i in 0..m {
                rhs[p + i] = mty[i];
            }

            let chol = c.clone().cholesky().ok_or(LmmError::NotPositiveDefinite)?;
            let sol = chol.solve(&rhs);

            let b_hat: Vec<f64> = sol.as_slice()[..p].to_vec();
            let u_hat: Vec<f64> = sol.as_slice()[p..].to_vec();

            // EM update for σ²_u
            let u_sum_sq: f64 = u_hat.iter().map(|u| u * u).sum();
            // tr((M'M + λI)⁻¹) ≈ using diagonal of C⁻¹
            let c_inv = chol.inverse();
            let trace_cinv_uu: f64 = (0..m).map(|i| c_inv[(p + i, p + i)]).sum();
            let old_sigma2_u = sigma2_u;
            let old_sigma2_e = sigma2_e;
            sigma2_u = ((u_sum_sq + sigma2_e * trace_cinv_uu) / m as f64).max(1e-10);

            // EM update for σ²_e
            let xb = &self.x_matrix * nalgebra::DVector::from_column_slice(&b_hat);
            let mu = &self.marker_matrix * nalgebra::DVector::from_column_slice(&u_hat);
            let resid: f64 = (0..n)
                .map(|i| (self.y[i] - xb[i] - mu[i]).powi(2))
                .sum();
            sigma2_e = (resid / (n - p) as f64).max(1e-10);

            // Convergence check
            let change = ((sigma2_u - old_sigma2_u).powi(2) + (sigma2_e - old_sigma2_e).powi(2)).sqrt()
                / (old_sigma2_u.powi(2) + old_sigma2_e.powi(2)).sqrt().max(1e-10);

            n_iter = iter + 1;
            if iter > 0 && change < tol {
                converged = true;
                break;
            }
        }

        // Final solve
        let lambda = sigma2_e / sigma2_u;
        let dim = p + m;
        let mut c = DMatrix::zeros(dim, dim);
        let mut rhs = nalgebra::DVector::zeros(dim);
        for i in 0..p { for j in 0..p { c[(i, j)] = xtx[(i, j)]; } }
        for i in 0..p { for j in 0..m { c[(i, p+j)] = xtm[(i, j)]; c[(p+j, i)] = xtm[(i, j)]; } }
        for i in 0..m { for j in 0..m { c[(p+i, p+j)] = mtm[(i, j)]; } c[(p+i, p+i)] += lambda; }
        for i in 0..p { rhs[i] = xty[i]; }
        for i in 0..m { rhs[p+i] = mty[i]; }

        let chol = c.clone().cholesky().ok_or(LmmError::NotPositiveDefinite)?;
        let sol = chol.solve(&rhs);
        let fixed_effects: Vec<f64> = sol.as_slice()[..p].to_vec();
        let marker_effects: Vec<f64> = sol.as_slice()[p..].to_vec();

        // Breeding values: g = Mû
        let u_vec = nalgebra::DVector::from_column_slice(&marker_effects);
        let gebv = &self.marker_matrix * &u_vec;
        let breeding_values: Vec<f64> = gebv.as_slice().to_vec();

        // Log-likelihood
        let n_eff = (n - p) as f64;
        let l_diag = chol.l();
        let log_det_c = 2.0 * (0..dim).map(|i| l_diag[(i, i)].ln()).sum::<f64>();
        let ypy: f64 = (0..n).map(|i| {
            let fitted = {
                let mut f = 0.0;
                for j in 0..p { f += self.x_matrix[(i, j)] * fixed_effects[j]; }
                for j in 0..m { f += self.marker_matrix[(i, j)] * marker_effects[j]; }
                f
            };
            (self.y[i] - fitted).powi(2)
        }).sum();
        let log_l = -0.5 * (n_eff * (2.0 * std::f64::consts::PI).ln()
            + n_eff * sigma2_e.ln() + m as f64 * sigma2_u.ln()
            + log_det_c + ypy / sigma2_e);

        let heritability = (sigma2_u * m as f64) / (sigma2_u * m as f64 + sigma2_e);

        let result = RrBlupResult {
            fixed_effects,
            marker_effects,
            breeding_values,
            sigma2_u,
            sigma2_e,
            lambda,
            log_likelihood: log_l,
            heritability,
            n_iterations: n_iter,
            converged,
        };
        self.result = Some(result.clone());
        Ok(result)
    }

    /// Fit with a known λ = σ²_e / σ²_u (skip REML).
    pub fn fit_with_lambda(&mut self, lambda: f64) -> Result<RrBlupResult> {
        let n = self.y.len();
        let p = self.x_matrix.ncols();
        let m = self.marker_matrix.ncols();

        let mt = self.marker_matrix.transpose();
        let xt = self.x_matrix.transpose();
        let dim = p + m;
        let mut c = DMatrix::zeros(dim, dim);
        let mut rhs = nalgebra::DVector::zeros(dim);

        let xtx = &xt * &self.x_matrix;
        let xtm = &xt * &self.marker_matrix;
        let mtm = &mt * &self.marker_matrix;
        let y_vec = nalgebra::DVector::from_column_slice(&self.y);
        let xty = &xt * &y_vec;
        let mty = &mt * &y_vec;

        for i in 0..p { for j in 0..p { c[(i, j)] = xtx[(i, j)]; } }
        for i in 0..p { for j in 0..m { c[(i, p+j)] = xtm[(i, j)]; c[(p+j, i)] = xtm[(i, j)]; } }
        for i in 0..m { for j in 0..m { c[(p+i, p+j)] = mtm[(i, j)]; } c[(p+i, p+i)] += lambda; }
        for i in 0..p { rhs[i] = xty[i]; }
        for i in 0..m { rhs[p+i] = mty[i]; }

        let chol = c.cholesky().ok_or(LmmError::NotPositiveDefinite)?;
        let sol = chol.solve(&rhs);
        let fixed_effects: Vec<f64> = sol.as_slice()[..p].to_vec();
        let marker_effects: Vec<f64> = sol.as_slice()[p..].to_vec();

        let u_vec = nalgebra::DVector::from_column_slice(&marker_effects);
        let gebv = &self.marker_matrix * &u_vec;
        let breeding_values: Vec<f64> = gebv.as_slice().to_vec();

        let resid: f64 = (0..n).map(|i| {
            let mut f = 0.0;
            for j in 0..p { f += self.x_matrix[(i, j)] * fixed_effects[j]; }
            for j in 0..m { f += self.marker_matrix[(i, j)] * marker_effects[j]; }
            (self.y[i] - f).powi(2)
        }).sum();
        let sigma2_e = resid / (n - p) as f64;
        let sigma2_u = sigma2_e / lambda;
        let heritability = (sigma2_u * m as f64) / (sigma2_u * m as f64 + sigma2_e);

        let result = RrBlupResult {
            fixed_effects, marker_effects, breeding_values,
            sigma2_u, sigma2_e, lambda,
            log_likelihood: 0.0,
            heritability,
            n_iterations: 1, converged: true,
        };
        self.result = Some(result.clone());
        Ok(result)
    }

    /// Predict breeding values for new individuals.
    pub fn predict(&self, new_genotypes: &DMatrix<f64>) -> Result<Vec<f64>> {
        let result = self.result.as_ref().ok_or_else(|| LmmError::ModelSpec("Model not fitted".into()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn simple_data() -> (DMatrix<f64>, DMatrix<f64>, Vec<f64>) {
        // 5 individuals, 10 markers
        let geno = DMatrix::from_row_slice(5, 10, &[
            0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0,
        ]);
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
        let m = DMatrix::from_row_slice(4, 3, &[
            -1.0, 0.0, 1.0,
             1.0, -1.0, 0.0,
             0.0, 1.0, -1.0,
            -1.0, 1.0, 0.0,
        ]);
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
}
