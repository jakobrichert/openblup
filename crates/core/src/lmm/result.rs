use nalgebra::DMatrix;
use serde::Serialize;

use crate::diagnostics::{
    compute_diagnostics, wald_tests_satterthwaite, ResidualDiagnostics, WaldTest,
};
use crate::model::MixedModel;

/// The result of fitting a mixed model via REML.
#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    /// Estimated variance components (final values).
    pub variance_components: Vec<VarianceEstimate>,
    /// Fixed effects (BLUE).
    pub fixed_effects: Vec<NamedEffect>,
    /// Random effects (BLUP), organized by random term.
    pub random_effects: Vec<RandomEffectBlock>,
    /// Restricted log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of REML iterations performed.
    pub n_iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Iteration history.
    pub history: Vec<RemlIteration>,
    /// Approximate standard errors of variance components
    /// (from inverse AI matrix at convergence; zeros when not available,
    /// e.g. after EM-REML).
    pub variance_se: Vec<f64>,
    /// Residuals: y - Xb - Zu.
    pub residuals: Vec<f64>,
    /// Variance-covariance matrix of the fixed effects, i.e. the
    /// fixed-effects block of C⁻¹ (p x p, stored row by row). Needed for
    /// Wald tests of multi-level terms and for contrasts between levels.
    pub fixed_cov: Vec<Vec<f64>>,
    /// Whether each variance parameter (same order as `variance_components`)
    /// ended at the boundary of the parameter space (effectively zero).
    /// Such parameters are reported without a standard error.
    pub at_boundary: Vec<bool>,
    /// Model dimensions.
    pub n_obs: usize,
    pub n_fixed_params: usize,
    pub n_variance_params: usize,
    /// Full inverse of the MME coefficient matrix at convergence
    /// (`(p + q) x (p + q)`), used for Satterthwaite degrees of freedom and
    /// residual diagnostics.
    #[serde(skip)]
    pub c_inv: Option<DMatrix<f64>>,
    /// Average information matrix of the variance parameters at convergence
    /// (`None` after EM-REML).
    #[serde(skip)]
    pub ai_matrix: Option<DMatrix<f64>>,
    /// Number of levels of each random term (block sizes of the random part
    /// of the MME).
    pub n_random_per_term: Vec<usize>,
}

/// A single variance component estimate.
#[derive(Debug, Clone, Serialize)]
pub struct VarianceEstimate {
    /// Term name (random term column, interaction `a:b`, or `residual`).
    pub name: String,
    /// Variance structure name (Identity, AR1, FactorAnalytic, ...).
    pub structure: String,
    /// Parameter names and estimates in the structure's parameter order.
    pub parameters: Vec<(String, f64)>,
    /// Approximate standard error per parameter (0 when unavailable).
    pub se: Vec<f64>,
    /// Whether each parameter converged to the boundary of its parameter space.
    pub at_boundary: Vec<bool>,
}

impl VarianceEstimate {
    /// The first parameter (sigma² for scaled structures), if any.
    pub fn sigma2(&self) -> Option<f64> {
        self.parameters.first().map(|(_, v)| *v)
    }
}

/// A named fixed or random effect estimate.
#[derive(Debug, Clone, Serialize)]
pub struct NamedEffect {
    pub term: String,
    pub level: String,
    pub estimate: f64,
    pub se: f64,
}

/// A block of random effects for a single random term.
#[derive(Debug, Clone, Serialize)]
pub struct RandomEffectBlock {
    pub term: String,
    pub effects: Vec<NamedEffect>,
}

/// Information about a single REML iteration.
#[derive(Debug, Clone, Serialize)]
pub struct RemlIteration {
    pub iteration: usize,
    pub log_likelihood: f64,
    pub variance_params: Vec<f64>,
    pub change: f64,
}

impl FitResult {
    /// AIC = -2 * logL + 2 * p (where p = number of variance parameters).
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_variance_params as f64
    }

    /// BIC = -2 * logL + p * ln(n - rank(X)).
    pub fn bic(&self) -> f64 {
        let n_eff = (self.n_obs - self.n_fixed_params) as f64;
        -2.0 * self.log_likelihood + self.n_variance_params as f64 * n_eff.ln()
    }

    /// Look up the first parameter (sigma²) of a variance component by name
    /// (e.g. `"genotype"` or `"residual"`).
    pub fn variance_component(&self, name: &str) -> Option<f64> {
        self.variance_components
            .iter()
            .find(|vc| vc.name == name)
            .and_then(|vc| vc.parameters.first().map(|(_, v)| *v))
    }

    /// Covariance between fixed effects `i` and `j` (indices into
    /// `fixed_effects`). Returns `None` if the covariance matrix is not
    /// available.
    pub fn fixed_effect_cov(&self, i: usize, j: usize) -> Option<f64> {
        self.fixed_cov.get(i).and_then(|row| row.get(j)).copied()
    }

    /// Whether every variance component is a single scaled variance
    /// (Identity / relationship-matrix terms and an IID residual).
    pub fn is_scaled_identity_model(&self) -> bool {
        self.variance_components
            .iter()
            .all(|vc| vc.parameters.len() == 1 && vc.structure == "Identity")
    }

    /// Wald F-tests with Satterthwaite denominator degrees of freedom.
    ///
    /// Available for scaled-identity models fitted by AI-REML (needs the
    /// average information matrix and `C⁻¹`); returns `None` otherwise, in
    /// which case [`wald_tests`](crate::diagnostics::wald_tests) with
    /// containment df applies.
    pub fn wald_tests_satterthwaite(&self) -> Option<Vec<WaldTest>> {
        if !self.is_scaled_identity_model() {
            return None;
        }
        let c_inv = self.c_inv.as_ref()?;
        let ai = self.ai_matrix.as_ref()?;
        wald_tests_satterthwaite(self, c_inv, ai, &self.n_random_per_term)
    }

    /// Residual diagnostics (conditional/marginal residuals, leverage,
    /// Cook's distance) for the model this result was fitted from. Requires
    /// an IID residual and `C⁻¹`.
    pub fn residual_diagnostics(&self, model: &MixedModel) -> Option<ResidualDiagnostics> {
        let c_inv = self.c_inv.as_ref()?;
        let residual = self.variance_components.last()?;
        if residual.structure != "Identity" || model.residual_grid.is_some() {
            return None;
        }
        let sigma2_e = residual.sigma2()?;
        let fixed: Vec<f64> = self.fixed_effects.iter().map(|e| e.estimate).collect();
        let random: Vec<f64> = self
            .random_effects
            .iter()
            .flat_map(|b| b.effects.iter().map(|e| e.estimate))
            .collect();
        if fixed.len() != model.x.cols() || random.len() != model.z_combined.cols() {
            return None;
        }
        Some(compute_diagnostics(
            &model.y,
            &model.x,
            &model.z_combined,
            &fixed,
            &random,
            c_inv,
            sigma2_e,
        ))
    }

    /// Print a formatted summary of the model fit.
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("=== Mixed Model Fit (REML) ===\n\n");
        s.push_str(&format!(
            "Observations: {}   Fixed params: {}   Variance params: {}\n",
            self.n_obs, self.n_fixed_params, self.n_variance_params
        ));
        s.push_str(&format!(
            "Converged: {}   Iterations: {}\n\n",
            self.converged, self.n_iterations
        ));

        s.push_str(&format!("Log-likelihood: {:.4}\n", self.log_likelihood));
        s.push_str(&format!("AIC: {:.4}\n", self.aic()));
        s.push_str(&format!("BIC: {:.4}\n\n", self.bic()));

        s.push_str("--- Variance Components ---\n");
        for vc in &self.variance_components {
            s.push_str(&format!("  {} ({}): ", vc.name, vc.structure));
            for (i, (pname, pval)) in vc.parameters.iter().enumerate() {
                s.push_str(&format!("{}={:.6}", pname, pval));
                if vc.at_boundary.get(i).copied().unwrap_or(false) {
                    s.push_str(" [boundary]");
                } else if let Some(se) = vc.se.get(i).filter(|se| **se > 0.0) {
                    s.push_str(&format!(" (SE: {:.6})", se));
                }
                s.push_str("  ");
            }
            s.push('\n');
        }

        s.push_str("\n--- Fixed Effects (BLUE) ---\n");
        for ef in &self.fixed_effects {
            s.push_str(&format!(
                "  {}.{}: {:.6} (SE: {:.6})\n",
                ef.term, ef.level, ef.estimate, ef.se
            ));
        }

        for block in &self.random_effects {
            s.push_str(&format!(
                "\n--- Random Effects: {} (BLUP) ---\n",
                block.term
            ));
            let mut sorted = block.effects.clone();
            sorted.sort_by(|a, b| {
                b.estimate
                    .partial_cmp(&a.estimate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let show = sorted.len().min(10);
            for ef in sorted.iter().take(show) {
                s.push_str(&format!(
                    "  {}: {:.6} (SE: {:.6})\n",
                    ef.level, ef.estimate, ef.se
                ));
            }
            if sorted.len() > 10 {
                s.push_str(&format!("  ... and {} more\n", sorted.len() - 10));
            }
        }

        s
    }
}
