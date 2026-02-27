/// The result of fitting a mixed model via REML.
#[derive(Debug, Clone)]
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
    /// (from inverse AI matrix at convergence).
    pub variance_se: Vec<f64>,
    /// Residuals: y - Xb - Zu.
    pub residuals: Vec<f64>,
    /// Model dimensions.
    pub n_obs: usize,
    pub n_fixed_params: usize,
    pub n_variance_params: usize,
}

/// A single variance component estimate.
#[derive(Debug, Clone)]
pub struct VarianceEstimate {
    pub name: String,
    pub structure: String,
    pub parameters: Vec<(String, f64)>,
}

/// A named fixed or random effect estimate.
#[derive(Debug, Clone)]
pub struct NamedEffect {
    pub term: String,
    pub level: String,
    pub estimate: f64,
    pub se: f64,
}

/// A block of random effects for a single random term.
#[derive(Debug, Clone)]
pub struct RandomEffectBlock {
    pub term: String,
    pub effects: Vec<NamedEffect>,
}

/// Information about a single REML iteration.
#[derive(Debug, Clone)]
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

    /// Print a formatted summary of the model fit.
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("=== Mixed Model Fit (AI-REML) ===\n\n");
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
            for (pname, pval) in &vc.parameters {
                s.push_str(&format!("{}={:.6}  ", pname, pval));
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
            s.push_str(&format!("\n--- Random Effects: {} (BLUP) ---\n", block.term));
            let mut sorted = block.effects.clone();
            sorted.sort_by(|a, b| b.estimate.partial_cmp(&a.estimate).unwrap_or(std::cmp::Ordering::Equal));
            let show = sorted.len().min(10);
            for ef in sorted.iter().take(show) {
                s.push_str(&format!("  {}: {:.6}\n", ef.level, ef.estimate));
            }
            if sorted.len() > 10 {
                s.push_str(&format!("  ... and {} more\n", sorted.len() - 10));
            }
        }

        s
    }
}
