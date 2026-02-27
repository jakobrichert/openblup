/// Model fit information criteria.
#[derive(Debug, Clone)]
pub struct ModelFit {
    pub log_likelihood: f64,
    pub n_obs: usize,
    pub n_fixed: usize,
    pub n_variance_params: usize,
}

impl ModelFit {
    /// AIC = -2 * logL + 2 * p (where p = number of variance parameters).
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_variance_params as f64
    }

    /// BIC = -2 * logL + p * ln(n - rank(X)).
    pub fn bic(&self) -> f64 {
        let n_eff = (self.n_obs - self.n_fixed) as f64;
        -2.0 * self.log_likelihood + self.n_variance_params as f64 * n_eff.ln()
    }

    /// Corrected AIC (AICc) for small sample sizes.
    pub fn aicc(&self) -> f64 {
        let p = self.n_variance_params as f64;
        let n_eff = (self.n_obs - self.n_fixed) as f64;
        self.aic() + 2.0 * p * (p + 1.0) / (n_eff - p - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_aic() {
        let fit = ModelFit {
            log_likelihood: -100.0,
            n_obs: 50,
            n_fixed: 3,
            n_variance_params: 2,
        };
        assert_relative_eq!(fit.aic(), 204.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bic() {
        let fit = ModelFit {
            log_likelihood: -100.0,
            n_obs: 50,
            n_fixed: 3,
            n_variance_params: 2,
        };
        // BIC = 200 + 2 * ln(47)
        let expected = 200.0 + 2.0 * 47.0_f64.ln();
        assert_relative_eq!(fit.bic(), expected, epsilon = 1e-10);
    }
}
