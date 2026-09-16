/// Monitors convergence of the REML algorithm.
#[derive(Debug)]
pub struct ConvergenceMonitor {
    tol: f64,
    max_iter: usize,
    history: Vec<ConvergenceRecord>,
}

#[derive(Debug, Clone)]
struct ConvergenceRecord {
    iteration: usize,
    log_likelihood: f64,
    param_change: f64,
    logl_change: f64,
}

impl ConvergenceMonitor {
    pub fn new(tol: f64, max_iter: usize) -> Self {
        Self {
            tol,
            max_iter,
            history: Vec::new(),
        }
    }

    /// Record a new iteration.
    pub fn record(&mut self, iteration: usize, log_likelihood: f64, param_change: f64) {
        let logl_change = if let Some(prev) = self.history.last() {
            (log_likelihood - prev.log_likelihood).abs() / (1.0 + log_likelihood.abs())
        } else {
            f64::INFINITY
        };

        self.history.push(ConvergenceRecord {
            iteration,
            log_likelihood,
            param_change,
            logl_change,
        });
    }

    /// Check if convergence criterion is met.
    pub fn is_converged(&self) -> bool {
        if let Some(last) = self.history.last() {
            last.param_change < self.tol && last.logl_change < self.tol
        } else {
            false
        }
    }

    /// Check if maximum iterations reached.
    pub fn max_reached(&self) -> bool {
        self.history.len() >= self.max_iter
    }

    /// Get the last log-likelihood value.
    pub fn last_logl(&self) -> Option<f64> {
        self.history.last().map(|r| r.log_likelihood)
    }

    /// Number of iterations recorded.
    pub fn n_iterations(&self) -> usize {
        self.history.len()
    }

    /// The iteration number of the most recent record, if any.
    pub fn last_iteration(&self) -> Option<usize> {
        self.history.last().map(|r| r.iteration)
    }

    /// Log-likelihood trace as `(iteration, log_likelihood)` pairs.
    pub fn log_likelihood_trace(&self) -> Vec<(usize, f64)> {
        self.history
            .iter()
            .map(|r| (r.iteration, r.log_likelihood))
            .collect()
    }

    /// Convergence tolerance.
    pub fn tolerance(&self) -> f64 {
        self.tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_converges_when_both_criteria_met() {
        let mut m = ConvergenceMonitor::new(1e-4, 3);
        assert!(!m.is_converged());
        m.record(1, -100.0, 1.0);
        assert!(!m.is_converged()); // first record: logl change is infinite
        m.record(2, -100.000001, 1e-6);
        assert!(m.is_converged());
        assert_eq!(m.n_iterations(), 2);
        assert_eq!(m.last_iteration(), Some(2));
        assert_eq!(m.last_logl(), Some(-100.000001));
        assert!(!m.max_reached());
        m.record(3, -100.000001, 1e-7);
        assert!(m.max_reached());
        assert_eq!(m.log_likelihood_trace().len(), 3);
        assert_eq!(m.tolerance(), 1e-4);
    }
}
