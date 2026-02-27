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
    pub fn record(
        &mut self,
        iteration: usize,
        log_likelihood: f64,
        param_change: f64,
    ) {
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
}
