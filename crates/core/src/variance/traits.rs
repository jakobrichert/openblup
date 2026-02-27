use crate::types::SparseMat;
use crate::error::Result;

/// Core trait that every variance structure must implement.
///
/// A variance structure defines the covariance matrix (and its inverse,
/// derivatives, and log-determinant) for a set of random effects or residuals.
///
/// For a random term with `q` levels, the covariance is `sigma^2 * Sigma(rho, ...)`
/// where `Sigma` is the correlation structure defined by this trait.
pub trait VarStruct: Send + Sync + std::fmt::Debug {
    /// Human-readable name: "Identity", "AR1", "AR1xAR1", "Unstructured", etc.
    fn name(&self) -> &str;

    /// Number of variance parameters this structure has.
    fn n_params(&self) -> usize;

    /// Get current parameter values.
    fn params(&self) -> Vec<f64>;

    /// Set parameter values (called during REML iteration).
    fn set_params(&mut self, params: &[f64]) -> Result<()>;

    /// Construct the covariance matrix for the given dimension.
    fn covariance_matrix(&self, dim: usize) -> SparseMat;

    /// Construct the inverse covariance matrix.
    /// Many structures have direct closed-form inverses that are cheaper
    /// than inverting the covariance matrix.
    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat;

    /// Compute log|Sigma| (the log-determinant).
    fn log_determinant(&self, dim: usize) -> f64;

    /// Compute dSigma^{-1}/d(theta_k) for each parameter.
    /// Returns one sparse matrix per parameter.
    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat>;

    /// Parameter bounds: (lower, upper) for each parameter.
    fn bounds(&self) -> Vec<(f64, f64)>;

    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn VarStruct>;

    /// Initial parameter values (reasonable defaults for starting REML).
    fn initial_params(&self) -> Vec<f64> {
        self.params()
    }
}

impl Clone for Box<dyn VarStruct> {
    fn clone(&self) -> Box<dyn VarStruct> {
        self.clone_boxed()
    }
}
