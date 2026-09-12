use crate::error::Result;
use crate::types::SparseMat;

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

    /// Human-readable parameter names, in the order of [`params`](Self::params).
    fn param_names(&self) -> Vec<String> {
        (0..self.n_params())
            .map(|i| format!("theta{}", i + 1))
            .collect()
    }

    /// Whether parameter `i` is a variance (scale) parameter, i.e. one whose
    /// natural starting value scales with the variance of the data. The
    /// REML engines use this to pick sensible starting values.
    fn is_variance_param(&self, i: usize) -> bool {
        let b = self.bounds();
        i < b.len() && b[i].0 >= 0.0 && b[i].1.is_infinite()
    }

    /// The dimension this structure is defined for, if it is intrinsic to the
    /// structure (e.g. a `Diagonal` with 4 variances). `None` means the
    /// structure works for any dimension (Identity, AR1).
    fn fixed_dim(&self) -> Option<usize> {
        None
    }

    /// `dSigma/dtheta_i` for every parameter, derived from the inverse
    /// derivatives as `-Sigma (dSigma^{-1}/dtheta_i) Sigma`.
    ///
    /// Only structured residuals need this; separable structures override
    /// it so that the product factorises.
    fn derivatives_of_covariance(&self, dim: usize) -> Vec<SparseMat> {
        let sigma = self.covariance_matrix(dim);
        self.derivatives_of_inverse(dim)
            .iter()
            .map(|d| {
                let sd: SparseMat = &sigma * d;
                let sds: SparseMat = &sd * &sigma;
                sds.map(|v| -v)
            })
            .collect()
    }

    /// `Sigma * dSigma^{-1}/dtheta_i` for every parameter (one matrix each).
    ///
    /// The REML score and average-information terms only need this product
    /// and `dSigma^{-1}/dtheta_i` itself, never `dSigma/dtheta_i`. The
    /// default forms the sparse product; separable (Kronecker) structures
    /// override it so that the product factorises and never touches the
    /// full covariance.
    fn sigma_times_derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        let sigma = self.covariance_matrix(dim);
        self.derivatives_of_inverse(dim)
            .iter()
            .map(|d| &sigma * d)
            .collect()
    }
}

impl Clone for Box<dyn VarStruct> {
    fn clone(&self) -> Box<dyn VarStruct> {
        self.clone_boxed()
    }
}
