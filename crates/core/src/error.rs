use thiserror::Error;

#[derive(Error, Debug)]
pub enum LmmError {
    #[error("Data error: {0}")]
    Data(String),

    #[error("Column '{0}' not found in DataFrame")]
    ColumnNotFound(String),

    #[error("Matrix is not positive definite")]
    NotPositiveDefinite,

    #[error("Cholesky factorization failed: {0}")]
    CholeskyFailed(String),

    #[error("Singular matrix encountered in {context}")]
    SingularMatrix { context: String },

    #[error("REML did not converge after {iterations} iterations (change = {change:.2e})")]
    NotConverged { iterations: usize, change: f64 },

    #[error("Dimension mismatch: expected {expected}, got {got} in {context}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    #[error("Pedigree error: {0}")]
    Pedigree(String),

    #[error("Invalid variance parameter: {0}")]
    InvalidParameter(String),

    #[error("Model specification error: {0}")]
    ModelSpec(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
}

pub type Result<T> = std::result::Result<T, LmmError>;
