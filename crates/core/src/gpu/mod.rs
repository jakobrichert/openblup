//! Optional GPU acceleration for compute-intensive operations.
//!
//! Enable with the `gpu` feature flag:
//! ```toml
//! plant-breeding-lmm-core = { version = "0.1", features = ["gpu"] }
//! ```
//!
//! Provides GPU-accelerated:
//! - G-matrix computation (ZZ'/(2*sum(pq)))
//! - Dense matrix multiplication
//!
//! Falls back to CPU automatically if no GPU is available.

#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
mod kernels;

#[cfg(feature = "gpu")]
pub use context::GpuContext;
#[cfg(feature = "gpu")]
pub use kernels::gpu_compute_g_matrix;

/// Compute G-matrix using GPU if available, CPU otherwise.
///
/// This function works regardless of whether the `gpu` feature is enabled.
/// With the feature enabled, it attempts GPU computation first and falls
/// back to CPU on failure. Without the feature, it always uses CPU.
pub fn compute_g_matrix_auto(
    marker_matrix: &nalgebra::DMatrix<f64>,
    allele_freqs: &[f64],
) -> crate::error::Result<nalgebra::DMatrix<f64>> {
    #[cfg(feature = "gpu")]
    {
        if let Some(ctx) = GpuContext::new_blocking() {
            match gpu_compute_g_matrix(&ctx, marker_matrix, allele_freqs) {
                Ok(g) => return Ok(g),
                Err(_) => {} // Fall through to CPU
            }
        }
    }

    // CPU fallback
    cpu_compute_g_matrix(marker_matrix, allele_freqs)
}

/// CPU implementation of G-matrix (VanRaden Method 1).
fn cpu_compute_g_matrix(
    marker_matrix: &nalgebra::DMatrix<f64>,
    allele_freqs: &[f64],
) -> crate::error::Result<nalgebra::DMatrix<f64>> {
    let n = marker_matrix.nrows();
    let m = marker_matrix.ncols();

    // Center markers: Z = M - 2p
    let mut z = marker_matrix.clone();
    for j in 0..m {
        let two_p = 2.0 * allele_freqs[j];
        for i in 0..n {
            z[(i, j)] -= two_p;
        }
    }

    // Denominator: 2 * sum(p_i * q_i)
    let denom: f64 = allele_freqs
        .iter()
        .map(|&p| 2.0 * p * (1.0 - p))
        .sum();

    if denom < 1e-10 {
        return Err(crate::error::LmmError::InvalidParameter(
            "All markers are monomorphic".into(),
        ));
    }

    // G = ZZ' / denom
    let g = (&z * z.transpose()) / denom;
    Ok(g)
}

#[cfg(feature = "gpu")]
mod context_impl {
    /// GPU compute context (placeholder for wgpu integration).
    pub struct GpuContext {
        pub device_name: String,
    }

    impl GpuContext {
        pub fn new_blocking() -> Option<Self> {
            // Attempt to initialize wgpu
            // For now, return None (no GPU available)
            None
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cpu_g_matrix() {
        let m = nalgebra::DMatrix::from_row_slice(3, 4, &[
            0.0, 1.0, 2.0, 1.0,
            2.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 2.0,
        ]);
        let freqs = vec![0.5, 1.0 / 3.0, 0.5, 2.0 / 3.0];
        let g = cpu_compute_g_matrix(&m, &freqs).unwrap();

        // G should be 3×3, symmetric
        assert_eq!(g.nrows(), 3);
        assert_eq!(g.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(g[(i, j)], g[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_auto_fallback() {
        let m = nalgebra::DMatrix::from_row_slice(2, 3, &[
            0.0, 1.0, 2.0,
            2.0, 1.0, 0.0,
        ]);
        let freqs = vec![0.5, 0.5, 0.5];
        let g = compute_g_matrix_auto(&m, &freqs).unwrap();
        assert_eq!(g.nrows(), 2);
    }
}
