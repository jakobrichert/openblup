//! Optional GPU acceleration for compute-intensive operations.
//!
//! Enable with the `gpu` feature flag:
//! ```toml
//! plant-breeding-lmm-core = { version = "0.1", features = ["gpu"] }
//! ```
//!
//! The public entry point is [`compute_g_matrix_auto`], which always works:
//! with the `gpu` feature enabled it first asks the GPU backend for a
//! device and falls back to the CPU implementation when none is available;
//! without the feature it runs on the CPU directly.
//!
//! # Status
//!
//! The `gpu` feature currently ships the backend *interface* only
//! (`GpuContext` and `gpu_compute_g_matrix`). No compute-shader
//! implementation is bundled yet, so `GpuContext::new_blocking` reports
//! that no device is available and every call takes the CPU path. This keeps
//! the feature compiling and lets downstream code opt in today without a
//! behavioural change once a `wgpu` backend lands.

use nalgebra::DMatrix;

use crate::error::{LmmError, Result};

#[cfg(feature = "gpu")]
pub use backend::{gpu_compute_g_matrix, GpuContext};

/// Whether this build of the library was compiled with the `gpu` feature.
pub const GPU_FEATURE_ENABLED: bool = cfg!(feature = "gpu");

/// Whether a GPU device is actually usable at runtime.
///
/// Always `false` when the `gpu` feature is disabled, and currently also
/// `false` with it enabled (see the module documentation).
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuContext::new_blocking().is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Compute G-matrix using GPU if available, CPU otherwise.
///
/// This function works regardless of whether the `gpu` feature is enabled.
/// With the feature enabled, it attempts GPU computation first and falls
/// back to CPU on failure. Without the feature, it always uses CPU.
pub fn compute_g_matrix_auto(
    marker_matrix: &DMatrix<f64>,
    allele_freqs: &[f64],
) -> Result<DMatrix<f64>> {
    #[cfg(feature = "gpu")]
    {
        if let Some(ctx) = GpuContext::new_blocking() {
            if let Ok(g) = gpu_compute_g_matrix(&ctx, marker_matrix, allele_freqs) {
                return Ok(g);
            }
            // Fall through to CPU on any GPU error.
        }
    }

    // CPU fallback
    cpu_compute_g_matrix(marker_matrix, allele_freqs)
}

/// CPU implementation of G-matrix (VanRaden Method 1).
fn cpu_compute_g_matrix(
    marker_matrix: &DMatrix<f64>,
    allele_freqs: &[f64],
) -> Result<DMatrix<f64>> {
    let n = marker_matrix.nrows();
    let m = marker_matrix.ncols();

    if allele_freqs.len() != m {
        return Err(LmmError::DimensionMismatch {
            expected: m,
            got: allele_freqs.len(),
            context: "allele frequencies must have one entry per marker".into(),
        });
    }

    // Center markers: Z = M - 2p
    let mut z = marker_matrix.clone();
    for (j, &p) in allele_freqs.iter().enumerate() {
        let two_p = 2.0 * p;
        for i in 0..n {
            z[(i, j)] -= two_p;
        }
    }

    // Denominator: 2 * sum(p_i * q_i)
    let denom: f64 = allele_freqs.iter().map(|&p| 2.0 * p * (1.0 - p)).sum();

    if denom < 1e-10 {
        return Err(LmmError::InvalidParameter(
            "All markers are monomorphic".into(),
        ));
    }

    // G = ZZ' / denom
    let g = (&z * z.transpose()) / denom;
    Ok(g)
}

/// GPU backend interface (compiled only with the `gpu` feature).
#[cfg(feature = "gpu")]
mod backend {
    use nalgebra::DMatrix;

    use crate::error::{LmmError, Result};

    /// Handle to a GPU compute device.
    ///
    /// This is the seam where a `wgpu` device/queue pair will live. Until a
    /// backend is implemented, [`GpuContext::new_blocking`] returns `None`.
    #[derive(Debug)]
    pub struct GpuContext {
        device_name: String,
    }

    impl GpuContext {
        /// Try to acquire a GPU device, blocking until initialisation completes.
        ///
        /// Returns `None` when no usable device exists (currently always).
        pub fn new_blocking() -> Option<Self> {
            None
        }

        /// Human-readable name of the device backing this context.
        pub fn device_name(&self) -> &str {
            &self.device_name
        }
    }

    /// Compute the VanRaden G-matrix on the GPU.
    ///
    /// Returns an error when the backend cannot run the computation, in
    /// which case [`super::compute_g_matrix_auto`] falls back to the CPU.
    pub fn gpu_compute_g_matrix(
        ctx: &GpuContext,
        _marker_matrix: &DMatrix<f64>,
        _allele_freqs: &[f64],
    ) -> Result<DMatrix<f64>> {
        Err(LmmError::InvalidParameter(format!(
            "GPU backend on device '{}' is not implemented yet",
            ctx.device_name()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cpu_g_matrix() {
        let m = DMatrix::from_row_slice(
            3,
            4,
            &[0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        );
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
    fn test_cpu_g_matrix_matches_genetics_module() {
        // The GPU module's CPU fallback must agree with the reference
        // implementation in `genetics::gmatrix`.
        let m = DMatrix::from_row_slice(
            4,
            5,
            &[
                0.0, 1.0, 2.0, 1.0, 0.0, //
                2.0, 1.0, 0.0, 1.0, 2.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, //
                0.0, 2.0, 1.0, 0.0, 1.0,
            ],
        );
        let freqs: Vec<f64> = (0..5).map(|j| m.column(j).sum() / 8.0).collect();
        let g_gpu_path = compute_g_matrix_auto(&m, &freqs).unwrap();
        let g_ref = crate::genetics::compute_g_matrix(&m, Some(&freqs)).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(g_gpu_path[(i, j)], g_ref[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_auto_fallback() {
        let m = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 2.0, 1.0, 0.0]);
        let freqs = vec![0.5, 0.5, 0.5];
        let g = compute_g_matrix_auto(&m, &freqs).unwrap();
        assert_eq!(g.nrows(), 2);
    }

    #[test]
    fn test_wrong_freq_length_errors() {
        let m = DMatrix::from_row_slice(2, 3, &[0.0, 1.0, 2.0, 2.0, 1.0, 0.0]);
        assert!(cpu_compute_g_matrix(&m, &[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_gpu_availability_is_consistent_with_feature() {
        if !GPU_FEATURE_ENABLED {
            assert!(!gpu_available());
        }
    }
}
