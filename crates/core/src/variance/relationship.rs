use std::sync::OnceLock;

use nalgebra::DMatrix;
use sprs::TriMat;

use crate::error::{LmmError, Result};
use crate::matrix::sparse::sparse_diagonal;
use crate::matrix::sparse_cholesky::SparseCholeskySolver;
use crate::types::SparseMat;

use super::traits::VarStruct;

/// A scaled relationship structure `G = sigma^2 * K`, with `K` given through
/// its inverse (a pedigree A⁻¹ or a genomic G⁻¹).
///
/// The sparse AI-REML path handles such terms as an [`Identity`](super::Identity)
/// variance plus the relationship inverse stored on the model; this structure
/// carries both, so the general engine (used when another term or the residual
/// is structured) fits the same model. It reports itself as `"Identity"` with
/// one `sigma2` parameter, exactly like the sparse path, and its
/// log-determinant includes `log|K|` (computed once with a sparse Cholesky
/// factorisation), as every engine's REML log-likelihood does.
#[derive(Debug, Clone)]
pub struct Relationship {
    sigma2: f64,
    inverse: SparseMat,
    /// log|K| = -log|K⁻¹|.
    log_det_k: f64,
    /// `K` itself, formed on first use (only needed for `covariance_matrix`).
    dense: OnceLock<SparseMat>,
}

impl Relationship {
    /// Create from a starting variance and the relationship inverse `K⁻¹`.
    pub fn new(sigma2: f64, inverse: SparseMat) -> Result<Self> {
        if inverse.rows() != inverse.cols() {
            return Err(LmmError::DimensionMismatch {
                expected: inverse.rows(),
                got: inverse.cols(),
                context: "relationship inverse must be square".into(),
            });
        }
        let log_det_k = -SparseCholeskySolver::new(&inverse)?.log_determinant();
        Ok(Self {
            sigma2,
            inverse,
            log_det_k,
            dense: OnceLock::new(),
        })
    }

    fn check_dim(&self, dim: usize) {
        assert_eq!(
            dim,
            self.inverse.rows(),
            "relationship structure is defined for {} levels",
            self.inverse.rows()
        );
    }

    fn relationship_matrix(&self) -> &SparseMat {
        self.dense.get_or_init(|| {
            let n = self.inverse.rows();
            let mut k_inv = DMatrix::<f64>::zeros(n, n);
            for (v, (i, j)) in self.inverse.iter() {
                k_inv[(i, j)] += *v;
            }
            let k = k_inv
                .cholesky()
                .map(|c| c.inverse())
                .unwrap_or_else(|| DMatrix::zeros(n, n));
            let mut tri = TriMat::new((n, n));
            for j in 0..n {
                for i in 0..n {
                    if k[(i, j)] != 0.0 {
                        tri.add_triplet(i, j, k[(i, j)]);
                    }
                }
            }
            tri.to_csc()
        })
    }
}

impl VarStruct for Relationship {
    fn name(&self) -> &str {
        "Identity"
    }

    fn n_params(&self) -> usize {
        1
    }

    fn params(&self) -> Vec<f64> {
        vec![self.sigma2]
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 1 {
            return Err(LmmError::InvalidParameter(format!(
                "Relationship expects 1 parameter, got {}",
                params.len()
            )));
        }
        self.sigma2 = params[0];
        Ok(())
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        self.check_dim(dim);
        let s = self.sigma2;
        self.relationship_matrix().map(|v| v * s)
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        self.check_dim(dim);
        let s = 1.0 / self.sigma2;
        self.inverse.map(|v| v * s)
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        dim as f64 * self.sigma2.ln() + self.log_det_k
    }

    fn relationship_log_det(&self) -> f64 {
        self.log_det_k
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        self.check_dim(dim);
        // d(K⁻¹ / sigma^2)/d(sigma^2) = -K⁻¹ / sigma^4
        let s = -1.0 / (self.sigma2 * self.sigma2);
        vec![self.inverse.map(|v| v * s)]
    }

    fn sigma_times_derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        // sigma^2 K * (-K⁻¹ / sigma^4) = -I / sigma^2, without forming K.
        vec![sparse_diagonal(&vec![-1.0 / self.sigma2; dim])]
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(1e-10, f64::INFINITY)]
    }

    fn param_names(&self) -> Vec<String> {
        vec!["sigma2".to_string()]
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }

    fn fixed_dim(&self) -> Option<usize> {
        Some(self.inverse.rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetics::{compute_a_inverse, Pedigree};
    use approx::assert_relative_eq;

    fn mrode_a_inverse() -> SparseMat {
        let triples: Vec<(String, Option<String>, Option<String>)> = [
            ("1", None, None),
            ("2", None, None),
            ("3", None, None),
            ("4", Some("1"), None),
            ("5", Some("3"), Some("2")),
            ("6", Some("1"), Some("2")),
            ("7", Some("4"), Some("5")),
            ("8", Some("3"), Some("6")),
        ]
        .iter()
        .map(|(a, s, d)| (a.to_string(), s.map(String::from), d.map(String::from)))
        .collect();
        let mut ped = Pedigree::from_triples(&triples).unwrap();
        ped.sort_pedigree().unwrap();
        compute_a_inverse(&ped).unwrap()
    }

    #[test]
    fn covariance_times_inverse_is_identity() {
        let r = Relationship::new(2.5, mrode_a_inverse()).unwrap();
        let g = r.covariance_matrix(8);
        let g_inv = r.inverse_covariance_matrix(8);
        let prod: SparseMat = &g * &g_inv;
        for i in 0..8 {
            for j in 0..8 {
                let v = prod.get(i, j).copied().unwrap_or(0.0);
                assert_relative_eq!(v, if i == j { 1.0 } else { 0.0 }, epsilon = 1e-10);
            }
        }
        // A[4,4] (animal 5, parents 3 and 2 unrelated) is 1.
        assert_relative_eq!(*g.get(4, 4).unwrap(), 2.5, epsilon = 1e-10);
    }

    #[test]
    fn log_determinant_includes_log_det_k() {
        let r = Relationship::new(2.0, mrode_a_inverse()).unwrap();
        let g = r.covariance_matrix(8);
        let mut dense = DMatrix::<f64>::zeros(8, 8);
        for (v, (i, j)) in g.iter() {
            dense[(i, j)] = *v;
        }
        let chol = dense.cholesky().unwrap();
        let expected = 2.0 * (0..8).map(|i| chol.l()[(i, i)].ln()).sum::<f64>();
        assert_relative_eq!(r.log_determinant(8), expected, epsilon = 1e-10);
        assert_relative_eq!(
            r.relationship_log_det(),
            expected - 8.0 * 2.0f64.ln(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn inverse_derivative_matches_finite_difference() {
        let a_inv = mrode_a_inverse();
        let r = Relationship::new(3.0, a_inv.clone()).unwrap();
        let d = &r.derivatives_of_inverse(8)[0];
        let h = 1e-6;
        let plus = Relationship::new(3.0 + h, a_inv.clone()).unwrap();
        let minus = Relationship::new(3.0 - h, a_inv).unwrap();
        let (p, m) = (
            plus.inverse_covariance_matrix(8),
            minus.inverse_covariance_matrix(8),
        );
        for (v, (i, j)) in d.iter() {
            let fd = (p.get(i, j).unwrap() - m.get(i, j).unwrap()) / (2.0 * h);
            assert_relative_eq!(*v, fd, epsilon = 1e-6);
        }
        // sigma_times_derivatives_of_inverse = G * dG⁻¹ = -I / sigma^2
        let gd = &r.sigma_times_derivatives_of_inverse(8)[0];
        let full: SparseMat = &r.covariance_matrix(8) * d;
        for i in 0..8 {
            for j in 0..8 {
                let a = gd.get(i, j).copied().unwrap_or(0.0);
                let b = full.get(i, j).copied().unwrap_or(0.0);
                assert_relative_eq!(a, b, epsilon = 1e-10);
            }
        }
    }
}
