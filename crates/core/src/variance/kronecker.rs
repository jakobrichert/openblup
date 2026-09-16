use crate::error::{LmmError, Result};
use crate::types::SparseMat;
use sprs::TriMat;

use super::traits::VarStruct;

/// Compute the Kronecker product A (x) B of two sparse matrices.
///
/// The Kronecker product of an (m x n) matrix A and a (p x q) matrix B
/// is the (mp x nq) block matrix where each block (i,j) is `A[i,j] * B`.
///
/// This is the fundamental operation for separable spatial models:
/// if rows have covariance Sigma_r and columns have covariance Sigma_c,
/// then the full field covariance is Sigma_r (x) Sigma_c.
///
/// Properties used in mixed models:
/// - (A (x) B)^{-1} = A^{-1} (x) B^{-1}
/// - |A (x) B| = |A|^q * |B|^n (for A: n x n, B: q x q)
/// - d(A (x) B)/dtheta = (dA/dtheta) (x) B (if B is constant w.r.t. theta)
pub fn kronecker_product(a: &SparseMat, b: &SparseMat) -> SparseMat {
    let (ra, ca) = (a.rows(), a.cols());
    let (rb, cb) = (b.rows(), b.cols());
    let mut tri = TriMat::new((ra * rb, ca * cb));

    for (&va, (ia, ja)) in a.iter() {
        for (&vb, (ib, jb)) in b.iter() {
            tri.add_triplet(ia * rb + ib, ja * cb + jb, va * vb);
        }
    }

    tri.to_csc()
}

/// Sparse identity matrix of dimension `n`.
fn sparse_eye(n: usize) -> SparseMat {
    let mut tri = TriMat::new((n, n));
    for i in 0..n {
        tri.add_triplet(i, i, 1.0);
    }
    tri.to_csc()
}

/// Separable (Kronecker) variance structure `Sigma = A ⊗ B` for an
/// interaction term or a gridded residual.
///
/// The levels of the term are ordered `a`-major: level `(i, j)` has index
/// `i * dim_b + j`, matching the row/column order of the Kronecker product.
/// Typical uses:
///
/// - `FA(env) ⊗ Known(A)` — factor analytic genotype-by-environment effects
///   with a pedigree relationship matrix;
/// - `Diagonal(env) ⊗ Known(I)` — heterogeneous genetic variance per environment;
/// - `AR1(row) ⊗ AR1::correlation(col)` — separable spatial residual.
///
/// Parameters are the concatenation of `a`'s and `b`'s parameters. Scale
/// identifiability is the caller's responsibility: at most one of the two
/// factors should carry a free variance (use [`AR1::correlation`](super::AR1::correlation)
/// or [`Known`](super::Known) for the other).
#[derive(Debug, Clone)]
pub struct KroneckerStruct {
    a: Box<dyn VarStruct>,
    dim_a: usize,
    b: Box<dyn VarStruct>,
    dim_b: usize,
    label_a: String,
    label_b: String,
    name: String,
}

impl KroneckerStruct {
    /// Combine two structures. `dim_a` and `dim_b` are the numbers of levels
    /// of the two factors; `label_a` / `label_b` name them in parameter names.
    pub fn new(
        a: Box<dyn VarStruct>,
        dim_a: usize,
        b: Box<dyn VarStruct>,
        dim_b: usize,
        label_a: &str,
        label_b: &str,
    ) -> Result<Self> {
        for (vs, dim, label) in [(&a, dim_a, label_a), (&b, dim_b, label_b)] {
            if let Some(d) = vs.fixed_dim() {
                if d != dim {
                    return Err(LmmError::DimensionMismatch {
                        expected: dim,
                        got: d,
                        context: format!(
                            "{} structure for '{}' is defined for {} levels but the factor has {}",
                            vs.name(),
                            label,
                            d,
                            dim
                        ),
                    });
                }
            }
        }
        let name = format!("{}({}) x {}({})", a.name(), label_a, b.name(), label_b);
        Ok(Self {
            a,
            dim_a,
            b,
            dim_b,
            label_a: label_a.to_string(),
            label_b: label_b.to_string(),
            name,
        })
    }

    /// The first (outer) factor's structure.
    pub fn outer(&self) -> &dyn VarStruct {
        self.a.as_ref()
    }

    /// The second (inner) factor's structure.
    pub fn inner(&self) -> &dyn VarStruct {
        self.b.as_ref()
    }

    /// Dimensions `(dim_a, dim_b)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.dim_a, self.dim_b)
    }

    fn check_dim(&self, dim: usize) {
        assert_eq!(
            dim,
            self.dim_a * self.dim_b,
            "Kronecker dim mismatch: {} x {} levels but dim={} requested",
            self.dim_a,
            self.dim_b,
            dim
        );
    }
}

impl VarStruct for KroneckerStruct {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_params(&self) -> usize {
        self.a.n_params() + self.b.n_params()
    }

    fn params(&self) -> Vec<f64> {
        let mut p = self.a.params();
        p.extend(self.b.params());
        p
    }

    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        let na = self.a.n_params();
        if params.len() != na + self.b.n_params() {
            return Err(LmmError::InvalidParameter(format!(
                "{} expects {} parameters, got {}",
                self.name,
                na + self.b.n_params(),
                params.len()
            )));
        }
        self.a.set_params(&params[..na])?;
        self.b.set_params(&params[na..])
    }

    fn covariance_matrix(&self, dim: usize) -> SparseMat {
        self.check_dim(dim);
        kronecker_product(
            &self.a.covariance_matrix(self.dim_a),
            &self.b.covariance_matrix(self.dim_b),
        )
    }

    fn inverse_covariance_matrix(&self, dim: usize) -> SparseMat {
        self.check_dim(dim);
        kronecker_product(
            &self.a.inverse_covariance_matrix(self.dim_a),
            &self.b.inverse_covariance_matrix(self.dim_b),
        )
    }

    fn log_determinant(&self, dim: usize) -> f64 {
        self.check_dim(dim);
        self.dim_b as f64 * self.a.log_determinant(self.dim_a)
            + self.dim_a as f64 * self.b.log_determinant(self.dim_b)
    }

    fn derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        self.check_dim(dim);
        let a_inv = self.a.inverse_covariance_matrix(self.dim_a);
        let b_inv = self.b.inverse_covariance_matrix(self.dim_b);
        let mut out = Vec::with_capacity(self.n_params());
        for da in self.a.derivatives_of_inverse(self.dim_a) {
            out.push(kronecker_product(&da, &b_inv));
        }
        for db in self.b.derivatives_of_inverse(self.dim_b) {
            out.push(kronecker_product(&a_inv, &db));
        }
        out
    }

    fn derivatives_of_covariance(&self, dim: usize) -> Vec<SparseMat> {
        self.check_dim(dim);
        let a_cov = self.a.covariance_matrix(self.dim_a);
        let b_cov = self.b.covariance_matrix(self.dim_b);
        let mut out = Vec::with_capacity(self.n_params());
        for da in self.a.derivatives_of_covariance(self.dim_a) {
            out.push(kronecker_product(&da, &b_cov));
        }
        for db in self.b.derivatives_of_covariance(self.dim_b) {
            out.push(kronecker_product(&a_cov, &db));
        }
        out
    }

    fn sigma_times_derivatives_of_inverse(&self, dim: usize) -> Vec<SparseMat> {
        self.check_dim(dim);
        let eye_a = sparse_eye(self.dim_a);
        let eye_b = sparse_eye(self.dim_b);
        let mut out = Vec::with_capacity(self.n_params());
        for m in self.a.sigma_times_derivatives_of_inverse(self.dim_a) {
            out.push(kronecker_product(&m, &eye_b));
        }
        for m in self.b.sigma_times_derivatives_of_inverse(self.dim_b) {
            out.push(kronecker_product(&eye_a, &m));
        }
        out
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut b = self.a.bounds();
        b.extend(self.b.bounds());
        b
    }

    fn is_standard_deviation_param(&self, i: usize) -> bool {
        let na = self.a.n_params();
        if i < na {
            self.a.is_standard_deviation_param(i)
        } else {
            self.b.is_standard_deviation_param(i - na)
        }
    }

    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .a
            .param_names()
            .into_iter()
            .map(|n| format!("{}.{}", self.label_a, n))
            .collect();
        names.extend(
            self.b
                .param_names()
                .into_iter()
                .map(|n| format!("{}.{}", self.label_b, n)),
        );
        names
    }

    fn is_variance_param(&self, i: usize) -> bool {
        let na = self.a.n_params();
        if i < na {
            self.a.is_variance_param(i)
        } else {
            self.b.is_variance_param(i - na)
        }
    }

    fn fixed_dim(&self) -> Option<usize> {
        Some(self.dim_a * self.dim_b)
    }

    fn clone_boxed(&self) -> Box<dyn VarStruct> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::sparse::{sparse_diagonal, sparse_identity, spmv};
    use approx::assert_relative_eq;

    fn dense(m: &SparseMat) -> nalgebra::DMatrix<f64> {
        let mut d = nalgebra::DMatrix::zeros(m.rows(), m.cols());
        for (v, (i, j)) in m.iter() {
            d[(i, j)] += *v;
        }
        d
    }

    #[test]
    fn test_kronecker_struct_ar1_x_known_consistency() {
        use crate::variance::{Known, AR1};
        let ks = KroneckerStruct::new(
            Box::new(AR1::new(2.0, 0.4)),
            3,
            Box::new(Known::identity(2)),
            2,
            "row",
            "col",
        )
        .unwrap();
        assert_eq!(ks.n_params(), 2);
        assert_eq!(ks.param_names(), vec!["row.sigma2", "row.rho"]);
        assert!(ks.is_variance_param(0));
        assert!(!ks.is_variance_param(1));

        let cov = dense(&ks.covariance_matrix(6));
        let inv = dense(&ks.inverse_covariance_matrix(6));
        let prod = &cov * &inv;
        for i in 0..6 {
            for j in 0..6 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(prod[(i, j)], e, epsilon = 1e-10);
            }
        }
        // log det
        assert_relative_eq!(
            ks.log_determinant(6),
            cov.determinant().ln(),
            epsilon = 1e-8
        );
        // Sigma * dSigma^-1 factorised == direct product
        let direct: Vec<_> = ks
            .derivatives_of_inverse(6)
            .iter()
            .map(|d| &cov * dense(d))
            .collect();
        let fact = ks.sigma_times_derivatives_of_inverse(6);
        for (m1, m2) in direct.iter().zip(fact.iter()) {
            let m2 = dense(m2);
            for i in 0..6 {
                for j in 0..6 {
                    assert_relative_eq!(m1[(i, j)], m2[(i, j)], epsilon = 1e-8);
                }
            }
        }
        // dSigma/dtheta factorised == -Sigma dSigma^-1 Sigma
        let dcov = ks.derivatives_of_covariance(6);
        for (d_inv, d_cov) in ks.derivatives_of_inverse(6).iter().zip(dcov.iter()) {
            let expected = -(&cov * dense(d_inv) * &cov);
            let got = dense(d_cov);
            for i in 0..6 {
                for j in 0..6 {
                    assert_relative_eq!(got[(i, j)], expected[(i, j)], epsilon = 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_kronecker_struct_rejects_dim_mismatch() {
        use crate::variance::{Diagonal, Known};
        let err = KroneckerStruct::new(
            Box::new(Diagonal::new(vec![1.0, 2.0, 3.0])),
            2,
            Box::new(Known::identity(4)),
            4,
            "env",
            "geno",
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_kronecker_identity_identity() {
        // I_2 (x) I_3 = I_6
        let i2 = sparse_identity(2);
        let i3 = sparse_identity(3);
        let result = kronecker_product(&i2, &i3);

        assert_eq!(result.rows(), 6);
        assert_eq!(result.cols(), 6);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = spmv(&result, &x);
        for i in 0..6 {
            assert_relative_eq!(y[i], x[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_kronecker_known_example() {
        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        // A (x) B = [[0, 5, 0, 10],
        //            [6, 7, 12, 14],
        //            [0, 15, 0, 20],
        //            [18, 21, 24, 28]]
        let mut tri_a = TriMat::new((2, 2));
        tri_a.add_triplet(0, 0, 1.0);
        tri_a.add_triplet(0, 1, 2.0);
        tri_a.add_triplet(1, 0, 3.0);
        tri_a.add_triplet(1, 1, 4.0);
        let a = tri_a.to_csc();

        let mut tri_b = TriMat::new((2, 2));
        tri_b.add_triplet(0, 1, 5.0);
        tri_b.add_triplet(1, 0, 6.0);
        tri_b.add_triplet(1, 1, 7.0);
        let b = tri_b.to_csc();

        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 4);
        assert_eq!(result.cols(), 4);

        // Test against known result: multiply by e1 = [1,0,0,0]
        let y = spmv(&result, &[1.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10); // row 0: 0*1 + 5*0 + 0*0 + 10*0 = 0
        assert_relative_eq!(y[1], 6.0, epsilon = 1e-10); // row 1: 6*1 + 7*0 + 12*0 + 14*0 = 6
        assert_relative_eq!(y[2], 0.0, epsilon = 1e-10); // row 2: 0*1 + 15*0 + 0*0 + 20*0 = 0
        assert_relative_eq!(y[3], 18.0, epsilon = 1e-10); // row 3: 18*1 + 21*0 + 24*0 + 28*0 = 18

        // Test with e4 = [0,0,0,1]
        let y = spmv(&result, &[0.0, 0.0, 0.0, 1.0]);
        assert_relative_eq!(y[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 14.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 20.0, epsilon = 1e-10);
        assert_relative_eq!(y[3], 28.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kronecker_dimension_check() {
        let a = sparse_identity(3);
        let b = sparse_identity(4);
        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 12);
        assert_eq!(result.cols(), 12);
    }

    #[test]
    fn test_kronecker_diagonal_diagonal() {
        // diag(2, 3) (x) diag(5, 7) = diag(10, 14, 15, 21)
        let a = sparse_diagonal(&[2.0, 3.0]);
        let b = sparse_diagonal(&[5.0, 7.0]);
        let result = kronecker_product(&a, &b);

        assert_eq!(result.rows(), 4);
        assert_eq!(result.cols(), 4);

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = spmv(&result, &x);
        assert_relative_eq!(y[0], 10.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 14.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 15.0, epsilon = 1e-10);
        assert_relative_eq!(y[3], 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kronecker_non_square() {
        // A: 2x3, B: 3x2 => result: 6x6
        let mut tri_a = TriMat::new((2, 3));
        tri_a.add_triplet(0, 0, 1.0);
        tri_a.add_triplet(1, 2, 2.0);
        let a = tri_a.to_csc();

        let mut tri_b = TriMat::new((3, 2));
        tri_b.add_triplet(0, 0, 3.0);
        tri_b.add_triplet(2, 1, 4.0);
        let b = tri_b.to_csc();

        let result = kronecker_product(&a, &b);
        assert_eq!(result.rows(), 6);
        assert_eq!(result.cols(), 6);
    }
}
