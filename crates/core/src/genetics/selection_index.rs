use crate::error::{LmmError, Result};
use nalgebra::DMatrix;

/// Smith-Hazel selection index for multi-trait breeding programs.
///
/// The index: I = b'p, where b = P⁻¹Ga and p is the vector of
/// phenotypic deviations.
///
/// References:
/// - Smith (1936). A discriminant function for plant selection.
/// - Hazel (1943). The genetic basis for constructing selection indexes.
/// - Pesek & Baker (1969). Desired improvement in relation to selection indices.
#[derive(Debug, Clone)]
pub struct SelectionIndex {
    /// Genetic covariance matrix G (t × t)
    g_matrix: DMatrix<f64>,
    /// Phenotypic covariance matrix P (t × t)
    p_matrix: DMatrix<f64>,
    /// Economic weights a (length t)
    economic_weights: Vec<f64>,
    /// Computed index coefficients b = P⁻¹Ga
    coefficients: Vec<f64>,
    /// Number of traits
    n_traits: usize,
}

impl SelectionIndex {
    /// Create a new selection index.
    pub fn new(
        g_matrix: DMatrix<f64>,
        p_matrix: DMatrix<f64>,
        economic_weights: Vec<f64>,
    ) -> Result<Self> {
        let t = economic_weights.len();
        if g_matrix.nrows() != t || g_matrix.ncols() != t {
            return Err(LmmError::DimensionMismatch {
                expected: t,
                got: g_matrix.nrows(),
                context: "G matrix dimensions must match number of traits".into(),
            });
        }
        if p_matrix.nrows() != t || p_matrix.ncols() != t {
            return Err(LmmError::DimensionMismatch {
                expected: t,
                got: p_matrix.nrows(),
                context: "P matrix dimensions must match number of traits".into(),
            });
        }

        // b = P⁻¹ * G * a
        let p_inv = p_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| LmmError::SingularMatrix {
                context: "Phenotypic covariance P is singular".into(),
            })?;

        let a = nalgebra::DVector::from_column_slice(&economic_weights);
        let ga = &g_matrix * &a;
        let b = &p_inv * &ga;
        let coefficients: Vec<f64> = b.as_slice().to_vec();

        Ok(Self {
            g_matrix,
            p_matrix,
            economic_weights,
            coefficients,
            n_traits: t,
        })
    }

    /// Index coefficients b = P⁻¹Ga.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Compute index value for one individual: I = b'p.
    pub fn index_value(&self, phenotypic_deviations: &[f64]) -> f64 {
        assert_eq!(phenotypic_deviations.len(), self.n_traits);
        self.coefficients
            .iter()
            .zip(phenotypic_deviations)
            .map(|(b, p)| b * p)
            .sum()
    }

    /// Compute index values for multiple individuals.
    /// `phenotypes`: n_individuals × n_traits matrix.
    pub fn index_values(&self, phenotypes: &DMatrix<f64>) -> Vec<f64> {
        let n = phenotypes.nrows();
        let b = nalgebra::DVector::from_column_slice(&self.coefficients);
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let p = phenotypes.row(i).transpose();
            result.push(b.dot(&p));
        }
        result
    }

    /// Selection accuracy: r_HI = √(b'Gb) / √(a'Ga).
    pub fn accuracy(&self) -> f64 {
        let b = nalgebra::DVector::from_column_slice(&self.coefficients);
        let a = nalgebra::DVector::from_column_slice(&self.economic_weights);
        let bgb = b.dot(&(&self.g_matrix * &b));
        let aga = a.dot(&(&self.g_matrix * &a));
        if aga <= 0.0 || bgb <= 0.0 {
            return 0.0;
        }
        (bgb / aga).sqrt().min(1.0)
    }

    /// Expected genetic gain per trait: ΔG = i * G * b / √(b'Pb).
    pub fn expected_genetic_gain(&self, selection_intensity: f64) -> Vec<f64> {
        let b = nalgebra::DVector::from_column_slice(&self.coefficients);
        let bpb = b.dot(&(&self.p_matrix * &b));
        if bpb <= 0.0 {
            return vec![0.0; self.n_traits];
        }
        let gb = &self.g_matrix * &b;
        let scale = selection_intensity / bpb.sqrt();
        gb.as_slice().iter().map(|v| v * scale).collect()
    }

    /// Index variance: Var(I) = b'Pb.
    pub fn index_variance(&self) -> f64 {
        let b = nalgebra::DVector::from_column_slice(&self.coefficients);
        b.dot(&(&self.p_matrix * &b))
    }

    /// Covariance between index and aggregate genotype: Cov(I, H) = b'Ga.
    pub fn index_genotype_covariance(&self) -> f64 {
        let b = nalgebra::DVector::from_column_slice(&self.coefficients);
        let a = nalgebra::DVector::from_column_slice(&self.economic_weights);
        b.dot(&(&self.g_matrix * &a))
    }

    /// Response per unit selection intensity: R = Gb / √(b'Pb).
    pub fn response_per_intensity(&self) -> Vec<f64> {
        self.expected_genetic_gain(1.0)
    }

    /// Restricted selection index (Kempthorne & Nordskog, 1959).
    ///
    /// Maximize gain in unrestricted traits while constraining
    /// restricted traits to zero genetic change.
    ///
    /// The constraint is: Gb must be zero for restricted traits,
    /// i.e., C'Gb = 0 where C selects restricted trait rows.
    pub fn restricted_index(
        g_matrix: DMatrix<f64>,
        p_matrix: DMatrix<f64>,
        economic_weights: Vec<f64>,
        restricted_traits: &[usize],
    ) -> Result<Self> {
        let t = economic_weights.len();

        let p_inv = p_matrix
            .clone()
            .try_inverse()
            .ok_or_else(|| LmmError::SingularMatrix {
                context: "P is singular".into(),
            })?;

        let r = restricted_traits.len();
        if r >= t {
            return Err(LmmError::ModelSpec(
                "Cannot restrict all traits".into(),
            ));
        }

        // C is t × r: columns are unit vectors selecting restricted traits
        let mut c_mat = DMatrix::zeros(t, r);
        for (j, &trait_idx) in restricted_traits.iter().enumerate() {
            c_mat[(trait_idx, j)] = 1.0;
        }

        // Unrestricted: b₀ = P⁻¹Ga
        let a = nalgebra::DVector::from_column_slice(&economic_weights);
        let ga = &g_matrix * &a;
        let b0 = &p_inv * &ga;

        // Restricted: b = b₀ - P⁻¹G C (C'G P⁻¹G C)⁻¹ C'G b₀
        // This ensures C'G b = 0 (zero genetic gain on restricted traits)
        let pinvg = &p_inv * &g_matrix;
        let gc = &g_matrix * &c_mat;
        let ct_g_pinvg_c = c_mat.transpose() * &g_matrix * &pinvg * &c_mat;

        let inner_inv = ct_g_pinvg_c
            .try_inverse()
            .ok_or_else(|| LmmError::SingularMatrix {
                context: "Restriction matrix is singular".into(),
            })?;

        let ct_g_b0 = c_mat.transpose() * &g_matrix * &b0;
        let correction = &pinvg * &c_mat * &inner_inv * &ct_g_b0;
        let b = &b0 - &correction;
        let coefficients: Vec<f64> = b.as_slice().to_vec();

        Ok(Self {
            g_matrix,
            p_matrix,
            economic_weights,
            coefficients,
            n_traits: t,
        })
    }

    /// Print index summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Selection Index Summary ===\n\n");
        s.push_str(&format!("Traits: {}\n", self.n_traits));
        s.push_str(&format!("Accuracy (r_HI): {:.4}\n", self.accuracy()));
        s.push_str(&format!("Index variance: {:.4}\n\n", self.index_variance()));

        s.push_str("Economic weights and index coefficients:\n");
        s.push_str(&format!("{:<10} {:>12} {:>12}\n", "Trait", "Weight", "Coeff"));
        for i in 0..self.n_traits {
            s.push_str(&format!(
                "{:<10} {:>12.4} {:>12.4}\n",
                format!("Trait_{}", i + 1),
                self.economic_weights[i],
                self.coefficients[i]
            ));
        }

        let gain = self.expected_genetic_gain(1.4); // i ≈ 1.4 for top 20%
        s.push_str(&format!(
            "\nExpected genetic gain (i=1.4, top 20%):\n"
        ));
        for (i, g) in gain.iter().enumerate() {
            s.push_str(&format!("  Trait_{}: {:.4}\n", i + 1, g));
        }

        s
    }
}

/// Desired gains selection index (Pesek & Baker, 1969).
///
/// b = P⁻¹G * d where d is the vector of desired genetic gains.
pub fn desired_gains_index(
    g_matrix: DMatrix<f64>,
    p_matrix: DMatrix<f64>,
    desired_gains: Vec<f64>,
) -> Result<SelectionIndex> {
    let t = desired_gains.len();

    let p_inv = p_matrix
        .clone()
        .try_inverse()
        .ok_or_else(|| LmmError::SingularMatrix {
            context: "P is singular".into(),
        })?;

    let g_inv = g_matrix
        .clone()
        .try_inverse()
        .ok_or_else(|| LmmError::SingularMatrix {
            context: "G is singular".into(),
        })?;

    // For desired gains, the "economic weights" are: a = G⁻¹d
    let d = nalgebra::DVector::from_column_slice(&desired_gains);
    let a = &g_inv * &d;
    let economic_weights: Vec<f64> = a.as_slice().to_vec();

    // b = P⁻¹Ga = P⁻¹G * G⁻¹d = P⁻¹d
    let b = &p_inv * &d;
    let coefficients: Vec<f64> = b.as_slice().to_vec();

    Ok(SelectionIndex {
        g_matrix,
        p_matrix,
        economic_weights,
        coefficients,
        n_traits: t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_matrices() -> (DMatrix<f64>, DMatrix<f64>) {
        // G = [[10, 5], [5, 20]]
        let g = DMatrix::from_row_slice(2, 2, &[10.0, 5.0, 5.0, 20.0]);
        // P = [[15, 8], [8, 30]]
        let p = DMatrix::from_row_slice(2, 2, &[15.0, 8.0, 8.0, 30.0]);
        (g, p)
    }

    #[test]
    fn test_basic_index() {
        let (g, p) = test_matrices();
        let a = vec![1.0, 1.0];
        let idx = SelectionIndex::new(g.clone(), p.clone(), a).unwrap();

        // Verify b = P⁻¹Ga numerically
        let p_inv = p.try_inverse().unwrap();
        let a_vec = nalgebra::DVector::from_column_slice(&[1.0, 1.0]);
        let expected = &p_inv * &(&g * &a_vec);

        assert_relative_eq!(idx.coefficients()[0], expected[0], epsilon = 1e-10);
        assert_relative_eq!(idx.coefficients()[1], expected[1], epsilon = 1e-10);

        // Coefficients should be reasonable positive values
        assert!(idx.coefficients()[0] > 0.0);
        assert!(idx.coefficients()[1] > 0.0);
    }

    #[test]
    fn test_index_value() {
        let (g, p) = test_matrices();
        let idx = SelectionIndex::new(g, p, vec![1.0, 1.0]).unwrap();
        let val = idx.index_value(&[2.0, 3.0]);
        let expected = idx.coefficients()[0] * 2.0 + idx.coefficients()[1] * 3.0;
        assert_relative_eq!(val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_accuracy_bounds() {
        let (g, p) = test_matrices();
        let idx = SelectionIndex::new(g, p, vec![1.0, 1.0]).unwrap();
        let acc = idx.accuracy();
        assert!(acc > 0.0 && acc <= 1.0, "accuracy={}", acc);
    }

    #[test]
    fn test_genetic_gain_direction() {
        let (g, p) = test_matrices();
        let idx = SelectionIndex::new(g, p, vec![1.0, 0.0]).unwrap();
        let gain = idx.expected_genetic_gain(1.4);
        // With weight on trait 1 only, trait 1 gain should be positive
        assert!(gain[0] > 0.0);
    }

    #[test]
    fn test_identity_matrices() {
        // With P = I and G = I, b should equal a
        let g = DMatrix::identity(3, 3);
        let p = DMatrix::identity(3, 3);
        let a = vec![2.0, 3.0, 1.0];
        let idx = SelectionIndex::new(g, p, a.clone()).unwrap();
        for i in 0..3 {
            assert_relative_eq!(idx.coefficients()[i], a[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_restricted_index() {
        let (g, p) = test_matrices();
        let idx = SelectionIndex::restricted_index(
            g.clone(), p.clone(), vec![1.0, 1.0], &[1],
        ).unwrap();

        // The restriction means C'Gb = 0 for the restricted trait
        // i.e., row 1 of G*b should be zero
        let b = nalgebra::DVector::from_column_slice(idx.coefficients());
        let gb = &g * &b;
        assert!(gb[1].abs() < 1e-8, "restricted trait Gb should be zero: {}", gb[1]);

        // Unrestricted trait should still have positive response
        assert!(gb[0] > 0.0, "unrestricted trait should have positive response");
    }

    #[test]
    fn test_desired_gains() {
        let (g, p) = test_matrices();
        let idx = desired_gains_index(g, p, vec![5.0, 10.0]).unwrap();
        assert_eq!(idx.coefficients().len(), 2);
        // Index should be constructible and have positive variance
        assert!(idx.index_variance() > 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let g = DMatrix::identity(2, 2);
        let p = DMatrix::identity(3, 3);
        assert!(SelectionIndex::new(g, p, vec![1.0, 1.0]).is_err());
    }

    #[test]
    fn test_summary() {
        let (g, p) = test_matrices();
        let idx = SelectionIndex::new(g, p, vec![1.0, 1.0]).unwrap();
        let summary = idx.summary();
        assert!(summary.contains("Selection Index"));
        assert!(summary.contains("Accuracy"));
    }
}
