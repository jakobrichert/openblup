//! K-fold cross-validation for genomic prediction accuracy assessment.
//!
//! Cross-validation partitions observations into k folds, fitting the model
//! on (k-1) folds and predicting the held-out fold. This is repeated for each
//! fold, yielding per-fold and overall accuracy metrics.
//!
//! The standard approach for genomic prediction (GBLUP / pedigree BLUP) is:
//!   1. Mask validation phenotypes
//!   2. Fit the mixed model on training phenotypes
//!   3. Predict validation animals: y_hat = X_val * b_hat + Z_val * u_hat
//!
//! # Example
//!
//! ```ignore
//! use plant_breeding_lmm_core::diagnostics::crossval::CrossValidator;
//!
//! let cv = CrossValidator::new(5).seed(42);
//! let result = cv.run(&y, &x, &z, Some(&ginv))?;
//! println!("{}", result.summary());
//! ```

use crate::error::{LmmError, Result};
use crate::lmm::MixedModelEquations;
use crate::matrix::sparse::{sparse_diagonal, spmv};

use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Configuration for k-fold cross-validation.
pub struct CrossValidator {
    /// Number of folds (default: 5).
    n_folds: usize,
    /// Random seed for reproducibility.
    seed: u64,
    /// Whether to stratify by a factor (strata indices).
    stratify_by: Option<String>,
    /// Maximum REML iterations per fold.
    max_iter: usize,
    /// REML convergence tolerance.
    tol: f64,
}

/// Results from cross-validation.
#[derive(Debug, Clone)]
pub struct CrossValResult {
    /// Per-fold results.
    pub folds: Vec<FoldResult>,
    /// Overall prediction accuracy (correlation between predicted and observed).
    pub accuracy: f64,
    /// Mean squared error of prediction.
    pub msep: f64,
    /// Prediction bias (regression slope of observed on predicted).
    pub bias: f64,
    /// Mean absolute error.
    pub mae: f64,
}

/// Result for a single fold.
#[derive(Debug, Clone)]
pub struct FoldResult {
    /// Fold index (0-based).
    pub fold: usize,
    /// Indices of validation observations.
    pub validation_indices: Vec<usize>,
    /// Predicted values for validation set.
    pub predicted: Vec<f64>,
    /// Observed values for validation set.
    pub observed: Vec<f64>,
    /// Correlation between predicted and observed.
    pub accuracy: f64,
    /// MSEP for this fold.
    pub msep: f64,
}

impl CrossValidator {
    /// Create a new cross-validator with the given number of folds.
    ///
    /// # Panics
    ///
    /// Panics if `n_folds` is less than 2.
    pub fn new(n_folds: usize) -> Self {
        assert!(n_folds >= 2, "Number of folds must be at least 2");
        Self {
            n_folds,
            seed: 0,
            stratify_by: None,
            max_iter: 50,
            tol: 1e-6,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the stratification column name (for stratified k-fold CV).
    pub fn stratify(mut self, column: &str) -> Self {
        self.stratify_by = Some(column.to_string());
        self
    }

    /// Set REML maximum iterations for each fold (default: 50).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set REML convergence tolerance for each fold (default: 1e-6).
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Run k-fold cross-validation on a mixed model defined by its raw
    /// components.
    ///
    /// For each fold:
    ///   1. Partition observations into training and validation sets
    ///   2. Subset X, Z, and y to the training rows
    ///   3. Fit REML on the training subset
    ///   4. Predict validation observations: y_hat = X_val * b_hat + Z_val * u_hat
    ///   5. Compute fold-level accuracy
    ///
    /// # Arguments
    ///
    /// * `y` - Full response vector (length n)
    /// * `x` - Fixed effects design matrix (n x p, sparse CSC)
    /// * `z` - Random effects design matrix (n x q, sparse CSC). For multiple
    ///         random terms, pass a single concatenated Z = [Z1 | Z2 | ...].
    /// * `ginv` - Optional relationship matrix inverse (q x q). Pass `None`
    ///           to use an identity matrix (IID random effects).
    ///
    /// # Returns
    ///
    /// A [`CrossValResult`] with per-fold and aggregate accuracy metrics.
    pub fn run(
        &self,
        y: &[f64],
        x: &sprs::CsMat<f64>,
        z: &sprs::CsMat<f64>,
        ginv: Option<&sprs::CsMat<f64>>,
    ) -> Result<CrossValResult> {
        let n = y.len();
        if n < self.n_folds {
            return Err(LmmError::ModelSpec(format!(
                "Number of observations ({}) must be >= number of folds ({})",
                n, self.n_folds
            )));
        }
        if x.rows() != n {
            return Err(LmmError::DimensionMismatch {
                expected: n,
                got: x.rows(),
                context: "X row count must match y length".into(),
            });
        }
        if z.rows() != n {
            return Err(LmmError::DimensionMismatch {
                expected: n,
                got: z.rows(),
                context: "Z row count must match y length".into(),
            });
        }

        let folds = create_folds(n, self.n_folds, self.seed);
        let mut fold_results = Vec::with_capacity(self.n_folds);

        for (fold_idx, val_indices) in folds.iter().enumerate() {
            let fold_result =
                self.run_fold(fold_idx, val_indices, y, x, z, ginv)?;
            fold_results.push(fold_result);
        }

        Ok(CrossValResult::from_folds(fold_results))
    }

    /// Leave-one-out cross-validation (LOOCV).
    ///
    /// This is equivalent to k-fold CV with k = n. Each observation is
    /// held out exactly once.
    pub fn leave_one_out(
        y: &[f64],
        x: &sprs::CsMat<f64>,
        z: &sprs::CsMat<f64>,
        ginv: Option<&sprs::CsMat<f64>>,
    ) -> Result<CrossValResult> {
        let n = y.len();
        CrossValidator::new(n).seed(0).run(y, x, z, ginv)
    }

    /// Run a single fold of cross-validation.
    fn run_fold(
        &self,
        fold_idx: usize,
        val_indices: &[usize],
        y: &[f64],
        x: &sprs::CsMat<f64>,
        z: &sprs::CsMat<f64>,
        ginv: Option<&sprs::CsMat<f64>>,
    ) -> Result<FoldResult> {
        let n = y.len();

        // Determine training indices
        let mut is_val = vec![false; n];
        for &vi in val_indices {
            is_val[vi] = true;
        }
        let train_indices: Vec<usize> = (0..n).filter(|i| !is_val[*i]).collect();
        let n_train = train_indices.len();

        // Subset matrices to training rows
        let x_train = subset_sparse_rows(x, &train_indices);
        let z_train = subset_sparse_rows(z, &train_indices);
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y[i]).collect();

        // Subset matrices to validation rows
        let x_val = subset_sparse_rows(x, val_indices);
        let z_val = subset_sparse_rows(z, val_indices);
        let y_val: Vec<f64> = val_indices.iter().map(|&i| y[i]).collect();

        // Fit REML on training set
        let q = z.cols();

        // Initialize variance parameters from data
        let y_mean: f64 = y_train.iter().sum::<f64>() / n_train as f64;
        let y_var: f64 = y_train
            .iter()
            .map(|&yi| (yi - y_mean).powi(2))
            .sum::<f64>()
            / (n_train - 1).max(1) as f64;
        let init_var = (y_var / 2.0).max(0.01);

        let mut sigma2_g = init_var;
        let mut sigma2_e = init_var;

        // REML iteration (EM-REML for robustness in CV context)
        for _iter in 0..self.max_iter {
            // Build G^{-1} block
            let g_inv_block = if let Some(ginv_mat) = ginv {
                ginv_mat.map(|v| v / sigma2_g)
            } else {
                sparse_diagonal(&vec![1.0 / sigma2_g; q])
            };

            let r_inv_scale = 1.0 / sigma2_e;

            // Assemble and solve MME
            let mme = MixedModelEquations::assemble(
                &x_train,
                &[z_train.clone()],
                &y_train,
                r_inv_scale,
                &[g_inv_block],
            );
            let sol = mme.solve()?;

            let c_inv = sol.c_inv.as_ref().ok_or(LmmError::CholeskyFailed(
                "C^{-1} not available".into(),
            ))?;

            let n_fixed = mme.n_fixed;
            let n_eff = (n_train - n_fixed) as f64;

            // EM update for residual variance
            let mut y_e_hat: f64 = y_train.iter().map(|yi| yi * yi).sum();
            let xty = crate::matrix::sparse::xt_y(&x_train, &y_train);
            for i in 0..n_fixed {
                y_e_hat -= sol.fixed_effects[i] * xty[i];
            }
            let zty = crate::matrix::sparse::xt_y(&z_train, &y_train);
            for j in 0..q {
                y_e_hat -= sol.random_effects[0][j] * zty[j];
            }
            let new_sigma2_e = (y_e_hat / n_eff).max(1e-10);

            // EM update for genetic variance
            let u = &sol.random_effects[0];
            let u_quadratic = if let Some(ginv_mat) = ginv {
                let kinv_u = spmv(ginv_mat, u);
                u.iter().zip(kinv_u.iter()).map(|(a, b)| a * b).sum::<f64>()
            } else {
                u.iter().map(|ui| ui * ui).sum::<f64>()
            };

            let n_fixed_cols = mme.n_fixed;
            let trace_term = if let Some(ginv_mat) = ginv {
                let mut tr = 0.0;
                for i in 0..q {
                    for j in 0..q {
                        let kinv_ij = ginv_mat.get(i, j).copied().unwrap_or(0.0);
                        let cinv_ji = c_inv[(n_fixed_cols + j, n_fixed_cols + i)];
                        tr += kinv_ij * cinv_ji;
                    }
                }
                tr
            } else {
                let mut tr = 0.0;
                for i in 0..q {
                    tr += c_inv[(n_fixed_cols + i, n_fixed_cols + i)];
                }
                tr
            };

            let new_sigma2_g =
                ((u_quadratic + trace_term) / q as f64).max(1e-10);

            // Check convergence
            let change = ((new_sigma2_e - sigma2_e).powi(2)
                + (new_sigma2_g - sigma2_g).powi(2))
            .sqrt()
                / (sigma2_e.powi(2) + sigma2_g.powi(2)).sqrt().max(1e-10);

            sigma2_e = new_sigma2_e;
            sigma2_g = new_sigma2_g;

            if change < self.tol {
                break;
            }
        }

        // Final solve with converged variances
        let g_inv_block = if let Some(ginv_mat) = ginv {
            ginv_mat.map(|v| v / sigma2_g)
        } else {
            sparse_diagonal(&vec![1.0 / sigma2_g; q])
        };

        let r_inv_scale = 1.0 / sigma2_e;
        let mme = MixedModelEquations::assemble(
            &x_train,
            &[z_train.clone()],
            &y_train,
            r_inv_scale,
            &[g_inv_block],
        );
        let sol = mme.solve()?;

        // Predict validation set: y_hat = X_val * b_hat + Z_val * u_hat
        let xb = spmv(&x_val, &sol.fixed_effects);
        let zu = spmv(&z_val, &sol.random_effects[0]);
        let predicted: Vec<f64> = xb.iter().zip(zu.iter()).map(|(a, b)| a + b).collect();

        let accuracy = correlation(&predicted, &y_val);
        let msep = mean_squared_error(&predicted, &y_val);

        Ok(FoldResult {
            fold: fold_idx,
            validation_indices: val_indices.to_vec(),
            predicted,
            observed: y_val,
            accuracy,
            msep,
        })
    }
}

impl CrossValResult {
    /// Aggregate fold-level results into overall cross-validation metrics.
    fn from_folds(folds: Vec<FoldResult>) -> Self {
        // Collect all predicted and observed values
        let mut all_predicted = Vec::new();
        let mut all_observed = Vec::new();
        for fold in &folds {
            all_predicted.extend_from_slice(&fold.predicted);
            all_observed.extend_from_slice(&fold.observed);
        }

        let accuracy = correlation(&all_predicted, &all_observed);
        let msep = mean_squared_error(&all_predicted, &all_observed);
        let bias = regression_slope(&all_predicted, &all_observed);
        let mae = mean_absolute_error(&all_predicted, &all_observed);

        CrossValResult {
            folds,
            accuracy,
            msep,
            bias,
            mae,
        }
    }

    /// Print a formatted summary of the cross-validation results.
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str("=== Cross-Validation Results ===\n\n");
        s.push_str(&format!("Number of folds: {}\n", self.folds.len()));

        let total_obs: usize =
            self.folds.iter().map(|f| f.validation_indices.len()).sum();
        s.push_str(&format!("Total observations: {}\n\n", total_obs));

        s.push_str("--- Per-fold Results ---\n");
        s.push_str(&format!(
            "{:<8} {:<12} {:<12} {:<12}\n",
            "Fold", "N_val", "Accuracy", "MSEP"
        ));
        for fold in &self.folds {
            s.push_str(&format!(
                "{:<8} {:<12} {:<12.4} {:<12.4}\n",
                fold.fold,
                fold.validation_indices.len(),
                fold.accuracy,
                fold.msep
            ));
        }

        s.push_str("\n--- Overall Metrics ---\n");
        s.push_str(&format!("Prediction accuracy (r): {:.4}\n", self.accuracy));
        s.push_str(&format!("MSEP:                    {:.4}\n", self.msep));
        s.push_str(&format!("Bias (slope):            {:.4}\n", self.bias));
        s.push_str(&format!("MAE:                     {:.4}\n", self.mae));

        s
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Partition `n` observations into `k` approximately equal folds.
///
/// Observations are shuffled using the given seed, then distributed
/// round-robin into folds. Each observation appears in exactly one fold.
pub(crate) fn create_folds(n: usize, k: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &idx) in indices.iter().enumerate() {
        folds[i % k].push(idx);
    }
    folds
}

/// Stratified partition: ensure each fold has similar proportions of each
/// stratum.
///
/// Within each stratum, observations are shuffled and distributed
/// round-robin. This guarantees that each fold has approximately the same
/// number of observations from each stratum.
pub(crate) fn create_stratified_folds(
    strata: &[usize],
    k: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Group indices by stratum
    let max_stratum = strata.iter().copied().max().unwrap_or(0);
    let mut stratum_indices: Vec<Vec<usize>> = vec![Vec::new(); max_stratum + 1];
    for (i, &s) in strata.iter().enumerate() {
        stratum_indices[s].push(i);
    }

    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];

    // For each stratum, shuffle and assign round-robin
    for indices in &mut stratum_indices {
        indices.shuffle(&mut rng);
        for (i, &idx) in indices.iter().enumerate() {
            folds[i % k].push(idx);
        }
    }

    folds
}

/// Compute the Pearson correlation coefficient between two vectors.
///
/// Returns 0.0 if either vector has zero variance.
pub(crate) fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

/// Compute the regression slope of `observed` on `predicted`.
///
/// The slope of the simple linear regression observed = a + b * predicted
/// provides a check for prediction bias. A slope of 1.0 indicates no bias;
/// values less than 1.0 indicate over-dispersion of predictions.
pub(crate) fn regression_slope(predicted: &[f64], observed: &[f64]) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }

    let mean_p = predicted.iter().sum::<f64>() / n as f64;
    let mean_o = observed.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut denom = 0.0;

    for i in 0..n {
        let dp = predicted[i] - mean_p;
        num += dp * (observed[i] - mean_o);
        denom += dp * dp;
    }

    if denom.abs() < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Compute the mean squared error of prediction.
fn mean_squared_error(predicted: &[f64], observed: &[f64]) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    predicted
        .iter()
        .zip(observed.iter())
        .map(|(p, o)| (p - o).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Compute the mean absolute error.
fn mean_absolute_error(predicted: &[f64], observed: &[f64]) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    predicted
        .iter()
        .zip(observed.iter())
        .map(|(p, o)| (p - o).abs())
        .sum::<f64>()
        / n as f64
}

/// Subset the rows of a sparse matrix by the given row indices.
///
/// Returns a new sparse matrix with rows corresponding to `indices`,
/// in the order they appear. The number of columns is unchanged.
fn subset_sparse_rows(mat: &sprs::CsMat<f64>, indices: &[usize]) -> sprs::CsMat<f64> {
    let n_new = indices.len();
    let ncols = mat.cols();
    let mut tri = sprs::TriMat::new((n_new, ncols));

    // Build a lookup from old row -> new row for efficiency
    let mut row_map = std::collections::HashMap::with_capacity(n_new);
    for (new_row, &old_row) in indices.iter().enumerate() {
        row_map.insert(old_row, new_row);
    }

    for (val, (row, col)) in mat.iter() {
        if let Some(&new_row) = row_map.get(&row) {
            tri.add_triplet(new_row, col, *val);
        }
    }

    tri.to_csc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ---------------------------------------------------------------
    // Fold creation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_create_folds_all_observations_appear_once() {
        let n = 20;
        let k = 5;
        let folds = create_folds(n, k, 42);

        assert_eq!(folds.len(), k);

        // Every index 0..n should appear exactly once across all folds
        let mut seen = vec![false; n];
        for fold in &folds {
            for &idx in fold {
                assert!(!seen[idx], "Index {} appeared more than once", idx);
                seen[idx] = true;
            }
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "Index {} was not assigned to any fold", i);
        }
    }

    #[test]
    fn test_create_folds_approximately_equal_size() {
        let n = 23;
        let k = 5;
        let folds = create_folds(n, k, 42);

        let min_size = n / k;       // 4
        let max_size = min_size + 1; // 5

        for (i, fold) in folds.iter().enumerate() {
            assert!(
                fold.len() >= min_size && fold.len() <= max_size,
                "Fold {} has size {}, expected between {} and {}",
                i,
                fold.len(),
                min_size,
                max_size
            );
        }
    }

    #[test]
    fn test_create_folds_reproducible() {
        let folds1 = create_folds(20, 5, 42);
        let folds2 = create_folds(20, 5, 42);
        assert_eq!(folds1, folds2);
    }

    #[test]
    fn test_create_folds_different_seeds_differ() {
        let folds1 = create_folds(20, 5, 42);
        let folds2 = create_folds(20, 5, 99);
        // With high probability, at least one fold differs
        assert_ne!(folds1, folds2);
    }

    // ---------------------------------------------------------------
    // Stratified fold tests
    // ---------------------------------------------------------------

    #[test]
    fn test_stratified_folds_all_observations_appear_once() {
        let strata = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        let k = 3;
        let folds = create_stratified_folds(&strata, k, 42);

        let n = strata.len();
        let mut seen = vec![false; n];
        for fold in &folds {
            for &idx in fold {
                assert!(!seen[idx]);
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_stratified_folds_proportions_maintained() {
        // 3 strata with 6, 6, 6 observations, 3 folds
        // Each fold should get 2 from each stratum
        let strata = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
        let k = 3;
        let folds = create_stratified_folds(&strata, k, 42);

        for (fi, fold) in folds.iter().enumerate() {
            let mut stratum_counts = vec![0usize; 3];
            for &idx in fold {
                stratum_counts[strata[idx]] += 1;
            }
            for (s, &count) in stratum_counts.iter().enumerate() {
                assert_eq!(
                    count, 2,
                    "Fold {} has {} obs from stratum {}, expected 2",
                    fi, count, s
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // Correlation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_relative_eq!(correlation(&x, &y), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert_relative_eq!(correlation(&x, &y), -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_zero_variance() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(correlation(&x, &y), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation_known_value() {
        // Known dataset: r should be very close to 1.0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.2, 1.8, 3.1, 3.9, 5.2];
        let r = correlation(&x, &y);
        assert!(r > 0.99 && r <= 1.0, "Expected r close to 1.0, got {}", r);
    }

    // ---------------------------------------------------------------
    // Regression slope tests
    // ---------------------------------------------------------------

    #[test]
    fn test_regression_slope_perfect() {
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            regression_slope(&predicted, &observed),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_regression_slope_double() {
        // observed = 2 * predicted
        let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let observed = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_relative_eq!(
            regression_slope(&predicted, &observed),
            2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_regression_slope_half() {
        // observed = 0.5 * predicted + constant
        let predicted = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let observed = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // 0.5*pred + 2
        assert_relative_eq!(
            regression_slope(&predicted, &observed),
            0.5,
            epsilon = 1e-10
        );
    }

    // ---------------------------------------------------------------
    // Subset sparse rows test
    // ---------------------------------------------------------------

    #[test]
    fn test_subset_sparse_rows() {
        // 4x3 matrix
        let mut tri = sprs::TriMat::new((4, 3));
        tri.add_triplet(0, 0, 1.0); // row 0
        tri.add_triplet(1, 1, 2.0); // row 1
        tri.add_triplet(2, 2, 3.0); // row 2
        tri.add_triplet(3, 0, 4.0); // row 3
        let mat = tri.to_csc();

        // Subset rows [1, 3]
        let sub = subset_sparse_rows(&mat, &[1, 3]);
        assert_eq!(sub.rows(), 2);
        assert_eq!(sub.cols(), 3);

        // row 0 of sub = original row 1: [0, 2, 0]
        let r0 = spmv(&sub, &[1.0, 1.0, 1.0]);
        assert_relative_eq!(r0[0], 2.0, epsilon = 1e-10);
        // row 1 of sub = original row 3: [4, 0, 0]
        assert_relative_eq!(r0[1], 4.0, epsilon = 1e-10);
    }

    // ---------------------------------------------------------------
    // Cross-validation integration tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cv_simple_model() {
        // Build a simple dataset: y = mu + genotype_effect + error
        // 3 genotypes, 4 reps each = 12 observations
        let n = 12;
        let q = 3; // genotypes

        // True effects: mu=10, g1=+2, g2=0, g3=-2
        let genotype_assignment = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let true_g = vec![2.0, 0.0, -2.0];
        let mu = 10.0;

        // Use fixed errors for reproducibility (instead of random)
        let errors = vec![
            0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.15, -0.15, 0.05, -0.25, 0.25, -0.05,
        ];

        let y: Vec<f64> = (0..n)
            .map(|i| mu + true_g[genotype_assignment[i]] + errors[i])
            .collect();

        // X = intercept column (n x 1)
        let mut x_tri = sprs::TriMat::new((n, 1));
        for i in 0..n {
            x_tri.add_triplet(i, 0, 1.0);
        }
        let x = x_tri.to_csc();

        // Z = genotype incidence (n x 3)
        let mut z_tri = sprs::TriMat::new((n, q));
        for i in 0..n {
            z_tri.add_triplet(i, genotype_assignment[i], 1.0);
        }
        let z = z_tri.to_csc();

        // Run 3-fold CV
        let cv = CrossValidator::new(3).seed(42);
        let result = cv.run(&y, &x, &z, None).unwrap();

        // With a well-structured dataset and low noise, accuracy should be
        // positive (predictions should correlate with observations)
        assert!(
            result.accuracy > 0.0,
            "Expected positive accuracy, got {}",
            result.accuracy
        );

        // All folds should have results
        assert_eq!(result.folds.len(), 3);

        // Total validation observations should equal n
        let total_val: usize =
            result.folds.iter().map(|f| f.validation_indices.len()).sum();
        assert_eq!(total_val, n);

        // Summary should be non-empty
        let summary = result.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Cross-Validation Results"));
    }

    #[test]
    fn test_cv_with_relationship_matrix() {
        // Same as above but with a non-identity G^{-1} (scaled identity)
        let n = 12;
        let q = 3;
        let genotype_assignment = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let true_g = vec![2.0, 0.0, -2.0];
        let mu = 10.0;
        let errors = vec![
            0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.15, -0.15, 0.05, -0.25, 0.25, -0.05,
        ];

        let y: Vec<f64> = (0..n)
            .map(|i| mu + true_g[genotype_assignment[i]] + errors[i])
            .collect();

        let mut x_tri = sprs::TriMat::new((n, 1));
        for i in 0..n {
            x_tri.add_triplet(i, 0, 1.0);
        }
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((n, q));
        for i in 0..n {
            z_tri.add_triplet(i, genotype_assignment[i], 1.0);
        }
        let z = z_tri.to_csc();

        // G^{-1} = identity (equivalent to no relationship info)
        let ginv = crate::matrix::sparse::sparse_identity(q);

        let cv = CrossValidator::new(3).seed(42);
        let result = cv.run(&y, &x, &z, Some(&ginv)).unwrap();

        assert!(result.accuracy > 0.0);
        assert_eq!(result.folds.len(), 3);
    }

    #[test]
    fn test_loocv_has_n_folds() {
        let n = 8;
        let q = 2;
        let genotype_assignment = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let y: Vec<f64> = (0..n)
            .map(|i| 10.0 + if genotype_assignment[i] == 0 { 1.0 } else { -1.0 })
            .collect();

        let mut x_tri = sprs::TriMat::new((n, 1));
        for i in 0..n {
            x_tri.add_triplet(i, 0, 1.0);
        }
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((n, q));
        for i in 0..n {
            z_tri.add_triplet(i, genotype_assignment[i], 1.0);
        }
        let z = z_tri.to_csc();

        let result = CrossValidator::leave_one_out(&y, &x, &z, None).unwrap();
        assert_eq!(result.folds.len(), n);

        // Each fold should have exactly 1 validation observation
        for fold in &result.folds {
            assert_eq!(fold.validation_indices.len(), 1);
        }
    }

    #[test]
    fn test_cv_too_few_observations() {
        let y = vec![1.0, 2.0];
        let mut x_tri = sprs::TriMat::new((2, 1));
        x_tri.add_triplet(0, 0, 1.0);
        x_tri.add_triplet(1, 0, 1.0);
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((2, 2));
        z_tri.add_triplet(0, 0, 1.0);
        z_tri.add_triplet(1, 1, 1.0);
        let z = z_tri.to_csc();

        // 5 folds with only 2 observations should fail
        let cv = CrossValidator::new(5);
        let result = cv.run(&y, &x, &z, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cv_dimension_mismatch_x() {
        let y = vec![1.0, 2.0, 3.0];
        // X has wrong number of rows
        let mut x_tri = sprs::TriMat::new((2, 1));
        x_tri.add_triplet(0, 0, 1.0);
        x_tri.add_triplet(1, 0, 1.0);
        let x = x_tri.to_csc();

        let mut z_tri = sprs::TriMat::new((3, 2));
        z_tri.add_triplet(0, 0, 1.0);
        z_tri.add_triplet(1, 1, 1.0);
        z_tri.add_triplet(2, 0, 1.0);
        let z = z_tri.to_csc();

        let cv = CrossValidator::new(2);
        let result = cv.run(&y, &x, &z, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_squared_error() {
        let predicted = vec![1.0, 2.0, 3.0];
        let observed = vec![1.5, 2.5, 3.5];
        // MSE = ((0.5)^2 + (0.5)^2 + (0.5)^2) / 3 = 0.75 / 3 = 0.25
        assert_relative_eq!(
            mean_squared_error(&predicted, &observed),
            0.25,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mean_absolute_error() {
        let predicted = vec![1.0, 2.0, 3.0];
        let observed = vec![1.5, 2.5, 3.5];
        // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert_relative_eq!(
            mean_absolute_error(&predicted, &observed),
            0.5,
            epsilon = 1e-10
        );
    }
}
