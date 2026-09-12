use nalgebra::DMatrix;

use crate::lmm::FitResult;

/// Result of a Wald F-test for a single fixed effect term.
#[derive(Debug, Clone)]
pub struct WaldTest {
    /// Name of the fixed effect term being tested.
    pub term: String,
    /// Wald F-statistic: F = beta_hat' (L C^{-1}_{bb} L')^{-1} beta_hat / rank(L).
    pub f_statistic: f64,
    /// Numerator degrees of freedom (number of levels tested for this term).
    pub num_df: usize,
    /// Denominator degrees of freedom (containment method: n - rank(X)).
    pub den_df: f64,
    /// Approximate p-value from the F distribution.
    pub p_value: f64,
}

/// Compute Wald F-tests for each fixed effect term.
///
/// For each term in the fixed effects formula, a Wald F-test is computed:
///
/// ```text
/// F = beta_hat' (L C^{-1}_{bb} L')^{-1} beta_hat / rank(L)
/// ```
///
/// where `L` is a contrast matrix that selects the columns of `X` corresponding
/// to that term, `C^{-1}_{bb}` is the fixed-effects block of the inverse of the
/// MME coefficient matrix, and `beta_hat` are the estimated fixed effects.
///
/// For multi-level terms the full fixed-effects covariance block
/// (`FitResult::fixed_cov`) is used, so correlated estimates (e.g. treatment
/// contrasts against a common reference level) are handled exactly. If the
/// covariance block is not available the test falls back to summing the
/// independent squared t-statistics, which is exact only for orthogonal
/// designs.
///
/// The denominator degrees of freedom use the simple containment method:
/// `den_df = n - rank(X)`, which is appropriate for balanced designs and
/// provides a conservative test for unbalanced designs. See
/// [`wald_tests_satterthwaite`](crate::diagnostics::wald_tests_satterthwaite)
/// for the Satterthwaite approximation.
///
/// # Arguments
///
/// * `result` - A fitted mixed model result containing fixed effects, SEs, and model dimensions.
///
/// # Returns
///
/// A vector of `WaldTest` results, one per fixed effect term. Terms with a
/// single level yield an F-test equivalent to a t-test. The intercept term
/// ("mu") is included if present.
pub fn wald_tests(result: &FitResult) -> Vec<WaldTest> {
    if result.fixed_effects.is_empty() {
        return Vec::new();
    }

    // Group fixed effects by term name
    let mut term_order: Vec<String> = Vec::new();
    let mut term_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, ef) in result.fixed_effects.iter().enumerate() {
        if !term_indices.contains_key(&ef.term) {
            term_order.push(ef.term.clone());
        }
        term_indices.entry(ef.term.clone()).or_default().push(i);
    }

    // Denominator df: containment method n - rank(X)
    let den_df = (result.n_obs - result.n_fixed_params) as f64;

    let mut tests = Vec::new();

    for term in &term_order {
        let indices = &term_indices[term];
        let num_df = indices.len();

        if num_df == 0 {
            continue;
        }

        if num_df == 1 {
            // Single-parameter term: F = (beta_hat / SE)^2
            let idx = indices[0];
            let ef = &result.fixed_effects[idx];
            let se = ef.se;
            if se > 0.0 {
                let t = ef.estimate / se;
                let f_stat = t * t;
                let p_value = f_distribution_sf(f_stat, 1.0, den_df);
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: f_stat,
                    num_df: 1,
                    den_df,
                    p_value,
                });
            } else {
                tests.push(WaldTest {
                    term: term.clone(),
                    f_statistic: 0.0,
                    num_df: 1,
                    den_df,
                    p_value: 1.0,
                });
            }
        } else {
            // Multi-parameter term: general Wald F-test
            //   F = beta' Var(beta)^- beta / rank(Var(beta))
            // using the sub-block of the fixed-effects covariance matrix.
            let beta: Vec<f64> = indices
                .iter()
                .map(|&idx| result.fixed_effects[idx].estimate)
                .collect();

            let (f_stat, rank) = match fixed_cov_block(result, indices) {
                Some(cov) => wald_f_general(&beta, &cov),
                None => {
                    // Fallback without covariances: sum of independent t².
                    let f: f64 = indices
                        .iter()
                        .map(|&idx| {
                            let ef = &result.fixed_effects[idx];
                            if ef.se > 0.0 {
                                (ef.estimate / ef.se).powi(2)
                            } else {
                                0.0
                            }
                        })
                        .sum::<f64>()
                        / num_df as f64;
                    (f, num_df)
                }
            };

            let num_df = rank.max(1);
            let p_value = if rank == 0 {
                1.0
            } else {
                f_distribution_sf(f_stat, num_df as f64, den_df)
            };

            tests.push(WaldTest {
                term: term.clone(),
                f_statistic: f_stat,
                num_df,
                den_df,
                p_value,
            });
        }
    }

    tests
}

/// Extract the covariance sub-block of `FitResult::fixed_cov` for the given
/// effect indices, or `None` if the covariance matrix is unavailable.
fn fixed_cov_block(result: &FitResult, indices: &[usize]) -> Option<DMatrix<f64>> {
    let p = result.fixed_effects.len();
    if result.fixed_cov.len() != p || result.fixed_cov.iter().any(|row| row.len() != p) {
        return None;
    }
    let k = indices.len();
    Some(DMatrix::from_fn(k, k, |a, b| {
        result.fixed_cov[indices[a]][indices[b]]
    }))
}

/// General Wald F-statistic for a vector of estimates with covariance `cov`:
///
/// ```text
/// F = beta' cov^- beta / rank(cov)
/// ```
///
/// `cov^-` is the Moore-Penrose pseudo-inverse obtained from the symmetric
/// eigendecomposition, so (near-)singular covariance blocks (e.g. from a
/// rank-deficient design) are handled by testing only the estimable
/// directions. Returns `(F, rank)`; `rank` is 0 (and `F` is 0) when the
/// covariance block is entirely singular.
pub(crate) fn wald_f_general(beta: &[f64], cov: &DMatrix<f64>) -> (f64, usize) {
    let k = beta.len();
    if k == 0 || cov.nrows() != k || cov.ncols() != k {
        return (0.0, 0);
    }
    // Symmetrise to guard against round-off asymmetry.
    let sym = (cov + cov.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let max_eig = eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    if max_eig <= 0.0 {
        return (0.0, 0);
    }
    let tol = max_eig * 1e-10;
    let b = nalgebra::DVector::from_column_slice(beta);
    let mut quad = 0.0;
    let mut rank = 0;
    for (i, &lambda) in eig.eigenvalues.iter().enumerate() {
        if lambda > tol {
            let proj = eig.eigenvectors.column(i).dot(&b);
            quad += proj * proj / lambda;
            rank += 1;
        }
    }
    if rank == 0 {
        (0.0, 0)
    } else {
        (quad / rank as f64, rank)
    }
}

/// Survival function (1 - CDF) of the F distribution.
///
/// Uses the regularised incomplete beta function relationship:
/// P(F > x | d1, d2) = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
///
/// The incomplete beta function is computed using a continued fraction
/// expansion (Lentz's algorithm).
pub(crate) fn f_distribution_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return 1.0;
    }
    if x.is_nan() || d1.is_nan() || d2.is_nan() {
        return 1.0;
    }

    let z = d1 * x / (d1 * x + d2);
    let a = d1 / 2.0;
    let b = d2 / 2.0;

    // P(F > x) = I_z(a, b) = 1 - I_z(a, b) ... wait.
    // Actually: P(F <= x) = I_z(a, b) where z = d1*x/(d1*x+d2)
    // So P(F > x) = 1 - I_z(a, b)
    1.0 - regularized_incomplete_beta(z, a, b)
}

/// Regularized incomplete beta function I_x(a, b).
///
/// Uses the continued fraction expansion (Lentz's method) for numerical stability.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    // Log of the beta function coefficient:
    // x^a * (1-x)^b / (a * B(a,b))
    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();

    let prefix = log_prefix.exp();

    // Continued fraction (Lentz's algorithm)
    let cf = beta_cf(x, a, b);

    prefix * cf
}

/// Log of the beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    // Coefficients for the Lanczos approximation (g=7, n=9)
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + 7.5; // g + 0.5
    let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

    log_sqrt_2pi + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Continued fraction expansion for the incomplete beta function.
///
/// Uses the modified Lentz's method with the standard DLMF 8.17.22 recurrence.
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..=max_iter {
        let m_f64 = m as f64;

        // Even step: d_{2m}
        let numerator_even =
            m_f64 * (b - m_f64) * x / ((a + 2.0 * m_f64 - 1.0) * (a + 2.0 * m_f64));
        d = 1.0 + numerator_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        f *= d * c;

        // Odd step: d_{2m+1}
        let numerator_odd =
            -((a + m_f64) * (a + b + m_f64) * x) / ((a + 2.0 * m_f64) * (a + 2.0 * m_f64 + 1.0));
        d = 1.0 + numerator_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    f
}

/// Format Wald test results as a table string.
pub fn format_wald_tests(tests: &[WaldTest]) -> String {
    let mut s = String::new();
    s.push_str("--- Wald Tests for Fixed Effects ---\n");
    s.push_str(&format!(
        "{:<20} {:>10} {:>8} {:>10} {:>12}\n",
        "Term", "F-stat", "NumDF", "DenDF", "Pr(>F)"
    ));
    s.push_str(&format!("{}\n", "-".repeat(62)));

    for test in tests {
        let significance = if test.p_value < 0.001 {
            "***"
        } else if test.p_value < 0.01 {
            "**"
        } else if test.p_value < 0.05 {
            "*"
        } else if test.p_value < 0.1 {
            "."
        } else {
            ""
        };

        s.push_str(&format!(
            "{:<20} {:>10.4} {:>8} {:>10.1} {:>12.4e} {}\n",
            test.term, test.f_statistic, test.num_df, test.den_df, test.p_value, significance
        ));
    }

    s.push_str("---\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ln_gamma_basic() {
        // Gamma(1) = 1, so ln(Gamma(1)) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // Gamma(2) = 1, so ln(Gamma(2)) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // Gamma(3) = 2, so ln(Gamma(3)) = ln(2)
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
        // Gamma(0.5) = sqrt(pi)
        let expected = 0.5 * std::f64::consts::PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_f_distribution_sf_extreme() {
        // F = 0 => p-value = 1
        assert!((f_distribution_sf(0.0, 1.0, 10.0) - 1.0).abs() < 1e-10);
        // Very large F => p-value ~ 0
        assert!(f_distribution_sf(1000.0, 1.0, 10.0) < 1e-6);
    }

    #[test]
    fn test_f_distribution_sf_known_values() {
        // F(1, 10) at F=4.96 should give approximately p=0.05
        // (critical value for alpha=0.05 is 4.9646)
        let p = f_distribution_sf(4.96, 1.0, 10.0);
        assert!(
            (p - 0.05).abs() < 0.005,
            "Expected p ~0.05 for F(1,10)=4.96, got {}",
            p
        );
    }

    #[test]
    fn test_f_distribution_sf_f_2_20() {
        // F(2, 20) critical value at 0.05 is approximately 3.49
        let p = f_distribution_sf(3.49, 2.0, 20.0);
        assert!(
            (p - 0.05).abs() < 0.01,
            "Expected p ~0.05 for F(2,20)=3.49, got {}",
            p
        );
    }

    #[test]
    fn test_wald_tests_empty() {
        use crate::lmm::FitResult;

        let result = FitResult {
            variance_components: vec![],
            fixed_effects: vec![],
            random_effects: vec![],
            log_likelihood: 0.0,
            n_iterations: 0,
            converged: true,
            history: vec![],
            variance_se: vec![],
            residuals: vec![],
            fixed_cov: vec![],
            at_boundary: vec![],
            n_obs: 10,
            n_fixed_params: 0,
            n_variance_params: 0,
            c_inv: None,
            fixed_cov_derivatives: Vec::new(),
            kenward_roger_terms: None,
            ai_matrix: None,
            n_random_per_term: vec![],
        };

        let tests = wald_tests(&result);
        assert!(tests.is_empty());
    }

    #[test]
    fn test_wald_tests_single_term() {
        use crate::lmm::{FitResult, NamedEffect, VarianceEstimate};

        let result = FitResult {
            variance_components: vec![VarianceEstimate {
                name: "residual".to_string(),
                structure: "Identity".to_string(),
                parameters: vec![("sigma2".to_string(), 1.0)],
                se: vec![],
                at_boundary: vec![],
            }],
            fixed_effects: vec![NamedEffect {
                term: "mu".to_string(),
                level: "intercept".to_string(),
                estimate: 5.0,
                se: 0.5,
            }],
            random_effects: vec![],
            log_likelihood: -10.0,
            n_iterations: 5,
            converged: true,
            history: vec![],
            variance_se: vec![0.1],
            residuals: vec![],
            fixed_cov: vec![],
            at_boundary: vec![],
            n_obs: 20,
            n_fixed_params: 1,
            n_variance_params: 1,
            c_inv: None,
            fixed_cov_derivatives: Vec::new(),
            kenward_roger_terms: None,
            ai_matrix: None,
            n_random_per_term: vec![],
        };

        let tests = wald_tests(&result);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);
        // F = (5.0/0.5)^2 = 100.0
        assert!((tests[0].f_statistic - 100.0).abs() < 1e-10);
        // den_df = 20 - 1 = 19
        assert!((tests[0].den_df - 19.0).abs() < 1e-10);
        // With F=100, df1=1, df2=19, p should be very small
        assert!(tests[0].p_value < 0.001);
    }

    #[test]
    fn test_wald_tests_multi_level_term() {
        use crate::lmm::{FitResult, NamedEffect, VarianceEstimate};

        let result = FitResult {
            variance_components: vec![VarianceEstimate {
                name: "residual".to_string(),
                structure: "Identity".to_string(),
                parameters: vec![("sigma2".to_string(), 1.0)],
                se: vec![],
                at_boundary: vec![],
            }],
            fixed_effects: vec![
                NamedEffect {
                    term: "mu".to_string(),
                    level: "intercept".to_string(),
                    estimate: 5.0,
                    se: 0.5,
                },
                NamedEffect {
                    term: "trt".to_string(),
                    level: "A".to_string(),
                    estimate: 2.0,
                    se: 0.8,
                },
                NamedEffect {
                    term: "trt".to_string(),
                    level: "B".to_string(),
                    estimate: -1.0,
                    se: 0.8,
                },
            ],
            random_effects: vec![],
            log_likelihood: -10.0,
            n_iterations: 5,
            converged: true,
            history: vec![],
            variance_se: vec![0.1],
            residuals: vec![],
            fixed_cov: vec![],
            at_boundary: vec![],
            n_obs: 30,
            n_fixed_params: 3,
            n_variance_params: 1,
            c_inv: None,
            fixed_cov_derivatives: Vec::new(),
            kenward_roger_terms: None,
            ai_matrix: None,
            n_random_per_term: vec![],
        };

        let tests = wald_tests(&result);
        assert_eq!(tests.len(), 2);

        // "mu" test
        assert_eq!(tests[0].term, "mu");
        assert_eq!(tests[0].num_df, 1);

        // "trt" test
        assert_eq!(tests[1].term, "trt");
        assert_eq!(tests[1].num_df, 2);
        assert!((tests[1].den_df - 27.0).abs() < 1e-10); // 30 - 3 = 27
    }

    #[test]
    fn test_wald_f_general_diagonal_matches_sum_of_t2() {
        let beta = [2.0, -1.0];
        let cov = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 1.0]);
        let (f, rank) = wald_f_general(&beta, &cov);
        // (2/2)^2 + (-1/1)^2 = 2, divided by rank 2
        assert!((f - 1.0).abs() < 1e-12);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_wald_f_general_uses_covariances() {
        // Strongly correlated estimates: the quadratic form differs from the
        // diagonal approximation.
        let beta = [1.0, 1.0];
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.9, 0.9, 1.0]);
        let (f, rank) = wald_f_general(&beta, &cov);
        // cov^-1 = 1/(1-0.81) * [[1,-0.9],[-0.9,1]]; b'cov^-1 b = 2*(1-0.9)/0.19
        let expected = 2.0 * 0.1 / 0.19 / 2.0;
        assert!((f - expected).abs() < 1e-10, "f = {}", f);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_wald_f_general_singular_block_reduces_rank() {
        let beta = [1.0, 1.0];
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let (f, rank) = wald_f_general(&beta, &cov);
        assert_eq!(rank, 1);
        // Only the direction (1,1)/sqrt(2) with eigenvalue 2: proj = sqrt(2)
        assert!((f - 1.0).abs() < 1e-10);
        let (f0, rank0) = wald_f_general(&beta, &DMatrix::zeros(2, 2));
        assert_eq!(rank0, 0);
        assert_eq!(f0, 0.0);
    }

    #[test]
    fn test_wald_tests_multi_level_term_uses_fixed_cov() {
        use crate::lmm::NamedEffect;
        let cov = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.9],
            vec![0.0, 0.9, 1.0],
        ];
        let result = FitResult {
            variance_components: vec![],
            fixed_effects: vec![
                NamedEffect {
                    term: "mu".into(),
                    level: "intercept".into(),
                    estimate: 5.0,
                    se: 1.0,
                },
                NamedEffect {
                    term: "rep".into(),
                    level: "R2".into(),
                    estimate: 1.0,
                    se: 1.0,
                },
                NamedEffect {
                    term: "rep".into(),
                    level: "R3".into(),
                    estimate: 1.0,
                    se: 1.0,
                },
            ],
            random_effects: vec![],
            log_likelihood: 0.0,
            n_iterations: 1,
            converged: true,
            history: vec![],
            variance_se: vec![],
            residuals: vec![],
            fixed_cov: cov,
            at_boundary: vec![],
            n_obs: 20,
            n_fixed_params: 3,
            n_variance_params: 1,
            c_inv: None,
            fixed_cov_derivatives: Vec::new(),
            kenward_roger_terms: None,
            ai_matrix: None,
            n_random_per_term: vec![],
        };
        let tests = wald_tests(&result);
        assert_eq!(tests.len(), 2);
        let rep = tests.iter().find(|t| t.term == "rep").unwrap();
        assert_eq!(rep.num_df, 2);
        let expected = 2.0 * 0.1 / 0.19 / 2.0;
        assert!((rep.f_statistic - expected).abs() < 1e-10);
        assert!(rep.p_value > 0.0 && rep.p_value < 1.0);
    }

    #[test]
    fn test_format_wald_tests() {
        let tests = vec![
            WaldTest {
                term: "mu".to_string(),
                f_statistic: 100.0,
                num_df: 1,
                den_df: 19.0,
                p_value: 1e-8,
            },
            WaldTest {
                term: "treatment".to_string(),
                f_statistic: 3.5,
                num_df: 2,
                den_df: 19.0,
                p_value: 0.042,
            },
        ];

        let output = format_wald_tests(&tests);
        assert!(output.contains("Wald Tests"));
        assert!(output.contains("mu"));
        assert!(output.contains("treatment"));
        assert!(output.contains("***")); // mu should be highly significant
        assert!(output.contains("*")); // treatment should be significant at 0.05
    }
}
