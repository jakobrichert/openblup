//! Integration test: Fit a simple mixed model and validate results.
//!
//! Model: yield = mu + rep (fixed) + genotype (random, IID) + error
//!
//! Data: A balanced trial with 3 genotypes x 2 reps = 6 observations.
//! This is deliberately simple so we can verify the results by hand.

use approx::assert_relative_eq;
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::model::MixedModelBuilder;
use plant_breeding_lmm_core::variance::Identity;

fn create_test_data() -> DataFrame {
    // Simple balanced trial:
    // Genotype G1: rep1=10, rep2=12  (mean=11)
    // Genotype G2: rep1=8, rep2=10   (mean=9)
    // Genotype G3: rep1=6, rep2=8    (mean=7)
    // Rep effect: rep2 = rep1 + 2
    // Overall mean = 9
    let mut df = DataFrame::new();
    df.add_float_column("yield", vec![10.0, 8.0, 6.0, 12.0, 10.0, 8.0])
        .unwrap();
    df.add_factor_column("genotype", &["G1", "G2", "G3", "G1", "G2", "G3"])
        .unwrap();
    df.add_factor_column("rep", &["R1", "R1", "R1", "R2", "R2", "R2"])
        .unwrap();
    df
}

#[test]
fn test_fit_simple_mixed_model() {
    let df = create_test_data();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("yield")
        .fixed("rep")  // factor levels (R1, R2) absorb the mean — full rank
        .random("genotype", Identity::new(1.0), None)
        .max_iterations(50)
        .convergence(1e-8)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    // Check basic properties
    assert_eq!(result.n_obs, 6);
    assert_eq!(result.n_fixed_params, 2); // 2 rep levels (absorb the mean)
    assert!(result.converged, "REML should converge");
    assert!(result.n_iterations > 0);
    assert!(result.n_iterations <= 50);

    // Variance components should be positive
    for vc in &result.variance_components {
        for (_, val) in &vc.parameters {
            assert!(*val > 0.0, "Variance component should be positive: {}", vc.name);
        }
    }

    // The genotype variance should be meaningful (there's clear genotype effect)
    let geno_var = result.variance_components[0].parameters[0].1;
    let resid_var = result.variance_components[1].parameters[0].1;
    assert!(geno_var > 0.0, "Genotype variance should be > 0");
    assert!(resid_var > 0.0, "Residual variance should be > 0");

    // BLUPs should reflect the genotype ranking: G1 > G2 > G3
    let blups = &result.random_effects[0].effects;
    let g1_blup = blups.iter().find(|e| e.level == "G1").unwrap().estimate;
    let g2_blup = blups.iter().find(|e| e.level == "G2").unwrap().estimate;
    let g3_blup = blups.iter().find(|e| e.level == "G3").unwrap().estimate;

    assert!(
        g1_blup > g2_blup,
        "G1 BLUP ({}) should be > G2 BLUP ({})",
        g1_blup,
        g2_blup
    );
    assert!(
        g2_blup > g3_blup,
        "G2 BLUP ({}) should be > G3 BLUP ({})",
        g2_blup,
        g3_blup
    );

    // BLUPs should sum to approximately zero (balanced data)
    let blup_sum: f64 = blups.iter().map(|e| e.estimate).sum();
    assert_relative_eq!(blup_sum, 0.0, epsilon = 0.1);

    // BLUPs should be shrunk toward zero compared to raw genotype means
    // Raw deviation of G1 from overall mean: 11 - 9 = 2
    assert!(
        g1_blup.abs() < 2.0,
        "BLUP should be shrunk: {} should be < 2.0",
        g1_blup.abs()
    );

    // Residuals should sum to approximately zero
    let resid_sum: f64 = result.residuals.iter().sum();
    assert_relative_eq!(resid_sum, 0.0, epsilon = 0.5);

    // Log-likelihood should be finite
    assert!(result.log_likelihood.is_finite());

    // AIC and BIC should be computable
    let aic = result.aic();
    let bic = result.bic();
    assert!(aic.is_finite());
    assert!(bic.is_finite());

    // Print summary for manual inspection
    println!("{}", result.summary());
}

#[test]
fn test_fit_intercept_only() {
    // Simplest possible model: y = mu + e
    let mut df = DataFrame::new();
    df.add_float_column("y", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    df.add_factor_column("group", &["A", "A", "B", "B", "B"])
        .unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("mu")
        .random("group", Identity::new(1.0), None)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();
    assert!(result.converged || result.n_iterations > 0);
    assert_eq!(result.n_obs, 5);
    println!("{}", result.summary());
}

#[test]
fn test_fit_larger_dataset() {
    // Simulate a larger dataset with 5 genotypes x 3 reps = 15 observations
    let genotypes = vec!["G1", "G2", "G3", "G4", "G5"];
    let reps = vec!["R1", "R2", "R3"];

    let mut yields = Vec::new();
    let mut geno_col = Vec::new();
    let mut rep_col = Vec::new();

    // True genotype effects: 2, 1, 0, -1, -2
    // True rep effects: -1, 0, 1
    let geno_effects = [2.0, 1.0, 0.0, -1.0, -2.0];
    let rep_effects = [-1.0, 0.0, 1.0];
    let mu = 10.0;
    // Small residuals for testing
    let residuals = [0.1, -0.2, 0.15, -0.1, 0.05, 0.2, -0.15, 0.1, -0.05, 0.12, -0.08, 0.03, 0.07, -0.11, 0.06];

    let mut idx = 0;
    for (r, rep) in reps.iter().enumerate() {
        for (g, geno) in genotypes.iter().enumerate() {
            yields.push(mu + geno_effects[g] + rep_effects[r] + residuals[idx]);
            geno_col.push(*geno);
            rep_col.push(*rep);
            idx += 1;
        }
    }

    let mut df = DataFrame::new();
    df.add_float_column("yield", yields).unwrap();
    df.add_factor_column("genotype", &geno_col).unwrap();
    df.add_factor_column("rep", &rep_col).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("yield")
        .fixed("rep")
        .random("genotype", Identity::new(1.0), None)
        .max_iterations(100)
        .convergence(1e-8)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    assert!(result.converged, "Should converge on well-behaved data");

    // Check genotype ranking is preserved
    let blups = &result.random_effects[0].effects;
    let g1 = blups.iter().find(|e| e.level == "G1").unwrap().estimate;
    let g5 = blups.iter().find(|e| e.level == "G5").unwrap().estimate;
    assert!(g1 > g5, "G1 ({}) should have higher BLUP than G5 ({})", g1, g5);

    println!("{}", result.summary());
}
