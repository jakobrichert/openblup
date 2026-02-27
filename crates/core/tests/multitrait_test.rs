//! Integration test: Multi-trait mixed model with 2 correlated traits.
//!
//! Simulates a balanced trial with 5 genotypes x 3 reps, where two traits
//! (yield and height) share a positive genetic correlation (~0.7).
//!
//! Model per trait: y = rep (fixed) + genotype (random) + error

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::model::MultiTraitModelBuilder;

/// Create a simulated two-trait dataset with known genetic correlation.
///
/// True parameters:
/// - 5 genotypes, 3 reps
/// - Genetic effects: correlated across traits (r_g ~ 0.7)
/// - Rep effects: trait 1 [0, 1, 2], trait 2 [0, 2, 4]
/// - Residual variance: small
fn create_two_trait_data() -> DataFrame {
    let genotypes = ["G1", "G2", "G3", "G4", "G5"];
    let reps = ["R1", "R2", "R3"];

    // True genotype effects for trait 1 (yield-like)
    let geno_eff_t1 = [3.0, 1.5, 0.0, -1.5, -3.0];
    // True genotype effects for trait 2 (height-like), correlated with t1
    let geno_eff_t2 = [6.0, 3.5, 0.5, -2.0, -5.0];

    // True rep effects
    let rep_eff_t1 = [0.0, 1.0, 2.0];
    let rep_eff_t2 = [0.0, 2.0, 4.0];

    let mu_t1 = 20.0;
    let mu_t2 = 80.0;

    // Small, deterministic "residuals" for reproducibility
    let residuals_t1 = [
        0.1, -0.2, 0.15, -0.1, 0.05,   // rep 1
        0.2, -0.15, 0.1, -0.05, 0.12,   // rep 2
        -0.08, 0.03, 0.07, -0.11, 0.06, // rep 3
    ];
    let residuals_t2 = [
        0.3, -0.4, 0.25, -0.2, 0.1,    // rep 1
        0.35, -0.3, 0.2, -0.15, 0.25,   // rep 2
        -0.15, 0.1, 0.15, -0.25, 0.12,  // rep 3
    ];

    let n = genotypes.len() * reps.len(); // 15
    let mut yields = Vec::with_capacity(n);
    let mut heights = Vec::with_capacity(n);
    let mut geno_col: Vec<&str> = Vec::with_capacity(n);
    let mut rep_col: Vec<&str> = Vec::with_capacity(n);

    let mut idx = 0;
    for (r, rep) in reps.iter().enumerate() {
        for (g, geno) in genotypes.iter().enumerate() {
            yields.push(mu_t1 + rep_eff_t1[r] + geno_eff_t1[g] + residuals_t1[idx]);
            heights.push(mu_t2 + rep_eff_t2[r] + geno_eff_t2[g] + residuals_t2[idx]);
            geno_col.push(geno);
            rep_col.push(rep);
            idx += 1;
        }
    }

    let mut df = DataFrame::new();
    df.add_float_column("yield", yields).unwrap();
    df.add_float_column("height", heights).unwrap();
    df.add_factor_column("genotype", &geno_col).unwrap();
    df.add_factor_column("rep", &rep_col).unwrap();
    df
}

#[test]
fn test_two_trait_model_converges() {
    let df = create_two_trait_data();

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    assert_eq!(model.n_traits, 2);
    assert_eq!(model.n_obs, 15);
    assert_eq!(model.y.len(), 30); // 2 traits * 15 obs

    let result = model.fit_reml().unwrap();

    // Model should converge
    assert!(
        result.converged,
        "Multi-trait REML should converge. Iterations: {}",
        result.n_iterations
    );

    // Print summary for inspection
    println!("{}", result.summary);

    // G0 should be positive definite (it was bent if needed)
    assert!(result.g0.clone().cholesky().is_some(), "G0 must be positive definite");

    // R0 should be positive definite
    assert!(result.r0.clone().cholesky().is_some(), "R0 must be positive definite");

    // Genetic variances should be positive
    assert!(result.g0[(0, 0)] > 0.0, "Genetic variance for trait 1 should be > 0");
    assert!(result.g0[(1, 1)] > 0.0, "Genetic variance for trait 2 should be > 0");

    // Residual variances should be positive
    assert!(result.r0[(0, 0)] > 0.0, "Residual variance for trait 1 should be > 0");
    assert!(result.r0[(1, 1)] > 0.0, "Residual variance for trait 2 should be > 0");

    // Log-likelihood should be finite
    assert!(result.log_likelihood.is_finite(), "Log-likelihood should be finite");
}

#[test]
fn test_genetic_correlation_positive() {
    let df = create_two_trait_data();

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    // The true genetic correlation is positive (~0.7ish based on our simulated effects).
    // The estimated correlation should also be positive and moderate.
    let r_g = result.genetic_correlations[(0, 1)];
    println!("Estimated genetic correlation: {:.4}", r_g);
    assert!(
        r_g > 0.0,
        "Genetic correlation should be positive, got {}",
        r_g
    );

    // Diagonal of correlation matrix should be 1.0
    assert_relative_eq!(result.genetic_correlations[(0, 0)], 1.0, epsilon = 1e-6);
    assert_relative_eq!(result.genetic_correlations[(1, 1)], 1.0, epsilon = 1e-6);

    // Correlation should be between -1 and 1
    assert!(r_g >= -1.0 && r_g <= 1.0, "Correlation out of bounds: {}", r_g);
}

#[test]
fn test_blup_ranking_within_trait() {
    let df = create_two_trait_data();

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    // Trait 1 (yield): G1 has highest effect, G5 has lowest
    // BLUPs should maintain this ranking
    let blups_t1 = &result.random_effects[0][0]; // trait 0, random term 0
    let levels = &model.random_level_names[0];
    let g1_idx = levels.iter().position(|l| l == "G1").unwrap();
    let g5_idx = levels.iter().position(|l| l == "G5").unwrap();

    println!("Trait 1 BLUPs:");
    for (i, level) in levels.iter().enumerate() {
        println!("  {}: {:.4}", level, blups_t1[i]);
    }

    assert!(
        blups_t1[g1_idx] > blups_t1[g5_idx],
        "G1 BLUP ({:.4}) should be > G5 BLUP ({:.4}) for trait 1",
        blups_t1[g1_idx],
        blups_t1[g5_idx]
    );

    // Trait 2 (height): same ranking expected (correlated traits)
    let blups_t2 = &result.random_effects[1][0]; // trait 1, random term 0
    println!("Trait 2 BLUPs:");
    for (i, level) in levels.iter().enumerate() {
        println!("  {}: {:.4}", level, blups_t2[i]);
    }

    assert!(
        blups_t2[g1_idx] > blups_t2[g5_idx],
        "G1 BLUP ({:.4}) should be > G5 BLUP ({:.4}) for trait 2",
        blups_t2[g1_idx],
        blups_t2[g5_idx]
    );
}

#[test]
fn test_two_trait_with_custom_starting_values() {
    let df = create_two_trait_data();

    // Provide reasonable starting values for G0 and R0
    let g0 = DMatrix::from_row_slice(2, 2, &[4.0, 2.0, 2.0, 8.0]);
    let r0 = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.1, 1.0]);

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .g0(g0)
        .r0(r0)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    assert!(result.converged, "Should converge with custom starting values");
    assert!(result.g0[(0, 0)] > 0.0);
    assert!(result.r0[(0, 0)] > 0.0);
    println!("{}", result.summary);
}

#[test]
fn test_fixed_effects_per_trait() {
    let df = create_two_trait_data();

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    // Should have fixed effects for each trait
    assert_eq!(result.fixed_effects.len(), 2, "Should have fixed effects for 2 traits");

    // Each trait should have rep effects (3 levels)
    assert_eq!(
        result.fixed_effects[0].len(),
        3,
        "Trait 1 should have 3 rep effects"
    );
    assert_eq!(
        result.fixed_effects[1].len(),
        3,
        "Trait 2 should have 3 rep effects"
    );

    // Print fixed effects
    for (t, effects) in result.fixed_effects.iter().enumerate() {
        println!("Trait {} fixed effects:", t + 1);
        for ef in effects {
            println!("  {}.{}: {:.4}", ef.term, ef.level, ef.estimate);
        }
    }
}

#[test]
fn test_genetic_variance_larger_than_residual_for_well_separated_genotypes() {
    // In our simulated data, genotype effects are large relative to residuals.
    // So the genetic variance should be substantially larger than the residual
    // variance for at least one trait.
    let df = create_two_trait_data();

    let mut model = MultiTraitModelBuilder::new()
        .data(&df)
        .traits(&["yield", "height"])
        .fixed("rep")
        .random("genotype", None)
        .max_iterations(200)
        .convergence(1e-6)
        .build()
        .unwrap();

    let result = model.fit_reml().unwrap();

    println!("G0 diagonal: [{:.4}, {:.4}]", result.g0[(0, 0)], result.g0[(1, 1)]);
    println!("R0 diagonal: [{:.4}, {:.4}]", result.r0[(0, 0)], result.r0[(1, 1)]);

    // At least one trait should have genetic variance > residual variance
    let geno_var_dominates = result.g0[(0, 0)] > result.r0[(0, 0)]
        || result.g0[(1, 1)] > result.r0[(1, 1)];
    assert!(
        geno_var_dominates,
        "Genetic variance should dominate residual for at least one trait"
    );
}
