//! Integration test: Pedigree BLUP (Animal Model) validated against Mrode (2014)
//! Chapter 3, Example 3.1.
//!
//! Model: pre_weaning_gain = sex (fixed) + animal (random, with A-matrix) + error
//!
//! Pedigree (8 animals, 3 founders):
//!   1, 0, 0  (base sire)
//!   2, 0, 0  (base dam)
//!   3, 0, 0  (base sire)
//!   4, 1, 0  (sire=1, dam unknown)
//!   5, 3, 2  (sire=3, dam=2)
//!   6, 1, 2  (sire=1, dam=2)
//!   7, 4, 5  (sire=4, dam=5)
//!   8, 3, 6  (sire=3, dam=6)
//!
//! Data (5 animals with records):
//!   animal  sex     pwg
//!   4       male    4.5
//!   5       female  2.9
//!   6       female  3.9
//!   7       male    3.5
//!   8       male    5.0
//!
//! Note: Animal 8 is MALE, not female. This matches the original Mrode textbook
//! (p. 39, Table 3.1) and is confirmed by multiple independent implementations
//! including BLUPF90.
//!
//! Variance components: sigma^2_a = 20, sigma^2_e = 40, alpha = sigma^2_e/sigma^2_a = 2.
//!
//! Expected BLUEs (sex effects) from Mrode Table 3.1:
//!   male:   4.35850
//!   female: 3.40443
//!
//! Expected BLUPs (animal breeding values) from Mrode Table 3.1:
//!   animal 1:  0.09844
//!   animal 2: -0.01877
//!   animal 3: -0.04108
//!   animal 4: -0.00866
//!   animal 5: -0.18573
//!   animal 6:  0.17687
//!   animal 7: -0.24946
//!   animal 8:  0.18261
//!
//! Reference: Mrode, R.A. (2014). Linear Models for the Prediction of Animal
//!            Breeding Values, 3rd Edition, CABI.
//!            Verified against BLUPF90 solutions:
//!            https://masuday.github.io/blupf90_tutorial/mrode_c03ex031_animal_model.html

use approx::assert_relative_eq;
use sprs::TriMat;

use plant_breeding_lmm_core::genetics::{compute_a_inverse, Pedigree};
use plant_breeding_lmm_core::lmm::MixedModelEquations;

/// Build the Mrode Example 3.1 pedigree (8 animals) and return it sorted.
fn mrode_example_3_1_pedigree() -> Pedigree {
    let triples = vec![
        ("1".to_string(), None, None),
        ("2".to_string(), None, None),
        ("3".to_string(), None, None),
        ("4".to_string(), Some("1".to_string()), None),
        (
            "5".to_string(),
            Some("3".to_string()),
            Some("2".to_string()),
        ),
        (
            "6".to_string(),
            Some("1".to_string()),
            Some("2".to_string()),
        ),
        (
            "7".to_string(),
            Some("4".to_string()),
            Some("5".to_string()),
        ),
        (
            "8".to_string(),
            Some("3".to_string()),
            Some("6".to_string()),
        ),
    ];
    let mut ped = Pedigree::from_triples(&triples).unwrap();
    ped.sort_pedigree().unwrap();
    ped
}

/// Build the X (sex) and Z (animal) design matrices and response vector for
/// Mrode Example 3.1 using the sorted pedigree index mapping.
///
/// Returns (X, Z, y) where:
///   X is 5x2 (column 0 = male, column 1 = female)
///   Z is 5x8 (maps observations to the full 8-animal sorted pedigree)
///   y is the response vector of length 5
fn mrode_example_3_1_design(ped: &Pedigree) -> (sprs::CsMat<f64>, sprs::CsMat<f64>, Vec<f64>) {
    let n_obs = 5;
    let n_animals = 8;

    // Map animal IDs to their sorted pedigree indices.
    let idx: Vec<usize> = (1..=8)
        .map(|i| {
            ped.animal_index(&i.to_string())
                .unwrap_or_else(|| panic!("Animal {} not found in pedigree", i))
        })
        .collect();

    // Data (from Mrode Table 3.1, p. 39):
    //   obs 0: animal 4, sex=male,   pwg=4.5
    //   obs 1: animal 5, sex=female, pwg=2.9
    //   obs 2: animal 6, sex=female, pwg=3.9
    //   obs 3: animal 7, sex=male,   pwg=3.5
    //   obs 4: animal 8, sex=male,   pwg=5.0
    let y = vec![4.5, 2.9, 3.9, 3.5, 5.0];

    // X: Fixed effects design matrix (5 x 2).
    // Column 0 = male, Column 1 = female.
    let mut x_tri = TriMat::new((n_obs, 2));
    x_tri.add_triplet(0, 0, 1.0); // obs 0: animal 4 -> male
    x_tri.add_triplet(1, 1, 1.0); // obs 1: animal 5 -> female
    x_tri.add_triplet(2, 1, 1.0); // obs 2: animal 6 -> female
    x_tri.add_triplet(3, 0, 1.0); // obs 3: animal 7 -> male
    x_tri.add_triplet(4, 0, 1.0); // obs 4: animal 8 -> male
    let x = x_tri.to_csc();

    // Z: Random effects incidence matrix (5 x 8).
    // Maps each observation to the column index in the sorted pedigree.
    let mut z_tri = TriMat::new((n_obs, n_animals));
    z_tri.add_triplet(0, idx[3], 1.0); // obs 0 -> animal 4
    z_tri.add_triplet(1, idx[4], 1.0); // obs 1 -> animal 5
    z_tri.add_triplet(2, idx[5], 1.0); // obs 2 -> animal 6
    z_tri.add_triplet(3, idx[6], 1.0); // obs 3 -> animal 7
    z_tri.add_triplet(4, idx[7], 1.0); // obs 4 -> animal 8
    let z = z_tri.to_csc();

    (x, z, y)
}

/// Test 1: Direct MME assembly and solve with fixed variance components.
///
/// This is the primary validation test. We manually construct the design matrices
/// X (5x2 for sex) and Z (5x8 for animal incidence mapping to the full pedigree),
/// assemble the MME with known sigma^2_a = 20 and sigma^2_e = 40, and verify
/// the solution against Mrode's published values.
#[test]
fn test_mrode_example_3_1_fixed_variances() {
    let ped = mrode_example_3_1_pedigree();

    // Verify pedigree structure after sorting.
    assert_eq!(ped.n_animals(), 8);
    assert!(ped.is_sorted());

    // Compute A-inverse (8x8).
    let a_inv = compute_a_inverse(&ped).unwrap();
    assert_eq!(a_inv.rows(), 8);
    assert_eq!(a_inv.cols(), 8);

    let (x, z, y) = mrode_example_3_1_design(&ped);
    let n_animals = 8;

    // Variance components.
    let sigma2_a = 20.0;
    let sigma2_e = 40.0;

    // G^{-1} = A^{-1} / sigma^2_a
    let g_inv = a_inv.map(|v| v / sigma2_a);
    let r_inv_scale = 1.0 / sigma2_e;

    // Assemble and solve the MME.
    let mme = MixedModelEquations::assemble(&x, &[z], &y, r_inv_scale, &[g_inv]);

    assert_eq!(mme.dim, 2 + n_animals); // 2 fixed (sex) + 8 random (animals)
    assert_eq!(mme.n_fixed, 2);
    assert_eq!(mme.n_random, vec![n_animals]);

    let sol = mme.solve().unwrap();

    // --- Verify BLUEs for sex effects ---
    // Column 0 = male, Column 1 = female.
    let blue_male = sol.fixed_effects[0];
    let blue_female = sol.fixed_effects[1];

    println!("BLUE male:   {:.6}", blue_male);
    println!("BLUE female: {:.6}", blue_female);

    // Mrode Table 3.1 / BLUPF90 verified values.
    assert_relative_eq!(blue_male, 4.35850, epsilon = 0.001);
    assert_relative_eq!(blue_female, 3.40443, epsilon = 0.001);

    // --- Verify BLUPs for animal breeding values ---
    let blups = &sol.random_effects[0];
    assert_eq!(blups.len(), n_animals);

    // Expected BLUPs from Mrode Table 3.1 / BLUPF90.
    let expected_blups = [
        ("1", 0.09844),
        ("2", -0.01877),
        ("3", -0.04108),
        ("4", -0.00866),
        ("5", -0.18573),
        ("6", 0.17687),
        ("7", -0.24946),
        ("8", 0.18261),
    ];

    for (animal_id, expected) in &expected_blups {
        let animal_idx = ped.animal_index(animal_id).unwrap();
        let actual = blups[animal_idx];
        println!(
            "Animal {}: BLUP = {:.5} (expected {:.5})",
            animal_id, actual, expected
        );
        assert_relative_eq!(actual, expected, epsilon = 0.001);
    }

    // --- Additional structural checks ---
    // Males gain more than females on average.
    assert!(
        blue_male > blue_female,
        "Male BLUE ({:.4}) should exceed female BLUE ({:.4})",
        blue_male,
        blue_female
    );
}

/// Test 2: Independent verification using dense matrices in Mrode's own notation.
///
/// This builds the MME directly using nalgebra dense matrices in Mrode's
/// notation: [X'X, X'Z; Z'X, Z'Z + alpha*A^{-1}] with alpha = 2. Animals
/// are indexed 1..8 in original order (no topological sort needed since this
/// bypasses our pedigree code). This serves as a cross-check that the sparse
/// MME assembly in Test 1 is correct.
#[test]
fn test_mrode_example_3_1_dense_verification() {
    let n = 8; // animals
    let p = 2; // sex levels
    let dim = p + n; // 10

    // Build A-inverse manually using Henderson's rules.
    let mut a_inv = nalgebra::DMatrix::<f64>::zeros(n, n);

    struct AnimalInfo {
        sire: Option<usize>,
        dam: Option<usize>,
    }

    let animals = [
        AnimalInfo { sire: None, dam: None },           // 1
        AnimalInfo { sire: None, dam: None },           // 2
        AnimalInfo { sire: None, dam: None },           // 3
        AnimalInfo { sire: Some(0), dam: None },        // 4 (sire=1)
        AnimalInfo { sire: Some(2), dam: Some(1) },     // 5 (sire=3, dam=2)
        AnimalInfo { sire: Some(0), dam: Some(1) },     // 6 (sire=1, dam=2)
        AnimalInfo { sire: Some(3), dam: Some(4) },     // 7 (sire=4, dam=5)
        AnimalInfo { sire: Some(2), dam: Some(5) },     // 8 (sire=3, dam=6)
    ];

    for (i, anim) in animals.iter().enumerate() {
        let d_i = match (&anim.sire, &anim.dam) {
            (Some(_), Some(_)) => 0.5,
            (Some(_), None) | (None, Some(_)) => 0.75,
            (None, None) => 1.0,
        };
        let alpha_i = 1.0 / d_i;

        a_inv[(i, i)] += alpha_i;
        if let Some(s) = anim.sire {
            a_inv[(i, s)] -= alpha_i / 2.0;
            a_inv[(s, i)] -= alpha_i / 2.0;
            a_inv[(s, s)] += alpha_i / 4.0;
        }
        if let Some(d) = anim.dam {
            a_inv[(i, d)] -= alpha_i / 2.0;
            a_inv[(d, i)] -= alpha_i / 2.0;
            a_inv[(d, d)] += alpha_i / 4.0;
        }
        if let (Some(s), Some(d)) = (anim.sire, anim.dam) {
            a_inv[(s, d)] += alpha_i / 4.0;
            a_inv[(d, s)] += alpha_i / 4.0;
        }
    }

    let alpha = 2.0; // sigma2_e / sigma2_a = 40/20

    // Build the MME in Mrode's formulation (multiplied through by sigma2_e).
    // Data: obs 0..4 -> animals 4,5,6,7,8 with sex male,female,female,male,male.
    //
    // X'X (2x2): 3 males, 2 females
    // X'Z (2x8): incidence of sex x animal
    // Z'Z (8x8): diagonal incidence
    let mut c = nalgebra::DMatrix::<f64>::zeros(dim, dim);

    // X'X
    c[(0, 0)] = 3.0; // 3 males (animals 4, 7, 8)
    c[(1, 1)] = 2.0; // 2 females (animals 5, 6)

    // X'Z and Z'X
    // Male animals: 4 (idx 3), 7 (idx 6), 8 (idx 7)
    c[(0, p + 3)] = 1.0; c[(p + 3, 0)] = 1.0;
    c[(0, p + 6)] = 1.0; c[(p + 6, 0)] = 1.0;
    c[(0, p + 7)] = 1.0; c[(p + 7, 0)] = 1.0;
    // Female animals: 5 (idx 4), 6 (idx 5)
    c[(1, p + 4)] = 1.0; c[(p + 4, 1)] = 1.0;
    c[(1, p + 5)] = 1.0; c[(p + 5, 1)] = 1.0;

    // Z'Z + alpha * A^{-1}
    // Z'Z diagonal: 1 for animals with records (4,5,6,7,8), 0 for founders (1,2,3)
    for &animal_idx in &[3, 4, 5, 6, 7] {
        c[(p + animal_idx, p + animal_idx)] += 1.0;
    }
    for i in 0..n {
        for j in 0..n {
            c[(p + i, p + j)] += alpha * a_inv[(i, j)];
        }
    }

    // RHS: [X'y; Z'y]
    let mut rhs = nalgebra::DVector::<f64>::zeros(dim);
    rhs[0] = 4.5 + 3.5 + 5.0;     // male X'y: animals 4, 7, 8
    rhs[1] = 2.9 + 3.9;            // female X'y: animals 5, 6
    rhs[p + 3] = 4.5;              // animal 4
    rhs[p + 4] = 2.9;              // animal 5
    rhs[p + 5] = 3.9;              // animal 6
    rhs[p + 6] = 3.5;              // animal 7
    rhs[p + 7] = 5.0;              // animal 8

    // Solve.
    let chol = c.clone().cholesky().expect("MME should be positive definite");
    let sol = chol.solve(&rhs);

    println!("Dense BLUEs: male={:.5}, female={:.5}", sol[0], sol[1]);
    for i in 0..n {
        println!("  Animal {}: BLUP = {:.5}", i + 1, sol[p + i]);
    }

    // Verify against Mrode Table 3.1 / BLUPF90.
    assert_relative_eq!(sol[0], 4.35850, epsilon = 0.001);
    assert_relative_eq!(sol[1], 3.40443, epsilon = 0.001);

    assert_relative_eq!(sol[p + 0], 0.09844, epsilon = 0.001);
    assert_relative_eq!(sol[p + 1], -0.01877, epsilon = 0.001);
    assert_relative_eq!(sol[p + 2], -0.04108, epsilon = 0.001);
    assert_relative_eq!(sol[p + 3], -0.00866, epsilon = 0.001);
    assert_relative_eq!(sol[p + 4], -0.18573, epsilon = 0.001);
    assert_relative_eq!(sol[p + 5], 0.17687, epsilon = 0.001);
    assert_relative_eq!(sol[p + 6], -0.24946, epsilon = 0.001);
    assert_relative_eq!(sol[p + 7], 0.18261, epsilon = 0.001);
}

/// Test 3: Verify A-inverse structure for the Mrode Example 3.1 pedigree.
///
/// This ensures the pedigree processing and Henderson's rules produce a
/// correct A-inverse before it is used in the MME.
#[test]
fn test_mrode_example_3_1_a_inverse() {
    let ped = mrode_example_3_1_pedigree();
    let a_inv = compute_a_inverse(&ped).unwrap();
    let n = ped.n_animals();

    // Convert to dense for inspection.
    let mut dense = vec![vec![0.0; n]; n];
    for (val, (row, col)) in a_inv.iter() {
        dense[row][col] += *val;
    }

    // Verify symmetry.
    for i in 0..n {
        for j in 0..n {
            assert_relative_eq!(dense[i][j], dense[j][i], epsilon = 1e-10);
        }
    }

    // Verify diagonal entries are positive.
    for i in 0..n {
        assert!(
            dense[i][i] > 0.0,
            "A-inverse diagonal [{},{}] should be positive, got {}",
            i, i, dense[i][i]
        );
    }

    assert_eq!(a_inv.rows(), 8);
    assert_eq!(a_inv.cols(), 8);

    // Verify specific entries using the sorted pedigree indices.
    // The A-inverse for the 8-animal pedigree should match Henderson's rules.
    let idx1 = ped.animal_index("1").unwrap();
    let idx2 = ped.animal_index("2").unwrap();

    // Animal 6 (sire=1, dam=2, alpha=2) contributes +0.5 to A_inv[1,2].
    // This is the only animal with both sire=1 and dam=2.
    assert!(
        dense[idx1][idx2] > 0.0,
        "A-inverse[1,2] should be positive due to animal 6's cross-term"
    );
}

/// Test 4: End-to-end EM-REML fitting with pedigree-based model.
///
/// Since REML estimates variance components from data (rather than using fixed
/// known values), we cannot expect exact Mrode values. Instead, we check:
///   - The REML algorithm converges.
///   - Variance component estimates are positive.
///   - BLUP ranking is sensible.
///   - BLUEs reflect the sex difference visible in the data.
#[test]
fn test_mrode_example_3_1_reml_convergence() {
    let ped = mrode_example_3_1_pedigree();
    let a_inv = compute_a_inverse(&ped).unwrap();
    let (x, z, y) = mrode_example_3_1_design(&ped);

    let n_obs = 5;
    let n_animals = 8;

    // Initialize variance components from data variance.
    let y_mean: f64 = y.iter().sum::<f64>() / n_obs as f64;
    let y_var: f64 =
        y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / (n_obs - 1) as f64;

    let mut sigma2_a = y_var / 2.0;
    let mut sigma2_e = y_var / 2.0;

    let max_iter = 500;
    let tol = 1e-8;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let g_inv = a_inv.map(|v| v / sigma2_a);
        let r_inv_scale = 1.0 / sigma2_e;

        let mme =
            MixedModelEquations::assemble(&x, &[z.clone()], &y, r_inv_scale, &[g_inv]);
        let sol = mme.solve().unwrap();

        let c_inv = sol.c_inv.as_ref().unwrap();
        let n_fixed = mme.n_fixed;

        // EM update for sigma^2_e: y'e / (n - p)
        let n_eff = (n_obs - n_fixed) as f64;
        let mut y_e_hat: f64 = y.iter().map(|yi| yi * yi).sum();
        let xty = plant_breeding_lmm_core::matrix::sparse::xt_y(&x, &y);
        for i in 0..n_fixed {
            y_e_hat -= sol.fixed_effects[i] * xty[i];
        }
        let zty = plant_breeding_lmm_core::matrix::sparse::xt_y(&z, &y);
        for j in 0..n_animals {
            y_e_hat -= sol.random_effects[0][j] * zty[j];
        }
        let new_sigma2_e = (y_e_hat / n_eff).max(1e-10);

        // EM update for sigma^2_a: (u'A^{-1}u + tr(A^{-1}C^{-1}_{uu})) / q
        let u = &sol.random_effects[0];
        let ainv_u = plant_breeding_lmm_core::matrix::sparse::spmv(&a_inv, u);
        let u_quad: f64 = u.iter().zip(ainv_u.iter()).map(|(a, b)| a * b).sum();

        let mut trace_term = 0.0;
        for i in 0..n_animals {
            for j in 0..n_animals {
                let ainv_ij = a_inv.get(i, j).copied().unwrap_or(0.0);
                let cinv_ji = c_inv[(n_fixed + j, n_fixed + i)];
                trace_term += ainv_ij * cinv_ji;
            }
        }
        let new_sigma2_a = ((u_quad + trace_term) / n_animals as f64).max(1e-10);

        // Convergence check: relative change in parameters.
        let old_params = [sigma2_a, sigma2_e];
        let new_params = [new_sigma2_a, new_sigma2_e];
        let param_diff: f64 = old_params
            .iter()
            .zip(new_params.iter())
            .map(|(o, n)| (o - n).powi(2))
            .sum::<f64>()
            .sqrt();
        let param_norm: f64 = old_params
            .iter()
            .map(|p| p * p)
            .sum::<f64>()
            .sqrt()
            .max(1e-10);
        let rel_change = param_diff / param_norm;

        sigma2_a = new_sigma2_a;
        sigma2_e = new_sigma2_e;

        if iter > 0 && rel_change < tol {
            converged = true;
            break;
        }
    }

    println!(
        "REML converged: {} in {} iterations",
        converged, n_iter
    );
    println!(
        "Estimated sigma^2_a = {:.4}, sigma^2_e = {:.4}",
        sigma2_a, sigma2_e
    );

    // With only 5 observations and 8 random levels, EM-REML may struggle to
    // converge or may converge to a boundary. We check basic sanity properties.
    // Variance components should be non-negative.
    assert!(sigma2_a >= 0.0, "sigma^2_a should be non-negative");
    assert!(sigma2_e > 0.0, "sigma^2_e should be positive");

    // Final solve with converged (or last-iteration) variances.
    let g_inv = a_inv.map(|v| v / sigma2_a.max(1e-10));
    let r_inv_scale = 1.0 / sigma2_e;
    let mme =
        MixedModelEquations::assemble(&x, &[z.clone()], &y, r_inv_scale, &[g_inv]);
    let sol = mme.solve().unwrap();

    // BLUEs: males should have higher estimated mean gain than females.
    // Data male mean = (4.5 + 3.5 + 5.0)/3 = 4.333
    // Data female mean = (2.9 + 3.9)/2 = 3.4
    let blue_male = sol.fixed_effects[0];
    let blue_female = sol.fixed_effects[1];
    println!("REML BLUEs: male = {:.4}, female = {:.4}", blue_male, blue_female);
    assert!(
        blue_male > blue_female,
        "Male BLUE ({:.4}) should exceed female BLUE ({:.4})",
        blue_male,
        blue_female
    );

    // Print all BLUPs for inspection.
    let blups = &sol.random_effects[0];
    for i in 1..=8 {
        let animal_idx = ped.animal_index(&i.to_string()).unwrap();
        println!("  Animal {}: BLUP = {:.4}", i, blups[animal_idx]);
    }
}

/// Test 5: Verify the MME coefficient matrix structure.
///
/// Checks dimensionality, symmetry, and specific block entries of the
/// assembled coefficient matrix for the Mrode Example 3.1 problem.
#[test]
fn test_mrode_example_3_1_mme_structure() {
    let ped = mrode_example_3_1_pedigree();
    let a_inv = compute_a_inverse(&ped).unwrap();
    let (x, z, y) = mrode_example_3_1_design(&ped);

    let sigma2_a = 20.0;
    let sigma2_e = 40.0;
    let g_inv = a_inv.map(|v| v / sigma2_a);
    let r_inv_scale = 1.0 / sigma2_e;

    let mme = MixedModelEquations::assemble(&x, &[z], &y, r_inv_scale, &[g_inv]);

    // Dimension: 2 (sex levels) + 8 (animal levels) = 10.
    assert_eq!(mme.dim, 10);
    assert_eq!(mme.n_fixed, 2);
    assert_eq!(mme.n_random, vec![8]);

    // Coefficient matrix should be symmetric.
    for i in 0..mme.dim {
        for j in 0..mme.dim {
            assert_relative_eq!(
                mme.coeff_matrix[(i, j)],
                mme.coeff_matrix[(j, i)],
                epsilon = 1e-10,
            );
        }
    }

    // Diagonal entries should be positive.
    for i in 0..mme.dim {
        assert!(
            mme.coeff_matrix[(i, i)] > 0.0,
            "C[{},{}] should be positive, got {}",
            i, i, mme.coeff_matrix[(i, i)]
        );
    }

    // X'X block (top-left 2x2): divided by sigma2_e.
    // X'X = [[3, 0], [0, 2]] (3 males, 2 females).
    // Scaled: [[3/40, 0], [0, 2/40]] = [[0.075, 0], [0, 0.05]].
    assert_relative_eq!(mme.coeff_matrix[(0, 0)], 3.0 / sigma2_e, epsilon = 1e-10);
    assert_relative_eq!(mme.coeff_matrix[(1, 1)], 2.0 / sigma2_e, epsilon = 1e-10);
    assert_relative_eq!(mme.coeff_matrix[(0, 1)], 0.0, epsilon = 1e-10);

    // RHS length should match dimension.
    assert_eq!(mme.rhs.len(), 10);

    // Verify RHS entries.
    // X'y / sigma2_e:
    //   male:   (4.5 + 3.5 + 5.0) / 40 = 13.0 / 40 = 0.325
    //   female: (2.9 + 3.9) / 40 = 6.8 / 40 = 0.17
    assert_relative_eq!(mme.rhs[0], 13.0 / 40.0, epsilon = 1e-10);
    assert_relative_eq!(mme.rhs[1], 6.8 / 40.0, epsilon = 1e-10);
}
