//! The sparse MME path: agreement with the dense equations at fixed
//! variances, a pedigree animal model at a scale the dense path cannot
//! handle, Satterthwaite derivatives for relationship-matrix terms and
//! residual diagnostics from the inverse subset.

use approx::assert_relative_eq;
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::diagnostics::compute_diagnostics;
use plant_breeding_lmm_core::genetics::{compute_a_inverse_with_inbreeding, Pedigree};
use plant_breeding_lmm_core::lmm::{MixedModelEquations, MmeInverse};
use plant_breeding_lmm_core::model::MixedModelBuilder;
use plant_breeding_lmm_core::variance::Identity;
use rand::{Rng, SeedableRng};

/// Standard normal via Box-Muller (rand has no normal distribution).
fn normal(rng: &mut rand::rngs::StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

struct Simulated {
    ped: Pedigree,
    df: DataFrame,
    n_records: usize,
}

/// Founders plus `n_gen` generations of `per_gen` animals with random
/// parents from the previous generation; records for every non-founder.
fn simulate(
    seed: u64,
    n_founders: usize,
    n_gen: usize,
    per_gen: usize,
    s2a: f64,
    s2e: f64,
) -> Simulated {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut ped = Pedigree::new();
    let mut ids: Vec<String> = Vec::new();
    let mut bv: Vec<f64> = Vec::new();
    for i in 0..n_founders {
        let id = format!("F{i}");
        ped.add_animal(&id, None, None).unwrap();
        ids.push(id);
        bv.push(normal(&mut rng) * s2a.sqrt());
    }
    let mut prev: Vec<usize> = (0..n_founders).collect();
    let mut records: Vec<(String, &'static str, f64)> = Vec::new();
    for g in 0..n_gen {
        let mut cur = Vec::with_capacity(per_gen);
        for i in 0..per_gen {
            let s = prev[rng.gen_range(0..prev.len())];
            let mut d = prev[rng.gen_range(0..prev.len())];
            while d == s {
                d = prev[rng.gen_range(0..prev.len())];
            }
            let id = format!("G{g}_{i}");
            ped.add_animal(&id, Some(&ids[s]), Some(&ids[d])).unwrap();
            let a = 0.5 * (bv[s] + bv[d]) + normal(&mut rng) * (0.5 * s2a).sqrt();
            ids.push(id.clone());
            bv.push(a);
            cur.push(ids.len() - 1);
            let sex = if rng.gen::<bool>() { "male" } else { "female" };
            let y =
                100.0 + if sex == "male" { 4.0 } else { 0.0 } + a + normal(&mut rng) * s2e.sqrt();
            records.push((id, sex, y));
        }
        prev = cur;
    }
    ped.sort_pedigree().unwrap();

    let mut df = DataFrame::new();
    df.add_float_column("y", records.iter().map(|r| r.2).collect())
        .unwrap();
    let animals: Vec<&str> = records.iter().map(|r| r.0.as_str()).collect();
    let sexes: Vec<&str> = records.iter().map(|r| r.1).collect();
    df.add_factor_column("animal", &animals).unwrap();
    df.add_factor_column("sex", &sexes).unwrap();
    Simulated {
        ped,
        n_records: records.len(),
        df,
    }
}

#[test]
fn sparse_fit_agrees_with_dense_equations_at_convergence() {
    let sim = simulate(1, 40, 3, 150, 20.0, 40.0);
    let mut model = MixedModelBuilder::new()
        .data(&sim.df)
        .response("y")
        .fixed("sex")
        .random_pedigree("animal", Identity::new(1.0), &sim.ped)
        .max_iterations(100)
        .convergence(1e-9)
        .build()
        .unwrap();
    let result = model.fit_reml().unwrap();
    assert!(result.converged);
    assert_eq!(result.n_obs, sim.n_records);
    let s2a = result.variance_component("animal").unwrap();
    let s2e = result.variance_component("residual").unwrap();
    eprintln!("converged: sigma2_a = {s2a:.4}, sigma2_e = {s2e:.4}");
    assert!(
        !result.at_boundary.iter().any(|b| *b),
        "a variance parameter converged to the boundary: {:?}",
        result.at_boundary
    );
    assert!(s2a > 1.0 && s2e > 1.0);

    // Re-solve the MME densely at the converged variances.
    let a_inv = compute_a_inverse_with_inbreeding(&sim.ped).unwrap();
    let g_inv = a_inv.map(|v| v / s2a);
    let mme =
        MixedModelEquations::assemble(&model.x, &model.z_blocks, &model.y, 1.0 / s2e, &[g_inv]);
    let dense = mme.solve().unwrap();
    let c_inv = dense.inverse().unwrap();
    let p = model.x.cols();
    for (i, fe) in result.fixed_effects.iter().enumerate() {
        assert_relative_eq!(fe.estimate, dense.fixed_effects[i], epsilon = 1e-8);
        assert_relative_eq!(fe.se, c_inv.entry(i, i).sqrt(), epsilon = 1e-8);
        for j in 0..p {
            assert_relative_eq!(result.fixed_cov[i][j], c_inv.entry(i, j), epsilon = 1e-10);
        }
    }
    for (j, re) in result.random_effects[0].effects.iter().enumerate() {
        assert_relative_eq!(re.estimate, dense.random_effects[0][j], epsilon = 1e-8);
        assert_relative_eq!(re.se, c_inv.entry(p + j, p + j).sqrt(), epsilon = 1e-8);
    }
    // The sparse inverse subset stored in the result is exact on the pattern.
    let sparse_inv = result.c_inv.as_ref().unwrap();
    assert!(matches!(sparse_inv, MmeInverse::Sparse { .. }));
    for (_, (i, j)) in a_inv.iter() {
        assert_relative_eq!(
            sparse_inv.entry(p + i, p + j),
            c_inv.entry(p + i, p + j),
            epsilon = 1e-10
        );
    }

    // Satterthwaite derivatives (which involve A⁻¹) against numerical
    // differentiation of the dense fixed-effects covariance.
    let phi_at = |s2a: f64, s2e: f64| {
        let g_inv = a_inv.map(|v| v / s2a);
        let mme =
            MixedModelEquations::assemble(&model.x, &model.z_blocks, &model.y, 1.0 / s2e, &[g_inv]);
        mme.solve().unwrap().c_inv.unwrap().fixed_block(p)
    };
    let ha = 1e-5 * s2a;
    let he = 1e-5 * s2e;
    let num_a = (phi_at(s2a + ha, s2e) - phi_at(s2a - ha, s2e)) / (2.0 * ha);
    let num_e = (phi_at(s2a, s2e + he) - phi_at(s2a, s2e - he)) / (2.0 * he);
    assert_eq!(result.fixed_cov_derivatives.len(), 2);
    for i in 0..p {
        for j in 0..p {
            assert_relative_eq!(
                result.fixed_cov_derivatives[0][i][j],
                num_a[(i, j)],
                epsilon = 1e-8,
                max_relative = 1e-4
            );
            assert_relative_eq!(
                result.fixed_cov_derivatives[1][i][j],
                num_e[(i, j)],
                epsilon = 1e-8,
                max_relative = 1e-4
            );
        }
    }
    let satt = result.wald_tests_satterthwaite().unwrap();
    assert_eq!(satt.len(), 1);
    assert!(satt[0].den_df >= 1.0 && satt[0].den_df <= (sim.n_records - p) as f64);
    // Kenward-Roger: P_i consistent with the derivatives (∂Φ/∂θ = −Φ P Φ)
    // and a finite adjusted test.
    let kr_terms = result.kenward_roger_terms.as_ref().unwrap();
    let phi = nalgebra::DMatrix::from_fn(p, p, |i, j| result.fixed_cov[i][j]);
    for (i, pm) in kr_terms.p.iter().enumerate() {
        let from_p = -(&phi * pm * &phi);
        for a in 0..p {
            for b in 0..p {
                assert_relative_eq!(
                    from_p[(a, b)],
                    result.fixed_cov_derivatives[i][a][b],
                    epsilon = 1e-9,
                    max_relative = 1e-6
                );
            }
        }
    }
    let kr = result.wald_tests_kenward_roger().unwrap();
    assert_eq!(kr.len(), 1);
    assert!(kr[0].f_statistic > 0.0 && kr[0].f_statistic.is_finite());
    assert!(kr[0].den_df >= 1.0 && kr[0].den_df <= (sim.n_records - p) as f64);

    // Residual diagnostics from the sparse inverse subset equal the dense ones.
    let fixed: Vec<f64> = result.fixed_effects.iter().map(|e| e.estimate).collect();
    let random: Vec<f64> = result.random_effects[0]
        .effects
        .iter()
        .map(|e| e.estimate)
        .collect();
    let d_sparse = result.residual_diagnostics(&model).unwrap();
    let d_dense = compute_diagnostics(
        &model.y,
        &model.x,
        &model.z_combined,
        &fixed,
        &random,
        c_inv,
        s2e,
    );
    for i in 0..sim.n_records {
        assert_relative_eq!(d_sparse.leverage[i], d_dense.leverage[i], epsilon = 1e-9);
        assert_relative_eq!(
            d_sparse.cooks_distance[i],
            d_dense.cooks_distance[i],
            epsilon = 1e-9
        );
    }
}

/// 4 000 animals in the pedigree, 3 600 records: far beyond what a dense
/// inverse per REML iteration could handle in a test. Each REML iteration
/// takes well under a second in a release build but several seconds with
/// the debug profile, so this test is ignored by default and CI runs it with
/// `cargo test --release -p plant-breeding-lmm-core --test sparse_mme_test -- --ignored`.
#[test]
#[ignore = "large-scale model; run with --release -- --ignored"]
fn animal_model_with_thousands_of_animals() {
    let sim = simulate(7, 400, 3, 1200, 20.0, 40.0);
    let mut model = MixedModelBuilder::new()
        .data(&sim.df)
        .response("y")
        .fixed("sex")
        .random_pedigree("animal", Identity::new(1.0), &sim.ped)
        .max_iterations(60)
        .convergence(1e-7)
        .build()
        .unwrap();
    let start = std::time::Instant::now();
    let result = model.fit_reml().unwrap();
    let elapsed = start.elapsed();
    assert!(
        result.converged,
        "did not converge: {:?}",
        result.history.last()
    );
    assert_eq!(result.random_effects[0].effects.len(), 4000);
    let s2a = result.variance_component("animal").unwrap();
    let s2e = result.variance_component("residual").unwrap();
    // Sampling error with 3 600 records is a few units.
    assert!((10.0..35.0).contains(&s2a), "sigma2_a = {s2a}");
    assert!((30.0..50.0).contains(&s2e), "sigma2_e = {s2e}");
    let male = result
        .fixed_effects
        .iter()
        .find(|e| e.level == "male")
        .unwrap()
        .estimate;
    let female = result
        .fixed_effects
        .iter()
        .find(|e| e.level == "female")
        .unwrap()
        .estimate;
    assert!(
        (male - female - 4.0).abs() < 1.5,
        "sex effect {}",
        male - female
    );
    // SEs and reliabilities are available for every animal.
    assert!(result.random_effects[0].effects.iter().all(|e| e.se > 0.0));
    assert!(result.variance_se.iter().all(|se| *se > 0.0));
    assert!(result.wald_tests_satterthwaite().is_some());
    eprintln!(
        "4 000-animal model: {} iterations in {:.2}s",
        result.n_iterations,
        elapsed.as_secs_f64()
    );
}

#[test]
fn em_reml_uses_the_sparse_path_too() {
    let sim = simulate(3, 20, 2, 80, 20.0, 40.0);
    let mut model = MixedModelBuilder::new()
        .data(&sim.df)
        .response("y")
        .fixed("sex")
        .random_pedigree("animal", Identity::new(1.0), &sim.ped)
        .max_iterations(500)
        .convergence(1e-9)
        .build()
        .unwrap();
    let em = model.fit_em_reml().unwrap();
    let ai = model.fit_reml().unwrap();
    assert!(matches!(em.c_inv, Some(MmeInverse::Sparse { .. })));
    assert_relative_eq!(
        em.variance_component("animal").unwrap(),
        ai.variance_component("animal").unwrap(),
        max_relative = 1e-4
    );
    assert_relative_eq!(
        em.variance_component("residual").unwrap(),
        ai.variance_component("residual").unwrap(),
        max_relative = 1e-4
    );
}
