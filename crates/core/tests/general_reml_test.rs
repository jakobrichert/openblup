//! Validation of the general AI-REML engine against an exact dense REML
//! log-likelihood.
//!
//! For every model the engine's fitted parameters must be a stationary
//! point of the *exact* restricted log-likelihood computed from the dense
//! covariance V = sum_k Z_k G_k Z_k' + R (central-difference gradient ~ 0),
//! and the engine's reported log-likelihood must equal the dense value.
//! This checks the scores, the MME identities, log-determinants (including
//! log|A| for pedigree Kronecker terms) and the gridded residual handling.

#![allow(clippy::needless_range_loop)]

use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};

use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::genetics::Pedigree;
use plant_breeding_lmm_core::lmm::{AiReml, EmReml, FitResult, GeneralReml};
use plant_breeding_lmm_core::model::{MixedModel, MixedModelBuilder};
use plant_breeding_lmm_core::types::SparseMat;
use plant_breeding_lmm_core::variance::{Diagonal, FactorAnalytic, Identity, VarStruct, AR1};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn dense_of(m: &SparseMat, rows: usize, cols: usize) -> DMatrix<f64> {
    let mut d = DMatrix::zeros(rows, cols);
    for (v, (i, j)) in m.iter() {
        d[(i, j)] += *v;
    }
    d
}

fn normal(rng: &mut rand::rngs::StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample a vector from N(0, cov) via Cholesky.
fn mvn(rng: &mut rand::rngs::StdRng, cov: &DMatrix<f64>) -> Vec<f64> {
    let l = cov.clone().cholesky().expect("covariance must be PD").l();
    let z = DVector::from_fn(cov.nrows(), |_, _| normal(rng));
    (l * z).as_slice().to_vec()
}

/// Flat parameter vector of the fitted model (term parameters, then residual).
fn result_theta(result: &FitResult) -> Vec<f64> {
    result
        .variance_components
        .iter()
        .flat_map(|vc| vc.parameters.iter().map(|(_, v)| *v))
        .collect()
}

fn set_theta(model: &mut MixedModel, theta: &[f64]) -> bool {
    let mut offset = 0;
    for vs in model.random_var_structs.iter_mut() {
        let n = vs.n_params();
        if vs.set_params(&theta[offset..offset + n]).is_err() {
            return false;
        }
        offset += n;
    }
    let n = model.residual_var_struct.n_params();
    model
        .residual_var_struct
        .set_params(&theta[offset..offset + n])
        .is_ok()
}

/// Exact REML log-likelihood from the dense V matrix (`None` if `theta` is
/// outside the parameter space).
fn dense_reml_logl_opt(model: &mut MixedModel, theta: &[f64]) -> Option<f64> {
    if !set_theta(model, theta) {
        return None;
    }
    Some(dense_reml_logl_inner(model))
}

fn dense_reml_logl(model: &mut MixedModel, theta: &[f64]) -> f64 {
    dense_reml_logl_opt(model, theta).expect("theta inside the parameter space")
}

fn dense_reml_logl_inner(model: &mut MixedModel) -> f64 {
    let n = model.n_obs;
    let p = model.x.cols();
    let x = dense_of(&model.x, n, p);

    let mut v = DMatrix::zeros(n, n);
    for (k, z) in model.z_blocks.iter().enumerate() {
        let q = z.cols();
        let g = dense_of(&model.random_var_structs[k].covariance_matrix(q), q, q);
        let zd = dense_of(z, n, q);
        v += &zd * g * zd.transpose();
    }
    let r = match &model.residual_grid {
        Some(grid) => {
            let full = dense_of(
                &model.residual_var_struct.covariance_matrix(grid.n_cells),
                grid.n_cells,
                grid.n_cells,
            );
            DMatrix::from_fn(n, n, |a, b| full[(grid.cell_index[a], grid.cell_index[b])])
        }
        None => dense_of(&model.residual_var_struct.covariance_matrix(n), n, n),
    };
    v += r;

    let chol_v = v.clone().cholesky().expect("V must be PD");
    let log_det_v = 2.0 * (0..n).map(|i| chol_v.l()[(i, i)].ln()).sum::<f64>();
    let v_inv = chol_v.inverse();
    let xtvix = x.transpose() * &v_inv * &x;
    let chol_x = xtvix.clone().cholesky().expect("X'V^-1X must be PD");
    let log_det_x = 2.0 * (0..p).map(|i| chol_x.l()[(i, i)].ln()).sum::<f64>();
    let y = DVector::from_column_slice(&model.y);
    let vinv_y = &v_inv * &y;
    let xtvy = x.transpose() * &vinv_y;
    let ypy = y.dot(&vinv_y) - xtvy.dot(&chol_x.solve(&xtvy));
    let log_2_pi = (2.0 * std::f64::consts::PI).ln();
    -0.5 * ((n - p) as f64 * log_2_pi + log_det_v + log_det_x + ypy)
}

/// Central-difference gradient; falls back to a forward difference when the
/// backward point leaves the parameter space (a parameter on its bound).
fn numerical_gradient(model: &mut MixedModel, theta: &[f64]) -> Vec<f64> {
    let f0 = dense_reml_logl(model, theta);
    (0..theta.len())
        .map(|i| {
            let h = 1e-5 * theta[i].abs().max(1.0);
            let mut plus = theta.to_vec();
            let mut minus = theta.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let fp = dense_reml_logl(model, &plus);
            match dense_reml_logl_opt(model, &minus) {
                Some(fm) => (fp - fm) / (2.0 * h),
                None => (fp - f0) / h,
            }
        })
        .collect()
}

/// The fitted parameters must be a stationary point of the exact REML
/// log-likelihood, and the engine's log-likelihood must match it.
fn assert_at_optimum(model: &mut MixedModel, result: &FitResult, grad_tol: f64) {
    if !result.converged {
        for h in result.history.iter().rev().take(6).rev() {
            println!(
                "it {} logl {:.6} change {:.2e} params {:?}",
                h.iteration, h.log_likelihood, h.change, h.variance_params
            );
        }
    }
    assert!(
        result.converged,
        "did not converge in {} iterations",
        result.n_iterations
    );
    let theta = result_theta(result);
    let logl_dense = dense_reml_logl(model, &theta);
    assert!(
        (logl_dense - result.log_likelihood).abs() < 1e-6 * logl_dense.abs().max(1.0),
        "engine logL {} != dense logL {}",
        result.log_likelihood,
        logl_dense
    );
    let grad = numerical_gradient(model, &theta);
    for (i, g) in grad.iter().enumerate() {
        if result.at_boundary[i] {
            continue;
        }
        assert!(
            g.abs() < grad_tol,
            "gradient of parameter {} is {} at the fitted point (theta = {:?}, grad = {:?})",
            i,
            g,
            theta,
            grad
        );
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

/// The general engine must reproduce the specialised engine on a plain
/// genotype + residual model.
#[test]
fn general_engine_matches_specialised_engine_on_identity_model() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let n_geno = 20;
    let reps = 3;
    let mut y = Vec::new();
    let mut geno = Vec::new();
    let mut rep = Vec::new();
    let g_eff: Vec<f64> = (0..n_geno).map(|_| 2.0 * normal(&mut rng)).collect();
    for r in 0..reps {
        for g in 0..n_geno {
            y.push(10.0 + r as f64 + g_eff[g] + normal(&mut rng));
            geno.push(format!("G{}", g));
            rep.push(format!("R{}", r));
        }
    }
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    let geno_refs: Vec<&str> = geno.iter().map(|s| s.as_str()).collect();
    let rep_refs: Vec<&str> = rep.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("geno", &geno_refs).unwrap();
    df.add_factor_column("rep", &rep_refs).unwrap();

    let build = || {
        MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .fixed("mu + rep")
            .random("geno", Identity::new(1.0), None)
            .build()
            .unwrap()
    };
    let mut m1 = build();
    let r1 = AiReml::new(100, 1e-10).fit(&mut m1).unwrap();
    let mut m2 = build();
    let r2 = GeneralReml::new(100, 1e-10).fit(&mut m2).unwrap();

    assert!(r1.converged && r2.converged);
    let t1 = result_theta(&r1);
    let t2 = result_theta(&r2);
    for (a, b) in t1.iter().zip(t2.iter()) {
        assert!((a - b).abs() < 1e-5 * a.abs(), "{} vs {}", a, b);
    }
    assert!((r1.log_likelihood - r2.log_likelihood).abs() < 1e-6);
    // SEs agree as well
    for (a, b) in r1.variance_se.iter().zip(r2.variance_se.iter()) {
        assert!((a - b).abs() < 1e-4 * a.abs().max(1.0), "se {} vs {}", a, b);
    }
    assert_at_optimum(&mut m2, &r2, 1e-3);
}

/// AR1 random row effects (a one-dimensional spatial trend) plus IID genotypes.
#[test]
fn ar1_random_term_is_estimated() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2);
    let n_rows = 40;
    let n_geno = 8;
    let (sigma2_row, rho) = (4.0, 0.6);
    let ar1 = AR1::new(sigma2_row, rho);
    let row_cov = dense_of(&ar1.covariance_matrix(n_rows), n_rows, n_rows);
    let row_eff = mvn(&mut rng, &row_cov);
    let g_eff: Vec<f64> = (0..n_geno).map(|_| 1.5 * normal(&mut rng)).collect();

    let mut y = Vec::new();
    let mut row = Vec::new();
    let mut geno = Vec::new();
    for r in 0..n_rows {
        for k in 0..3 {
            let g = (r * 3 + k) % n_geno;
            y.push(20.0 + row_eff[r] + g_eff[g] + normal(&mut rng));
            row.push((r + 1) as f64);
            geno.push(format!("G{}", g));
        }
    }
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    df.add_float_column("row", row).unwrap();
    let geno_refs: Vec<&str> = geno.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("geno", &geno_refs).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("mu")
        .random("row", AR1::new(1.0, 0.2), None)
        .random("geno", Identity::new(1.0), None)
        .max_iterations(100)
        .convergence(1e-9)
        .build()
        .unwrap();
    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());

    let row_vc = &result.variance_components[0];
    assert_eq!(row_vc.structure, "AR1");
    assert_eq!(row_vc.parameters[0].0, "sigma2");
    assert_eq!(row_vc.parameters[1].0, "rho");
    let rho_hat = row_vc.parameters[1].1;
    assert!(rho_hat > 0.2 && rho_hat < 0.9, "rho estimate {}", rho_hat);
    assert!(row_vc.se[1] > 0.0);
    assert_eq!(result.n_variance_params, 4);
    assert!(
        result.n_iterations < 40,
        "took {} iterations",
        result.n_iterations
    );
    assert_at_optimum(&mut model, &result, 1e-3);
}

/// Heterogeneous genetic variance per environment: Diagonal(env) ⊗ I(genotype).
#[test]
fn diagonal_env_by_genotype_interaction() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let n_env = 3;
    let n_geno = 25;
    let env_var: [f64; 3] = [1.0, 4.0, 9.0];
    let mut y = Vec::new();
    let mut env = Vec::new();
    let mut geno = Vec::new();
    for e in 0..n_env {
        for g in 0..n_geno {
            for _rep in 0..2 {
                y.push(50.0 + 5.0 * e as f64 + normal(&mut rng));
                env.push(format!("E{}", e));
                geno.push(format!("G{:02}", g));
            }
        }
    }
    // add genotype-by-environment effects with per-environment variance
    let mut ge = vec![vec![0.0; n_geno]; n_env];
    for e in 0..n_env {
        for g in 0..n_geno {
            ge[e][g] = env_var[e].sqrt() * normal(&mut rng);
        }
    }
    for (i, yi) in y.iter_mut().enumerate() {
        let e = i / (n_geno * 2);
        let g = (i / 2) % n_geno;
        *yi += ge[e][g];
    }
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    let env_refs: Vec<&str> = env.iter().map(|s| s.as_str()).collect();
    let geno_refs: Vec<&str> = geno.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("env", &env_refs).unwrap();
    df.add_factor_column("geno", &geno_refs).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("env")
        .random_interaction("env", Diagonal::new(vec![1.0; n_env]), "geno")
        .max_iterations(100)
        .convergence(1e-9)
        .build()
        .unwrap();
    assert_eq!(model.z_blocks[0].cols(), n_env * n_geno);
    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());

    let vc = &result.variance_components[0];
    assert_eq!(vc.name, "env:geno");
    assert_eq!(vc.parameters.len(), 3);
    assert_eq!(vc.parameters[0].0, "env.sigma2_1");
    let v: Vec<f64> = vc.parameters.iter().map(|(_, v)| *v).collect();
    assert!(
        v[0] < v[1] && v[1] < v[2],
        "variances should increase: {:?}",
        v
    );
    assert!(v[2] > 4.0 && v[2] < 16.0, "sigma2_3 = {}", v[2]);
    assert_eq!(result.random_effects[0].effects.len(), n_env * n_geno);
    assert!(result.random_effects[0].effects[0]
        .level
        .starts_with("E0:G00"));
    assert_at_optimum(&mut model, &result, 1e-3);
}

fn two_generation_pedigree(n_founders: usize, n_offspring: usize) -> Pedigree {
    let mut triples = Vec::new();
    for i in 0..n_founders {
        triples.push((format!("F{}", i), None, None));
    }
    for j in 0..n_offspring {
        let s = j % (n_founders / 2);
        let d = n_founders / 2 + (j * 7 % (n_founders / 2));
        triples.push((
            format!("O{}", j),
            Some(format!("F{}", s)),
            Some(format!("F{}", d)),
        ));
    }
    let mut ped = Pedigree::from_triples(&triples).unwrap();
    ped.sort_pedigree().unwrap();
    ped
}

/// FA1(env) ⊗ A: factor analytic genotype-by-environment effects with a
/// pedigree relationship matrix. Founders have no records.
#[test]
fn fa1_env_by_pedigree_interaction() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let n_env = 3;
    let ped = two_generation_pedigree(8, 24);
    let ids: Vec<String> = (0..ped.n_animals())
        .map(|i| ped.animal_id(i).to_string())
        .collect();
    let offspring: Vec<&String> = ids.iter().filter(|id| id.starts_with('O')).collect();

    // simulate: loadings (1, 2, 1.5), psi (0.5, 0.5, 0.5), residual 1
    let lambda = [1.0, 2.0, 1.5];
    let n_anim = ped.n_animals();
    // breeding-value factor per animal via pedigree (parent average + Mendelian)
    let mut factor = vec![0.0; n_anim];
    for i in 0..n_anim {
        match (ped.sire(i), ped.dam(i)) {
            (Some(s), Some(d)) => {
                factor[i] = 0.5 * (factor[s] + factor[d]) + (0.5f64).sqrt() * normal(&mut rng)
            }
            _ => factor[i] = normal(&mut rng),
        }
    }
    let mut y = Vec::new();
    let mut env = Vec::new();
    let mut animal = Vec::new();
    for e in 0..n_env {
        for id in &offspring {
            let i = ped.animal_index(id).unwrap();
            // genotype-by-environment effect shared by the two replicate records
            let ge = lambda[e] * factor[i] + (0.5f64).sqrt() * normal(&mut rng);
            for _rep in 0..2 {
                y.push(30.0 + 2.0 * e as f64 + ge + normal(&mut rng));
                env.push(format!("E{}", e));
                animal.push((*id).clone());
            }
        }
    }
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    let env_refs: Vec<&str> = env.iter().map(|s| s.as_str()).collect();
    let animal_refs: Vec<&str> = animal.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("env", &env_refs).unwrap();
    df.add_factor_column("animal", &animal_refs).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("env")
        .random_interaction_pedigree("env", FactorAnalytic::new(n_env, 1), "animal", &ped)
        .max_iterations(300)
        .convergence(1e-8)
        .build()
        .unwrap();
    assert_eq!(model.z_blocks[0].cols(), n_env * n_anim);
    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());

    let vc = &result.variance_components[0];
    assert_eq!(vc.name, "env:animal");
    assert_eq!(vc.parameters.len(), 6); // 3 loadings + 3 psi
    assert_eq!(vc.parameters[0].0, "env.lambda_1_1");
    assert_eq!(vc.parameters[3].0, "env.psi_1");
    // Genetic variances lambda^2 + psi should rank like the simulation (env 2 largest).
    let lam: Vec<f64> = (0..3).map(|e| vc.parameters[e].1).collect();
    let psi: Vec<f64> = (0..3).map(|e| vc.parameters[3 + e].1).collect();
    let gvar: Vec<f64> = (0..3).map(|e| lam[e] * lam[e] + psi[e]).collect();
    assert!(gvar[1] > gvar[0], "genetic variances {:?}", gvar);
    // founders get breeding values in every environment
    assert_eq!(result.random_effects[0].effects.len(), n_env * n_anim);
    assert!(result.random_effects[0]
        .effects
        .iter()
        .any(|e| e.level == "E0:F0"));
    assert_at_optimum(&mut model, &result, 2e-3);
}

/// A two-trait animal model expressed with the general structures: the
/// records are stacked in long format (one row per trait x animal), the
/// genetic term is `trait:animal` with `US(trait) ⊗ A` and the residual is
/// `US(trait) ⊗ I` over the (trait, unit) grid — i.e. G0 ⊗ A and R0 ⊗ I.
#[test]
fn two_trait_animal_model_via_kronecker_structures() {
    use plant_breeding_lmm_core::variance::StructureSpec;

    let mut rng = rand::rngs::StdRng::seed_from_u64(11);
    let ped = two_generation_pedigree(8, 40);
    let n_anim = ped.n_animals();
    let ids: Vec<String> = (0..n_anim).map(|i| ped.animal_id(i).to_string()).collect();
    let offspring: Vec<&String> = ids.iter().filter(|id| id.starts_with('O')).collect();

    let g0 = DMatrix::from_row_slice(2, 2, &[2.0, 0.8, 0.8, 1.0]);
    let r0 = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 0.5]);
    let mut bv = vec![[0.0f64; 2]; n_anim];
    for i in 0..n_anim {
        match (ped.sire(i), ped.dam(i)) {
            (Some(sire), Some(dam)) => {
                let m = mvn(&mut rng, &(0.5 * &g0));
                bv[i] = [
                    0.5 * (bv[sire][0] + bv[dam][0]) + m[0],
                    0.5 * (bv[sire][1] + bv[dam][1]) + m[1],
                ];
            }
            _ => {
                let m = mvn(&mut rng, &g0);
                bv[i] = [m[0], m[1]];
            }
        }
    }
    let mu = [20.0, 5.0];
    let (mut y, mut trait_col, mut animal, mut unit) = (vec![], vec![], vec![], vec![]);
    for id in &offspring {
        let i = ped.animal_index(id).unwrap();
        let e = mvn(&mut rng, &r0);
        for t in 0..2 {
            y.push(mu[t] + bv[i][t] + e[t]);
            trait_col.push(format!("T{}", t));
            animal.push((*id).clone());
            unit.push((*id).clone());
        }
    }
    let n_obs = y.len();
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    let t_refs: Vec<&str> = trait_col.iter().map(|s| s.as_str()).collect();
    let a_refs: Vec<&str> = animal.iter().map(|s| s.as_str()).collect();
    let u_refs: Vec<&str> = unit.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("trait", &t_refs).unwrap();
    df.add_factor_column("animal", &a_refs).unwrap();
    df.add_factor_column("unit", &u_refs).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("trait")
        .random_interaction_spec(
            "trait",
            StructureSpec::Unstructured,
            "animal",
            None,
            Some(&ped),
        )
        .residual_interaction_spec(
            "trait",
            StructureSpec::Unstructured,
            "unit",
            StructureSpec::Known,
        )
        .max_iterations(300)
        .convergence(1e-8)
        .build()
        .unwrap();
    assert_eq!(model.n_obs, n_obs);
    assert_eq!(model.z_blocks[0].cols(), 2 * n_anim);
    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());
    // Reconstruct G0 and R0 from the Cholesky parameters L_r_c.
    let cov_from = |vc: &plant_breeding_lmm_core::lmm::VarianceEstimate| -> DMatrix<f64> {
        let get = |name: &str| {
            vc.parameters
                .iter()
                .find(|(n, _)| n.ends_with(name))
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("parameter {} missing in {:?}", name, vc.parameters))
        };
        let l = DMatrix::from_row_slice(2, 2, &[get("L_1_1"), 0.0, get("L_2_1"), get("L_2_2")]);
        &l * l.transpose()
    };
    assert_eq!(result.variance_components[0].name, "trait:animal");
    assert_eq!(result.variance_components[0].parameters.len(), 3);
    assert_eq!(result.variance_components[1].parameters.len(), 3);
    let g0_hat = cov_from(&result.variance_components[0]);
    let r0_hat = cov_from(&result.variance_components[1]);
    println!("G0 = {} R0 = {}", g0_hat, r0_hat);
    for m in [&g0_hat, &r0_hat] {
        assert!(m[(0, 0)] > 0.0 && m[(1, 1)] > 0.0);
        let corr = m[(0, 1)] / (m[(0, 0)] * m[(1, 1)]).sqrt();
        assert!((-1.0..=1.0).contains(&corr));
    }
    // Trait 1 has the larger genetic variance and a positive genetic correlation.
    assert!(g0_hat[(0, 0)] > g0_hat[(1, 1)]);
    assert!(g0_hat[(0, 1)] > 0.0);
    // Breeding values for both traits and every animal, founders included.
    assert_eq!(result.random_effects[0].effects.len(), 2 * n_anim);
    assert!(result.random_effects[0]
        .effects
        .iter()
        .any(|e| e.level == "T1:F0"));
    assert_at_optimum(&mut model, &result, 2e-3);
}

/// Separable AR1 x AR1 residual on a field grid with missing plots.
#[test]
fn ar1_x_ar1_residual_with_missing_plots() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(5);
    let (n_rows, n_cols) = (12, 10);
    let (sigma2, rho_r, rho_c) = (2.0, 0.6, 0.4);
    let n_geno = 20;
    let r_cov = dense_of(
        &AR1::new(sigma2, rho_r).covariance_matrix(n_rows),
        n_rows,
        n_rows,
    );
    let c_cov = dense_of(
        &AR1::correlation(rho_c).covariance_matrix(n_cols),
        n_cols,
        n_cols,
    );
    let full = r_cov.kronecker(&c_cov);
    let e = mvn(&mut rng, &full);
    let g_eff: Vec<f64> = (0..n_geno).map(|_| normal(&mut rng)).collect();

    let missing = [3usize, 17, 44, 45, 78, 99, 100, 113];
    let mut y = Vec::new();
    let mut row = Vec::new();
    let mut col = Vec::new();
    let mut geno = Vec::new();
    for r in 0..n_rows {
        for c in 0..n_cols {
            let cell = r * n_cols + c;
            if missing.contains(&cell) {
                continue;
            }
            let g = (cell * 7) % n_geno;
            y.push(100.0 + g_eff[g] + e[cell]);
            row.push((r + 1) as f64);
            col.push((c + 1) as f64);
            geno.push(format!("G{}", g));
        }
    }
    let n = y.len();
    assert_eq!(n, n_rows * n_cols - missing.len());
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    df.add_float_column("row", row).unwrap();
    df.add_float_column("col", col).unwrap();
    let geno_refs: Vec<&str> = geno.iter().map(|s| s.as_str()).collect();
    df.add_factor_column("geno", &geno_refs).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("mu")
        .random("geno", Identity::new(1.0), None)
        .residual_interaction("row", AR1::new(1.0, 0.3), "col", AR1::correlation(0.3))
        .max_iterations(100)
        .convergence(1e-9)
        .build()
        .unwrap();
    let grid = model.residual_grid.as_ref().unwrap();
    assert_eq!(grid.n_cells, n_rows * n_cols);
    assert_eq!(grid.cell_index.len(), n);

    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());
    let res = result.variance_components.last().unwrap();
    assert_eq!(res.name, "residual");
    assert_eq!(res.parameters.len(), 3);
    assert_eq!(res.parameters[0].0, "row.sigma2");
    assert_eq!(res.parameters[1].0, "row.rho");
    assert_eq!(res.parameters[2].0, "col.rho");
    let rho_r_hat = res.parameters[1].1;
    let rho_c_hat = res.parameters[2].1;
    assert!(
        rho_r_hat > 0.3 && rho_r_hat < 0.85,
        "rho_row = {}",
        rho_r_hat
    );
    assert!(
        rho_c_hat > 0.1 && rho_c_hat < 0.7,
        "rho_col = {}",
        rho_c_hat
    );
    assert!(res.se.iter().all(|s| *s > 0.0));
    assert_at_optimum(&mut model, &result, 1e-3);
}

/// AR1 residual on the observations in data order (no grid).
#[test]
fn ar1_residual_in_data_order() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(6);
    let n = 60;
    let cov = dense_of(&AR1::new(3.0, 0.5).covariance_matrix(n), n, n);
    let e = mvn(&mut rng, &cov);
    let y: Vec<f64> = (0..n)
        .map(|i| 5.0 + if i % 2 == 0 { 1.0 } else { 0.0 } + e[i])
        .collect();
    let trt: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    df.add_factor_column("trt", &trt).unwrap();

    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .fixed("trt")
        .residual(AR1::new(1.0, 0.1))
        .max_iterations(100)
        .convergence(1e-9)
        .build()
        .unwrap();
    let result = model.fit_reml().unwrap();
    println!("{}", result.summary());
    assert_eq!(result.variance_components.len(), 1);
    let rho = result.variance_components[0].parameters[1].1;
    assert!(rho > 0.2 && rho < 0.8, "rho = {}", rho);
    assert_at_optimum(&mut model, &result, 1e-3);
}

#[test]
fn structured_models_reject_em_and_bad_specs() {
    let mut df = DataFrame::new();
    df.add_float_column("y", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    df.add_factor_column("g", &["a", "b", "c", "a", "b", "c"])
        .unwrap();
    df.add_factor_column("r", &["1", "1", "1", "2", "2", "2"])
        .unwrap();

    // EM cannot handle AR1
    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .random("g", AR1::new(1.0, 0.2), None)
        .build()
        .unwrap();
    assert!(model.needs_general_engine());
    assert!(EmReml::new(10, 1e-6).fit(&mut model).is_err());

    // Diagonal with the wrong number of levels
    let err = MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .random("g", Diagonal::new(vec![1.0, 1.0]), None)
        .build()
        .err()
        .expect("dimension mismatch");
    assert!(err.to_string().contains("3"));

    // pedigree combined with a non-identity structure
    let ped = two_generation_pedigree(4, 2);
    assert!(MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .random_pedigree("g", AR1::new(1.0, 0.1), &ped)
        .build()
        .is_err());

    // duplicate residual grid cells
    assert!(MixedModelBuilder::new()
        .data(&df)
        .response("y")
        .residual_interaction("r", AR1::new(1.0, 0.1), "g", AR1::correlation(0.1))
        .build()
        .is_ok());
    let mut df2 = df.clone();
    df2.add_factor_column("dup", &["1", "1", "1", "1", "1", "1"])
        .unwrap();
    assert!(MixedModelBuilder::new()
        .data(&df2)
        .response("y")
        .residual_interaction("dup", AR1::new(1.0, 0.1), "g", AR1::correlation(0.1))
        .build()
        .is_err());
}
