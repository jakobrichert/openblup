//! Scaling benchmark for the sparse animal-model path (AI-REML with a
//! pedigree relationship matrix).
//!
//! Run from the repository root, always in release mode:
//!
//! ```text
//! cargo run --release -p plant-breeding-lmm-core --example animal_model_benchmark
//! cargo run --release -p plant-breeding-lmm-core --example animal_model_benchmark -- 5000 20000 60000
//! cargo run --release -p plant-breeding-lmm-core --example animal_model_benchmark -- --mating random 20000
//! ```
//!
//! The data are simulated deterministically: 10% founders, then nine
//! generations of equal size with a record (`y = sex + a + e`, h² = 1/3) on
//! every non-founder. Two mating designs are available:
//!
//! * `sires` (default): each generation uses 2% of the previous generation as
//!   sires and random dams, as in livestock populations;
//! * `random`: both parents are drawn at random from the previous generation,
//!   which gives the least structured pedigree and the most fill-in.

use std::time::Instant;

use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::genetics::Pedigree;
use plant_breeding_lmm_core::lmm::SparseMmeStructure;
use plant_breeding_lmm_core::matrix::sparse_cholesky::CholeskyAnalysis;
use plant_breeding_lmm_core::model::MixedModelBuilder;
use plant_breeding_lmm_core::variance::Identity;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, PartialEq)]
enum Mating {
    Sires,
    Random,
}

fn normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn simulate(n_animals: usize, mating: Mating, seed: u64) -> (Pedigree, DataFrame) {
    const S2A: f64 = 20.0;
    const S2E: f64 = 40.0;
    let mut rng = StdRng::seed_from_u64(seed);
    let n_founders = (n_animals / 10).max(2);
    let per_gen = ((n_animals - n_founders) / 9).max(2);

    let mut ped = Pedigree::new();
    let mut ids = Vec::with_capacity(n_animals);
    let mut bv = Vec::with_capacity(n_animals);
    for i in 0..n_founders {
        let id = format!("F{i}");
        ped.add_animal(&id, None, None).unwrap();
        ids.push(id);
        bv.push(normal(&mut rng) * S2A.sqrt());
    }
    let mut prev: Vec<usize> = (0..n_founders).collect();
    let (mut y, mut animal, mut sex) = (Vec::new(), Vec::new(), Vec::new());
    for g in 0..9 {
        let n_sires = match mating {
            Mating::Sires => (prev.len() / 50).max(1),
            Mating::Random => prev.len(),
        };
        let sires = &prev[..n_sires];
        let mut cur = Vec::with_capacity(per_gen);
        for i in 0..per_gen {
            let s = sires[rng.gen_range(0..sires.len())];
            let mut d = prev[rng.gen_range(0..prev.len())];
            while d == s {
                d = prev[rng.gen_range(0..prev.len())];
            }
            let id = format!("G{g}_{i}");
            ped.add_animal(&id, Some(&ids[s]), Some(&ids[d])).unwrap();
            let a = 0.5 * (bv[s] + bv[d]) + normal(&mut rng) * (0.5 * S2A).sqrt();
            let male = rng.gen::<bool>();
            y.push(100.0 + if male { 4.0 } else { 0.0 } + a + normal(&mut rng) * S2E.sqrt());
            animal.push(id.clone());
            sex.push(if male { "male" } else { "female" });
            ids.push(id);
            bv.push(a);
            cur.push(ids.len() - 1);
        }
        prev = cur;
    }
    ped.sort_pedigree().unwrap();

    let mut df = DataFrame::new();
    df.add_float_column("y", y).unwrap();
    let animal: Vec<&str> = animal.iter().map(String::as_str).collect();
    df.add_factor_column("animal", &animal).unwrap();
    df.add_factor_column("sex", &sex).unwrap();
    (ped, df)
}

fn main() -> plant_breeding_lmm_core::Result<()> {
    let mut mating = Mating::Sires;
    let mut sizes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mating" => {
                mating = match args.next().as_deref() {
                    Some("sires") => Mating::Sires,
                    Some("random") => Mating::Random,
                    other => panic!("--mating sires|random, got {other:?}"),
                }
            }
            n => sizes.push(n.parse::<usize>().expect("number of animals")),
        }
    }
    if sizes.is_empty() {
        sizes = vec![5_000, 20_000, 60_000];
    }
    println!(
        "Mating: {}\n",
        if mating == Mating::Sires {
            "sires (2% of each generation)"
        } else {
            "random"
        }
    );
    println!(
        "| animals | records | equations | nnz(L) | setup | fit | iterations | per iteration | σ²a | σ²e |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for n in sizes {
        let (ped, df) = simulate(n, mating, 2024);
        let start = Instant::now();
        let mut model = MixedModelBuilder::new()
            .data(&df)
            .response("y")
            .fixed("sex")
            .random_pedigree("animal", Identity::new(1.0), &ped)
            .max_iterations(50)
            .convergence(1e-6)
            .build()?;
        let setup = start.elapsed().as_secs_f64();

        let mme = SparseMmeStructure::for_model(&model).assemble(1.0, &[1.0]);
        let factor_nnz = CholeskyAnalysis::new(&mme.coeff_matrix)?.factor_nnz();

        let start = Instant::now();
        let result = model.fit_reml()?;
        let fit = start.elapsed().as_secs_f64();
        println!(
            "| {} | {} | {} | {:.1}M | {:.2}s | {:.2}s | {}{} | {:.2}s | {:.2} | {:.2} |",
            ped.n_animals(),
            model.n_obs,
            mme.dim,
            factor_nnz as f64 / 1e6,
            setup,
            fit,
            result.n_iterations,
            if result.converged {
                ""
            } else {
                " (not converged)"
            },
            fit / result.n_iterations as f64,
            result.variance_component("animal").unwrap_or(f64::NAN),
            result.variance_component("residual").unwrap_or(f64::NAN),
        );
    }
    Ok(())
}
