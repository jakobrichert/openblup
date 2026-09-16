//! The README "Quick Start (Rust)" example.
//!
//! Run from the repository root:
//! `cargo run -p plant-breeding-lmm-core --example quick_start`

use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::diagnostics::{format_wald_tests, wald_tests};
use plant_breeding_lmm_core::genetics::Pedigree;
use plant_breeding_lmm_core::model::MixedModelBuilder;
use plant_breeding_lmm_core::variance::Identity;

fn main() -> plant_breeding_lmm_core::Result<()> {
    // Load data; numeric columns with NA become NaN, so drop missing plots first.
    let mut df = DataFrame::from_csv("examples/field_trial.csv")?.drop_missing("yield")?;
    df.as_factor("rep")?; // rep is coded 1, 2, 3 in the file

    // yield = mu + rep (fixed, treatment contrasts) + genotype (random, IID) + error
    let mut model = MixedModelBuilder::new()
        .data(&df)
        .response("yield")
        .fixed("mu + rep")
        .random("genotype", Identity::new(1.0), None)
        .build()?;
    let result = model.fit_reml()?;
    println!("{}", result.summary());
    println!("{}", format_wald_tests(&wald_tests(&result)));

    // Animal model: the random term follows the pedigree (all animals get a BLUP,
    // including the 12 founders without records)
    let records = DataFrame::from_csv("examples/animal_records.csv")?;
    let ped = Pedigree::from_csv("examples/animal_pedigree.csv")?;
    let mut model = MixedModelBuilder::new()
        .data(&records)
        .response("weight")
        .fixed("sex")
        .random_pedigree("animal", Identity::new(1.0), &ped)
        .build()?;
    let result = model.fit_reml()?;
    println!("{}", result.summary());
    Ok(())
}
