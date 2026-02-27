use clap::{Parser, Subcommand};
use anyhow::{Context, Result};

use plant_breeding_lmm_core as core;
use core::data::DataFrame;
use core::diagnostics::{wald_tests, format_wald_tests};
use core::genetics::{compute_a_inverse, Pedigree};
use core::lmm::{AiReml, EmReml};
use core::model::MixedModelBuilder;
use core::variance::Identity;

#[derive(Parser)]
#[command(name = "openblup")]
#[command(version)]
#[command(about = "Open-source REML and BLUP for plant and animal breeding")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fit a linear mixed model via REML
    Fit {
        /// Path to data CSV file
        #[arg(short, long)]
        data: String,

        /// Response variable column name
        #[arg(short, long)]
        response: String,

        /// Fixed effects formula (e.g. "mu + rep + block")
        #[arg(short, long)]
        fixed: String,

        /// Random effects (repeatable, e.g. --random genotype --random sire)
        #[arg(long)]
        random: Vec<String>,

        /// Path to pedigree CSV (columns: id, sire, dam).
        /// When provided, the first random term uses the A-matrix.
        #[arg(long)]
        pedigree: Option<String>,

        /// REML algorithm: "ai" (default) or "em"
        #[arg(long, default_value = "ai")]
        algorithm: String,

        /// Maximum number of REML iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,

        /// Convergence tolerance
        #[arg(long, default_value = "1e-6")]
        tolerance: f64,

        /// Output format: "text" (default) or "json"
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Compute the inverse of the additive relationship matrix from a pedigree
    Ainverse {
        /// Path to pedigree CSV (columns: id, sire, dam)
        #[arg(short, long)]
        pedigree: String,

        /// Include inbreeding coefficients in output
        #[arg(long)]
        inbreeding: bool,
    },
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Fit {
            data,
            response,
            fixed,
            random,
            pedigree,
            algorithm,
            max_iter,
            tolerance,
            format,
        } => {
            cmd_fit(
                &data, &response, &fixed, &random, pedigree.as_deref(),
                &algorithm, max_iter, tolerance, &format,
            )
        }
        Commands::Ainverse {
            pedigree,
            inbreeding,
        } => cmd_ainverse(&pedigree, inbreeding),
    }
}

fn cmd_fit(
    data_path: &str,
    response: &str,
    fixed: &str,
    random_terms: &[String],
    pedigree_path: Option<&str>,
    algorithm: &str,
    max_iter: usize,
    tolerance: f64,
    output_format: &str,
) -> Result<()> {
    // Load data
    let df = DataFrame::from_csv(data_path)
        .with_context(|| format!("Failed to load data from '{}'", data_path))?;

    eprintln!(
        "Loaded {} observations, {} columns from '{}'",
        df.nrows(),
        df.ncols(),
        data_path
    );

    // Load pedigree if provided
    let pedigree = if let Some(ped_path) = pedigree_path {
        let mut ped = Pedigree::from_csv(ped_path)
            .with_context(|| format!("Failed to load pedigree from '{}'", ped_path))?;
        ped.sort_pedigree()
            .context("Failed to sort pedigree")?;
        eprintln!(
            "Loaded pedigree with {} animals from '{}'",
            ped.n_animals(),
            ped_path
        );
        Some(ped)
    } else {
        None
    };

    // Build model
    let mut builder = MixedModelBuilder::new()
        .data(&df)
        .response(response)
        .fixed(fixed)
        .max_iterations(max_iter)
        .convergence(tolerance);

    for (i, term) in random_terms.iter().enumerate() {
        let ginv = if i == 0 {
            if let Some(ref ped) = pedigree {
                // Use A-inverse for the first random term when pedigree is provided
                match compute_a_inverse(ped) {
                    Ok(ainv) => {
                        eprintln!(
                            "Using A-inverse ({0}x{0}) for random term '{1}'",
                            ainv.rows(),
                            term
                        );
                        Some(ainv)
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not compute A-inverse: {}. Using identity.", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        builder = builder.random(term, Identity::new(1.0), ginv);
    }

    let mut model = builder
        .build()
        .context("Failed to build mixed model")?;

    eprintln!(
        "Model: {} fixed params, {} random terms, algorithm={}",
        model.x.cols(),
        model.z_blocks.len(),
        algorithm
    );

    // Fit model
    let result = match algorithm.to_lowercase().as_str() {
        "ai" | "ai-reml" => {
            let solver = AiReml::new(max_iter, tolerance);
            solver.fit(&mut model).context("AI-REML fitting failed")?
        }
        "em" | "em-reml" => {
            let solver = EmReml::new(max_iter, tolerance);
            solver.fit(&mut model).context("EM-REML fitting failed")?
        }
        other => {
            anyhow::bail!(
                "Unknown algorithm '{}'. Use 'ai' (default) or 'em'.",
                other
            );
        }
    };

    // Output results
    match output_format.to_lowercase().as_str() {
        "json" => print_json(&result)?,
        _ => print_text(&result),
    }

    Ok(())
}

fn print_text(result: &core::lmm::FitResult) {
    println!("{}", result.summary());

    // Wald tests
    let tests = wald_tests(result);
    if !tests.is_empty() {
        println!("{}", format_wald_tests(&tests));
    }
}

fn print_json(result: &core::lmm::FitResult) -> Result<()> {
    // Build a JSON-serializable representation
    let mut map = serde_json::Map::new();

    map.insert(
        "converged".to_string(),
        serde_json::Value::Bool(result.converged),
    );
    map.insert(
        "n_iterations".to_string(),
        serde_json::json!(result.n_iterations),
    );
    map.insert(
        "log_likelihood".to_string(),
        serde_json::json!(result.log_likelihood),
    );
    map.insert("aic".to_string(), serde_json::json!(result.aic()));
    map.insert("bic".to_string(), serde_json::json!(result.bic()));
    map.insert("n_obs".to_string(), serde_json::json!(result.n_obs));

    // Variance components
    let vc: Vec<serde_json::Value> = result
        .variance_components
        .iter()
        .zip(result.variance_se.iter().chain(std::iter::repeat(&0.0)))
        .map(|(v, se)| {
            serde_json::json!({
                "name": v.name,
                "structure": v.structure,
                "sigma2": v.parameters[0].1,
                "se": se,
            })
        })
        .collect();
    map.insert("variance_components".to_string(), serde_json::json!(vc));

    // Fixed effects
    let fe: Vec<serde_json::Value> = result
        .fixed_effects
        .iter()
        .map(|e| {
            serde_json::json!({
                "term": e.term,
                "level": e.level,
                "estimate": e.estimate,
                "se": e.se,
            })
        })
        .collect();
    map.insert("fixed_effects".to_string(), serde_json::json!(fe));

    // Wald tests
    let tests = wald_tests(result);
    let wt: Vec<serde_json::Value> = tests
        .iter()
        .map(|t| {
            serde_json::json!({
                "term": t.term,
                "f_statistic": t.f_statistic,
                "num_df": t.num_df,
                "den_df": t.den_df,
                "p_value": t.p_value,
            })
        })
        .collect();
    map.insert("wald_tests".to_string(), serde_json::json!(wt));

    // Random effects (top 10 per term)
    let re: Vec<serde_json::Value> = result
        .random_effects
        .iter()
        .map(|block| {
            let mut sorted = block.effects.clone();
            sorted.sort_by(|a, b| {
                b.estimate
                    .partial_cmp(&a.estimate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top: Vec<serde_json::Value> = sorted
                .iter()
                .take(10)
                .map(|e| {
                    serde_json::json!({
                        "level": e.level,
                        "estimate": e.estimate,
                        "se": e.se,
                    })
                })
                .collect();
            serde_json::json!({
                "term": block.term,
                "n_levels": block.effects.len(),
                "top_effects": top,
            })
        })
        .collect();
    map.insert("random_effects".to_string(), serde_json::json!(re));

    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(map))?;
    println!("{}", json_str);
    Ok(())
}

fn cmd_ainverse(pedigree_path: &str, _inbreeding: bool) -> Result<()> {
    let mut ped = Pedigree::from_csv(pedigree_path)
        .with_context(|| format!("Failed to load pedigree from '{}'", pedigree_path))?;

    ped.sort_pedigree()
        .context("Failed to sort pedigree")?;

    let n = ped.n_animals();
    println!("Pedigree loaded: {} animals", n);

    let ainv = compute_a_inverse(&ped)
        .context("Failed to compute A-inverse")?;

    println!("A-inverse dimensions: {} x {}", ainv.rows(), ainv.cols());
    println!("A-inverse non-zeros:  {}", ainv.nnz());
    println!(
        "Density: {:.2}%",
        100.0 * ainv.nnz() as f64 / (n * n) as f64
    );

    // Print a few diagonal elements as a sanity check
    println!("\nFirst diagonal entries of A-inverse:");
    for i in 0..n.min(5) {
        let val = ainv.get(i, i).copied().unwrap_or(0.0);
        println!("  {}: {:.4}", ped.animal_id(i), val);
    }

    Ok(())
}
