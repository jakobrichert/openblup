//! `openblup` — command-line interface to the OpenBLUP mixed model engine.

use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use plant_breeding_lmm_core as core;

use core::data::DataFrame;
use core::diagnostics::{format_wald_tests, wald_tests};
use core::genetics::{
    compute_a_inverse, compute_a_inverse_with_inbreeding, compute_inbreeding, Pedigree,
};
use core::lmm::{reliability_from_se, AiReml, EmReml, FitResult};
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

/// REML algorithm.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Algorithm {
    /// Average Information REML (fast, quadratic convergence)
    Ai,
    /// Expectation-Maximisation REML (slow but very robust)
    Em,
}

/// Output format for `fit`.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    /// Human-readable summary
    Text,
    /// Machine-readable JSON (all effects included)
    Json,
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

        /// Fixed effects formula, e.g. "mu + rep + block" (factors use
        /// treatment contrasts against their first level)
        #[arg(short, long, default_value = "mu")]
        fixed: String,

        /// Random effect column (repeatable, e.g. --random genotype --random sire).
        /// Numeric columns are automatically treated as factors.
        #[arg(long, value_name = "COLUMN")]
        random: Vec<String>,

        /// Treat a numeric column as a factor (repeatable), e.g. --factor rep
        #[arg(long, value_name = "COLUMN")]
        factor: Vec<String>,

        /// Path to pedigree CSV (columns: animal, sire, dam; 0/NA = unknown).
        /// The pedigree A-inverse (with inbreeding) is attached to the random
        /// term named by --pedigree-term, or to the first --random term.
        #[arg(long)]
        pedigree: Option<String>,

        /// Random term that uses the pedigree relationship matrix
        #[arg(long, value_name = "COLUMN", requires = "pedigree")]
        pedigree_term: Option<String>,

        /// REML algorithm
        #[arg(long, value_enum, default_value_t = Algorithm::Ai)]
        algorithm: Algorithm,

        /// Maximum number of REML iterations
        #[arg(long, default_value_t = 50)]
        max_iter: usize,

        /// Convergence tolerance (relative change in variance parameters)
        #[arg(long, default_value_t = 1e-6)]
        tolerance: f64,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,

        /// Write all random effects (BLUPs) with SE and reliability to a CSV file
        #[arg(long, value_name = "PATH")]
        blups: Option<String>,
    },

    /// Compute the inverse of the additive relationship matrix from a pedigree
    Ainverse {
        /// Path to pedigree CSV (columns: animal, sire, dam; 0/NA = unknown)
        #[arg(short, long)]
        pedigree: String,

        /// Account for inbreeding (Meuwissen & Luo 1992) and report
        /// inbreeding coefficients
        #[arg(long)]
        inbreeding: bool,

        /// Write the A-inverse as a triplet CSV (row, col, value) using animal IDs
        #[arg(long, value_name = "PATH")]
        output: Option<String>,
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
            factor,
            pedigree,
            pedigree_term,
            algorithm,
            max_iter,
            tolerance,
            format,
            blups,
        } => cmd_fit(FitOptions {
            data,
            response,
            fixed,
            random,
            factor,
            pedigree,
            pedigree_term,
            algorithm,
            max_iter,
            tolerance,
            format,
            blups,
        }),
        Commands::Ainverse {
            pedigree,
            inbreeding,
            output,
        } => cmd_ainverse(&pedigree, inbreeding, output.as_deref()),
    }
}

struct FitOptions {
    data: String,
    response: String,
    fixed: String,
    random: Vec<String>,
    factor: Vec<String>,
    pedigree: Option<String>,
    pedigree_term: Option<String>,
    algorithm: Algorithm,
    max_iter: usize,
    tolerance: f64,
    format: OutputFormat,
    blups: Option<String>,
}

/// Load, validate and sort a pedigree file.
fn load_pedigree(path: &str) -> Result<Pedigree> {
    let mut ped = Pedigree::from_csv(path)
        .with_context(|| format!("Failed to load pedigree from '{}'", path))?;
    ped.validate()
        .with_context(|| format!("Pedigree '{}' is inconsistent", path))?;
    ped.sort_pedigree().context("Failed to sort pedigree")?;
    eprintln!(
        "Loaded pedigree with {} animals from '{}'",
        ped.n_animals(),
        path
    );
    Ok(ped)
}

fn cmd_fit(opts: FitOptions) -> Result<()> {
    // ---- data ----
    let mut df = DataFrame::from_csv(&opts.data)
        .with_context(|| format!("Failed to load data from '{}'", opts.data))?;
    eprintln!(
        "Loaded {} observations, {} columns from '{}'",
        df.nrows(),
        df.ncols(),
        opts.data
    );

    if df.get_float(&opts.response).is_err() {
        bail!(
            "Response column '{}' is missing or not numeric (columns: {})",
            opts.response,
            df.column_names().join(", ")
        );
    }
    if df.has_missing(&opts.response)? {
        let before = df.nrows();
        df = df.drop_missing(&opts.response)?;
        eprintln!(
            "Dropped {} rows with missing '{}' ({} remain)",
            before - df.nrows(),
            opts.response,
            df.nrows()
        );
    }
    if df.nrows() == 0 {
        bail!("No observations with a non-missing response");
    }

    for col in &opts.factor {
        df.as_factor(col)
            .with_context(|| format!("Cannot treat column '{}' as a factor", col))?;
    }
    for col in &opts.random {
        if df.get_factor(col).is_err() {
            df.as_factor(col)
                .with_context(|| format!("Random term '{}' must be a categorical column", col))?;
            eprintln!("Treating numeric column '{}' as a factor", col);
        }
    }

    // ---- pedigree ----
    let pedigree = match &opts.pedigree {
        Some(path) => Some(load_pedigree(path)?),
        None => None,
    };
    let pedigree_term = match (&pedigree, &opts.pedigree_term) {
        (None, _) => None,
        (Some(_), Some(term)) => {
            if !opts.random.contains(term) {
                bail!(
                    "--pedigree-term '{}' is not one of the --random terms",
                    term
                );
            }
            Some(term.clone())
        }
        (Some(_), None) => match opts.random.first() {
            Some(first) => Some(first.clone()),
            None => bail!("--pedigree requires at least one --random term"),
        },
    };

    // ---- model ----
    let mut builder = MixedModelBuilder::new()
        .data(&df)
        .response(&opts.response)
        .fixed(&opts.fixed)
        .max_iterations(opts.max_iter)
        .convergence(opts.tolerance);

    for term in &opts.random {
        match (&pedigree, &pedigree_term) {
            (Some(ped), Some(pt)) if pt == term => {
                eprintln!(
                    "Using pedigree A-inverse ({0}x{0}, with inbreeding) for random term '{1}'",
                    ped.n_animals(),
                    term
                );
                builder = builder.random_pedigree(term, Identity::new(1.0), ped);
            }
            _ => {
                builder = builder.random(term, Identity::new(1.0), None);
            }
        }
    }

    let mut model = builder.build().context("Failed to build mixed model")?;

    eprintln!(
        "Model: {} fixed params, {} random terms, algorithm={:?}",
        model.x.cols(),
        model.z_blocks.len(),
        opts.algorithm
    );

    // ---- fit ----
    let result = match opts.algorithm {
        Algorithm::Ai => AiReml::new(opts.max_iter, opts.tolerance)
            .fit(&mut model)
            .context("AI-REML fitting failed")?,
        Algorithm::Em => EmReml::new(opts.max_iter, opts.tolerance)
            .fit(&mut model)
            .context("EM-REML fitting failed")?,
    };

    if !result.converged {
        let last_change = result.history.last().map(|h| h.change).unwrap_or(f64::NAN);
        eprintln!(
            "Warning: REML did not converge in {} iterations (last relative change {:.2e}). \
             Consider --max-iter or --algorithm em.",
            result.n_iterations, last_change
        );
    }

    // ---- output (files first, so a closed stdout pipe cannot lose them) ----
    if let Some(path) = &opts.blups {
        let n = write_blups_csv(&result, path)?;
        eprintln!("Wrote {} random effects to '{}'", n, path);
    }

    match opts.format {
        OutputFormat::Text => print_text(&result),
        OutputFormat::Json => print_json(&result)?,
    }

    Ok(())
}

fn print_text(result: &FitResult) {
    println!("{}", result.summary());

    // Wald tests
    let tests = wald_tests(result);
    if !tests.is_empty() {
        println!("{}", format_wald_tests(&tests));
    }
}

fn result_to_json(result: &FitResult) -> serde_json::Value {
    let variance_components: Vec<serde_json::Value> = result
        .variance_components
        .iter()
        .enumerate()
        .map(|(k, v)| {
            serde_json::json!({
                "name": v.name,
                "structure": v.structure,
                "sigma2": v.parameters.first().map(|p| p.1),
                "parameters": v.parameters,
                "se": result.variance_se.get(k).copied().filter(|se| *se > 0.0),
            })
        })
        .collect();

    let fixed_effects: Vec<serde_json::Value> = result
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

    let wald: Vec<serde_json::Value> = wald_tests(result)
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

    let random_effects: Vec<serde_json::Value> = result
        .random_effects
        .iter()
        .map(|block| {
            let sigma2 = result.variance_component(&block.term);
            let effects: Vec<serde_json::Value> = block
                .effects
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "level": e.level,
                        "estimate": e.estimate,
                        "se": e.se,
                        "reliability": sigma2.map(|s2| reliability_from_se(e.se, s2)),
                    })
                })
                .collect();
            serde_json::json!({
                "term": block.term,
                "sigma2": sigma2,
                "n_levels": block.effects.len(),
                "effects": effects,
            })
        })
        .collect();

    serde_json::json!({
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "log_likelihood": result.log_likelihood,
        "aic": result.aic(),
        "bic": result.bic(),
        "n_obs": result.n_obs,
        "n_fixed_params": result.n_fixed_params,
        "n_variance_params": result.n_variance_params,
        "variance_components": variance_components,
        "fixed_effects": fixed_effects,
        "wald_tests": wald,
        "random_effects": random_effects,
    })
}

fn print_json(result: &FitResult) -> Result<()> {
    let json_str = serde_json::to_string_pretty(&result_to_json(result))?;
    println!("{}", json_str);
    Ok(())
}

/// Write every random effect to a CSV file. Returns the number of rows written.
fn write_blups_csv(result: &FitResult, path: &str) -> Result<usize> {
    let file = File::create(path).with_context(|| format!("Cannot create '{}'", path))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "term,level,estimate,se,reliability")?;
    let mut n = 0;
    for block in &result.random_effects {
        let sigma2 = result.variance_component(&block.term);
        for e in &block.effects {
            let rel = sigma2
                .map(|s2| format!("{:.6}", reliability_from_se(e.se, s2)))
                .unwrap_or_default();
            writeln!(
                w,
                "{},{},{:.6},{:.6},{}",
                csv_field(&block.term),
                csv_field(&e.level),
                e.estimate,
                e.se,
                rel
            )?;
            n += 1;
        }
    }
    w.flush()?;
    Ok(n)
}

/// Quote a CSV field if it contains separators or quotes.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn cmd_ainverse(pedigree_path: &str, inbreeding: bool, output: Option<&str>) -> Result<()> {
    let ped = load_pedigree(pedigree_path)?;
    let n = ped.n_animals();
    println!("Pedigree loaded: {} animals", n);

    let ainv = if inbreeding {
        compute_a_inverse_with_inbreeding(&ped)
            .context("Failed to compute A-inverse with inbreeding")?
    } else {
        compute_a_inverse(&ped).context("Failed to compute A-inverse")?
    };

    println!(
        "A-inverse method: Henderson (1976){}",
        if inbreeding {
            " with Meuwissen & Luo (1992) inbreeding"
        } else {
            ", inbreeding ignored (use --inbreeding)"
        }
    );
    println!("A-inverse dimensions: {} x {}", ainv.rows(), ainv.cols());
    println!("A-inverse non-zeros:  {}", ainv.nnz());
    if n > 0 {
        println!(
            "Density: {:.2}%",
            100.0 * ainv.nnz() as f64 / (n * n) as f64
        );
    }

    // Print a few diagonal elements as a sanity check
    println!("\nFirst diagonal entries of A-inverse:");
    for i in 0..n.min(5) {
        let val = ainv.get(i, i).copied().unwrap_or(0.0);
        println!("  {}: {:.4}", ped.animal_id(i), val);
    }

    if inbreeding {
        let f = compute_inbreeding(&ped).context("Failed to compute inbreeding")?;
        let n_inbred = f.iter().filter(|&&x| x > 1e-12).count();
        let mean_f = if n > 0 {
            f.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let max_f = f.iter().cloned().fold(0.0_f64, f64::max);
        println!(
            "\nInbreeding: {} of {} animals inbred, mean F = {:.4}, max F = {:.4}",
            n_inbred, n, mean_f, max_f
        );
        println!("{:<12} {:<12} {:<12} {:>8}", "animal", "sire", "dam", "F");
        for (i, f_i) in f.iter().enumerate() {
            let sire = ped.sire(i).map(|s| ped.animal_id(s)).unwrap_or("0");
            let dam = ped.dam(i).map(|d| ped.animal_id(d)).unwrap_or("0");
            println!(
                "{:<12} {:<12} {:<12} {:>8.4}",
                ped.animal_id(i),
                sire,
                dam,
                f_i
            );
        }
    }

    if let Some(path) = output {
        let file = File::create(path).with_context(|| format!("Cannot create '{}'", path))?;
        let mut w = BufWriter::new(file);
        writeln!(w, "row,col,value")?;
        let mut count = 0;
        for (val, (i, j)) in ainv.iter() {
            if *val != 0.0 {
                writeln!(
                    w,
                    "{},{},{}",
                    csv_field(ped.animal_id(i)),
                    csv_field(ped.animal_id(j)),
                    val
                )?;
                count += 1;
            }
        }
        w.flush()?;
        println!("\nWrote {} non-zero entries to '{}'", count, path);
    }

    Ok(())
}
