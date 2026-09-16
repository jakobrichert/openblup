//! `openblup` — command-line interface to the OpenBLUP mixed model engine.

use std::fs::File;
use std::io::{BufWriter, Write};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use plant_breeding_lmm_core as core;

use core::data::DataFrame;
use core::diagnostics::{format_wald_tests, wald_tests, CrossValResult, WaldTest};
use core::genetics::{
    compute_a_inverse, compute_a_inverse_with_inbreeding, compute_inbreeding, Pedigree,
};
use core::lmm::{reliability_from_se, AiReml, EmReml, FitResult};
use core::model::{MixedModel, MixedModelBuilder};
use core::variance::StructureSpec;

#[derive(Parser)]
#[command(name = "openblup")]
#[command(version)]
#[command(about = "Open-source REML and BLUP for plant and animal breeding")]
#[command(after_help = "\
TERM SPECIFICATIONS (--random, --residual)
  A random term is `factor[:structure]` or an interaction
  `factor[:structure]*factor[:structure]`. Structures: idv (default, one
  variance), ar1 / ar1(rho), ar1c / ar1c(rho) (correlation only), diag,
  us, fa1, fa2, ... A pedigree (--pedigree) attaches to the factor named by
  --pedigree-term (default: the first term's factor, or the inner factor of
  an interaction).

  Examples:
    --random genotype
    --random animal --pedigree ped.csv
    --random \"env:fa1*genotype\"              genotype-by-environment, FA1 across environments
    --random \"env:diag*animal\" --pedigree ped.csv --pedigree-term animal
    --random \"row:ar1*col:ar1c\"              separable spatial random effect
    --residual \"row:ar1*col:ar1c\"            AR1 x AR1 residual over the field grid
    --residual ar1                            AR1 residual in data order")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// REML algorithm.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Algorithm {
    /// Average Information REML (fast, quadratic convergence)
    Ai,
    /// Expectation-Maximisation REML (slow but very robust; identity structures only)
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

/// Denominator degrees of freedom for Wald F-tests.
#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Ddf {
    /// n - rank(X)
    Containment,
    /// Satterthwaite approximation (AI-REML fits with no variance parameter
    /// on the boundary; otherwise falls back to containment)
    Satterthwaite,
    /// Kenward-Roger bias-adjusted F-test (AI-REML fits of models with
    /// scaled-identity / relationship-matrix terms and an IID residual;
    /// otherwise falls back to Satterthwaite, then containment)
    KenwardRoger,
}

impl Ddf {
    fn name(self) -> &'static str {
        match self {
            Ddf::Containment => "containment",
            Ddf::Satterthwaite => "satterthwaite",
            Ddf::KenwardRoger => "kenward-roger",
        }
    }
}

#[derive(Subcommand)]
// The enum is built once from argv; the size difference between variants
// is irrelevant here.
#[allow(clippy::large_enum_variant)]
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

        /// Random term (repeatable), e.g. --random genotype --random "env:fa1*genotype".
        /// Numeric columns are automatically treated as factors.
        #[arg(long, value_name = "TERM")]
        random: Vec<String>,

        /// Residual structure, e.g. "row:ar1*col:ar1c" or "ar1" (default: IID)
        #[arg(long, value_name = "TERM")]
        residual: Option<String>,

        /// Treat a numeric column as a factor (repeatable), e.g. --factor rep
        #[arg(long, value_name = "COLUMN")]
        factor: Vec<String>,

        /// Path to pedigree CSV (columns: animal, sire, dam; 0/NA = unknown).
        /// The pedigree A-inverse (with inbreeding) is attached to the factor
        /// named by --pedigree-term.
        #[arg(long)]
        pedigree: Option<String>,

        /// Factor that uses the pedigree relationship matrix
        #[arg(long, value_name = "FACTOR", requires = "pedigree")]
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

        /// Denominator degrees of freedom for the Wald F-tests
        #[arg(long, value_enum, default_value_t = Ddf::Containment)]
        ddf: Ddf,

        /// Output format
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,

        /// Write all random effects (BLUPs) with SE and reliability to a CSV file
        #[arg(long, value_name = "PATH")]
        blups: Option<String>,

        /// Write residual diagnostics (fitted values, residuals, leverage,
        /// Cook's distance) to a CSV file
        #[arg(long, value_name = "PATH")]
        diagnostics: Option<String>,

        /// Run k-fold cross-validation of prediction accuracy (single random
        /// term models with an IID residual; skipped with a warning otherwise)
        #[arg(long, value_name = "K")]
        cv: Option<usize>,

        /// Random seed for the cross-validation folds
        #[arg(long, default_value_t = 0)]
        cv_seed: u64,
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
            residual,
            factor,
            pedigree,
            pedigree_term,
            algorithm,
            max_iter,
            tolerance,
            ddf,
            format,
            blups,
            diagnostics,
            cv,
            cv_seed,
        } => cmd_fit(FitOptions {
            data,
            response,
            fixed,
            random,
            residual,
            factor,
            pedigree,
            pedigree_term,
            algorithm,
            max_iter,
            tolerance,
            ddf,
            format,
            blups,
            diagnostics,
            cv,
            cv_seed,
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
    residual: Option<String>,
    factor: Vec<String>,
    pedigree: Option<String>,
    pedigree_term: Option<String>,
    algorithm: Algorithm,
    max_iter: usize,
    tolerance: f64,
    ddf: Ddf,
    format: OutputFormat,
    blups: Option<String>,
    diagnostics: Option<String>,
    cv: Option<usize>,
    cv_seed: u64,
}

/// One factor of a term specification: `name[:structure]`.
#[derive(Debug, Clone)]
struct FactorSpec {
    name: String,
    structure: StructureSpec,
    explicit_structure: bool,
}

/// A random or residual term: one factor or an interaction of two.
#[derive(Debug, Clone)]
struct TermSpec {
    outer: FactorSpec,
    inner: Option<FactorSpec>,
}

impl TermSpec {
    /// The factors' column names.
    fn columns(&self) -> Vec<&str> {
        let mut v = vec![self.outer.name.as_str()];
        if let Some(inner) = &self.inner {
            v.push(inner.name.as_str());
        }
        v
    }

    fn label(&self) -> String {
        match &self.inner {
            Some(inner) => format!("{}:{}", self.outer.name, inner.name),
            None => self.outer.name.clone(),
        }
    }
}

fn parse_factor_spec(s: &str) -> Result<FactorSpec> {
    let s = s.trim();
    if s.is_empty() {
        bail!("Empty factor in term specification");
    }
    match s.split_once(':') {
        Some((name, structure)) => Ok(FactorSpec {
            name: name.trim().to_string(),
            structure: StructureSpec::parse(structure)
                .with_context(|| format!("Invalid structure in '{}'", s))?,
            explicit_structure: true,
        }),
        None => Ok(FactorSpec {
            name: s.to_string(),
            structure: StructureSpec::Identity { sigma2: 1.0 },
            explicit_structure: false,
        }),
    }
}

/// Parse `factor[:struct]` or `factor[:struct]*factor[:struct]`.
fn parse_term_spec(s: &str) -> Result<TermSpec> {
    let parts: Vec<&str> = s.split('*').collect();
    match parts.as_slice() {
        [single] => Ok(TermSpec {
            outer: parse_factor_spec(single)?,
            inner: None,
        }),
        [outer, inner] => Ok(TermSpec {
            outer: parse_factor_spec(outer)?,
            inner: Some(parse_factor_spec(inner)?),
        }),
        _ => bail!(
            "Term '{}' has more than two factors; only `a*b` interactions are supported",
            s
        ),
    }
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
    if opts.cv.is_some_and(|k| k < 2) {
        bail!("--cv needs at least 2 folds");
    }

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

    // ---- terms ----
    let random_terms: Vec<TermSpec> = opts
        .random
        .iter()
        .map(|s| parse_term_spec(s).with_context(|| format!("Invalid --random '{}'", s)))
        .collect::<Result<_>>()?;
    let residual_term = match &opts.residual {
        Some(s) => Some(parse_term_spec(s).with_context(|| format!("Invalid --residual '{}'", s))?),
        None => None,
    };
    for term in random_terms.iter().chain(residual_term.iter()) {
        for col in term.columns() {
            if df.get_column(col).is_err() {
                bail!(
                    "Column '{}' (term '{}') not found in the data (columns: {})",
                    col,
                    term.label(),
                    df.column_names().join(", ")
                );
            }
            if df.get_factor(col).is_err() {
                df.as_factor(col)
                    .with_context(|| format!("Term '{}' must use categorical columns", col))?;
                eprintln!("Treating numeric column '{}' as a factor", col);
            }
        }
    }

    // ---- pedigree ----
    let pedigree = match &opts.pedigree {
        Some(path) => Some(load_pedigree(path)?),
        None => None,
    };
    let pedigree_factor: Option<String> = match (&pedigree, &opts.pedigree_term) {
        (None, _) => None,
        (Some(_), Some(term)) => {
            let known = random_terms
                .iter()
                .any(|t| t.columns().contains(&term.as_str()));
            if !known {
                bail!(
                    "--pedigree-term '{}' is not a factor of any --random term",
                    term
                );
            }
            Some(term.clone())
        }
        (Some(_), None) => match random_terms.first() {
            Some(t) => Some(
                t.inner
                    .as_ref()
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| t.outer.name.clone()),
            ),
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

    for term in &random_terms {
        match &term.inner {
            None => {
                let uses_ped = pedigree_factor.as_deref() == Some(term.outer.name.as_str());
                if uses_ped && term.outer.explicit_structure && !term.outer.structure.is_identity()
                {
                    bail!(
                        "Term '{}': a pedigree can only be combined with the default (idv) \
                         structure; use an interaction such as env:fa1*{} for structured \
                         genetic effects",
                        term.label(),
                        term.outer.name
                    );
                }
                if uses_ped {
                    let ped = pedigree.as_ref().unwrap();
                    eprintln!(
                        "Using pedigree A-inverse ({0}x{0}, with inbreeding) for random term '{1}'",
                        ped.n_animals(),
                        term.outer.name
                    );
                    builder = builder.random_spec(
                        &term.outer.name,
                        term.outer.structure.clone(),
                        Some(ped),
                    );
                } else {
                    builder =
                        builder.random_spec(&term.outer.name, term.outer.structure.clone(), None);
                }
            }
            Some(inner) => {
                if pedigree_factor.as_deref() == Some(term.outer.name.as_str()) {
                    bail!(
                        "Term '{}': the pedigree factor must be the second (inner) factor of \
                         an interaction, e.g. {}:{}*{}",
                        term.label(),
                        inner.name,
                        "fa1",
                        term.outer.name
                    );
                }
                let uses_ped = pedigree_factor.as_deref() == Some(inner.name.as_str());
                let ped = if uses_ped { pedigree.as_ref() } else { None };
                if let Some(p) = ped {
                    eprintln!(
                        "Using pedigree A-inverse ({0}x{0}, with inbreeding) for factor '{1}' of term '{2}'",
                        p.n_animals(),
                        inner.name,
                        term.label()
                    );
                }
                let inner_spec = if inner.explicit_structure {
                    Some(inner.structure.clone())
                } else {
                    None
                };
                builder = builder.random_interaction_spec(
                    &term.outer.name,
                    term.outer.structure.clone(),
                    &inner.name,
                    inner_spec,
                    ped,
                );
            }
        }
    }

    if let Some(res) = &residual_term {
        match &res.inner {
            None => {
                builder = builder.residual_spec(res.outer.structure.clone());
            }
            Some(inner) => {
                builder = builder.residual_interaction_spec(
                    &res.outer.name,
                    res.outer.structure.clone(),
                    &inner.name,
                    inner.structure.clone(),
                );
            }
        }
    }

    let mut model = builder.build().context("Failed to build mixed model")?;

    eprintln!(
        "Model: {} fixed params, {} random terms, {} variance parameters, algorithm={:?}{}",
        model.x.cols(),
        model.z_blocks.len(),
        model
            .random_var_structs
            .iter()
            .map(|v| v.n_params())
            .sum::<usize>()
            + model.residual_var_struct.n_params(),
        opts.algorithm,
        if model.needs_general_engine() {
            " (general engine)"
        } else {
            ""
        }
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

    // ---- Wald tests ----
    let (tests, ddf_used) = wald_tests_for(&result, opts.ddf);
    if ddf_used != opts.ddf {
        eprintln!(
            "Warning: {} df need an AI-REML fit with no variance parameter on the boundary \
             (Kenward-Roger also needs scaled-identity terms and an IID residual); \
             reporting {} df instead.",
            opts.ddf.name(),
            ddf_used.name()
        );
    }

    // ---- cross-validation ----
    // An unsupported model (several random terms, structured residual) only
    // skips the cross-validation, like an unavailable ddf method, so the fit
    // itself is still reported.
    let cv_result = match opts.cv.map(|k| model.cross_validate(k, opts.cv_seed)) {
        Some(Ok(cv)) => Some(cv),
        Some(Err(core::LmmError::ModelSpec(msg))) => {
            eprintln!("Warning: skipping cross-validation: {}.", msg);
            None
        }
        Some(Err(e)) => return Err(e).context("Cross-validation failed"),
        None => None,
    };

    // ---- output (files first, so a closed stdout pipe cannot lose them) ----
    if let Some(path) = &opts.blups {
        let n = write_blups_csv(&result, path)?;
        eprintln!("Wrote {} random effects to '{}'", n, path);
    }
    if let Some(path) = &opts.diagnostics {
        let n = write_diagnostics_csv(&result, &model, path)?;
        eprintln!("Wrote diagnostics for {} observations to '{}'", n, path);
    }

    match opts.format {
        OutputFormat::Text => {
            println!("{}", result.summary());
            if !tests.is_empty() {
                println!("{}", format_wald_tests(&tests));
                println!("Denominator df: {}\n", ddf_used.name());
            }
            if let Some(cv) = &cv_result {
                println!("{}", cv.summary());
            }
        }
        OutputFormat::Json => {
            let mut json = result_to_json(&result, &tests, ddf_used);
            if let Some(cv) = &cv_result {
                json["cross_validation"] = cv_to_json(cv);
            }
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }

    Ok(())
}

fn wald_tests_for(result: &FitResult, ddf: Ddf) -> (Vec<WaldTest>, Ddf) {
    if ddf == Ddf::KenwardRoger {
        if let Some(tests) = result.wald_tests_kenward_roger() {
            return (tests, Ddf::KenwardRoger);
        }
    }
    if ddf != Ddf::Containment {
        if let Some(tests) = result.wald_tests_satterthwaite() {
            return (tests, Ddf::Satterthwaite);
        }
    }
    (wald_tests(result), Ddf::Containment)
}

fn result_to_json(result: &FitResult, tests: &[WaldTest], ddf: Ddf) -> serde_json::Value {
    let variance_components: Vec<serde_json::Value> = result
        .variance_components
        .iter()
        .map(|v| {
            let parameters: Vec<serde_json::Value> = v
                .parameters
                .iter()
                .enumerate()
                .map(|(i, (name, value))| {
                    serde_json::json!({
                        "name": name,
                        "value": value,
                        "se": v.se.get(i).copied().filter(|se| *se > 0.0),
                        "at_boundary": v.at_boundary.get(i).copied().unwrap_or(false),
                    })
                })
                .collect();
            serde_json::json!({
                "name": v.name,
                "structure": v.structure,
                "sigma2": v.sigma2(),
                "se": v.se.first().copied().filter(|se| *se > 0.0),
                "parameters": parameters,
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

    let wald: Vec<serde_json::Value> = tests
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
        "ddf_method": ddf.name(),
        "wald_tests": wald,
        "random_effects": random_effects,
    })
}

fn cv_to_json(cv: &CrossValResult) -> serde_json::Value {
    let folds: Vec<serde_json::Value> = cv
        .folds
        .iter()
        .map(|f| {
            serde_json::json!({
                "fold": f.fold,
                "n_validation": f.validation_indices.len(),
                "accuracy": f.accuracy,
                "msep": f.msep,
            })
        })
        .collect();
    serde_json::json!({
        "n_folds": cv.folds.len(),
        "accuracy": cv.accuracy,
        "msep": cv.msep,
        "bias": cv.bias,
        "mae": cv.mae,
        "folds": folds,
    })
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

/// Write residual diagnostics to a CSV file. Returns the number of rows.
fn write_diagnostics_csv(result: &FitResult, model: &MixedModel, path: &str) -> Result<usize> {
    let diag = result
        .residual_diagnostics(model)
        .ok_or_else(|| anyhow::anyhow!("Residual diagnostics need an IID residual structure"))?;
    let file = File::create(path).with_context(|| format!("Cannot create '{}'", path))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "obs,observed,fitted,residual,marginal_residual,standardized,leverage,cooks_distance"
    )?;
    for i in 0..diag.fitted.len() {
        writeln!(
            w,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            i + 1,
            model.y[i],
            diag.fitted[i],
            diag.conditional[i],
            diag.marginal[i],
            diag.standardized[i],
            diag.leverage[i],
            diag.cooks_distance[i]
        )?;
    }
    w.flush()?;
    Ok(diag.fitted.len())
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
