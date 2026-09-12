# OpenBLUP

**Open-source REML and BLUP for plant and animal breeding** — a modern linear mixed model engine written in Rust with Python bindings, a command-line tool and a WebAssembly build.

## Why This Project?

**The breeding community deserves modern, open tools.**

For decades, variance component estimation and breeding value prediction in genetics have depended on [ASReml](https://vsni.co.uk/software/asreml), a proprietary Fortran-based tool. While ASReml is an excellent piece of software that has powered thousands of research papers, its closed-source nature and Fortran codebase create real barriers:

- **No transparency**: Researchers cannot inspect, audit, or modify the algorithms they depend on for scientific conclusions.
- **No extensibility**: Adding new variance structures, integrating with modern ML pipelines, or embedding in larger systems is impossible without vendor support.
- **Fortran lock-in**: The Fortran ecosystem lacks modern tooling (package managers, CI/CD, cross-compilation, WebAssembly targets). Maintaining and extending Fortran code requires increasingly rare expertise.
- **Licence costs**: ASReml licences are expensive, creating inequity between well-funded programs and breeding efforts in developing countries where genetic gain matters most.
- **Reproducibility**: Closed-source software is a weak link in reproducible science. When the solver is a black box, "reproducing" a result means "having the same licence."

Open alternatives exist (e.g., [sommer](https://cran.r-project.org/package=sommer) in R, [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) in Julia), but none combine ASReml's full feature set (pedigree BLUP, genomic BLUP, spatial models, multi-trait, factor analytic structures) with the performance needed for national-scale evaluations.

**This project aims to change that** by implementing the core algorithms from scratch in Rust — a language with C-level performance, memory safety without garbage collection, and first-class tooling — and exposing them through an ergonomic Python API.

## Features

### Core Engine
- **AI-REML** (Average Information) with exact REML scores, a likelihood safeguard and EM burn-in/fallback; parameters that reach zero are fixed at the boundary and reported as such (like ASReml's `B`)
- **General multi-parameter REML engine**: any combination of the variance structures below, for random terms, `outer:inner` interaction terms (`Σ_outer ⊗ Σ_inner`, optionally with a pedigree inner factor) and the residual, all validated against dense reference likelihoods and numerical gradients
- **Henderson's Mixed Model Equations** (MME): sparse assembly, sparse Cholesky (faer, AMD ordering) and a Takahashi inverse subset for the traces, prediction error variances and leverages REML needs; dense assembly for structured residuals
- **BLUP/BLUE** extraction with standard errors, reliabilities and the full fixed-effects covariance matrix
- **Treatment contrasts** for factors (`mu + rep` is full rank, like R's `model.matrix`)
- **Wald F-tests** for fixed effects using the full covariance block, with containment, Satterthwaite or Kenward-Roger (bias-adjusted F, matched denominator df) degrees of freedom
- **Diagnostics**: log-likelihood, AIC, BIC, convergence monitoring, residual diagnostics
- **Missing data**: `NA`/empty fields in CSV files become `NaN`; rows with a missing response are dropped

### Pedigree BLUP (Animal Model)
- Pedigree parsing and validation (CSV, programmatic)
- Henderson's A-inverse with Meuwissen & Luo (1992) inbreeding
- Topological sort for correct pedigree ordering
- Animals without records (ancestors) get breeding values — the random term follows the pedigree ordering, not the data
- Validated against Mrode (2005) Example 3.1, through the raw MME *and* through the public builder API

### Genomic BLUP (GBLUP)
- VanRaden Method 1 (2008) G-matrix construction
- G-matrix blending with A22 (Misztal et al. 2010)
- Single-step H-inverse (Legarra et al. 2009, Aguilar et al. 2010)
- Full A-matrix computation and A22 extraction

### Spatial & Variance Structures
- **AR1** (first-order autoregressive) with closed-form tridiagonal inverse, as a variance (`ar1`) or a correlation-only (`ar1c`) structure
- **Kronecker product** for separable models: AR1 x AR1 spatial residuals over a row x column grid (missing plots allowed), FA x genotype/pedigree for genotype-by-environment
- **Diagonal** (heterogeneous variances)
- **Unstructured** covariance (Cholesky parameterized)
- **Factor analytic** (FA1, FA2, ...) with Woodbury inverse (Smith et al. 2001)
- **Identity** (IID random effects) and **known** (fixed) matrices such as a pedigree A-inverse
- Every parameter (variances, correlations, loadings, specific variances) is reported with its standard error from the inverse average-information matrix and a boundary flag

### Multi-trait Models
- Kronecker-structured MME for correlated traits
- EM-REML for trait covariance estimation (G0, R0)
- Genetic correlation estimation
- Positive-definiteness enforcement via eigenvalue bending

### Selection Indices, Marker Models, Cross-Validation
- Smith-Hazel, restricted (Kempthorne & Nordskog) and desired-gains (Pesek & Baker) indices
- RR-BLUP marker effect model with EM-REML
- k-fold, stratified and leave-one-out cross-validation with prediction accuracy, bias and MSEP

### Python Package (`openblup`)
- `MixedModel`, `Pedigree`, `FitResult` with numpy, scipy.sparse and pandas interop
- Structured terms (`add_random(..., structure="ar1")`, `add_random_interaction("env", "genotype", "fa1")`, `set_residual_interaction("row", "col")`), Satterthwaite Wald tests, residual diagnostics and k-fold cross-validation
- Full type stubs for IDE autocompletion
- Install with `pip install .` (maturin)

### CLI Tool (`openblup`)
- `openblup fit` — fit models from CSV; term specs such as `--random "env:fa1*genotype"` and `--residual "row:ar1*col:ar1c"`, `--ddf satterthwaite|kenward-roger`, `--cv K`, `--diagnostics`, text or JSON output, BLUP export to CSV
- `openblup ainverse` — compute, inspect and export the A-inverse and inbreeding coefficients

### WebAssembly Target
- Self-contained `openblup-wasm` crate (no rayon, no faer, no file I/O) with `wasm-bindgen` exports
- JSON API for A-inverse (with inbreeding), single-random-term EM-REML and the G-matrix
- Browser demo page in `crates/wasm/www`

### Current limitations (honest status)
- Models with an IID residual (animal, genomic and plant-trial models) assemble the MME **sparsely** and solve them with a sparse Cholesky factorization (fill-reducing ordering) plus a Takahashi inverse subset, so the number of equations is limited by memory for the factor rather than by a dense inverse. Models with structured residuals or dense variance structures (AR1 x AR1 residuals, FA / unstructured terms) use the general engine, which assembles and inverts `C` densely; keep those to a few thousand equations.
- The `gpu` feature compiles a backend *interface* only; all computation runs on the CPU until a `wgpu` backend is contributed.
- Multi-trait models use EM-REML only.
- Satterthwaite denominator df need the average-information matrix, so they are available after an AI-REML fit with no variance parameter on the boundary (any variance structure). Kenward-Roger additionally needs scaled-identity / relationship-matrix terms and an IID residual. Unavailable methods fall back to the next simpler one (the output says which). Leverage-based residual diagnostics need an IID residual.
- `--cv` / `cross_validate()` handle models with a single random term (genomic or pedigree prediction).

## Quick Start (Rust)

This is `crates/core/examples/quick_start.rs`; run it with
`cargo run -p plant-breeding-lmm-core --example quick_start`.

```rust
use plant_breeding_lmm_core::data::DataFrame;
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
```

## Quick Start (Python)

```bash
# Install (requires the Rust toolchain)
pip install .            # or: pip install maturin && maturin develop --release
```

```python
from openblup import MixedModel, Pedigree, compute_a_inverse, compute_g_matrix

# Fit a simple mixed model
model = MixedModel()
model.load_csv("examples/field_trial.csv")   # rows with a missing response are dropped at fit time
model.as_factor("rep")                        # numeric codes -> categorical
model.set_response("yield")
model.add_fixed("mu + rep")
model.add_random("genotype")
result = model.fit()
print(result.summary())
print(result.variance_components())           # {'genotype': ..., 'residual': ...}
print(result.wald_tests())

# Animal model with a pedigree relationship matrix
ped = Pedigree.from_csv("examples/animal_pedigree.csv")
model = MixedModel()
model.load_csv("examples/animal_records.csv")
model.set_response("weight")
model.add_fixed("sex")
model.add_random_pedigree("animal", ped)      # A-inverse with inbreeding, pedigree ordering
result = model.fit()
blups = dict(zip(result.random_effect_levels()["animal"], result.random_effects()["animal"]))
print(result.variance_components_se())        # SEs from the inverse AI matrix
print(result.at_boundary())                   # parameters that converged to zero

# Multi-environment trial: factor-analytic genotype-by-environment term
model = MixedModel()
model.load_csv("examples/met_trial.csv")
model.set_response("yield")
model.add_fixed("mu + env")
model.add_random_interaction("env", "genotype", outer_structure="fa1")
result = model.fit()
for p in result.variance_parameters():        # loadings, specific variances, residual: value, se, at_boundary
    print(p["component"], p["name"], round(p["value"], 3), round(p["se"], 3))

# Spatial analysis: AR1 x AR1 residual over the field grid, genotype random
model = MixedModel()
model.load_csv("examples/field_trial.csv")
model.as_factor("rep")
model.set_response("yield")
model.add_fixed("mu + rep")
model.add_random("genotype")
model.set_residual_interaction("row", "col", "ar1", "ar1c")
print(model.fit().summary())

# Satterthwaite df, residual diagnostics and cross-validation
model = MixedModel()
model.load_csv("examples/field_trial.csv")
model.as_factor("rep")
model.set_response("yield")
model.add_fixed("mu + rep")
model.add_random("genotype")
result = model.fit()
print(result.wald_tests(ddf="satterthwaite"))
print(result.wald_tests(ddf="kenward-roger"))
print(result.residual_diagnostics()["cooks_distance"])
print(model.cross_validate(n_folds=5, seed=1)["accuracy"])

# Relationship matrices directly
a_inv = compute_a_inverse(ped)                # scipy.sparse.csc_matrix (rows = ped.animal_ids())
g = compute_g_matrix(markers_0_1_2)           # numpy (n x n)

# pandas
model = MixedModel()
model.set_dataframe(df)                       # numeric -> float, everything else -> factor
```

## Quick Start (CLI)

```bash
# Fit a model from the command line (rep is numeric in the file, so mark it as a factor)
openblup fit --data examples/field_trial.csv --response yield \
    --fixed "mu + rep" --factor rep --random genotype

# Animal model with pedigree: all 63 pedigree animals get a breeding value
openblup fit --data examples/animal_records.csv --response weight \
    --fixed sex --random animal --pedigree examples/animal_pedigree.csv \
    --format json --blups breeding_values.csv

# Multi-environment trial: FA1 genotype-by-environment interaction
openblup fit --data examples/met_trial.csv --response yield \
    --fixed "mu + env" --random "env:fa1*genotype"

# Spatial analysis: AR1 x AR1 residual over the row x column grid
openblup fit --data examples/field_trial.csv --response yield \
    --fixed "mu + rep" --factor rep --random genotype --residual "row:ar1*col:ar1c"

# Satterthwaite df, 5-fold cross-validation and residual diagnostics
openblup fit --data examples/field_trial.csv --response yield \
    --fixed "mu + rep" --factor rep --random genotype \
    --ddf satterthwaite --cv 5 --diagnostics diagnostics.csv

# Inspect the A-inverse and inbreeding coefficients
openblup ainverse --pedigree examples/mrode_pedigree.csv --inbreeding --output ainv.csv
```

A random term is `factor[:structure]` or an interaction `factor[:structure]*factor[:structure]`
with structures `idv` (default), `ar1`/`ar1(rho)`, `ar1c` (correlation only), `diag`, `us`,
`fa1`, `fa2`, ... A pedigree attaches to the factor named by `--pedigree-term`
(e.g. `--random "env:diag*animal" --pedigree ped.csv --pedigree-term animal`).
Run `openblup fit --help` for all options (`--algorithm em`, `--max-iter`, `--tolerance`, ...).

## Building

```bash
# Requires Rust 1.70+ (CI runs stable on Linux, macOS and Windows)
cargo build --release

# Run all Rust tests (core unit tests, integration tests, CLI end-to-end tests, wasm crate)
cargo test --workspace

# Large-scale check (4 000-animal pedigree model) — ignored by default, run in release mode
cargo test --release -p plant-breeding-lmm-core --test sparse_mme_test -- --ignored

# Lints used in CI
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Build the CLI (binary: target/release/openblup)
cargo build --release -p plant-breeding-lmm-cli

# Build and test the Python package
pip install maturin numpy scipy
maturin develop --release
python -m pytest python/tests

# Build the WebAssembly demo
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build crates/wasm --target web --out-dir www/pkg
python3 -m http.server --directory crates/wasm/www   # then open http://localhost:8000
```

## Architecture

```
openblup/
├── crates/
│   ├── core/                 # Pure Rust library (plant-breeding-lmm-core)
│   │   ├── data/             # DataFrame, Factor columns, CSV I/O (NA handling)
│   │   ├── matrix/           # Sparse ops, dense helpers, faer Cholesky, sparse inverse
│   │   ├── model/            # Builder API, design matrices (treatment contrasts), multi-trait
│   │   ├── lmm/              # MME, AI-REML, EM-REML, BLUP utilities, structured R
│   │   ├── variance/         # AR1, Diagonal, Unstructured, Kronecker, Factor Analytic
│   │   ├── genetics/         # Pedigree, A/G/H matrices, RR-BLUP, selection indices
│   │   ├── diagnostics/      # Wald tests, Satterthwaite df, residuals, cross-validation
│   │   └── gpu/              # Feature-gated GPU backend interface (CPU fallback)
│   ├── python-bindings/      # PyO3 extension module (openblup._internal)
│   ├── cli/                  # Command-line tool (clap) + end-to-end tests
│   └── wasm/                 # WebAssembly target (wasm-bindgen) + browser demo
├── python/
│   ├── openblup/             # Python package (wrappers, type stubs)
│   └── tests/                # Python test suite
├── examples/                 # Example data (field trial, Mrode pedigree)
└── .github/workflows/        # CI: fmt, clippy, tests, wasm, Python on 3 platforms
```

## Algorithms & References

The algorithms implemented here are based on well-established quantitative genetics literature:

### Core REML & Mixed Model Theory
- **Henderson, C.R.** (1984). *Applications of Linear Models in Animal Breeding*. University of Guelph. — The foundational text for mixed model equations in breeding.
- **Patterson, H.D. & Thompson, R.** (1971). Recovery of inter-block information when block sizes are unequal. *Biometrika*, 58(3), 545-554. — Original REML paper.
- **Searle, S.R., Casella, G. & McCulloch, C.E.** (1992). *Variance Components*. Wiley. — Comprehensive treatment of variance component estimation.
- **Gilmour, A.R., Thompson, R. & Cullis, B.R.** (1995). Average Information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440-1450. — AI-REML algorithm used in ASReml.
- **Kenward, M.G. & Roger, J.H.** (1997). Small sample inference for fixed effects from restricted maximum likelihood. *Biometrics*, 53(3), 983-997. — Bias-adjusted covariance, scaled F and denominator df; Satterthwaite df follow Giesbrecht & Burns (1985) and Fai & Cornelius (1996) for multi-df terms.
- **Takahashi, K., Fagan, J. & Chin, M.-S.** (1973). Formation of a sparse bus impedance matrix and its application to short circuit study. *Proc. 8th PICA Conference*, 63-69. — The sparse inverse subset used for the traces and prediction error variances.

### Pedigree & Relationship Matrices
- **Henderson, C.R.** (1976). A simple method for computing the inverse of a numerator relationship matrix used in prediction of breeding values. *Biometrics*, 32(1), 69-83. — Henderson's rules for A-inverse.
- **Meuwissen, T.H.E. & Luo, Z.** (1992). Computing inbreeding coefficients in large populations. *Genetics, Selection, Evolution*, 24(4), 305-313. — Efficient inbreeding algorithm.
- **Mrode, R.A.** (2005). *Linear Models for the Prediction of Animal Breeding Values* (2nd ed.). CABI Publishing. — Standard textbook; our integration tests validate against examples from this book.
- **Quaas, R.L.** (1976). Computing the diagonal elements and inverse of a large numerator relationship matrix. *Biometrics*, 32(4), 949-953.

### Genomic Selection
- **VanRaden, P.M.** (2008). Efficient methods to compute genomic predictions. *Journal of Dairy Science*, 91(11), 4414-4423. — G-matrix construction (Method 1).
- **Legarra, A., Aguilar, I. & Misztal, I.** (2009). A relationship matrix including full pedigree and genomic information. *Journal of Dairy Science*, 92(9), 4656-4663. — Single-step H-matrix.
- **Aguilar, I., Misztal, I., Johnson, D.L., Legarra, A., Tsuruta, S. & Lawlor, T.J.** (2010). Hot topic: A unified approach to utilize phenotypic, full pedigree, and genomic information for genetic evaluation of Holstein final score. *Journal of Dairy Science*, 93(2), 743-752.

### Spatial Analysis
- **Gilmour, A.R., Cullis, B.R. & Verbyla, A.P.** (1997). Accounting for natural and extraneous variation in the analysis of field experiments. *Journal of Agricultural, Biological, and Environmental Statistics*, 2(3), 269-293. — AR1xAR1 spatial models.
- **Smith, A., Cullis, B.R. & Thompson, R.** (2005). The analysis of crop cultivar breeding and evaluation trials: an overview of current mixed model approaches. *Journal of Agricultural Science*, 143(6), 449-462.

### Multi-Environment & Factor Analytic Models
- **Smith, A., Cullis, B.R. & Thompson, R.** (2001). Analyzing variety by environment data using multiplicative mixed models and adjustments for spatial field trend. *Biometrics*, 57(4), 1138-1147. — Factor analytic models for MET.
- **Thompson, R., Cullis, B.R., Smith, A. & Gilmour, A.R.** (2003). A sparse implementation of the Average Information algorithm for factor analytic and reduced rank variance models. *Australian & New Zealand Journal of Statistics*, 45(4), 445-459.

### Software Comparison
- **Butler, D.G., Cullis, B.R., Gilmour, A.R. & Gogel, B.J.** (2009). *ASReml-R Reference Manual*. VSN International. — The proprietary reference implementation.
- **Covarrubias-Pazaran, G.** (2016). Genome-assisted prediction of quantitative traits using the R package sommer. *PLoS ONE*, 11(6), e0156744. — Open-source R alternative.

## Comparison with Existing Tools

| Feature | ASReml | sommer (R) | MixedModels.jl | **OpenBLUP** |
|---------|--------|------------|-----------------|----------------------|
| Language | Fortran | R | Julia | **Rust + Python** |
| Open source | No | Yes | Yes | **Yes** |
| AI-REML | Yes | No | No | **Yes** |
| Pedigree BLUP | Yes | Yes | No | **Yes** |
| Genomic BLUP | Yes | Yes | No | **Yes** |
| Single-step (H) | Yes | Yes | No | **Yes** |
| Spatial (AR1xAR1) | Yes | Yes | No | **Yes** |
| Multi-trait | Yes | Yes | Yes | **Yes (EM-REML)** |
| Factor analytic | Yes | Limited | No | **Yes** |
| Sparse solver | Yes | No | Yes | **Yes** (sparse Cholesky + Takahashi inverse subset; dense for structured residuals) |
| Python API | No | No | No | **Yes (PyO3)** |
| CLI tool | Yes | No | No | **Yes** |
| Wald tests | Yes | Yes | Yes | **Yes** |
| WebAssembly target | No | No | No | **Yes** |
| Memory safe | No | N/A | Yes (GC) | **Yes (ownership)** |
| Performance | Excellent | Slow | Good | **Good for small/medium problems (see limitations)** |

## Contributing

Contributions are welcome! See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide, including:

- How to set up your development environment
- Code style and naming conventions (follows standard QG notation)
- Testing requirements (all new algorithms must be validated)
- Pull request process and templates

**Areas where help is especially valuable:**

| Area | What's Needed |
|------|---------------|
| **Validation** | Run the same model in OpenBLUP + ASReml/sommer, compare variance components and BLUPs |
| **Sparse MME solve** | Wire the existing sparse Cholesky / Takahashi code into the REML engines for large evaluations |
| **Tutorials** | Worked examples from real breeding programs (dairy, wheat, maize, forestry) |
| **GPU backends** | wgpu compute shader implementations behind the existing `gpu` feature interface |

Even if you don't write code — validation reports, bug reports, and feature requests are extremely valuable. See our [issue templates](.github/ISSUE_TEMPLATE/).

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

## Acknowledgements

This project draws on decades of quantitative genetics research. We are grateful to the scientists who developed and published these algorithms, and to the ASReml team whose software set the standard for mixed model analysis in breeding.
