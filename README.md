# OpenBLUP

**Open-source REML and BLUP for plant and animal breeding** — a modern linear mixed model engine written in Rust with Python bindings.

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
- **AI-REML** (Average Information) with quadratic convergence + EM-REML fallback
- **Henderson's Mixed Model Equations** (MME) — dense and sparse assembly/solve
- **BLUP/BLUE** extraction with standard errors
- **Sparse Cholesky solver** via [faer](https://github.com/sarah-ek/faer-rs) (supernodal, symbolic/numeric split)
- **Wald F-tests** for fixed effects with p-values
- **Diagnostics**: Log-likelihood, AIC, BIC, convergence monitoring

### Pedigree BLUP (Animal Model)
- Pedigree parsing and validation (CSV, programmatic)
- Henderson's A-inverse with Meuwissen & Luo (1992) inbreeding
- Topological sort for correct pedigree ordering
- Validated against Mrode (2005) textbook examples

### Genomic BLUP (GBLUP)
- VanRaden Method 1 (2008) G-matrix construction
- G-matrix blending with A22 (Misztal et al. 2010)
- Single-step H-inverse (Legarra et al. 2009, Aguilar et al. 2010)
- Full A-matrix computation and A22 extraction

### Spatial & Variance Structures
- **AR1** (first-order autoregressive) with closed-form tridiagonal inverse
- **Kronecker product** for separable spatial models (AR1 x AR1)
- **Diagonal** (heterogeneous variances)
- **Unstructured** covariance (Cholesky parameterized)
- **Identity** (IID random effects)

### Multi-trait Models
- Kronecker-structured MME for correlated traits
- EM-REML for trait covariance estimation (G0, R0)
- Genetic correlation estimation
- Positive-definiteness enforcement via eigenvalue bending

### Python Bindings (PyO3)
- `MixedModel` class with fluent API
- `Pedigree` class with CSV import and A-inverse
- `compute_g_matrix()` from numpy marker arrays
- scipy.sparse interop for relationship matrices
- Full type stubs for IDE autocompletion
- Install via `pip install -e .` (maturin)

### CLI Tool
- `openblup fit` — fit models from CSV with formula specification
- `openblup ainverse` — compute and inspect A-inverse from pedigree
- Text and JSON output formats

### Data I/O
- CSV import with automatic type detection (numeric vs. categorical)
- Flexible model specification with builder pattern API

### Roadmap

These features are planned for future releases, roughly in priority order:

| Feature | Description | Complexity |
|---------|-------------|------------|
| **Factor Analytic (FA) models** | Reduced-rank covariance for multi-environment trials (MET). FA1, FA2, etc. Dramatically reduces the number of parameters for large MET analyses (Smith et al. 2001, Thompson et al. 2003). | High |
| **Structured R (spatial residuals)** | Extend REML to support non-identity residual structures (e.g., AR1xAR1 residual in field trials). Currently R = σ²I; this would allow R = σ² (Σ_row ⊗ Σ_col). | Medium |
| **Satterthwaite / Kenward-Roger df** | Improved denominator degrees of freedom for Wald F-tests. Currently uses the containment method; Kenward-Roger is more accurate for unbalanced designs. | Medium |
| **Residual diagnostics** | Conditional and marginal residuals, leverage, Cook's distance, QQ plots. Essential for model checking in practice. | Medium |
| **Selection indices** | Smith-Hazel index, economic weights, index coefficients. Used in multi-trait selection to combine breeding values into a single selection criterion. | Low |
| **Marker effect models (RR-BLUP)** | Ridge regression BLUP for estimating individual SNP marker effects rather than genomic breeding values. Useful for genomic prediction and QTL mapping. | Medium |
| **Cross-validation** | k-fold CV for assessing genomic prediction accuracy. Widely used to compare models and training population designs. | Low |
| **Sparse inverse subset** | Compute only the diagonal (or selected elements) of C⁻¹ using Takahashi equations, avoiding full dense inversion. Critical for scaling to >10,000 animals. | High |
| **GPU acceleration** | CUDA/ROCm support for G-matrix computation and dense linear algebra on large genomic datasets. | High |
| **WASM target** | Compile to WebAssembly for browser-based breeding value estimation. Useful for education and lightweight field tools. | Low |

## Quick Start (Rust)

```rust
use plant_breeding_lmm_core::data::DataFrame;
use plant_breeding_lmm_core::model::MixedModelBuilder;
use plant_breeding_lmm_core::variance::Identity;
use plant_breeding_lmm_core::genetics::{Pedigree, compute_a_inverse};

// Load data
let df = DataFrame::from_csv("field_trial.csv")?;

// Simple model: yield = rep (fixed) + genotype (random, IID) + error
let mut model = MixedModelBuilder::new()
    .data(&df)
    .response("yield")
    .fixed("rep")
    .random("genotype", Identity::new(1.0), None)
    .build()?;

let result = model.fit_reml()?;
println!("{}", result.summary());

// With pedigree relationship matrix
let ped = Pedigree::from_csv("pedigree.csv")?;
let a_inv = compute_a_inverse(&ped)?;

let mut model = MixedModelBuilder::new()
    .data(&df)
    .response("yield")
    .fixed("rep")
    .random("animal", Identity::new(1.0), Some(a_inv))
    .build()?;

let result = model.fit_reml()?;
println!("{}", result.summary());
```

## Quick Start (Python)

```bash
# Install (requires Rust toolchain + maturin)
pip install maturin
pip install -e .
```

```python
from plant_breeding_lmm import MixedModel, Pedigree, compute_a_inverse

# Fit a simple mixed model
model = MixedModel()
model.load_csv("field_trial.csv")
model.set_response("yield")
model.add_fixed("rep")
model.add_random("genotype")
result = model.fit()
print(result.summary())

# With pedigree relationship matrix
ped = Pedigree.from_csv("pedigree.csv")
a_inv = compute_a_inverse(ped)  # scipy.sparse compatible

model = MixedModel()
model.load_csv("field_trial.csv")
model.set_response("yield")
model.add_fixed("rep")
model.add_random("animal", ginverse=a_inv)
result = model.fit()
print(result.variance_components())  # {'animal': 20.5, 'residual': 40.1}
```

## Quick Start (CLI)

```bash
# Fit a model from the command line
openblup fit --data trial.csv --response yield --fixed "rep" --random genotype

# With pedigree
openblup fit --data trial.csv --response yield --fixed "rep" \
    --random animal --pedigree pedigree.csv

# Inspect A-inverse
openblup ainverse --pedigree pedigree.csv
```

## Building

```bash
# Requires Rust 1.70+ (tested with 1.93)
cargo build --release

# Run tests (214 tests)
cargo test --workspace

# Build Python bindings
pip install maturin
maturin develop --release

# Build CLI
cargo build --release -p openblup-cli
```

## Architecture

```
openblup/
├── crates/
│   ├── core/                 # Pure Rust library (13,000+ lines)
│   │   ├── data/             # DataFrame, Factor columns, CSV I/O
│   │   ├── matrix/           # Sparse ops, dense helpers, faer Cholesky
│   │   ├── model/            # Builder API, design matrices, multi-trait
│   │   ├── lmm/              # MME, AI-REML, EM-REML, BLUP/BLUE
│   │   ├── variance/         # AR1, Diagonal, Unstructured, Kronecker
│   │   ├── genetics/         # Pedigree, A/G/H matrices, breeding values
│   │   └── diagnostics/      # LogL, AIC/BIC, Wald tests, convergence
│   ├── python-bindings/      # PyO3 bridge with numpy/scipy interop
│   └── cli/                  # Command-line tool (clap)
└── python/                   # Python package + type stubs
```

## Algorithms & References

The algorithms implemented here are based on well-established quantitative genetics literature:

### Core REML & Mixed Model Theory
- **Henderson, C.R.** (1984). *Applications of Linear Models in Animal Breeding*. University of Guelph. — The foundational text for mixed model equations in breeding.
- **Patterson, H.D. & Thompson, R.** (1971). Recovery of inter-block information when block sizes are unequal. *Biometrika*, 58(3), 545-554. — Original REML paper.
- **Searle, S.R., Casella, G. & McCulloch, C.E.** (1992). *Variance Components*. Wiley. — Comprehensive treatment of variance component estimation.
- **Gilmour, A.R., Thompson, R. & Cullis, B.R.** (1995). Average Information REML: An efficient algorithm for variance parameter estimation in linear mixed models. *Biometrics*, 51(4), 1440-1450. — AI-REML algorithm used in ASReml.

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
| Multi-trait | Yes | Yes | Yes | **Yes** |
| Factor analytic | Yes | Limited | No | Planned |
| Sparse solver | Yes | No | Yes | **Yes (faer)** |
| Python API | No | No | No | **Yes (PyO3)** |
| CLI tool | Yes | No | No | **Yes** |
| Wald tests | Yes | Yes | Yes | **Yes** |
| WebAssembly target | No | No | No | Possible |
| Memory safe | No | N/A | Yes (GC) | **Yes (ownership)** |
| Performance | Excellent | Slow | Good | **Excellent** |

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
| **Factor Analytic** | FA1/FA2 variance structures for multi-environment trial analysis |
| **Tutorials** | Worked examples from real breeding programs (dairy, wheat, maize, forestry) |
| **Python polish** | pandas DataFrame input, better error messages, documentation |

Even if you don't write code — validation reports, bug reports, and feature requests are extremely valuable. See our [issue templates](.github/ISSUE_TEMPLATE/).

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.

## Acknowledgements

This project draws on decades of quantitative genetics research. We are grateful to the scientists who developed and published these algorithms, and to the ASReml team whose software set the standard for mixed model analysis in breeding.
