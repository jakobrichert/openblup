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

### Implemented (Phase 1-2)

- **REML variance component estimation** via EM-REML algorithm
- **Henderson's Mixed Model Equations** (MME) assembly and solve
- **BLUP/BLUE** extraction with standard errors
- **Pedigree BLUP (Animal Model)**
  - Pedigree parsing and validation (CSV, programmatic)
  - Henderson's A-inverse with Meuwissen & Luo (1992) inbreeding
  - Topological sort for correct pedigree ordering
- **Sparse Cholesky solver** via [faer](https://github.com/sarah-ek/faer-rs) (supernodal, with symbolic/numeric split)
- **Flexible model specification** with builder pattern API
- **Data I/O**: CSV import with automatic type detection (numeric vs. categorical)
- **Diagnostics**: Log-likelihood, AIC, BIC, convergence monitoring

### Planned

- **Genomic BLUP (GBLUP)**: VanRaden (2008) G-matrix, single-step H-matrix (Legarra et al. 2009)
- **Spatial analysis**: AR1, AR1xAR1 field trial models
- **Multi-trait models**: Unstructured, diagonal, compound symmetry covariance
- **Factor analytic structures**: For multi-environment trial analysis
- **AI-REML**: Average Information REML for faster convergence
- **Python bindings**: Full PyO3/maturin integration with numpy/scipy interop
- **CLI tool**: Command-line interface for batch analyses

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

## Building

```bash
# Requires Rust 1.70+ (tested with 1.93)
cargo build --release

# Run tests
cargo test --workspace

# Run benchmarks (coming soon)
cargo bench
```

## Architecture

```
plant-breeding-lmm/
├── crates/
│   ├── core/                 # Pure Rust library
│   │   ├── data/             # DataFrame, Factor columns, CSV I/O
│   │   ├── matrix/           # Sparse ops, dense helpers, faer Cholesky
│   │   ├── model/            # Builder API, design matrix construction
│   │   ├── lmm/              # MME assembly, EM-REML, BLUP/BLUE
│   │   ├── variance/         # VarStruct trait + implementations
│   │   ├── genetics/         # Pedigree, A-inverse, (G-matrix, H-matrix)
│   │   └── diagnostics/      # LogL, AIC/BIC, convergence
│   ├── python-bindings/      # PyO3 bridge (planned)
│   └── cli/                  # Command-line tool (planned)
└── python/                   # Python package
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
| Pedigree BLUP | Yes | Yes | No | **Yes** |
| Genomic BLUP | Yes | Yes | No | **Planned** |
| Spatial (AR1xAR1) | Yes | Yes | No | **Planned** |
| Multi-trait | Yes | Yes | Yes | **Planned** |
| Factor analytic | Yes | Limited | No | **Planned** |
| Sparse solver | Yes | No | Yes | **Yes (faer)** |
| Python API | No | No | No | **Planned** |
| WebAssembly target | No | No | No | **Possible** |
| Memory safe | No | N/A | Yes (GC) | **Yes (ownership)** |
| Performance | Excellent | Slow | Good | **Excellent** |

## Contributing

Contributions are welcome! This project is in active development. Areas where help is especially valuable:

- **Validation**: Comparing results against ASReml/sommer on real datasets
- **Genomic features**: G-matrix, ssGBLUP, marker effects
- **Spatial models**: AR1, AR1xAR1, spline-based
- **Python bindings**: PyO3 integration, pandas/numpy interop
- **Documentation**: Tutorials, worked examples from breeding programs

## Licence

MIT OR Apache-2.0 (dual-licensed)

## Acknowledgements

This project draws on decades of quantitative genetics research. We are grateful to the scientists who developed and published these algorithms, and to the ASReml team whose software set the standard for mixed model analysis in breeding.
