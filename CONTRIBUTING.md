# Contributing to OpenBLUP

Thank you for considering contributing to OpenBLUP! This project aims to provide the plant and animal breeding community with a modern, open-source alternative to proprietary mixed model software. Every contribution — whether code, documentation, validation, or bug reports — helps make breeding tools more accessible worldwide.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Where to Help](#where-to-help)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behaviour to the maintainers.

## How Can I Contribute?

### Report Bugs

Found a numerical issue, a convergence failure, or unexpected results? Please open an issue with:

1. A minimal reproducible example (data + model specification)
2. Expected results (e.g., from ASReml, sommer, or textbook)
3. Actual results from OpenBLUP
4. Your Rust version (`rustc --version`) and OS

Numerical correctness is critical for breeding software — we take every discrepancy seriously.

### Validate Against Other Software

One of the most valuable contributions is running the same model in OpenBLUP and another tool (ASReml-R, sommer, BLUPF90, MixedModels.jl, Wombat) and comparing:

- Variance component estimates
- BLUP rankings and magnitudes
- Log-likelihood values
- Convergence behaviour

If you have access to ASReml and can share comparison results (even on simulated data), that is extremely helpful.

### Suggest Features

Open an issue describing:

- What you need (e.g., "compound symmetry covariance for multi-trait")
- Why it matters for your breeding program
- References to the relevant statistical method

### Improve Documentation

- Tutorials and worked examples (especially from real breeding programs)
- Better docstrings on public API functions
- Corrections to the academic references
- Translations of documentation

### Write Code

See [Where to Help](#where-to-help) for priority areas.

## Getting Started

### Prerequisites

- **Rust** 1.70 or later (`rustup update stable`)
- **Git**
- (Optional) **Python 3.8+** and **maturin** for Python bindings
- (Optional) **R** with `sommer` or `ASReml-R` for validation

### Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/openblup.git
cd openblup

# Build
cargo build --workspace

# Run tests
cargo test --workspace

# Run a specific test
cargo test -p plant-breeding-lmm-core test_mrode_example

# Build Python bindings (optional)
pip install maturin
maturin develop
```

### Project Structure

```
openblup/
├── crates/
│   ├── core/                 # Main library (start here)
│   │   ├── src/
│   │   │   ├── data/         # DataFrame, CSV I/O
│   │   │   ├── genetics/     # Pedigree, A/G/H matrices
│   │   │   ├── lmm/          # REML engines, MME, BLUP
│   │   │   ├── matrix/       # Sparse/dense ops, Cholesky
│   │   │   ├── model/        # Model builder, design matrices
│   │   │   ├── variance/     # Variance structure trait + impls
│   │   │   └── diagnostics/  # Wald tests, information criteria
│   │   └── tests/            # Integration tests
│   ├── python-bindings/      # PyO3 bridge
│   └── cli/                  # Command-line tool
└── python/                   # Python package + type stubs
```

## Development Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** Keep commits focused and atomic.

3. **Add tests.** Every new algorithm or public function should have tests. For numerical methods, validate against known solutions (textbook examples, other software).

4. **Run the full test suite:**
   ```bash
   cargo test --workspace
   ```

5. **Check formatting and lints:**
   ```bash
   cargo fmt --check
   cargo clippy --workspace -- -W warnings
   ```

6. **Open a pull request** with a clear description of what and why.

## Code Style

### Rust

- Follow standard Rust conventions (`cargo fmt` formatting)
- Use `snake_case` for functions and variables, `CamelCase` for types
- Document public APIs with `///` doc comments
- Include references to the statistical literature when implementing algorithms:
  ```rust
  /// Compute Henderson's A-inverse using the rules from
  /// Henderson (1976) with Meuwissen & Luo (1992) inbreeding.
  pub fn compute_a_inverse(ped: &Pedigree) -> Result<SparseMat> { ... }
  ```
- Prefer clarity over cleverness — breeding scientists will read this code
- Use `thiserror` for error types, `Result<T>` from our error module

### Naming Conventions

Follow standard quantitative genetics notation where possible:

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| X | `x` | Fixed effects design matrix |
| Z | `z` | Random effects design matrix |
| G | `g0` | Genetic (co)variance matrix |
| R | `r0` | Residual (co)variance matrix |
| A | `a_matrix` | Numerator relationship matrix |
| A⁻¹ | `a_inv` | Inverse of A |
| G (genomic) | `g_matrix` | Genomic relationship matrix |
| H⁻¹ | `h_inv` | Single-step inverse |
| b-hat | `fixed_effects` | BLUE of fixed effects |
| u-hat | `random_effects` | BLUP of random effects |
| σ² | `sigma2` | Variance component |

## Testing

### Test Categories

1. **Unit tests** (`#[cfg(test)]` in source files): Test individual functions against known inputs/outputs.

2. **Integration tests** (`crates/core/tests/`): Test full model fitting pipelines.

3. **Validation tests**: Compare against published textbook results (e.g., Mrode 2005).

### Writing Good Tests for Numerical Code

```rust
use approx::assert_relative_eq;

#[test]
fn test_my_algorithm() {
    // 1. State the reference: "Mrode (2005) Table 3.1"
    // 2. Set up known inputs
    // 3. Compare against published values with appropriate tolerance
    assert_relative_eq!(result, expected, epsilon = 1e-4);
}
```

- Use `approx::assert_relative_eq!` for floating-point comparisons
- Tolerances: `1e-4` for values from textbooks (often rounded), `1e-8` for exact analytical results
- Always include the source of expected values in a comment

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific module
cargo test -p plant-breeding-lmm-core genetics

# With output (see BLUP values, variance components)
cargo test -p plant-breeding-lmm-core --test integration_test -- --nocapture
```

## Pull Request Process

1. **All tests must pass.** No exceptions.
2. **New features need tests.** Preferably validated against another tool or textbook.
3. **Keep PRs focused.** One feature or fix per PR. Large PRs are hard to review.
4. **Update documentation** if you change public APIs.
5. **Reference issues** in your PR description (e.g., "Closes #42").
6. A maintainer will review and may request changes. This is normal and constructive.

### PR Title Convention

```
feat: add compound symmetry variance structure
fix: correct AR1 inverse for rho near boundary
docs: add multi-trait tutorial
test: validate GBLUP against sommer on wheat data
refactor: extract common Kronecker assembly into shared function
```

## Where to Help

### High Priority

| Area | What's Needed | Skills |
|------|---------------|--------|
| **Validation** | Run same models in OpenBLUP + ASReml/sommer, compare results | R + breeding knowledge |
| **Factor Analytic models** | FA1/FA2 variance structures for MET | Rust + linear algebra |
| **Structured residuals** | Non-identity R (spatial AR1xAR1 residual) in REML | Rust + REML theory |
| **Sparse inverse subset** | Takahashi equations for scalability >10k animals | Rust + sparse LA |

### Medium Priority

| Area | What's Needed | Skills |
|------|---------------|--------|
| **Kenward-Roger df** | Better denominator df for Wald tests | Rust + statistics |
| **Residual diagnostics** | Leverage, Cook's D, conditional residuals | Rust |
| **RR-BLUP** | Ridge regression marker effects model | Rust + genomics |
| **Tutorials** | Worked examples: dairy, wheat, maize, forestry | Breeding + writing |
| **Python API polish** | pandas DataFrame input, better error messages | Python + PyO3 |

### Good First Issues

- Add more CSV parsing tests (edge cases: quoted fields, Unicode)
- Improve error messages (include column names, dimensions)
- Add `Display` trait implementations for result types
- Write docstring examples for public functions
- Add `serde::Serialize` to result types for JSON export

## Questions?

Open a [discussion](https://github.com/jakobrichert/openblup/discussions) or an issue. There are no stupid questions — the intersection of quantitative genetics and systems programming is niche, and we're happy to explain either side.
