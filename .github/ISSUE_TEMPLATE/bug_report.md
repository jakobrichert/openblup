---
name: Bug Report
about: Report incorrect results, convergence failures, or crashes
title: ''
labels: bug
assignees: ''
---

## Description

A clear description of the bug.

## Minimal Reproducible Example

```rust
// Rust code to reproduce, OR
// describe the CSV data + model specification
```

## Expected Behaviour

What should happen? If you have results from another tool (ASReml, sommer, etc.), include them here.

## Actual Behaviour

What actually happens? Include error messages, incorrect values, etc.

## Environment

- **OS**: (e.g., Ubuntu 22.04, macOS 14, Windows 11)
- **Rust version**: (`rustc --version`)
- **OpenBLUP version/commit**: (`git rev-parse --short HEAD`)

## Additional Context

- Dataset size (n observations, p fixed, q random levels)
- Variance structure used
- Whether the model converges
