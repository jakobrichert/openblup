//! # OpenBLUP core
//!
//! ASReml-like linear mixed model engine for plant and animal breeding:
//! REML variance component estimation (AI-REML / EM-REML), Henderson's
//! mixed model equations, pedigree and genomic relationship matrices,
//! spatial and multi-trait variance structures, and model diagnostics.
//!
//! Start with [`data::DataFrame`], [`model::MixedModelBuilder`] and
//! [`genetics::Pedigree`].

// Index-based loops mirror the matrix algebra of the cited references and
// are clearer than iterator chains for this code base; the REML engines
// legitimately thread many parameters through their update functions.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod data;
pub mod diagnostics;
pub mod error;
pub mod genetics;
pub mod gpu;
pub mod lmm;
pub mod matrix;
pub mod model;
pub mod types;
pub mod variance;

pub use error::{LmmError, Result};
