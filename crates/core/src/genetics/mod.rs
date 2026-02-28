// Genetics module - Phase 2+
// Pedigree, A-matrix, G-matrix, H-matrix, breeding values

pub mod ainverse;
pub mod gmatrix;
pub mod hmatrix;
pub mod pedigree;
pub mod rrblup;
pub mod selection_index;

pub use ainverse::{compute_a_inverse, compute_a_inverse_with_inbreeding, compute_inbreeding};
pub use gmatrix::{blend_g_matrix, compute_g_matrix, invert_g_matrix};
pub use hmatrix::{compute_a_matrix, compute_h_inverse, extract_a22};
pub use pedigree::Pedigree;
pub use rrblup::{RrBlup, RrBlupResult};
pub use selection_index::{desired_gains_index, SelectionIndex};
