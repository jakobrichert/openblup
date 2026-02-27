// Genetics module - Phase 2+
// Pedigree, A-matrix, G-matrix, H-matrix, breeding values

pub mod ainverse;
pub mod pedigree;

pub use ainverse::{compute_a_inverse, compute_a_inverse_with_inbreeding, compute_inbreeding};
pub use pedigree::Pedigree;
