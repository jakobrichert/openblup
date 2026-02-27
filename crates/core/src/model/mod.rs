mod builder;
mod design;
pub mod multitrait;

pub use builder::{MixedModel, MixedModelBuilder};
pub use design::{build_fixed_design, build_random_design, FixedEffectLabel};
pub use multitrait::{MultiTraitModel, MultiTraitModelBuilder};
