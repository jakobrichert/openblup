mod builder;
mod design;
pub mod multitrait;
mod spec;

pub use builder::{MixedModel, MixedModelBuilder, ResidualGrid};
pub use design::{
    build_fixed_design, build_random_design, build_random_design_interaction,
    build_random_design_with_levels, FixedEffectLabel, FixedTerm,
};
pub use multitrait::{MultiTraitModel, MultiTraitModelBuilder};
pub use spec::{FactorSpec, FitSpec, PreparedModel, TermSpec};
