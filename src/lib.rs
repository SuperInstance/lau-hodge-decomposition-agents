//! # lau-hodge-decomposition-agents
//!
//! Hodge decomposition for agent systems.
//!
//! Every agent signal decomposes into exact + coexact + harmonic:
//! `ω = dα + δβ + h`
//!
//! This is the fundamental theorem of Hodge theory applied to agents:
//! any agent behavior = what it learned (exact) + what it was told (coexact) +
//! what it already knew (harmonic).

pub mod forms;
pub mod complex;
pub mod hodge_star;
pub mod laplacian;
pub mod decomposition;
pub mod betti;
pub mod serre;
pub mod spectral;
pub mod agent;

pub use forms::*;
pub use complex::*;
pub use hodge_star::*;
pub use laplacian::*;
pub use decomposition::*;
pub use betti::*;
pub use serre::*;
pub use spectral::*;
pub use agent::*;
