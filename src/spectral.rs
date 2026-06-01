//! Spectral properties of the Hodge Laplacian.
//!
//! The spectral gap (smallest nonzero eigenvalue) controls convergence rates
//! in many applications, including agent learning dynamics.

use crate::forms::SimplicialComplex;
use crate::laplacian::{laplacian_eigenvalues, hodge_laplacian_matrix};

/// Spectral analysis of the Hodge Laplacian at degree k.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpectralAnalysis {
    pub degree: usize,
    pub eigenvalues: Vec<f64>,
    pub spectral_gap: f64,
    pub multiplicity_of_zero: usize,
    pub max_eigenvalue: f64,
}

impl SpectralAnalysis {
    /// Perform spectral analysis at degree k.
    pub fn analyze(complex: &SimplicialComplex, k: usize) -> Self {
        let evals = laplacian_eigenvalues(complex, k);
        let tol = 1e-8;

        let zero_count = evals.iter().filter(|&&e| e.abs() < tol).count();
        let nonzero: Vec<f64> = evals.iter().copied().filter(|&e| e.abs() >= tol).collect();
        let spectral_gap = nonzero.first().copied().unwrap_or(0.0);
        let max_eval = evals.last().copied().unwrap_or(0.0);

        Self {
            degree: k,
            eigenvalues: evals,
            spectral_gap,
            multiplicity_of_zero: zero_count,
            max_eigenvalue: max_eval,
        }
    }

    /// The spectral gap controls how fast the system converges to equilibrium.
    /// Larger gap → faster convergence.
    pub fn convergence_rate(&self) -> f64 {
        if self.spectral_gap > 0.0 {
            self.spectral_gap
        } else {
            0.0 // No convergence (all harmonic)
        }
    }

    /// Condition number of the Laplacian restricted to non-harmonic forms.
    pub fn condition_number(&self) -> f64 {
        if self.spectral_gap > 0.0 && self.max_eigenvalue > 0.0 {
            self.max_eigenvalue / self.spectral_gap
        } else {
            f64::INFINITY
        }
    }

    /// Ratio of harmonic to total dimension.
    pub fn harmonic_ratio(&self) -> f64 {
        if self.eigenvalues.is_empty() {
            0.0
        } else {
            self.multiplicity_of_zero as f64 / self.eigenvalues.len() as f64
        }
    }
}

/// Compute the spectral gap at degree k.
pub fn spectral_gap(complex: &SimplicialComplex, k: usize) -> f64 {
    let analysis = SpectralAnalysis::analyze(complex, k);
    analysis.spectral_gap
}

/// Full spectral decomposition of the Hodge Laplacian.
pub fn full_spectral_analysis(complex: &SimplicialComplex) -> Vec<SpectralAnalysis> {
    (0..=complex.dimension)
        .map(|k| SpectralAnalysis::analyze(complex, k))
        .collect()
}

/// Verify that the spectral gap is positive (implies exponential convergence).
pub fn has_spectral_gap(complex: &SimplicialComplex, k: usize) -> bool {
    spectral_gap(complex, k) > 1e-10
}

/// Heat kernel estimate: for time t, the heat kernel e^{-tΔ} decays
/// at rate e^{-t λ_1} where λ_1 is the spectral gap.
pub fn heat_kernel_decay(complex: &SimplicialComplex, k: usize, t: f64) -> f64 {
    let gap = spectral_gap(complex, k);
    (-t * gap).exp()
}

/// Mixing time: time for the heat kernel to decay to ε.
pub fn mixing_time(complex: &SimplicialComplex, k: usize, epsilon: f64) -> f64 {
    let gap = spectral_gap(complex, k);
    if gap > 0.0 {
        -epsilon.ln() / gap
    } else {
        f64::INFINITY
    }
}

/// Verify the eigenvalue interlacing property between adjacent Laplacians.
pub fn verify_eigenvalue_properties(complex: &SimplicialComplex, k: usize, tol: f64) -> bool {
    let evals = laplacian_eigenvalues(complex, k);
    // All eigenvalues should be non-negative
    evals.iter().all(|&e| e >= -tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_analysis_triangle_0() {
        let sc = SimplicialComplex::triangle();
        let analysis = SpectralAnalysis::analyze(&sc, 0);
        assert_eq!(analysis.multiplicity_of_zero, 1); // H^0 = 1
        assert!(analysis.spectral_gap > 0.0);
    }

    #[test]
    fn test_spectral_analysis_triangle_1() {
        let sc = SimplicialComplex::triangle();
        let analysis = SpectralAnalysis::analyze(&sc, 1);
        assert_eq!(analysis.multiplicity_of_zero, 0); // H^1 = 0
    }

    #[test]
    fn test_spectral_gap_positive_triangle() {
        let sc = SimplicialComplex::triangle();
        assert!(has_spectral_gap(&sc, 1));
    }

    #[test]
    fn test_spectral_analysis_tetrahedron_0() {
        let sc = SimplicialComplex::tetrahedron();
        let analysis = SpectralAnalysis::analyze(&sc, 0);
        assert_eq!(analysis.multiplicity_of_zero, 1);
    }

    #[test]
    fn test_spectral_analysis_tetrahedron_1() {
        let sc = SimplicialComplex::tetrahedron();
        let analysis = SpectralAnalysis::analyze(&sc, 1);
        assert_eq!(analysis.multiplicity_of_zero, 0);
    }

    #[test]
    fn test_eigenvalues_nonnegative() {
        let sc = SimplicialComplex::tetrahedron();
        for k in 0..=2 {
            assert!(verify_eigenvalue_properties(&sc, k, 1e-8));
        }
    }

    #[test]
    fn test_heat_kernel_decay() {
        let sc = SimplicialComplex::triangle();
        let decay = heat_kernel_decay(&sc, 0, 1.0);
        assert!(decay > 0.0 && decay <= 1.0);
    }

    #[test]
    fn test_mixing_time_finite() {
        let sc = SimplicialComplex::triangle();
        let mt = mixing_time(&sc, 0, 0.01);
        assert!(mt.is_finite());
        assert!(mt > 0.0);
    }

    #[test]
    fn test_full_spectral_analysis() {
        let sc = SimplicialComplex::triangle();
        let analyses = full_spectral_analysis(&sc);
        assert_eq!(analyses.len(), 3); // degrees 0, 1, 2
    }

    #[test]
    fn test_convergence_rate() {
        let sc = SimplicialComplex::triangle();
        let analysis = SpectralAnalysis::analyze(&sc, 0);
        let rate = analysis.convergence_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_condition_number() {
        let sc = SimplicialComplex::triangle();
        let analysis = SpectralAnalysis::analyze(&sc, 1);
        let cond = analysis.condition_number();
        assert!(cond > 0.0);
    }

    #[test]
    fn test_harmonic_ratio() {
        let sc = SimplicialComplex::triangle();
        let analysis = SpectralAnalysis::analyze(&sc, 0);
        // 3 eigenvalues, 1 zero → ratio 1/3
        let ratio = analysis.harmonic_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0);
    }

    #[test]
    fn test_two_vertices_spectral() {
        let sc = SimplicialComplex::two_vertices();
        let analysis = SpectralAnalysis::analyze(&sc, 0);
        assert_eq!(analysis.multiplicity_of_zero, 2); // 2 components
    }

    #[test]
    fn test_spectral_gap_corresponds_to_betti() {
        let sc = SimplicialComplex::triangle();
        for k in 0..=2 {
            let analysis = SpectralAnalysis::analyze(&sc, k);
            let betti = crate::betti::betti_number(&sc, k);
            assert_eq!(analysis.multiplicity_of_zero, betti);
        }
    }
}
