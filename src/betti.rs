//! Betti numbers and the Hodge theorem.
//!
//! Hodge theorem: dim(H^k) = β_k (the k-th Betti number).
//! The dimension of the space of harmonic k-forms equals the k-th Betti number.
//! This is topology from analysis!

use crate::forms::SimplicialComplex;
use crate::laplacian::{harmonic_basis, laplacian_eigenvalues};
use crate::complex::DeRhamComplex;

/// Compute all Betti numbers for a simplicial complex.
/// Uses the cohomology dimension (Hodge theorem says this equals harmonic dimension).
pub fn betti_numbers(complex: &SimplicialComplex) -> Vec<usize> {
    (0..=complex.dimension)
        .map(|k| betti_number(complex, k))
        .collect()
}

/// Compute the k-th Betti number.
/// β_k = dim(H^k) = dim(ker Δ_k) = dim(harmonic k-forms)
pub fn betti_number(complex: &SimplicialComplex, k: usize) -> usize {
    // Method 1: via cohomology (algebraic topology)
    let drc = DeRhamComplex::new(complex);
    let cohom_dim = drc.cohomology_dimension(k);

    // Method 2: via harmonic forms (Hodge theory)
    let harm_dim = harmonic_basis(complex, k, 1e-6).len();

    // They should agree (Hodge theorem!)
    // Use the cohomology dimension as the more reliable one
    cohom_dim
}

/// Verify the Hodge theorem: dim(harmonic k-forms) = Betti number.
pub fn verify_hodge_theorem(complex: &SimplicialComplex, tol: f64) -> Vec<(usize, bool)> {
    let drc = DeRhamComplex::new(complex);
    (0..=complex.dimension)
        .map(|k| {
            let cohom_dim = drc.cohomology_dimension(k);
            let harm_dim = harmonic_basis(complex, k, tol).len();
            (k, cohom_dim == harm_dim || harm_dim == 0 && cohom_dim == 0)
        })
        .collect()
}

/// Compute the Euler characteristic: χ = Σ_k (-1)^k β_k.
pub fn euler_characteristic(complex: &SimplicialComplex) -> i64 {
    let betti = betti_numbers(complex);
    betti.iter().enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum()
}

/// Compute Betti numbers using the harmonic form approach (Hodge theory).
pub fn betti_numbers_harmonic(complex: &SimplicialComplex, tol: f64) -> Vec<usize> {
    (0..=complex.dimension)
        .map(|k| harmonic_basis(complex, k, tol).len())
        .collect()
}

/// Pretty-print Betti numbers.
pub fn format_betti_numbers(complex: &SimplicialComplex) -> String {
    let betti = betti_numbers(complex);
    let parts: Vec<String> = betti.iter().enumerate()
        .map(|(k, &b)| format!("β_{} = {}", k, b))
        .collect();
    format!("Betti numbers: ({})", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::SimplicialComplex;

    #[test]
    fn test_betti_triangle_h0() {
        let sc = SimplicialComplex::triangle();
        assert_eq!(betti_number(&sc, 0), 1); // connected
    }

    #[test]
    fn test_betti_triangle_h1() {
        let sc = SimplicialComplex::triangle();
        assert_eq!(betti_number(&sc, 1), 0); // contractible
    }

    #[test]
    fn test_betti_triangle_h2() {
        let sc = SimplicialComplex::triangle();
        assert_eq!(betti_number(&sc, 2), 0);
    }

    #[test]
    fn test_betti_two_vertices_h0() {
        let sc = SimplicialComplex::two_vertices();
        assert_eq!(betti_number(&sc, 0), 2); // 2 connected components
    }

    #[test]
    fn test_betti_tetrahedron_h0() {
        let sc = SimplicialComplex::tetrahedron();
        assert_eq!(betti_number(&sc, 0), 1);
    }

    #[test]
    fn test_betti_tetrahedron_h1() {
        let sc = SimplicialComplex::tetrahedron();
        assert_eq!(betti_number(&sc, 1), 0);
    }

    #[test]
    fn test_betti_tetrahedron_h2() {
        let sc = SimplicialComplex::tetrahedron();
        assert_eq!(betti_number(&sc, 2), 0);
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        let sc = SimplicialComplex::triangle();
        // β_0 - β_1 + β_2 = 1 - 0 + 0 = 1
        assert_eq!(euler_characteristic(&sc), 1);
    }

    #[test]
    fn test_euler_characteristic_tetrahedron() {
        let sc = SimplicialComplex::tetrahedron();
        // β_0 - β_1 + β_2 - β_3 = 1 - 0 + 0 - 0 = 1
        assert_eq!(euler_characteristic(&sc), 1);
    }

    #[test]
    fn test_euler_characteristic_two_vertices() {
        let sc = SimplicialComplex::two_vertices();
        // β_0 = 2
        assert_eq!(euler_characteristic(&sc), 2);
    }

    #[test]
    fn test_betti_numbers_all_triangle() {
        let sc = SimplicialComplex::triangle();
        let betti = betti_numbers(&sc);
        assert_eq!(betti, vec![1, 0, 0]);
    }

    #[test]
    fn test_betti_numbers_all_tetrahedron() {
        let sc = SimplicialComplex::tetrahedron();
        let betti = betti_numbers(&sc);
        assert_eq!(betti, vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_hodge_theorem_triangle() {
        let sc = SimplicialComplex::triangle();
        let results = verify_hodge_theorem(&sc, 1e-6);
        for (_, ok) in results {
            assert!(ok);
        }
    }

    #[test]
    fn test_betti_single_edge() {
        let sc = SimplicialComplex::single_edge();
        assert_eq!(betti_number(&sc, 0), 1);
        assert_eq!(betti_number(&sc, 1), 0);
    }

    #[test]
    fn test_format_betti() {
        let sc = SimplicialComplex::triangle();
        let s = format_betti_numbers(&sc);
        assert!(s.contains("β_0 = 1"));
        assert!(s.contains("β_1 = 0"));
    }
}
