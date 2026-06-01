//! Hodge Laplacian: Δ = dδ + δd.
//!
//! The Hodge Laplacian is the fundamental operator of Hodge theory.
//! Its kernel consists exactly of the harmonic forms.

use nalgebra::DMatrix;
use crate::forms::{DifferentialForm, SimplicialComplex};
use crate::hodge_star::codifferential_matrix;
use crate::complex::exterior_derivative;

/// Build the Hodge Laplacian matrix Δ_k for k-forms.
/// Δ_k = d_{k-1} δ_k + δ_{k+1} d_k
pub fn hodge_laplacian_matrix(complex: &SimplicialComplex, k: usize) -> DMatrix<f64> {
    let nk = complex.num_simplices(k);
    if nk == 0 {
        return DMatrix::zeros(0, 0);
    }

    // d_{k-1} δ_k
    let d_prev = if k > 0 {
        let dm = complex.exterior_derivative_matrix(k - 1);
        if dm.is_empty() {
            DMatrix::zeros(nk, nk)
        } else {
            // d_{k-1}: Ω^{k-1} → Ω^k, so d_{k-1} is nk × n_{k-1}
            // δ_k: Ω^k → Ω^{k-1}, so δ_k is n_{k-1} × nk
            let delta_k = codifferential_matrix(complex, k);
            &dm * &delta_k
        }
    } else {
        DMatrix::zeros(nk, nk)
    };

    // δ_{k+1} d_k
    let dd_next = {
        let dk = complex.exterior_derivative_matrix(k);
        if dk.is_empty() {
            DMatrix::zeros(nk, nk)
        } else {
            // d_k: Ω^k → Ω^{k+1}, dk is n_{k+1} × nk
            // δ_{k+1}: Ω^{k+1} → Ω^k, δ_{k+1} is nk × n_{k+1}
            let delta_next = codifferential_matrix(complex, k + 1);
            &delta_next * &dk
        }
    };

    &d_prev + &dd_next
}

/// Apply the Hodge Laplacian to a k-form.
pub fn apply_hodge_laplacian(complex: &SimplicialComplex, form: &DifferentialForm) -> DifferentialForm {
    let k = form.degree;
    let lap = hodge_laplacian_matrix(complex, k);
    if lap.is_empty() {
        return DifferentialForm::zero(k, 0);
    }
    let v = form.to_vector();
    let result = &lap * &v;
    DifferentialForm::from_vector(k, &result)
}

/// Check if a form is harmonic: Δω = 0.
pub fn is_harmonic(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let lap = apply_hodge_laplacian(complex, form);
    lap.is_zero(tol)
}

/// A harmonic form must satisfy both dω = 0 and δω = 0.
/// This is equivalent to Δω = 0 on a closed manifold.
pub fn is_harmonic_ddelta(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let dw = exterior_derivative(complex, form);
    let delta_w = crate::hodge_star::apply_codifferential(complex, form);
    dw.is_zero(tol) && delta_w.is_zero(tol)
}

/// Find a basis for the harmonic k-forms.
/// Returns eigenvectors of the Hodge Laplacian with eigenvalue ≈ 0.
pub fn harmonic_basis(complex: &SimplicialComplex, k: usize, tol: f64) -> Vec<DifferentialForm> {
    let lap = hodge_laplacian_matrix(complex, k);
    if lap.is_empty() {
        return vec![];
    }

    let nk = complex.num_simplices(k);
    let eigendecomp = lap.clone().symmetric_eigen();

    let mut basis = Vec::new();
    for i in 0..eigendecomp.eigenvalues.len() {
        if eigendecomp.eigenvalues[i].abs() < tol {
            let col = eigendecomp.eigenvectors.column(i);
            let coeffs: Vec<f64> = col.iter().copied().collect();
            basis.push(DifferentialForm::new(k, coeffs));
        }
    }

    basis
}

/// Compute the eigenvalues of the Hodge Laplacian at degree k.
pub fn laplacian_eigenvalues(complex: &SimplicialComplex, k: usize) -> Vec<f64> {
    let lap = hodge_laplacian_matrix(complex, k);
    if lap.is_empty() {
        return vec![];
    }

    let nk = complex.num_simplices(k);
    let eigendecomp = lap.clone().symmetric_eigen();
    let mut evals: Vec<f64> = eigendecomp.eigenvalues.iter().copied().collect();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evals
}

/// Verify that the Hodge Laplacian is positive semi-definite.
pub fn verify_positive_semidefinite(complex: &SimplicialComplex, k: usize, tol: f64) -> bool {
    let evals = laplacian_eigenvalues(complex, k);
    evals.iter().all(|&e| e > -tol)
}

/// Verify Δ = dδ + δd: check that harmonic forms are exactly closed + co-closed.
pub fn verify_hodge_identity(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let k = form.degree;
    let d = complex.exterior_derivative_matrix(k);
    let delta = codifferential_matrix(complex, k);

    let d_prev = if k > 0 {
        complex.exterior_derivative_matrix(k - 1)
    } else {
        DMatrix::zeros(0, 0)
    };
    let delta_prev = codifferential_matrix(complex, k + 1);

    // Δω = d(δω) + δ(dω)
    let v = form.to_vector();

    let d_delta_w = if !d_prev.is_empty() && !delta.is_empty() {
        &d_prev * &(&delta * &v)
    } else {
        DVector::zeros(v.len())
    };

    let delta_d_w = if !delta_prev.is_empty() && !d.is_empty() {
        &delta_prev * &(&d * &v)
    } else {
        DVector::zeros(v.len())
    };

    let delta_w = d_delta_w + delta_d_w;
    let lap = apply_hodge_laplacian(complex, form);

    let diff: Vec<f64> = delta_w.iter().zip(lap.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();

    diff.iter().all(|d| *d < tol)
}

use nalgebra::DVector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_symmetric() {
        let sc = SimplicialComplex::triangle();
        let lap = hodge_laplacian_matrix(&sc, 0);
        assert_eq!(lap.nrows(), lap.ncols());
        // Check symmetry
        for i in 0..lap.nrows() {
            for j in i+1..lap.ncols() {
                assert!((lap[(i,j)] - lap[(j,i)]).abs() < 1e-10,
                    "Not symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_laplacian_positive_semidefinite_0() {
        let sc = SimplicialComplex::triangle();
        assert!(verify_positive_semidefinite(&sc, 0, 1e-8));
    }

    #[test]
    fn test_laplacian_positive_semidefinite_1() {
        let sc = SimplicialComplex::triangle();
        assert!(verify_positive_semidefinite(&sc, 1, 1e-8));
    }

    #[test]
    fn test_constant_0form_harmonic_on_triangle() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 1.0, 1.0]);
        assert!(is_harmonic(&sc, &f, 1e-8));
    }

    #[test]
    fn test_nonconstant_0form_not_harmonic_on_triangle() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        assert!(!is_harmonic(&sc, &f, 1e-6));
    }

    #[test]
    fn test_laplacian_tetrahedron_psd_0() {
        let sc = SimplicialComplex::tetrahedron();
        assert!(verify_positive_semidefinite(&sc, 0, 1e-8));
    }

    #[test]
    fn test_laplacian_tetrahedron_psd_1() {
        let sc = SimplicialComplex::tetrahedron();
        assert!(verify_positive_semidefinite(&sc, 1, 1e-8));
    }

    #[test]
    fn test_laplacian_tetrahedron_psd_2() {
        let sc = SimplicialComplex::tetrahedron();
        assert!(verify_positive_semidefinite(&sc, 2, 1e-8));
    }

    #[test]
    fn test_harmonic_basis_triangle_0() {
        let sc = SimplicialComplex::triangle();
        let basis = harmonic_basis(&sc, 0, 1e-6);
        assert_eq!(basis.len(), 1); // H^0 = 1 for connected
    }

    #[test]
    fn test_harmonic_basis_triangle_1() {
        let sc = SimplicialComplex::triangle();
        let basis = harmonic_basis(&sc, 1, 1e-6);
        assert_eq!(basis.len(), 0); // H^1 = 0 for contractible
    }

    #[test]
    fn test_harmonic_form_is_closed_and_coclosed() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 1.0, 1.0]);
        assert!(is_harmonic_ddelta(&sc, &f, 1e-8));
    }

    #[test]
    fn test_eigenvalues_nonnegative() {
        let sc = SimplicialComplex::tetrahedron();
        for k in 0..=2 {
            let evals = laplacian_eigenvalues(&sc, k);
            for &e in &evals {
                assert!(e > -1e-8, "Negative eigenvalue {} at degree {}", e, k);
            }
        }
    }

    #[test]
    fn test_laplacian_linearity() {
        let sc = SimplicialComplex::triangle();
        let f1 = DifferentialForm::new(0, vec![1.0, 0.0, 0.0]);
        let f2 = DifferentialForm::new(0, vec![0.0, 1.0, 0.0]);
        let l1 = apply_hodge_laplacian(&sc, &f1);
        let l2 = apply_hodge_laplacian(&sc, &f2);
        let sum = f1.add(&f2);
        let l_sum = apply_hodge_laplacian(&sc, &sum);
        let expected = l1.add(&l2);
        assert!(l_sum.sub(&expected).is_zero(1e-10));
    }

    #[test]
    fn test_two_vertices_harmonic() {
        let sc = SimplicialComplex::two_vertices();
        // Each vertex is its own harmonic 0-form
        let f = DifferentialForm::new(0, vec![1.0, 0.0]);
        assert!(is_harmonic(&sc, &f, 1e-8));
        let g = DifferentialForm::new(0, vec![0.0, 1.0]);
        assert!(is_harmonic(&sc, &g, 1e-8));
    }

    #[test]
    fn test_single_edge_harmonic() {
        let sc = SimplicialComplex::single_edge();
        // The constant function is harmonic
        let f = DifferentialForm::new(0, vec![1.0, 1.0]);
        assert!(is_harmonic(&sc, &f, 1e-8));
    }

    #[test]
    fn test_laplacian_matrix_square() {
        // The Laplacian should be square
        let sc = SimplicialComplex::triangle();
        for k in 0..=2 {
            let lap = hodge_laplacian_matrix(&sc, k);
            assert_eq!(lap.nrows(), lap.ncols());
            assert_eq!(lap.nrows(), sc.num_simplices(k));
        }
    }
}
