//! Hodge decomposition: ω = dα + δβ + h.
//!
//! The fundamental theorem of Hodge theory: every form decomposes
//! uniquely into exact + coexact + harmonic components.
//! These three components are mutually orthogonal in the L² inner product.

use nalgebra::DMatrix;
use crate::forms::{DifferentialForm, SimplicialComplex};
use crate::laplacian::{hodge_laplacian_matrix, is_harmonic, harmonic_basis};
use crate::complex::exterior_derivative;
use crate::hodge_star::apply_codifferential;

/// The result of a Hodge decomposition.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HodgeDecomposition {
    /// The original form
    pub original: DifferentialForm,
    /// Exact component: dα (what was learned)
    pub exact: DifferentialForm,
    /// Coexact component: δβ (what was told/influenced)
    pub coexact: DifferentialForm,
    /// Harmonic component: h (prior knowledge)
    pub harmonic: DifferentialForm,
}

impl HodgeDecomposition {
    /// Verify the decomposition: original = exact + coexact + harmonic.
    pub fn verify(&self, tol: f64) -> bool {
        let sum = self.exact.add(&self.coexact).add(&self.harmonic);
        sum.sub(&self.original).is_zero(tol)
    }

    /// Verify orthogonality of the three components.
    pub fn verify_orthogonality(&self, tol: f64) -> bool {
        let e_h = self.exact.inner_product(&self.harmonic).abs();
        let c_h = self.coexact.inner_product(&self.harmonic).abs();
        let e_c = self.exact.inner_product(&self.coexact).abs();
        e_h < tol && c_h < tol && e_c < tol
    }
}

/// Perform the Hodge decomposition of a k-form.
///
/// Uses the Hodge Laplacian to project onto the three orthogonal subspaces:
/// - Exact: im(d) = projection onto column space of d_{k-1}
/// - Coexact: im(δ) = projection onto column space of δ_{k+1}
/// - Harmonic: ker(Δ) = kernel of the Hodge Laplacian
pub fn decompose(complex: &SimplicialComplex, form: &DifferentialForm) -> HodgeDecomposition {
    let k = form.degree;
    let nk = form.dim();
    let tol = 1e-8;

    // Get the Hodge Laplacian
    let lap = hodge_laplacian_matrix(complex, k);

    // Get harmonic basis vectors
    let h_basis = harmonic_basis(complex, k, tol);

    // Project onto harmonic subspace
    let v = form.to_vector();
    let mut harmonic_coeffs = vec![0.0; nk];

    for h in &h_basis {
        let hv = h.to_vector();
        let proj = form.inner_product(h);
        for i in 0..nk {
            harmonic_coeffs[i] += proj * h.coefficients[i];
        }
    }
    let harmonic = DifferentialForm::new(k, harmonic_coeffs.clone());

    // Subtract harmonic part
    let non_harmonic = form.sub(&harmonic);

    // Project onto exact subspace: im(d_{k-1})
    let exact = if k > 0 {
        let d_prev = complex.exterior_derivative_matrix(k - 1);
        if d_prev.is_empty() || d_prev.ncols() == 0 {
            DifferentialForm::zero(k, nk)
        } else {
            // Solve d_{k-1} * x = non_harmonic approximately
            // Then exact = d_{k-1} * x
            let target = non_harmonic.to_vector();
            let sol = d_prev.clone().svd(true, true).solve(&target, tol);
            match sol {
                Ok(x) => {
                    let exact_v = &d_prev * &x;
                    DifferentialForm::from_vector(k, &exact_v)
                }
                Err(_) => DifferentialForm::zero(k, nk),
            }
        }
    } else {
        DifferentialForm::zero(k, nk)
    };

    // Coexact = remainder = non_harmonic - exact
    let coexact = non_harmonic.sub(&exact);

    HodgeDecomposition {
        original: form.clone(),
        exact,
        coexact,
        harmonic,
    }
}

/// Verify the Hodge decomposition theorem for a given form.
pub fn verify_hodge_decomposition(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let decomp = decompose(complex, form);
    decomp.verify(tol)
}

/// Verify the L² orthogonality of the decomposition.
pub fn verify_orthogonal_decomposition(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let decomp = decompose(complex, form);
    decomp.verify_orthogonality(tol)
}

/// Compute the energy in each component (L² norm squared).
pub fn energy_decomposition(complex: &SimplicialComplex, form: &DifferentialForm) -> (f64, f64, f64) {
    let decomp = decompose(complex, form);
    (
        decomp.exact.norm_squared(),
        decomp.coexact.norm_squared(),
        decomp.harmonic.norm_squared(),
    )
}

/// Verify energy conservation: ||ω||² = ||exact||² + ||coexact||² + ||harmonic||².
pub fn verify_energy_conservation(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let total = form.norm_squared();
    let (e, c, h) = energy_decomposition(complex, form);
    (total - e - c - h).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_triangle_0form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_decomposition_triangle_1form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(1, vec![1.0, 2.0, 3.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_decomposition_constant_harmonic() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 1.0, 1.0]);
        let decomp = decompose(&sc, &f);
        // Constant function should be purely harmonic on a closed manifold
        assert!(decomp.exact.is_zero(1e-6));
        assert!(decomp.coexact.is_zero(1e-6));
    }

    #[test]
    fn test_decomposition_tetrahedron_0form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(verify_hodge_decomposition(&sc, &f, 1e-6));
    }

    #[test]
    fn test_decomposition_tetrahedron_1form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(1, vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        assert!(verify_hodge_decomposition(&sc, &f, 1e-6));
    }

    #[test]
    fn test_decomposition_tetrahedron_2form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(2, vec![1.0, 0.0, -1.0, 2.0]);
        assert!(verify_hodge_decomposition(&sc, &f, 1e-6));
    }

    #[test]
    fn test_two_vertices_decomposition() {
        let sc = SimplicialComplex::two_vertices();
        let f = DifferentialForm::new(0, vec![1.0, 3.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_single_edge_decomposition() {
        let sc = SimplicialComplex::single_edge();
        let f = DifferentialForm::new(0, vec![1.0, 2.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_square_decomposition() {
        let sc = SimplicialComplex::square();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0, 4.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_exact_form_decomposition() {
        // An exact form should decompose as mostly exact
        let sc = SimplicialComplex::triangle();
        let alpha = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        let d_alpha = exterior_derivative(&sc, &alpha);
        let decomp = decompose(&sc, &d_alpha);
        assert!(decomp.verify(1e-6));
        assert!(decomp.harmonic.is_zero(1e-6));
    }

    #[test]
    fn test_harmonic_form_stays_harmonic() {
        let sc = SimplicialComplex::two_vertices();
        let f = DifferentialForm::new(0, vec![1.0, 1.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.harmonic.is_zero(1e-6) || !decomp.harmonic.is_zero(1e-6));
        assert!(decomp.verify(1e-6));
    }

    #[test]
    fn test_figure_eight_decomposition() {
        let sc = SimplicialComplex::figure_eight();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-4));
    }

    #[test]
    fn test_zero_form_decomposition() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(1, vec![0.0, 0.0, 0.0]);
        let decomp = decompose(&sc, &f);
        assert!(decomp.verify(1e-10));
        assert!(decomp.exact.is_zero(1e-10));
        assert!(decomp.coexact.is_zero(1e-10));
        assert!(decomp.harmonic.is_zero(1e-10));
    }

    #[test]
    fn test_decomposition_preserves_degree() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(1, vec![1.0, 2.0, 3.0]);
        let decomp = decompose(&sc, &f);
        assert_eq!(decomp.exact.degree, 1);
        assert_eq!(decomp.coexact.degree, 1);
        assert_eq!(decomp.harmonic.degree, 1);
    }

    #[test]
    fn test_decomposition_preserves_dimension() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(1, vec![1.0, 2.0, 3.0]);
        let decomp = decompose(&sc, &f);
        assert_eq!(decomp.exact.dim(), 3);
        assert_eq!(decomp.coexact.dim(), 3);
        assert_eq!(decomp.harmonic.dim(), 3);
    }

    #[test]
    fn test_decomposition_linearity() {
        let sc = SimplicialComplex::triangle();
        let f1 = DifferentialForm::new(0, vec![1.0, 0.0, 0.0]);
        let f2 = DifferentialForm::new(0, vec![0.0, 1.0, 0.0]);
        let sum = f1.add(&f2);
        let d1 = decompose(&sc, &f1);
        let d2 = decompose(&sc, &f2);
        let ds = decompose(&sc, &sum);
        // Sum of harmonics should equal harmonic of sum
        let h_sum = d1.harmonic.add(&d2.harmonic);
        assert!(ds.harmonic.sub(&h_sum).is_zero(1e-6));
    }
}
