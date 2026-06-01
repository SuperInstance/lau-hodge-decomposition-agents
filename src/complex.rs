//! De Rham complex: 0 → Ω⁰ → Ω¹ → Ω² → ... with d² = 0.
//!
//! The exterior derivative d_k maps k-forms to (k+1)-forms.
//! The fundamental identity d² = 0 ensures exact forms are closed.

use crate::forms::{DifferentialForm, SimplicialComplex};

/// Apply the exterior derivative d_k to a k-form.
/// d_k = (∂_{k+1})^T, the transpose of the boundary operator.
pub fn exterior_derivative(complex: &SimplicialComplex, form: &DifferentialForm) -> DifferentialForm {
    let k = form.degree;
    let dm = complex.exterior_derivative_matrix(k);
    if dm.is_empty() {
        return DifferentialForm::zero(k + 1, 0);
    }
    let v = form.to_vector();
    let result = &dm * &v;
    DifferentialForm::from_vector(k + 1, &result)
}

/// Verify that d² = 0 on a given form.
pub fn verify_d_squared_zero(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let d1 = exterior_derivative(complex, form);
    let d2 = exterior_derivative(complex, &d1);
    d2.is_zero(tol)
}

/// Verify d² = 0 as a matrix identity: d_{k+1} * d_k = 0.
pub fn verify_d_squared_matrix(complex: &SimplicialComplex, k: usize, tol: f64) -> bool {
    let dk = complex.exterior_derivative_matrix(k);
    let dk1 = complex.exterior_derivative_matrix(k + 1);
    if dk.is_empty() || dk1.is_empty() {
        return true;
    }
    let product = &dk1 * &dk;
    product.iter().all(|x| x.abs() < tol)
}

/// Check if a form is closed: dω = 0.
pub fn is_closed(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    let dw = exterior_derivative(complex, form);
    dw.is_zero(tol)
}

/// Check if a form is exact: ω = dα for some α.
/// A form is exact iff it is in the image of d_{k-1}.
pub fn is_exact(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    if form.degree == 0 {
        // 0-forms are exact iff they are zero
        return form.is_zero(tol);
    }
    // Check if form is in the column space of d_{k-1}
    let dm = complex.exterior_derivative_matrix(form.degree - 1);
    if dm.is_empty() {
        return form.is_zero(tol);
    }
    // Solve dm * x = form using least squares
    let target = form.to_vector();
    let svd = dm.clone().svd(true, true).solve(&target, tol);
    match svd {
        Ok(x) => {
            let residual = &target - &(dm * &x);
            residual.iter().all(|v| v.abs() < tol * 100.0)
        }
        Err(_) => false,
    }
}

/// Poincaré lemma: on a contractible complex, closed forms are exact.
/// For a simplex (contractible), any closed form should be exact.
pub fn poincare_lemma_check(complex: &SimplicialComplex, form: &DifferentialForm, tol: f64) -> bool {
    if is_closed(complex, form, tol) {
        is_exact(complex, form, tol)
    } else {
        true // not closed, lemma doesn't apply
    }
}

/// The full de Rham complex as matrices.
pub struct DeRhamComplex<'a> {
    pub complex: &'a SimplicialComplex,
}

impl<'a> DeRhamComplex<'a> {
    pub fn new(complex: &'a SimplicialComplex) -> Self {
        Self { complex }
    }

    /// Get the exterior derivative matrix at degree k.
    pub fn d(&self, k: usize) -> nalgebra::DMatrix<f64> {
        self.complex.exterior_derivative_matrix(k)
    }

    /// Verify the full cochain complex property: d_{k+1} * d_k = 0 for all k.
    pub fn verify_complex(&self, tol: f64) -> Vec<(usize, bool)> {
        (0..self.complex.dimension)
            .map(|k| (k, verify_d_squared_matrix(self.complex, k, tol)))
            .collect()
    }

    /// Compute the k-th cohomology dimension (as a vector space dimension).
    /// dim H^k = dim(ker d_k) - dim(im d_{k-1})
    pub fn cohomology_dimension(&self, k: usize) -> usize {
        let dk = self.d(k);
        let dim_k = self.complex.num_simplices(k);
        if dim_k == 0 {
            return 0;
        }

        // Rank of d_k = dim(im d_k)
        let rank_dk = if dk.is_empty() { 0 } else { rank_of_matrix(&dk) };

        // Rank of d_{k-1} = dim(im d_{k-1})
        let rank_d_prev = if k == 0 {
            0
        } else {
            let d_prev = self.d(k - 1);
            if d_prev.is_empty() { 0 } else { rank_of_matrix(&d_prev) }
        };

        // dim H^k = dim(Ω^k) - rank(d_k) - rank(d_{k-1})
        dim_k - rank_dk - rank_d_prev
    }
}

/// Compute the rank of a matrix using SVD.
fn rank_of_matrix(m: &nalgebra::DMatrix<f64>) -> usize {
    let svd = m.clone().svd(false, false);
    let tol = 1e-10 * m.nrows().max(m.ncols()) as f64;
    svd.singular_values.iter().filter(|&&s| s > tol).count()
}

/// Compute rank from SVD singular values.
pub fn rank_from_svd(singular_values: &nalgebra::DVector<f64>, tol: f64) -> usize {
    singular_values.iter().filter(|&&s| s > tol).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forms::SimplicialComplex;

    #[test]
    fn test_d_squared_zero_triangle_0form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        assert!(verify_d_squared_zero(&sc, &f, 1e-10));
    }

    #[test]
    fn test_d_squared_zero_triangle_1form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(1, vec![1.0, 2.0, 3.0]);
        assert!(verify_d_squared_zero(&sc, &f, 1e-10));
    }

    #[test]
    fn test_d_squared_matrix_triangle() {
        let sc = SimplicialComplex::triangle();
        assert!(verify_d_squared_matrix(&sc, 0, 1e-10));
    }

    #[test]
    fn test_d_squared_zero_tetrahedron_0form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(verify_d_squared_zero(&sc, &f, 1e-10));
    }

    #[test]
    fn test_d_squared_zero_tetrahedron_1form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(verify_d_squared_zero(&sc, &f, 1e-10));
    }

    #[test]
    fn test_d_squared_zero_tetrahedron_2form() {
        let sc = SimplicialComplex::tetrahedron();
        let f = DifferentialForm::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(verify_d_squared_zero(&sc, &f, 1e-10));
    }

    #[test]
    fn test_closed_constant_0form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![5.0, 5.0, 5.0]);
        assert!(is_closed(&sc, &f, 1e-10));
    }

    #[test]
    fn test_not_closed_varying_0form() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        let d = exterior_derivative(&sc, &f);
        assert!(!d.is_zero(1e-10));
    }

    #[test]
    fn test_exact_on_simplex() {
        let sc = SimplicialComplex::triangle();
        // On a simplex, closed 1-forms are exact (Poincaré lemma)
        let f0 = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);
        let d_f0 = exterior_derivative(&sc, &f0);
        assert!(is_closed(&sc, &d_f0, 1e-10));
        assert!(is_exact(&sc, &d_f0, 1e-6));
    }

    #[test]
    fn test_poincare_lemma_simplex() {
        let sc = SimplicialComplex::triangle();
        let f = DifferentialForm::new(0, vec![3.0, 7.0, 2.0]);
        let df = exterior_derivative(&sc, &f);
        assert!(poincare_lemma_check(&sc, &df, 1e-6));
    }

    #[test]
    fn test_de_rham_complex_triangle() {
        let sc = SimplicialComplex::triangle();
        let drc = DeRhamComplex::new(&sc);
        let results = drc.verify_complex(1e-10);
        for (_, ok) in results {
            assert!(ok);
        }
    }

    #[test]
    fn test_de_rham_complex_tetrahedron() {
        let sc = SimplicialComplex::tetrahedron();
        let drc = DeRhamComplex::new(&sc);
        let results = drc.verify_complex(1e-10);
        for (_, ok) in results {
            assert!(ok);
        }
    }

    #[test]
    fn test_cohomology_triangle_h0() {
        let sc = SimplicialComplex::triangle();
        let drc = DeRhamComplex::new(&sc);
        // Triangle is connected, so H^0 = 1
        assert_eq!(drc.cohomology_dimension(0), 1);
    }

    #[test]
    fn test_cohomology_triangle_h1() {
        let sc = SimplicialComplex::triangle();
        let drc = DeRhamComplex::new(&sc);
        // Triangle is contractible, so H^1 = 0
        assert_eq!(drc.cohomology_dimension(1), 0);
    }

    #[test]
    fn test_cohomology_two_vertices_h0() {
        let sc = SimplicialComplex::two_vertices();
        let drc = DeRhamComplex::new(&sc);
        // Two disconnected vertices, so H^0 = 2
        assert_eq!(drc.cohomology_dimension(0), 2);
    }

    #[test]
    fn test_cohomology_tetrahedron_h0() {
        let sc = SimplicialComplex::tetrahedron();
        let drc = DeRhamComplex::new(&sc);
        assert_eq!(drc.cohomology_dimension(0), 1);
    }

    #[test]
    fn test_cohomology_tetrahedron_h1() {
        let sc = SimplicialComplex::tetrahedron();
        let drc = DeRhamComplex::new(&sc);
        assert_eq!(drc.cohomology_dimension(1), 0);
    }

    #[test]
    fn test_exterior_derivative_linearity() {
        let sc = SimplicialComplex::triangle();
        let f1 = DifferentialForm::new(0, vec![1.0, 0.0, 0.0]);
        let f2 = DifferentialForm::new(0, vec![0.0, 1.0, 0.0]);
        let d_f1 = exterior_derivative(&sc, &f1);
        let d_f2 = exterior_derivative(&sc, &f2);
        let sum = f1.add(&f2);
        let d_sum = exterior_derivative(&sc, &sum);
        let expected = d_f1.add(&d_f2);
        assert!(d_sum.sub(&expected).is_zero(1e-10));
    }
}
