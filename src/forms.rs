//! Differential forms on simplicial complexes.
//!
//! A k-form is represented as a vector of coefficients over the k-simplices.
//! We use nalgebra for the linear algebra backbone.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// A differential k-form: a vector of coefficients over k-simplices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentialForm {
    /// The degree k of this form
    pub degree: usize,
    /// Coefficients indexed by k-simplex
    pub coefficients: Vec<f64>,
}

impl DifferentialForm {
    /// Create a new k-form with the given coefficients.
    pub fn new(degree: usize, coefficients: Vec<f64>) -> Self {
        Self { degree, coefficients }
    }

    /// Create a zero k-form of given degree and dimension.
    pub fn zero(degree: usize, dim: usize) -> Self {
        Self { degree, coefficients: vec![0.0; dim] }
    }

    /// Dimension (number of k-simplices).
    pub fn dim(&self) -> usize {
        self.coefficients.len()
    }

    /// Convert to a nalgebra DVector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.coefficients.clone())
    }

    /// Convert from a nalgebra DVector.
    pub fn from_vector(degree: usize, v: &DVector<f64>) -> Self {
        Self { degree, coefficients: v.iter().copied().collect() }
    }

    /// Add two forms of the same degree.
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.degree, other.degree);
        let coeffs: Vec<f64> = self.coefficients.iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.degree, coeffs)
    }

    /// Subtract two forms of the same degree.
    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.degree, other.degree);
        let coeffs: Vec<f64> = self.coefficients.iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(self.degree, coeffs)
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.degree, self.coefficients.iter().map(|c| c * s).collect())
    }

    /// L² inner product with another form.
    pub fn inner_product(&self, other: &Self) -> f64 {
        self.coefficients.iter()
            .zip(other.coefficients.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// L² norm squared.
    pub fn norm_squared(&self) -> f64 {
        self.inner_product(self)
    }

    /// L² norm.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Check if the form is (approximately) zero.
    pub fn is_zero(&self, tol: f64) -> bool {
        self.norm() < tol
    }

    /// Check exact degree.
    pub fn degree(&self) -> usize {
        self.degree
    }
}

/// A simplicial complex that provides the combinatorial backbone.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimplicialComplex {
    /// Dimension of the complex (max simplex dimension)
    pub dimension: usize,
    /// Simplex counts by degree: simplex_count[k] = number of k-simplices
    pub simplex_count: Vec<usize>,
    /// Boundary matrices: boundary[k] maps (k+1)-simplices to k-simplices
    /// Stored as (rows, cols, entries) where entries[row * cols + col] = value
    pub boundary_matrices: Vec<BoundaryMatrix>,
}

/// A sparse-ish boundary matrix stored densely.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<f64>,
}

impl BoundaryMatrix {
    pub fn new(rows: usize, cols: usize, entries: Vec<f64>) -> Self {
        assert_eq!(entries.len(), rows * cols);
        Self { rows, cols, entries }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, entries: vec![0.0; rows * cols] }
    }

    pub fn to_dmatrix(&self) -> DMatrix<f64> {
        // Convert from row-major storage to column-major for nalgebra
        let mut col_major = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                col_major[c * self.rows + r] = self.entries[r * self.cols + c];
            }
        }
        DMatrix::from_vec_generic(
            nalgebra::Dyn(self.rows),
            nalgebra::Dyn(self.cols),
            col_major,
        )
    }

    pub fn from_dmatrix(m: &DMatrix<f64>) -> Self {
        let rows = m.nrows();
        let cols = m.ncols();
        // Convert from column-major to row-major
        let mut entries = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                entries[r * cols + c] = m[(r, c)];
            }
        }
        Self { rows, cols, entries }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.entries[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.entries[row * self.cols + col] = val;
    }
}

impl SimplicialComplex {
    /// Create a complex from boundary matrices.
    /// boundary_matrices[k] is the boundary from (k+1)-simplices to k-simplices.
    pub fn new(simplex_count: Vec<usize>, boundary_matrices: Vec<BoundaryMatrix>) -> Self {
        let dimension = simplex_count.len().saturating_sub(1);
        Self { dimension, simplex_count, boundary_matrices }
    }

    /// Number of k-simplices.
    pub fn num_simplices(&self, k: usize) -> usize {
        self.simplex_count.get(k).copied().unwrap_or(0)
    }

    /// Get the boundary matrix ∂_{k+1}: C_{k+1} → C_k as a DMatrix.
    pub fn boundary(&self, k_plus_1: usize) -> Option<DMatrix<f64>> {
        self.boundary_matrices.get(k_plus_1).map(|bm| bm.to_dmatrix())
    }

    /// The exterior derivative d_k: Ω^k → Ω^{k+1} is the transpose of ∂_{k+1}.
    pub fn exterior_derivative_matrix(&self, k: usize) -> DMatrix<f64> {
        if k + 1 <= self.dimension {
            if let Some(b) = self.boundary(k + 1) {
                return b.transpose();
            }
        }
        DMatrix::zeros(0, 0)
    }

    /// Build a standard simplex (triangle).
    /// Edges: e0=(0,1), e1=(1,2), e2=(2,0)
    /// Face: [0,1,2], boundary = (1,2)-(0,2)+(0,1) = e1+e2+e0
    pub fn triangle() -> Self {
        let simplex_count = vec![3, 3, 1];
        let mut d1 = BoundaryMatrix::zeros(3, 3);
        // e0=(0,1): v1 - v0
        d1.set(0, 0, -1.0); d1.set(1, 0, 1.0);
        // e1=(1,2): v2 - v1
        d1.set(1, 1, -1.0); d1.set(2, 1, 1.0);
        // e2=(2,0): v0 - v2
        d1.set(2, 2, -1.0); d1.set(0, 2, 1.0);

        let mut d2 = BoundaryMatrix::zeros(3, 1);
        // ∂[0,1,2] = (1,2)-(0,2)+(0,1)
        // (1,2)=e1:+1, (0,2)=-e2 so -(0,2)=+e2, (0,1)=e0:+1
        d2.set(0, 0, 1.0);
        d2.set(1, 0, 1.0);
        d2.set(2, 0, 1.0);

        Self::new(simplex_count, vec![d1, d2])
    }

    /// Build a tetrahedron (3-simplex) with correct boundary matrices.
    /// Edges: e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3), e5=(2,3)
    /// Faces: f0=(0,1,2), f1=(0,1,3), f2=(0,2,3), f3=(1,2,3)
    pub fn tetrahedron() -> Self {
        let simplex_count = vec![4, 6, 4, 1];
        // d1: 4×6 boundary matrix (vertices ← edges)
        let mut d1 = BoundaryMatrix::zeros(4, 6);
        // e0=(0,1)
        d1.set(0, 0, -1.0); d1.set(1, 0, 1.0);
        // e1=(0,2)
        d1.set(0, 1, -1.0); d1.set(2, 1, 1.0);
        // e2=(0,3)
        d1.set(0, 2, -1.0); d1.set(3, 2, 1.0);
        // e3=(1,2)
        d1.set(1, 3, -1.0); d1.set(2, 3, 1.0);
        // e4=(1,3)
        d1.set(1, 4, -1.0); d1.set(3, 4, 1.0);
        // e5=(2,3)
        d1.set(2, 5, -1.0); d1.set(3, 5, 1.0);

        // d2: 6×4 boundary matrix (edges ← faces)
        // ∂(i,j,k) = (j,k) - (i,k) + (i,j)
        // In edge indices: (j,k)→idx(j,k), (i,k)→idx(i,k), (i,j)→idx(i,j)
        // Reversed pair (a,b) maps to -edge(b,a) where edge(b,a) is in our list
        let mut d2 = BoundaryMatrix::zeros(6, 4);
        // f0=(0,1,2): ∂ = (1,2)-(0,2)+(0,1) = e3 - e1 + e0
        d2.set(0, 0, 1.0); d2.set(1, 0, -1.0); d2.set(3, 0, 1.0);
        // f1=(0,1,3): ∂ = (1,3)-(0,3)+(0,1) = e4 - e2 + e0
        d2.set(0, 1, 1.0); d2.set(2, 1, -1.0); d2.set(4, 1, 1.0);
        // f2=(0,2,3): ∂ = (2,3)-(0,3)+(0,2) = e5 - e2 + e1
        d2.set(1, 2, 1.0); d2.set(2, 2, -1.0); d2.set(5, 2, 1.0);
        // f3=(1,2,3): ∂ = (2,3)-(1,3)+(1,2) = e5 - e4 + e3
        d2.set(3, 3, 1.0); d2.set(4, 3, -1.0); d2.set(5, 3, 1.0);

        // d3: 4×1 boundary matrix (faces ← volume)
        // ∂[0,1,2,3] = (1,2,3) - (0,2,3) + (0,1,3) - (0,1,2)
        // = f3 - f2 + f1 - f0
        let mut d3 = BoundaryMatrix::zeros(4, 1);
        d3.set(0, 0, -1.0); d3.set(1, 0, 1.0);
        d3.set(2, 0, -1.0); d3.set(3, 0, 1.0);

        Self::new(simplex_count, vec![d1, d2, d3])
    }

    /// Two isolated vertices (disconnected).
    pub fn two_vertices() -> Self {
        let simplex_count = vec![2];
        Self::new(simplex_count, vec![])
    }

    /// A single edge (two vertices connected).
    pub fn single_edge() -> Self {
        let simplex_count = vec![2, 1];
        let mut d1 = BoundaryMatrix::zeros(2, 1);
        d1.set(0, 0, -1.0);
        d1.set(1, 0, 1.0);
        Self::new(simplex_count, vec![d1])
    }

    /// A square (4 vertices, 4 edges, possibly 1 face).
    pub fn square() -> Self {
        let simplex_count = vec![4, 4, 1];
        let mut d1 = BoundaryMatrix::zeros(4, 4);
        // Edges: (0,1), (1,2), (2,3), (3,0)
        d1.set(0, 0, -1.0); d1.set(1, 0, 1.0);
        d1.set(1, 1, -1.0); d1.set(2, 1, 1.0);
        d1.set(2, 2, -1.0); d1.set(3, 2, 1.0);
        d1.set(3, 3, -1.0); d1.set(0, 3, 1.0);

        let mut d2 = BoundaryMatrix::zeros(4, 1);
        d2.set(0, 0, 1.0); d2.set(1, 0, 1.0);
        d2.set(2, 0, 1.0); d2.set(3, 0, 1.0);

        Self::new(simplex_count, vec![d1, d2])
    }

    /// Figure eight: two triangles sharing a vertex.
    pub fn figure_eight() -> Self {
        // 5 vertices: 0=shared, 1,2 triangle A, 3,4 triangle B
        // 6 edges: (0,1), (1,2), (2,0), (0,3), (3,4), (4,0)
        // 2 faces
        let simplex_count = vec![5, 6, 2];
        let mut d1 = BoundaryMatrix::zeros(5, 6);
        // (0,1)
        d1.set(0, 0, -1.0); d1.set(1, 0, 1.0);
        // (1,2)
        d1.set(1, 1, -1.0); d1.set(2, 1, 1.0);
        // (2,0)
        d1.set(2, 2, -1.0); d1.set(0, 2, 1.0);
        // (0,3)
        d1.set(0, 3, -1.0); d1.set(3, 3, 1.0);
        // (3,4)
        d1.set(3, 4, -1.0); d1.set(4, 4, 1.0);
        // (4,0)
        d1.set(4, 5, -1.0); d1.set(0, 5, 1.0);

        let mut d2 = BoundaryMatrix::zeros(6, 2);
        // Face A: edges (0,1), (1,2), (2,0) = cols 0,1,2
        d2.set(0, 0, 1.0); d2.set(1, 0, 1.0); d2.set(2, 0, 1.0);
        // Face B: edges (0,3), (3,4), (4,0) = cols 3,4,5
        d2.set(3, 1, 1.0); d2.set(4, 1, 1.0); d2.set(5, 1, 1.0);

        Self::new(simplex_count, vec![d1, d2])
    }

    /// Torus-like: represented as a 2x2 grid with identifications (simplified).
    /// Uses a triangulated torus with 9 vertices, 27 edges, 18 faces.
    /// For simplicity, we use a smaller model.
    pub fn torus_small() -> Self {
        // Simplified: use the square with opposite edges identified
        // 1 vertex, 2 edges (a, b), 1 face
        let simplex_count = vec![1, 2, 1];
        let mut d1 = BoundaryMatrix::zeros(1, 2);
        // Both edges start and end at the same vertex
        d1.set(0, 0, 0.0);
        d1.set(0, 1, 0.0);
        let mut d2 = BoundaryMatrix::zeros(2, 1);
        d2.set(0, 0, 0.0);
        d2.set(1, 0, 0.0);
        Self::new(simplex_count, vec![d1, d2])
    }

    /// Empty complex.
    pub fn empty() -> Self {
        Self::new(vec![0], vec![])
    }
}
