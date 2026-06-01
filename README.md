# lau-hodge-decomposition-agents

> Hodge decomposition for agent systems — every agent signal decomposes into **exact + coexact + harmonic**: `ω = dα + δβ + h`

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## What This Does

This crate applies the **Hodge decomposition theorem** from differential geometry to agent behavior analysis. Given any agent signal (represented as a differential form on a simplicial complex), it decomposes it into three orthogonal components:

| Component | Mathematical | Intuitive Meaning |
|-----------|-------------|-------------------|
| **Exact** (dα) | Image of the exterior derivative d | What the agent *learned* — gradient-driven change |
| **Coexact** (δβ) | Image of the codifferential δ | What the agent was *told* — externally imposed signal |
| **Harmonic** (h) | Kernel of the Hodge Laplacian Δ | What the agent *already knew* — prior knowledge |

The three components are mutually orthogonal in the L² inner product and sum exactly to the original signal, giving you an interpretable decomposition of any agent behavior.

## Key Idea

In algebraic topology, the Hodge decomposition theorem states that on a compact Riemannian manifold, every differential k-form ω decomposes uniquely as:

```
ω = dα + δβ + h
```

where d is the exterior derivative (exact part), δ is the codifferential (coexact part), and h is harmonic (in the kernel of the Laplacian). This crate discretizes this theorem using **simplicial complexes** (triangles, tetrahedra, etc.) and sparse linear algebra, then applies it to agent signals.

The decomposition is computed via the Hodge Laplacian Δ = dδ + δd, using SVD to project onto the exact subspace and subtracting out the harmonic component.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-hodge-decomposition-agents = "0.1"
```

Requires Rust 2021 edition.

### Dependencies

- **nalgebra** 0.33 — linear algebra (matrices, SVD, eigendecomposition)
- **serde** 1 (with `derive`) — serialization of all types

## Quick Start

```rust
use lau_hodge_decomposition_agents::*;

// Build a simplicial complex (triangle = 3 vertices, 3 edges, 1 face)
let complex = SimplicialComplex::triangle();

// Define a 0-form (signal on vertices)
let signal = DifferentialForm::new(0, vec![1.0, 2.0, 3.0]);

// Decompose it
let decomp = decompose(&complex, &signal);

println!("Exact:    {:?}", decomp.exact.coefficients);    // learned
println!("Coexact:  {:?}", decomp.coexact.coefficients);  // told
println!("Harmonic: {:?}", decomp.harmonic.coefficients); // known

// Verify the decomposition: original = exact + coexact + harmonic
assert!(decomp.verify(1e-6));

// Verify L² orthogonality of the three components
assert!(decomp.verify_orthogonality(1e-6));

// Check energy conservation: ||ω||² = ||exact||² + ||coexact||² + ||harmonic||²
assert!(verify_energy_conservation(&complex, &signal, 1e-6));
```

### Agent Learning Analysis

```rust
// Analyze what an agent learned vs was told vs already knew
let analysis = AgentLearningAnalysis::new(&complex);

// Feed in a behavior signal
let behavior = DifferentialForm::new(1, vec![0.5, -0.3, 0.8]);
let result = analysis.analyze(&behavior);

println!("Learning fraction: {:.2}%", result.exact_fraction * 100.0);
println!("Instruction fraction: {:.2}%", result.coexact_fraction * 100.0);
println!("Prior fraction: {:.2}%", result.harmonic_fraction * 100.0);
```

### Topological Features

```rust
// Compute Betti numbers (topological invariants)
let betti = compute_betti_numbers(&complex);
// betti[0] = connected components, betti[1] = loops, betti[2] = voids

// Serre duality: relate k-forms to (n-k)-forms
let dual = serre_dual(&complex, &harmonic_form);

// Spectral analysis: eigenvalues of the Hodge Laplacian
let spectrum = hodge_spectrum(&complex, 1); // 1-forms
```

## API Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `DifferentialForm` | `forms` | A discrete k-form: degree + coefficient vector |
| `SimplicialComplex` | `complex` | A simplicial complex with boundary operators |
| `HodgeDecomposition` | `decomposition` | Result of decomposing a form: exact + coexact + harmonic |

### Core Functions

| Function | Module | Description |
|----------|--------|-------------|
| `decompose()` | `decomposition` | Perform Hodge decomposition of a k-form |
| `exterior_derivative()` | `complex` | Apply the exterior derivative d to a k-form |
| `apply_codifferential()` | `hodge_star` | Apply the codifferential δ to a k-form |
| `hodge_laplacian_matrix()` | `laplacian` | Build the Hodge Laplacian Δ = dδ + δd |
| `compute_betti_numbers()` | `betti` | Compute topological Betti numbers β₀, β₁, β₂… |
| `harmonic_basis()` | `laplacian` | Find basis vectors for the harmonic subspace |
| `is_harmonic()` | `laplacian` | Check if a form is in ker(Δ) |
| `hodge_spectrum()` | `spectral` | Eigenvalues of the Hodge Laplacian |

### Agent Analysis

| Type | Description |
|------|-------------|
| `AgentLearningAnalysis` | Decompose agent behaviors into learned/told/known fractions |
| `LearningResult` | Fractions and norms for each decomposition component |

## How It Works

1. **Build the complex**: A `SimplicialComplex` stores vertices, edges, triangles, etc. and precomputes boundary operator matrices Bₖ for each dimension.

2. **Exterior derivative matrix**: The exterior derivative dₖ is the transpose of the boundary operator: dₖ = Bₖ₊₁ᵀ. This maps k-forms to (k+1)-forms.

3. **Hodge Laplacian**: Δₖ = dₖ₋₁ dₖ₋₁ᵀ + dₖᵀ dₖ. This is an nₖ × nₖ matrix where nₖ is the number of k-simplices.

4. **Harmonic projection**: Find the null space of Δₖ using eigendecomposition. Project the form onto this subspace to get h.

5. **Exact projection**: Solve dₖ₋₁ · x ≈ (ω − h) using SVD. The exact component is dₖ₋₁ · x.

6. **Coexact = remainder**: β = ω − dα − h. By construction, this lies in im(δ).

7. **Verification**: The crate verifies ω = dα + δβ + h and checks orthogonality ‖⟨dα, δβ⟩‖ < ε, etc.

## The Math

### Hodge Decomposition Theorem

On a compact orientable Riemannian manifold M, for any k-form ω ∈ Ωᵏ(M):

```
Ωᵏ(M) = im(dₖ₋₁) ⊕ im(δₖ₊₁) ⊕ ker(Δₖ)
```

This gives the unique decomposition ω = dα + δβ + h where:
- **dα** is *exact* (gradient of a (k−1)-form)
- **δβ** is *coexact* (divergence of a (k+1)-form)
- **h** is *harmonic* (Δh = 0, both closed and coclosed)

### Betti Numbers

The k-th Betti number βₖ equals the dimension of the harmonic subspace ker(Δₖ):

```
βₖ = dim ker(Δₖ) = dim {ω : Δₖω = 0}
```

These are topological invariants: β₀ = #connected components, β₁ = #independent loops, β₂ = #voids.

### Serre Duality

For a complex of dimension n, Serre duality establishes an isomorphism between harmonic k-forms and harmonic (n−k)-forms:

```
Hᵏ(M) ≅ Hⁿ⁻ᵏ(M)
```

### Hodge Star

The Hodge star operator ⋆ maps k-forms to (n−k)-forms and is used to define the codifferential:

```
δ = (−1)ⁿᵏ⁺ⁿ⁺¹ ⋆ d ⋆
```

## Tests

121 unit tests covering:
- Decomposition correctness on triangles, tetrahedra, and custom complexes
- Orthogonality verification for all component pairs
- Energy conservation: ‖ω‖² = ‖exact‖² + ‖coexact‖² + ‖harmonic‖²
- Betti number computation (β₀ for connected/disconnected complexes, β₁ for loops)
- Serre duality isomorphism
- Spectral analysis of the Hodge Laplacian
- Agent learning analysis with known behavior patterns

Run with: `cargo test`

## License

MIT
