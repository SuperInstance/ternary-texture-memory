# ternary-texture-memory

Texture memory access patterns for ternary data.

## Overview

`ternary-texture-memory` provides abstractions for GPU texture memory access with support for 1D, 2D, and 3D textures, nearest and linear interpolation sampling, boundary modes (clamp/wrap/mirror), and spatial coherence analysis. Designed for ternary (three-valued logic) data with spatial locality — where texture caching significantly improves access performance compared to global memory.

## Features

- **Texture1D, Texture2D, Texture3D** — Spatially cached texture access for ternary data in 1, 2, and 3 dimensions with integer reads and floating-point sampling.
- **Sampler** — Configurable interpolation mode (nearest or linear) and boundary mode (clamp, wrap, mirror) for out-of-range coordinate handling.
- **Ternary Values** — Native support for three-valued logic (True/False/Unknown) with Kleene logic operations (AND, OR), numeric conversion for interpolation, and round-trip fidelity.
- **Spatial Coherence Analysis** — Measure neighbor similarity ratios, Shannon entropy, and coherence metrics for 1D and 2D data to quantify spatial locality.
- **Boundary Modes** — Three addressing modes for handling coordinates outside the texture bounds, enabling seamless tiling and edge handling.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-texture-memory = { git = "https://github.com/SuperInstance/ternary-texture-memory" }
```

### Basic Example

```rust
use ternary_texture_memory::*;

// Create a 1D texture
let tex = Texture1D::new(vec![Ternary::True, Ternary::False, Ternary::Unknown, Ternary::True]);

// Read with nearest sampling
let sampler = Sampler::nearest_clamp();
let val = tex.read(1, &sampler);
assert_eq!(val, Ternary::False);

// Sample with linear interpolation
let linear_sampler = Sampler::new(SampleMode::Linear, BoundaryMode::Clamp);
let interpolated = tex.sample(0.5, &linear_sampler);
// Lerp between True(1.0) and False(0.0) at 0.5 → 0.5 → Unknown
assert_eq!(interpolated, Ternary::Unknown);
```

### 2D Texture with Boundary Handling

```rust
use ternary_texture_memory::*;

let mut tex = Texture2D::constant(8, 8, Ternary::Unknown);
tex.write(3, 3, Ternary::True).unwrap();
tex.write(4, 3, Ternary::True).unwrap();

// Wrap mode wraps coordinates around
let wrap_sampler = Sampler::new(SampleMode::Nearest, BoundaryMode::Wrap);
let val = tex.read(8, 0, &wrap_sampler); // wraps to (0, 0)
assert_eq!(val, Ternary::Unknown);

// Mirror mode reflects at boundaries
let mirror_sampler = Sampler::new(SampleMode::Nearest, BoundaryMode::Mirror);
let val = tex.read(-1, 0, &mirror_sampler); // mirrors to (1, 0)
```

### Spatial Coherence Analysis

```rust
use ternary_texture_memory::*;

// High coherence data (runs of same values)
let data = vec![
    Ternary::True, Ternary::True, Ternary::True,
    Ternary::False, Ternary::False,
];
let report = coherence_1d(&data);
println!("Same-neighbor ratio: {:.1}%", report.same_neighbor_ratio * 100.0);
println!("Entropy: {:.3} bits", report.entropy);
```

## Architecture

### Ternary Values

The `Ternary` enum represents three-valued logic:
- `True` → 1.0
- `False` → 0.0
- `Unknown` → 0.5

Supports Kleene logic AND/OR operations and numeric conversion for interpolation. The `from_f64` threshold maps values below 0.25 to False, above 0.75 to True, and in between to Unknown.

### Texture Types

Each texture dimension (1D, 2D, 3D) supports:
- **Integer reads** with boundary handling via the sampler's boundary mode
- **Float sampling** with nearest or linear (bilinear/trilinear) interpolation
- **Writes** at integer coordinates with bounds checking

### Boundary Modes

- **Clamp** — Coordinates are clamped to `[0, size-1]`. Safe for edge-aware kernels.
- **Wrap** — Coordinates wrap using modulo. Useful for periodic/tiling patterns.
- **Mirror** — Coordinates reflect at boundaries. Good for symmetric data access.

### Spatial Coherence

The coherence analysis measures how spatially local neighboring values are:
- `same_neighbor_ratio` — Fraction of adjacent pairs with matching values (1.0 = perfectly coherent, 0.0 = no neighbors match)
- `entropy` — Shannon entropy in bits (0.0 = all same, log₂(3) ≈ 1.585 = uniform random)

High coherence means texture caching will be very effective; low coherence suggests random-access patterns where caching provides less benefit.

## License

MIT
