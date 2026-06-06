//! # ternary-texture-memory
//!
//! Texture memory access patterns for ternary data.
//!
//! This crate provides abstractions for GPU texture memory access with support
//! for 1D, 2D, and 3D textures, nearest and linear sampling, boundary modes
//! (clamp/wrap/mirror), and spatial coherence analysis. Designed for ternary
//! (three-valued logic) data with spatial locality.

use std::fmt;

/// Boundary mode for texture addressing when coordinates fall outside [0, size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Clamp coordinates to the valid range.
    Clamp,
    /// Wrap coordinates using modulo.
    Wrap,
    /// Mirror coordinates at boundaries.
    Mirror,
}

impl BoundaryMode {
    /// Apply the boundary mode to a coordinate within the given size.
    pub fn apply(&self, coord: i64, size: u64) -> u64 {
        let size = size as i64;
        if size <= 0 {
            return 0;
        }
        match self {
            BoundaryMode::Clamp => {
                coord.clamp(0, size - 1) as u64
            }
            BoundaryMode::Wrap => {
                let result = coord % size;
                if result < 0 {
                    (result + size) as u64
                } else {
                    result as u64
                }
            }
            BoundaryMode::Mirror => {
                let mut c = coord;
                let period = 2 * size;
                c = ((c % period) + period) % period;
                if c >= size {
                    (2 * size - 1 - c) as u64
                } else {
                    c as u64
                }
            }
        }
    }
}

/// Interpolation mode for texture sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMode {
    /// Nearest-neighbor sampling.
    Nearest,
    /// Linear interpolation.
    Linear,
}

/// A ternary value: True, False, or Unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ternary {
    True,
    False,
    Unknown,
}

impl Ternary {
    /// Convert to a numeric value for interpolation.
    pub fn to_f64(&self) -> f64 {
        match self {
            Ternary::True => 1.0,
            Ternary::False => 0.0,
            Ternary::Unknown => 0.5,
        }
    }

    /// Convert from a numeric value back to ternary.
    pub fn from_f64(v: f64) -> Self {
        if v < 0.25 {
            Ternary::False
        } else if v > 0.75 {
            Ternary::True
        } else {
            Ternary::Unknown
        }
    }

    /// Logical AND for ternary values (Kleene logic).
    pub fn and(&self, other: &Ternary) -> Ternary {
        match (self, other) {
            (Ternary::False, _) | (_, Ternary::False) => Ternary::False,
            (Ternary::True, Ternary::True) => Ternary::True,
            _ => Ternary::Unknown,
        }
    }

    /// Logical OR for ternary values (Kleene logic).
    pub fn or(&self, other: &Ternary) -> Ternary {
        match (self, other) {
            (Ternary::True, _) | (_, Ternary::True) => Ternary::True,
            (Ternary::False, Ternary::False) => Ternary::False,
            _ => Ternary::Unknown,
        }
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ternary::True => write!(f, "T"),
            Ternary::False => write!(f, "F"),
            Ternary::Unknown => write!(f, "U"),
        }
    }
}

/// Sampler configuration for texture reads.
#[derive(Debug, Clone)]
pub struct Sampler {
    /// Interpolation mode.
    pub mode: SampleMode,
    /// Boundary mode for out-of-range coordinates.
    pub boundary: BoundaryMode,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(mode: SampleMode, boundary: BoundaryMode) -> Self {
        Self { mode, boundary }
    }

    /// Create a nearest-neighbor sampler with clamp boundary.
    pub fn nearest_clamp() -> Self {
        Self::new(SampleMode::Nearest, BoundaryMode::Clamp)
    }

    /// Create a linear interpolation sampler with wrap boundary.
    pub fn linear_wrap() -> Self {
        Self::new(SampleMode::Linear, BoundaryMode::Wrap)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::nearest_clamp()
    }
}

/// 1D texture for ternary data with spatially cached access.
#[derive(Debug, Clone)]
pub struct Texture1D {
    /// Data elements.
    data: Vec<Ternary>,
    /// Default sampler.
    default_sampler: Sampler,
}

impl Texture1D {
    /// Create a 1D texture from data.
    pub fn new(data: Vec<Ternary>) -> Self {
        Self {
            data,
            default_sampler: Sampler::default(),
        }
    }

    /// Create a 1D texture filled with a constant value.
    pub fn constant(size: usize, value: Ternary) -> Self {
        Self::new(vec![value; size])
    }

    /// Width of the texture.
    pub fn width(&self) -> usize {
        self.data.len()
    }

    /// Direct read at an integer coordinate with boundary handling.
    pub fn read(&self, x: i64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        let ux = sampler.boundary.apply(x, self.data.len() as u64) as usize;
        self.data[ux]
    }

    /// Sample the texture with the given sampler.
    pub fn sample(&self, x: f64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        match sampler.mode {
            SampleMode::Nearest => {
                let ix = x.round() as i64;
                self.read(ix, sampler)
            }
            SampleMode::Linear => {
                let x0 = x.floor() as i64;
                let x1 = x0 + 1;
                let frac = x - x.floor();
                let v0 = self.read(x0, sampler).to_f64();
                let v1 = self.read(x1, sampler).to_f64();
                Ternary::from_f64(v0 * (1.0 - frac) + v1 * frac)
            }
        }
    }

    /// Sample using the default sampler.
    pub fn sample_default(&self, x: f64) -> Ternary {
        self.sample(x, &self.default_sampler)
    }

    /// Write a value at an integer coordinate.
    pub fn write(&mut self, x: usize, value: Ternary) -> Result<(), TextureError> {
        if x >= self.data.len() {
            return Err(TextureError::OutOfBounds {
                coord: format!("x={}", x),
                size: format!("width={}", self.data.len()),
            });
        }
        self.data[x] = value;
        Ok(())
    }

    /// Get raw data reference.
    pub fn data(&self) -> &[Ternary] {
        &self.data
    }
}

/// 2D texture for ternary data with spatially cached access.
#[derive(Debug, Clone)]
pub struct Texture2D {
    data: Vec<Ternary>,
    width: usize,
    height: usize,
    default_sampler: Sampler,
}

impl Texture2D {
    /// Create a 2D texture from flat data with given dimensions.
    pub fn new(data: Vec<Ternary>, width: usize, height: usize) -> Result<Self, TextureError> {
        if data.len() != width * height {
            return Err(TextureError::SizeMismatch {
                expected: width * height,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            default_sampler: Sampler::default(),
        })
    }

    /// Create a 2D texture filled with a constant value.
    pub fn constant(width: usize, height: usize, value: Ternary) -> Self {
        Self {
            data: vec![value; width * height],
            width,
            height,
            default_sampler: Sampler::default(),
        }
    }

    /// Dimensions of the texture.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn flat_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Direct read at integer coordinates with boundary handling.
    pub fn read(&self, x: i64, y: i64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        let ux = sampler.boundary.apply(x, self.width as u64) as usize;
        let uy = sampler.boundary.apply(y, self.height as u64) as usize;
        self.data[self.flat_index(ux, uy)]
    }

    /// Sample the texture with bilinear interpolation.
    pub fn sample(&self, x: f64, y: f64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        match sampler.mode {
            SampleMode::Nearest => {
                let ix = x.round() as i64;
                let iy = y.round() as i64;
                self.read(ix, iy, sampler)
            }
            SampleMode::Linear => {
                let x0 = x.floor() as i64;
                let y0 = y.floor() as i64;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let fx = x - x.floor();
                let fy = y - y.floor();

                let v00 = self.read(x0, y0, sampler).to_f64();
                let v10 = self.read(x1, y0, sampler).to_f64();
                let v01 = self.read(x0, y1, sampler).to_f64();
                let v11 = self.read(x1, y1, sampler).to_f64();

                let top = v00 * (1.0 - fx) + v10 * fx;
                let bottom = v01 * (1.0 - fx) + v11 * fx;
                Ternary::from_f64(top * (1.0 - fy) + bottom * fy)
            }
        }
    }

    /// Write a value at integer coordinates.
    pub fn write(&mut self, x: usize, y: usize, value: Ternary) -> Result<(), TextureError> {
        if x >= self.width || y >= self.height {
            return Err(TextureError::OutOfBounds {
                coord: format!("x={}, y={}", x, y),
                size: format!("{}x{}", self.width, self.height),
            });
        }
        let idx = self.flat_index(x, y);
        self.data[idx] = value;
        Ok(())
    }

    /// Get raw data reference.
    pub fn data(&self) -> &[Ternary] {
        &self.data
    }
}

/// 3D texture for ternary data with spatially cached access.
#[derive(Debug, Clone)]
pub struct Texture3D {
    data: Vec<Ternary>,
    width: usize,
    height: usize,
    depth: usize,
    default_sampler: Sampler,
}

impl Texture3D {
    /// Create a 3D texture from flat data with given dimensions.
    pub fn new(data: Vec<Ternary>, width: usize, height: usize, depth: usize) -> Result<Self, TextureError> {
        let expected = width * height * depth;
        if data.len() != expected {
            return Err(TextureError::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            depth,
            default_sampler: Sampler::default(),
        })
    }

    /// Create a constant-filled 3D texture.
    pub fn constant(width: usize, height: usize, depth: usize, value: Ternary) -> Self {
        Self {
            data: vec![value; width * height * depth],
            width,
            height,
            depth,
            default_sampler: Sampler::default(),
        }
    }

    /// Dimensions of the texture.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    fn flat_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    /// Direct read at integer coordinates with boundary handling.
    pub fn read(&self, x: i64, y: i64, z: i64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        let ux = sampler.boundary.apply(x, self.width as u64) as usize;
        let uy = sampler.boundary.apply(y, self.height as u64) as usize;
        let uz = sampler.boundary.apply(z, self.depth as u64) as usize;
        self.data[self.flat_index(ux, uy, uz)]
    }

    /// Sample with trilinear interpolation.
    pub fn sample(&self, x: f64, y: f64, z: f64, sampler: &Sampler) -> Ternary {
        if self.data.is_empty() {
            return Ternary::Unknown;
        }
        match sampler.mode {
            SampleMode::Nearest => {
                self.read(x.round() as i64, y.round() as i64, z.round() as i64, sampler)
            }
            SampleMode::Linear => {
                let x0 = x.floor() as i64;
                let y0 = y.floor() as i64;
                let z0 = z.floor() as i64;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let z1 = z0 + 1;
                let fx = x - x.floor();
                let fy = y - y.floor();
                let fz = z - z.floor();

                let v000 = self.read(x0, y0, z0, sampler).to_f64();
                let v100 = self.read(x1, y0, z0, sampler).to_f64();
                let v010 = self.read(x0, y1, z0, sampler).to_f64();
                let v110 = self.read(x1, y1, z0, sampler).to_f64();
                let v001 = self.read(x0, y0, z1, sampler).to_f64();
                let v101 = self.read(x1, y0, z1, sampler).to_f64();
                let v011 = self.read(x0, y1, z1, sampler).to_f64();
                let v111 = self.read(x1, y1, z1, sampler).to_f64();

                let c00 = v000 * (1.0 - fx) + v100 * fx;
                let c10 = v010 * (1.0 - fx) + v110 * fx;
                let c01 = v001 * (1.0 - fx) + v101 * fx;
                let c11 = v011 * (1.0 - fx) + v111 * fx;

                let c0 = c00 * (1.0 - fy) + c10 * fy;
                let c1 = c01 * (1.0 - fy) + c11 * fy;

                Ternary::from_f64(c0 * (1.0 - fz) + c1 * fz)
            }
        }
    }

    /// Write a value at integer coordinates.
    pub fn write(&mut self, x: usize, y: usize, z: usize, value: Ternary) -> Result<(), TextureError> {
        if x >= self.width || y >= self.height || z >= self.depth {
            return Err(TextureError::OutOfBounds {
                coord: format!("x={}, y={}, z={}", x, y, z),
                size: format!("{}x{}x{}", self.width, self.height, self.depth),
            });
        }
        let idx = self.flat_index(x, y, z);
        self.data[idx] = value;
        Ok(())
    }
}

/// Texture errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureError {
    /// Size mismatch when creating texture.
    SizeMismatch { expected: usize, actual: usize },
    /// Coordinate out of bounds.
    OutOfBounds { coord: String, size: String },
}

impl fmt::Display for TextureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureError::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {} elements, got {}", expected, actual)
            }
            TextureError::OutOfBounds { coord, size } => {
                write!(f, "coordinate {} out of bounds for {}", coord, size)
            }
        }
    }
}

impl std::error::Error for TextureError {}

/// Result of spatial coherence analysis.
#[derive(Debug, Clone)]
pub struct CoherenceReport {
    /// Percentage of neighboring pairs that have the same value.
    pub same_neighbor_ratio: f64,
    /// Number of same-value neighbor pairs.
    pub same_pairs: usize,
    /// Total neighbor pairs examined.
    pub total_pairs: usize,
    /// Shannon entropy of the data.
    pub entropy: f64,
}

/// Measure spatial coherence of a 1D texture (how similar neighboring values are).
pub fn coherence_1d(data: &[Ternary]) -> CoherenceReport {
    if data.len() < 2 {
        return CoherenceReport {
            same_neighbor_ratio: 1.0,
            same_pairs: 0,
            total_pairs: 0,
            entropy: 0.0,
        };
    }

    let total_pairs = data.len() - 1;
    let same_pairs = data.windows(2).filter(|w| w[0] == w[1]).count();

    let entropy = compute_entropy(data);

    CoherenceReport {
        same_neighbor_ratio: same_pairs as f64 / total_pairs as f64,
        same_pairs,
        total_pairs,
        entropy,
    }
}

/// Measure spatial coherence of a 2D texture.
pub fn coherence_2d(data: &[Ternary], width: usize, height: usize) -> CoherenceReport {
    if width == 0 || height == 0 {
        return CoherenceReport {
            same_neighbor_ratio: 1.0,
            same_pairs: 0,
            total_pairs: 0,
            entropy: 0.0,
        };
    }

    let mut same_pairs = 0usize;
    let mut total_pairs = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            // Right neighbor
            if x + 1 < width {
                total_pairs += 1;
                if data[idx] == data[idx + 1] {
                    same_pairs += 1;
                }
            }
            // Bottom neighbor
            if y + 1 < height {
                total_pairs += 1;
                if data[idx] == data[(y + 1) * width + x] {
                    same_pairs += 1;
                }
            }
        }
    }

    let entropy = compute_entropy(data);

    CoherenceReport {
        same_neighbor_ratio: if total_pairs > 0 {
            same_pairs as f64 / total_pairs as f64
        } else {
            1.0
        },
        same_pairs,
        total_pairs,
        entropy,
    }
}

/// Compute Shannon entropy of ternary data.
fn compute_entropy(data: &[Ternary]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let total = data.len() as f64;
    let true_count = data.iter().filter(|v| **v == Ternary::True).count() as f64;
    let false_count = data.iter().filter(|v| **v == Ternary::False).count() as f64;
    let unknown_count = data.iter().filter(|v| **v == Ternary::Unknown).count() as f64;

    let mut entropy = 0.0;
    for count in [true_count, false_count, unknown_count] {
        if count > 0.0 {
            let p = count / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    // === 1D Addressing Tests ===

    #[test]
    fn test_texture_1d_basic() {
        let tex = Texture1D::new(vec![Ternary::True, Ternary::False, Ternary::Unknown, Ternary::True]);
        assert_eq!(tex.width(), 4);

        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(0, &sampler), Ternary::True);
        assert_eq!(tex.read(1, &sampler), Ternary::False);
        assert_eq!(tex.read(2, &sampler), Ternary::Unknown);
        assert_eq!(tex.read(3, &sampler), Ternary::True);
    }

    #[test]
    fn test_texture_1d_write() {
        let mut tex = Texture1D::new(vec![Ternary::False; 4]);
        tex.write(2, Ternary::True).unwrap();
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(2, &sampler), Ternary::True);
        assert!(tex.write(10, Ternary::True).is_err());
    }

    #[test]
    fn test_texture_1d_constant() {
        let tex = Texture1D::constant(10, Ternary::Unknown);
        assert_eq!(tex.width(), 10);
        let sampler = Sampler::nearest_clamp();
        for i in 0..10 {
            assert_eq!(tex.read(i as i64, &sampler), Ternary::Unknown);
        }
    }

    // === 2D Addressing Tests ===

    #[test]
    fn test_texture_2d_basic() {
        let data = vec![
            Ternary::True, Ternary::False,
            Ternary::Unknown, Ternary::True,
        ];
        let tex = Texture2D::new(data, 2, 2).unwrap();
        assert_eq!(tex.dimensions(), (2, 2));

        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(0, 0, &sampler), Ternary::True);
        assert_eq!(tex.read(1, 0, &sampler), Ternary::False);
        assert_eq!(tex.read(0, 1, &sampler), Ternary::Unknown);
        assert_eq!(tex.read(1, 1, &sampler), Ternary::True);
    }

    #[test]
    fn test_texture_2d_size_mismatch() {
        let data = vec![Ternary::True; 3];
        let result = Texture2D::new(data, 2, 2);
        assert!(matches!(result, Err(TextureError::SizeMismatch { .. })));
    }

    #[test]
    fn test_texture_2d_write() {
        let mut tex = Texture2D::constant(3, 3, Ternary::False);
        tex.write(1, 2, Ternary::True).unwrap();
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(1, 2, &sampler), Ternary::True);
        assert!(tex.write(5, 0, Ternary::True).is_err());
    }

    // === Nearest Sampling Tests ===

    #[test]
    fn test_nearest_sampling_1d() {
        let tex = Texture1D::new(vec![Ternary::True, Ternary::False, Ternary::Unknown]);
        let sampler = Sampler::nearest_clamp();
        // 0.4 rounds to 0 → True
        assert_eq!(tex.sample(0.4, &sampler), Ternary::True);
        // 0.6 rounds to 1 → False
        assert_eq!(tex.sample(0.6, &sampler), Ternary::False);
        // 2.4 rounds to 2 → Unknown
        assert_eq!(tex.sample(2.4, &sampler), Ternary::Unknown);
    }

    #[test]
    fn test_nearest_sampling_2d() {
        let data = vec![
            Ternary::True, Ternary::False,
            Ternary::Unknown, Ternary::True,
        ];
        let tex = Texture2D::new(data, 2, 2).unwrap();
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.sample(0.3, 0.3, &sampler), Ternary::True);
        assert_eq!(tex.sample(1.0, 1.0, &sampler), Ternary::True);
    }

    #[test]
    fn test_linear_sampling_1d() {
        let tex = Texture1D::new(vec![Ternary::True, Ternary::False]); // 1.0 and 0.0
        let sampler = Sampler::new(SampleMode::Linear, BoundaryMode::Clamp);
        // At 0.5: lerp(1.0, 0.0, 0.5) = 0.5 → Unknown
        assert_eq!(tex.sample(0.5, &sampler), Ternary::Unknown);
        // At 0.0: lerp(1.0, 0.0, 0.0) = 1.0 → True
        assert_eq!(tex.sample(0.0, &sampler), Ternary::True);
    }

    // === Boundary Handling Tests ===

    #[test]
    fn test_boundary_clamp() {
        let mode = BoundaryMode::Clamp;
        assert_eq!(mode.apply(-1, 4), 0);
        assert_eq!(mode.apply(0, 4), 0);
        assert_eq!(mode.apply(3, 4), 3);
        assert_eq!(mode.apply(5, 4), 3);
    }

    #[test]
    fn test_boundary_wrap() {
        let mode = BoundaryMode::Wrap;
        assert_eq!(mode.apply(0, 4), 0);
        assert_eq!(mode.apply(4, 4), 0);
        assert_eq!(mode.apply(5, 4), 1);
        assert_eq!(mode.apply(-1, 4), 3);
        assert_eq!(mode.apply(-2, 4), 2);
    }

    #[test]
    fn test_boundary_mirror() {
        let mode = BoundaryMode::Mirror;
        assert_eq!(mode.apply(0, 4), 0);
        assert_eq!(mode.apply(4, 4), 3); // mirror: 4→3
        assert_eq!(mode.apply(5, 4), 2); // mirror: 5→2
        assert_eq!(mode.apply(-1, 4), 0); // mirror: -1→0
        assert_eq!(mode.apply(7, 4), 0); // 7 → period=8, 7 mod 8 = 7, 7>=4 → 2*4-1-7=0
    }

    #[test]
    fn test_clamp_with_texture() {
        let tex = Texture1D::new(vec![Ternary::True, Ternary::False, Ternary::Unknown]);
        let sampler = Sampler::new(SampleMode::Nearest, BoundaryMode::Clamp);
        assert_eq!(tex.read(-5, &sampler), Ternary::True);  // clamps to 0
        assert_eq!(tex.read(100, &sampler), Ternary::Unknown); // clamps to 2
    }

    #[test]
    fn test_wrap_with_texture() {
        let tex = Texture1D::new(vec![Ternary::True, Ternary::False, Ternary::Unknown]);
        let sampler = Sampler::new(SampleMode::Nearest, BoundaryMode::Wrap);
        assert_eq!(tex.read(3, &sampler), Ternary::True);    // wraps to 0
        assert_eq!(tex.read(-1, &sampler), Ternary::Unknown); // wraps to 2
    }

    // === 3D Texture Tests ===

    #[test]
    fn test_texture_3d_basic() {
        let data = vec![Ternary::True; 8];
        let tex = Texture3D::new(data, 2, 2, 2).unwrap();
        assert_eq!(tex.dimensions(), (2, 2, 2));
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(0, 0, 0, &sampler), Ternary::True);
        assert_eq!(tex.read(1, 1, 1, &sampler), Ternary::True);
    }

    #[test]
    fn test_texture_3d_write() {
        let mut tex = Texture3D::constant(2, 2, 2, Ternary::False);
        tex.write(1, 1, 1, Ternary::True).unwrap();
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(1, 1, 1, &sampler), Ternary::True);
    }

    #[test]
    fn test_texture_3d_size_mismatch() {
        let data = vec![Ternary::True; 7];
        let result = Texture3D::new(data, 2, 2, 2);
        assert!(matches!(result, Err(TextureError::SizeMismatch { .. })));
    }

    // === Spatial Coherence Tests ===

    #[test]
    fn test_coherence_1d_perfect() {
        let data = vec![Ternary::True; 10];
        let report = coherence_1d(&data);
        assert_eq!(report.total_pairs, 9);
        assert_eq!(report.same_pairs, 9);
        assert!((report.same_neighbor_ratio - 1.0).abs() < 1e-9);
        assert!(report.entropy.abs() < 1e-9); // All same → 0 entropy
    }

    #[test]
    fn test_coherence_1d_alternating() {
        let data = vec![Ternary::True, Ternary::False, Ternary::True, Ternary::False];
        let report = coherence_1d(&data);
        assert_eq!(report.total_pairs, 3);
        assert_eq!(report.same_pairs, 0);
        assert!((report.same_neighbor_ratio - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_coherence_1d_mixed() {
        let data = vec![Ternary::True, Ternary::True, Ternary::False, Ternary::False];
        let report = coherence_1d(&data);
        assert_eq!(report.total_pairs, 3);
        assert_eq!(report.same_pairs, 2);
        assert!((report.same_neighbor_ratio - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_coherence_1d_single() {
        let report = coherence_1d(&[Ternary::True]);
        assert_eq!(report.total_pairs, 0);
        assert_eq!(report.same_neighbor_ratio, 1.0);
    }

    #[test]
    fn test_coherence_2d_perfect() {
        let data = vec![Ternary::True; 16];
        let report = coherence_2d(&data, 4, 4);
        assert!(report.same_neighbor_ratio > 0.99);
    }

    #[test]
    fn test_coherence_2d_checkerboard() {
        let data: Vec<Ternary> = (0..16)
            .map(|i| if (i % 2 + (i / 4) % 2) % 2 == 0 { Ternary::True } else { Ternary::False })
            .collect();
        let report = coherence_2d(&data, 4, 4);
        // Checkerboard has very low coherence
        assert!(report.same_neighbor_ratio < 0.5);
    }

    #[test]
    fn test_entropy_all_same() {
        let data = vec![Ternary::True; 100];
        let entropy = compute_entropy(&data);
        assert!(entropy.abs() < 1e-9);
    }

    #[test]
    fn test_entropy_uniform() {
        // Equal thirds of each value
        let mut data = Vec::new();
        data.extend(std::iter::repeat(Ternary::True).take(33));
        data.extend(std::iter::repeat(Ternary::False).take(33));
        data.extend(std::iter::repeat(Ternary::Unknown).take(34));
        let entropy = compute_entropy(&data);
        // log2(3) ≈ 1.585
        assert!((entropy - 3.0f64.log2()).abs() < 0.01);
    }

    #[test]
    fn test_ternary_and() {
        assert_eq!(Ternary::True.and(&Ternary::True), Ternary::True);
        assert_eq!(Ternary::True.and(&Ternary::False), Ternary::False);
        assert_eq!(Ternary::False.and(&Ternary::True), Ternary::False);
        assert_eq!(Ternary::False.and(&Ternary::Unknown), Ternary::False);
        assert_eq!(Ternary::Unknown.and(&Ternary::True), Ternary::Unknown);
        assert_eq!(Ternary::Unknown.and(&Ternary::Unknown), Ternary::Unknown);
    }

    #[test]
    fn test_ternary_or() {
        assert_eq!(Ternary::True.or(&Ternary::False), Ternary::True);
        assert_eq!(Ternary::False.or(&Ternary::False), Ternary::False);
        assert_eq!(Ternary::Unknown.or(&Ternary::False), Ternary::Unknown);
        assert_eq!(Ternary::Unknown.or(&Ternary::True), Ternary::True);
    }

    #[test]
    fn test_ternary_roundtrip() {
        for val in [Ternary::True, Ternary::False, Ternary::Unknown] {
            let rt = Ternary::from_f64(val.to_f64());
            assert_eq!(val, rt, "roundtrip failed for {:?}", val);
        }
    }

    #[test]
    fn test_empty_texture_1d() {
        let tex = Texture1D::new(vec![]);
        let sampler = Sampler::nearest_clamp();
        assert_eq!(tex.read(0, &sampler), Ternary::Unknown);
        assert_eq!(tex.sample(0.0, &sampler), Ternary::Unknown);
    }

    #[test]
    fn test_error_display() {
        let err = TextureError::SizeMismatch { expected: 4, actual: 3 };
        assert!(format!("{}", err).contains("expected 4"));
    }

    #[test]
    fn test_sampler_default() {
        let s = Sampler::default();
        assert_eq!(s.mode, SampleMode::Nearest);
        assert_eq!(s.boundary, BoundaryMode::Clamp);
    }

    #[test]
    fn test_ternary_display() {
        assert_eq!(format!("{}", Ternary::True), "T");
        assert_eq!(format!("{}", Ternary::False), "F");
        assert_eq!(format!("{}", Ternary::Unknown), "U");
    }
}
