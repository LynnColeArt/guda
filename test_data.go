package guda

import (
	"math"
)

// GenerateFloat32 generates deterministic float32 test data using a linear
// congruential generator (LCG). This ensures reproducible tests across runs.
//
// Parameters:
//   - size: Number of elements to generate
//   - seed: Random seed for reproducibility
//
// Example:
//   data := GenerateFloat32(1024, 12345)
func GenerateFloat32(size int, seed uint64) []float32 {
	data := make([]float32, size)
	rng := seed
	for i := range data {
		rng = rng*1103515245 + 12345 // LCG parameters from Numerical Recipes
		data[i] = float32(rng) / float32(1<<32) // Normalize to [0, 1)
	}
	return data
}

// GenerateFloat32Range generates deterministic float32 data in a specific range.
//
// Parameters:
//   - size: Number of elements
//   - seed: Random seed
//   - min: Minimum value (inclusive)
//   - max: Maximum value (exclusive)
//
// Example:
//   data := GenerateFloat32Range(1024, 42, -1.0, 1.0) // Generate values in [-1, 1)
func GenerateFloat32Range(size int, seed uint64, min, max float32) []float32 {
	data := GenerateFloat32(size, seed)
	scale := max - min
	for i := range data {
		data[i] = data[i]*scale + min
	}
	return data
}

// GenerateFloat32EdgeCases generates test data with edge cases for floating point.
// Includes zero, denormals, infinity, NaN, and extreme values.
// Useful for testing numerical stability and special case handling.
func GenerateFloat32EdgeCases() []float32 {
	return []float32{
		0.0,
		-0.0,
		1.0,
		-1.0,
		math.SmallestNonzeroFloat32,
		-math.SmallestNonzeroFloat32,
		math.MaxFloat32,
		-math.MaxFloat32,
		float32(math.Inf(1)),
		float32(math.Inf(-1)),
		float32(math.NaN()),
		1e-38,  // Near denormal
		-1e-38,
		1e38,   // Large but not max
		-1e38,
	}
}

// GenerateMatrixFloat32 generates a deterministic matrix in row-major order.
//
// Parameters:
//   - rows: Number of rows
//   - cols: Number of columns  
//   - seed: Random seed
//
// Example:
//   A := GenerateMatrixFloat32(128, 256, 12345) // 128x256 matrix
func GenerateMatrixFloat32(rows, cols int, seed uint64) []float32 {
	return GenerateFloat32(rows*cols, seed)
}

// GenerateIdentityMatrix generates an identity matrix of the specified size.
// Diagonal elements are 1.0, all others are 0.0.
func GenerateIdentityMatrix(size int) []float32 {
	data := make([]float32, size*size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1.0
	}
	return data
}

// GenerateDiagonalMatrix generates a diagonal matrix with specified diagonal values.
// All non-diagonal elements are 0.0.
func GenerateDiagonalMatrix(diagonal []float32) []float32 {
	size := len(diagonal)
	data := make([]float32, size*size)
	for i := 0; i < size; i++ {
		data[i*size+i] = diagonal[i]
	}
	return data
}

// TestDataSizes returns common test sizes for benchmarking cache effects.
// Includes sizes that fit in L1, L2, L3, and main memory.
func TestDataSizes() []int {
	return []int{
		16,               // Tiny (alignment test)
		256,              // Small (1KB)
		1024,             // 4KB (typical page)
		8 * 1024,         // 32KB (L1 cache size)
		64 * 1024,        // 256KB (L2 cache size)
		256 * 1024,       // 1MB (L3 portion)
		1024 * 1024,      // 4MB (L3 stress)
		16 * 1024 * 1024, // 64MB (RAM)
	}
}

// TestMatrixSizes returns common matrix sizes for GEMM testing.
// Includes square and rectangular matrices of various sizes.
func TestMatrixSizes() [][3]int {
	return [][3]int{
		// {M, N, K}
		{16, 16, 16},       // Tiny
		{64, 64, 64},       // Small
		{128, 128, 128},    // L1-friendly
		{256, 256, 256},    // L2-friendly
		{512, 512, 512},    // Medium
		{1024, 1024, 1024}, // Large
		// Rectangular cases
		{128, 256, 64},     // Wide
		{256, 128, 64},     // Tall
		{1024, 64, 256},    // Skinny
		{64, 1024, 256},    // Fat
		// Edge cases
		{1, 1, 1},          // Scalar
		{1, 1024, 1024},    // Vector-matrix
		{1024, 1, 1024},    // Matrix-vector
	}
}

// GenerateSequence generates a simple arithmetic sequence for debugging.
// Useful when you need predictable patterns.
//
// Example:
//   data := GenerateSequence(10, 0, 2) // [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
func GenerateSequence(size int, start, step float32) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = start + float32(i)*step
	}
	return data
}

// AlmostEqual checks if two float32 values are approximately equal
// within the specified tolerance. Handles special cases like NaN and Inf.
func AlmostEqual(a, b, tolerance float32) bool {
	// Handle NaN
	if math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	// Handle Inf
	if math.IsInf(float64(a), 0) && math.IsInf(float64(b), 0) {
		return math.Signbit(float64(a)) == math.Signbit(float64(b))
	}
	// Regular comparison
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= tolerance
}

// SlicesAlmostEqual checks if two float32 slices are approximately equal
// element-wise within the specified tolerance.
func SlicesAlmostEqual(a, b []float32, tolerance float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !AlmostEqual(a[i], b[i], tolerance) {
			return false
		}
	}
	return true
}