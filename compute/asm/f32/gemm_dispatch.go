//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	// No imports needed for now
)

// HasAVX512Support checks if the CPU supports AVX-512
var HasAVX512Support bool

// HasAVX2Support checks if the CPU supports AVX2
var HasAVX2Support bool

// CPU feature detection is done at package init time
func init() {
	// This will be populated by the main guda package
	// We'll set up the proper linkage later
}

// GemmKernel represents a generic GEMM kernel interface
type GemmKernel func(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int)

// SelectGemmKernel returns the best available GEMM kernel for the current CPU
func SelectGemmKernel() GemmKernel {
	if HasAVX512Support {
		return gemmAVX512Wrapper
	}
	if HasAVX2Support {
		return gemmAVX2Wrapper
	}
	// Fall back to scalar implementation
	return gemmScalarWrapper
}

// gemmAVX512Wrapper wraps the AVX-512 implementation for the generic interface
func gemmAVX512Wrapper(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	// For now, only handle non-transposed case
	// The AVX-512 kernel expects pre-packed data, so we use the high-level function
	GemmAVX512(false, false, m, n, k, alpha, a, lda, b, ldb, 1.0, c, ldc)
}

// gemmAVX2Wrapper wraps the AVX2 implementation (existing gonum kernels)
func gemmAVX2Wrapper(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	// Use the existing optimized scalar loop for now
	// TODO: Integrate menthar's AVX2 kernels
	gemmScalarWrapper(m, n, k, alpha, a, lda, b, ldb, c, ldc)
}

// gemmScalarWrapper provides the scalar fallback
func gemmScalarWrapper(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	// Direct scalar implementation without overhead
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0.0)
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
}

// Optimized kernel for no-transpose case
func GemmOptimizedNotNot(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	kernel := SelectGemmKernel()
	kernel(m, n, k, alpha, a, lda, b, ldb, c, ldc)
}

// SetCPUFeatures allows the main package to inform us about CPU capabilities
func SetCPUFeatures(hasAVX512, hasAVX2 bool) {
	HasAVX512Support = hasAVX512
	HasAVX2Support = hasAVX2
}