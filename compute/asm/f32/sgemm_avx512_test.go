//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"fmt"
	"math"
	"testing"
)

// TestPackAMatrixAVX512 tests the A matrix packing routine
func TestPackAMatrixAVX512(t *testing.T) {
	// Test a simple 4x3 matrix
	m, k := 4, 3
	lda := k
	
	// Input matrix A (row-major)
	// [1 2 3]
	// [4 5 6]
	// [7 8 9]
	// [10 11 12]
	a := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	
	// Pack A (should get 16x3 with padding)
	packedSize := ((m + MR_AVX512 - 1) / MR_AVX512) * MR_AVX512 * k
	aPacked := make([]float32, packedSize)
	PackAMatrixAVX512(aPacked, a, lda, m, k)
	
	// Expected layout: for each k, store 16 contiguous rows (with padding)
	expected := []float32{
		// k=0
		1, 4, 7, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// k=1
		2, 5, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		// k=2
		3, 6, 9, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	}
	
	for i, v := range expected {
		if math.Abs(float64(aPacked[i]-v)) > 1e-6 {
			t.Errorf("PackAMatrixAVX512: index %d, got %f, want %f", i, aPacked[i], v)
		}
	}
}

// TestPackBMatrixAVX512 tests the B matrix packing routine
func TestPackBMatrixAVX512(t *testing.T) {
	// Test a simple 3x4 matrix
	k, n := 3, 4
	ldb := n
	
	// Input matrix B (row-major)
	// [1 2 3 4]
	// [5 6 7 8]
	// [9 10 11 12]
	b := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	
	// Pack B (should get panels of KC x 4)
	packedSize := ((n + NR_AVX512 - 1) / NR_AVX512) * NR_AVX512 * k
	bPacked := make([]float32, packedSize)
	PackBMatrixAVX512(bPacked, b, ldb, k, n)
	
	// Expected layout: for each panel of 4 columns, store KC elements per column
	expected := []float32{
		// Column 0: [1, 5, 9]
		1, 5, 9,
		// Column 1: [2, 6, 10]
		2, 6, 10,
		// Column 2: [3, 7, 11]
		3, 7, 11,
		// Column 3: [4, 8, 12]
		4, 8, 12,
	}
	
	for i, v := range expected {
		if math.Abs(float64(bPacked[i]-v)) > 1e-6 {
			t.Errorf("PackBMatrixAVX512: index %d, got %f, want %f", i, bPacked[i], v)
		}
	}
}

// TestGemmAVX512Small tests the AVX-512 GEMM with a small matrix
func TestGemmAVX512Small(t *testing.T) {
	if !HasAVX512Support {
		t.Skip("AVX-512 not supported on this CPU")
	}
	
	// Test C = A * B where
	// A is 4x3, B is 3x4, C is 4x4
	m, n, k := 4, 4, 3
	
	// A matrix (row-major)
	// [1 2 3]
	// [4 5 6]
	// [7 8 9]
	// [10 11 12]
	a := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	
	// B matrix (row-major)
	// [1 2 3 4]
	// [5 6 7 8]
	// [9 10 11 12]
	b := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	
	// Expected result C = A * B
	// Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
	//      = [38, 44, 50, 56]
	// Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
	//      = [83, 98, 113, 128]
	// Row 2: [128, 152, 176, 200]
	// Row 3: [173, 206, 239, 272]
	expected := []float32{
		38, 44, 50, 56,
		83, 98, 113, 128,
		128, 152, 176, 200,
		173, 206, 239, 272,
	}
	
	// Compute C = 1.0 * A * B + 0.0 * C
	c := make([]float32, m*n)
	GemmAVX512(false, false, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
	
	// Check result
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			if math.Abs(float64(c[idx]-expected[idx])) > 1e-4 {
				t.Errorf("GemmAVX512: C[%d,%d] = %f, want %f", i, j, c[idx], expected[idx])
			}
		}
	}
}

// TestGemmAVX512Large tests the AVX-512 GEMM with larger matrices
func TestGemmAVX512Large(t *testing.T) {
	if !HasAVX512Support {
		t.Skip("AVX-512 not supported on this CPU")
	}
	
	// Test with matrices that exercise the blocking
	m, n, k := 128, 128, 128
	
	// Initialize matrices
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	cRef := make([]float32, m*n)
	
	// Fill with simple patterns
	for i := 0; i < m*k; i++ {
		a[i] = float32(i%7 + 1)
	}
	for i := 0; i < k*n; i++ {
		b[i] = float32(i%5 + 1)
	}
	
	// Compute with AVX-512
	GemmAVX512(false, false, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
	
	// Compute reference
	gemmRef(m, n, k, 1.0, a, k, b, n, 0.0, cRef, n)
	
	// Compare results
	maxDiff := float32(0.0)
	for i := 0; i < m*n; i++ {
		diff := float32(math.Abs(float64(c[i] - cRef[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	
	// Allow for some floating point error accumulation
	tolerance := float32(1e-3)
	if maxDiff > tolerance {
		t.Errorf("GemmAVX512 large: max difference %f exceeds tolerance %f", maxDiff, tolerance)
	}
}

// gemmRef is a reference scalar GEMM implementation
func gemmRef(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	// Handle beta
	if beta == 0 {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] = 0
			}
		}
	} else if beta != 1 {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] *= beta
			}
		}
	}
	
	// Compute C += alpha * A * B
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

// BenchmarkGemmAVX512 benchmarks the AVX-512 implementation
func BenchmarkGemmAVX512(b *testing.B) {
	if !HasAVX512Support {
		b.Skip("AVX-512 not supported on this CPU")
	}
	
	sizes := []int{128, 256, 512, 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			m, n, k := size, size, size
			
			// Allocate matrices
			a := make([]float32, m*k)
			bMat := make([]float32, k*n)
			c := make([]float32, m*n)
			
			// Initialize
			for i := range a {
				a[i] = 1.0
			}
			for i := range bMat {
				bMat[i] = 1.0
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GemmAVX512(false, false, m, n, k, 1.0, a, k, bMat, n, 0.0, c, n)
			}
			
			// Report GFLOPS
			flops := 2 * int64(m) * int64(n) * int64(k)
			b.SetBytes(int64(m*k+k*n+m*n) * 4) // float32 = 4 bytes
			
			seconds := b.Elapsed().Seconds() / float64(b.N)
			gflops := float64(flops) / (seconds * 1e9)
			b.ReportMetric(gflops, "GFLOPS")
		})
	}
}