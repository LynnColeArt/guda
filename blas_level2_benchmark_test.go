package guda

import (
	"math/rand"
	"testing"
	"time"

	"github.com/LynnColeArt/guda/compute"
	"github.com/LynnColeArt/guda/compute/blas"
)

// BenchmarkBLASLevel2 comprehensively benchmarks all BLAS Level 2 operations
// with different matrix sizes to identify optimization opportunities
func BenchmarkBLASLevel2(b *testing.B) {
	sizes := []struct {
		name string
		m, n int
	}{
		{"Small_32x32", 32, 32},
		{"Medium_128x128", 128, 128},
		{"Large_512x512", 512, 512},
		{"Transformer_768x768", 768, 768},   // Common transformer size
		{"Wide_64x1024", 64, 1024},          // Wide matrices common in ML
		{"Tall_1024x64", 1024, 64},          // Tall matrices
	}

	for _, sz := range sizes {
		b.Run("GEMV_"+sz.name, func(b *testing.B) {
			benchmarkGEMV(b, sz.m, sz.n)
		})
		b.Run("GER_"+sz.name, func(b *testing.B) {
			benchmarkGER(b, sz.m, sz.n)
		})
		b.Run("SYMV_"+sz.name, func(b *testing.B) {
			benchmarkSYMV(b, sz.n)
		})
	}
}

// benchmarkGEMV tests matrix-vector multiplication performance
func benchmarkGEMV(b *testing.B, m, n int) {
	// Initialize test data
	rng := rand.New(rand.NewSource(42))
	a := make([]float32, m*n)
	x := make([]float32, n)
	y := make([]float32, m)
	
	for i := range a {
		a[i] = rng.Float32()
	}
	for i := range x {
		x[i] = rng.Float32()
	}
	for i := range y {
		y[i] = rng.Float32()
	}
	
	alpha := float32(1.5)
	beta := float32(0.5)
	
	// Create BLAS implementation
	impl := compute.Implementation{}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	// Track operations for performance metrics
	ops := int64(2 * m * n) // multiply-add operations
	
	for i := 0; i < b.N; i++ {
		// y = alpha * A * x + beta * y
		impl.Sgemv(blas.NoTrans, m, n, alpha, a, n, x, 1, beta, y, 1)
	}
	
	// Calculate GFLOPS
	elapsed := b.Elapsed()
	if elapsed > 0 {
		gflops := float64(ops*int64(b.N)) / elapsed.Seconds() / 1e9
		b.ReportMetric(gflops, "GFLOPS")
	}
}

// benchmarkGER tests rank-1 update performance
func benchmarkGER(b *testing.B, m, n int) {
	rng := rand.New(rand.NewSource(42))
	a := make([]float32, m*n)
	x := make([]float32, m)
	y := make([]float32, n)
	
	for i := range a {
		a[i] = rng.Float32()
	}
	for i := range x {
		x[i] = rng.Float32()
	}
	for i := range y {
		y[i] = rng.Float32()
	}
	
	alpha := float32(1.2)
	impl := compute.Implementation{}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	ops := int64(2 * m * n) // multiply-add operations
	
	for i := 0; i < b.N; i++ {
		// A += alpha * x * y^T
		impl.Sger(m, n, alpha, x, 1, y, 1, a, n)
	}
	
	elapsed := b.Elapsed()
	if elapsed > 0 {
		gflops := float64(ops*int64(b.N)) / elapsed.Seconds() / 1e9
		b.ReportMetric(gflops, "GFLOPS")
	}
}

// benchmarkSYMV tests symmetric matrix-vector multiplication
func benchmarkSYMV(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(42))
	a := make([]float32, n*n)
	x := make([]float32, n)
	y := make([]float32, n)
	
	// Create symmetric matrix
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := rng.Float32()
			a[i*n+j] = val
			a[j*n+i] = val // Make symmetric
		}
	}
	for i := range x {
		x[i] = rng.Float32()
	}
	for i := range y {
		y[i] = rng.Float32()
	}
	
	alpha := float32(1.3)
	beta := float32(0.7)
	impl := compute.Implementation{}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	ops := int64(2 * n * n) // Can be optimized to n*(n+1) for symmetric
	
	for i := 0; i < b.N; i++ {
		// y = alpha * A * x + beta * y (A is symmetric)
		impl.Ssymv(blas.Upper, n, alpha, a, n, x, 1, beta, y, 1)
	}
	
	elapsed := b.Elapsed()
	if elapsed > 0 {
		gflops := float64(ops*int64(b.N)) / elapsed.Seconds() / 1e9
		b.ReportMetric(gflops, "GFLOPS")
	}
}

// TestBLASLevel2Correctness ensures all operations produce correct results
func TestBLASLevel2Correctness(t *testing.T) {
	impl := compute.Implementation{}
	
	t.Run("GEMV_Correctness", func(t *testing.T) {
		testGEMVCorrectness(t, impl)
	})
	
	t.Run("GER_Correctness", func(t *testing.T) {
		testGERCorrectness(t, impl)
	})
	
	t.Run("SYMV_Correctness", func(t *testing.T) {
		testSYMVCorrectness(t, impl)
	})
}

func testGEMVCorrectness(t *testing.T, impl compute.Implementation) {
	// Simple 3x3 test case
	m, n := 3, 3
	a := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	x := []float32{1, 2, 3}
	y := []float32{1, 1, 1}
	alpha := float32(1.0)
	beta := float32(0.0)
	
	// Expected: [14, 32, 50] (A * x)
	expected := []float32{14, 32, 50}
	
	impl.Sgemv(blas.NoTrans, m, n, alpha, a, n, x, 1, beta, y, 1)
	
	for i := 0; i < m; i++ {
		if absFloat32(y[i]-expected[i]) > 1e-5 {
			t.Errorf("GEMV result[%d]: expected %f, got %f", i, expected[i], y[i])
		}
	}
}

func testGERCorrectness(t *testing.T, impl compute.Implementation) {
	// Simple 2x2 test case
	m, n := 2, 2
	a := []float32{1, 2, 3, 4}
	x := []float32{1, 2}
	y := []float32{1, 1}
	alpha := float32(1.0)
	
	// A += alpha * x * y^T
	// A += [1, 1; 2, 2] = [2, 3; 5, 6]
	expected := []float32{2, 3, 5, 6}
	
	impl.Sger(m, n, alpha, x, 1, y, 1, a, n)
	
	for i := 0; i < m*n; i++ {
		if absFloat32(a[i]-expected[i]) > 1e-5 {
			t.Errorf("GER result[%d]: expected %f, got %f", i, expected[i], a[i])
		}
	}
}

func testSYMVCorrectness(t *testing.T, impl compute.Implementation) {
	// Simple 2x2 symmetric test case
	n := 2
	a := []float32{
		1, 2,
		2, 3,
	}
	x := []float32{1, 2}
	y := []float32{0, 0}
	alpha := float32(1.0)
	beta := float32(0.0)
	
	// Expected: [5, 8] (symmetric A * x)
	expected := []float32{5, 8}
	
	impl.Ssymv(blas.Upper, n, alpha, a, n, x, 1, beta, y, 1)
	
	for i := 0; i < n; i++ {
		if absFloat32(y[i]-expected[i]) > 1e-5 {
			t.Errorf("SYMV result[%d]: expected %f, got %f", i, expected[i], y[i])
		}
	}
}

// Helper function for floating point comparison
func absFloat32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// BenchmarkCompareBLASLevel2VsNaive compares optimized vs naive implementations
func BenchmarkCompareBLASLevel2VsNaive(b *testing.B) {
	m, n := 256, 256
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	
	// Initialize test data
	a := make([]float32, m*n)
	x := make([]float32, n)
	y1 := make([]float32, m) // For optimized
	y2 := make([]float32, m) // For naive
	
	for i := range a {
		a[i] = rng.Float32()
	}
	for i := range x {
		x[i] = rng.Float32()
	}
	
	alpha := float32(1.0)
	beta := float32(0.0)
	impl := compute.Implementation{}
	
	b.Run("Optimized_GEMV", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(y1, y2) // Reset
			impl.Sgemv(blas.NoTrans, m, n, alpha, a, n, x, 1, beta, y1, 1)
		}
	})
	
	b.Run("Naive_GEMV", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(y2, y1) // Reset
			naiveGEMV(m, n, alpha, a, n, x, beta, y2)
		}
	})
}

// naiveGEMV provides a simple reference implementation
func naiveGEMV(m, n int, alpha float32, a []float32, lda int, x []float32, beta float32, y []float32) {
	for i := 0; i < m; i++ {
		sum := float32(0)
		for j := 0; j < n; j++ {
			sum += a[i*lda+j] * x[j]
		}
		y[i] = beta*y[i] + alpha*sum
	}
}