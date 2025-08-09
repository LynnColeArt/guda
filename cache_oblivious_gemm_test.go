package guda

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestCacheObliviousGEMM tests correctness of the cache-oblivious implementation
func TestCacheObliviousGEMM(t *testing.T) {
	sizes := []struct {
		m, n, k int
	}{
		{16, 16, 16},
		{64, 64, 64},
		{127, 129, 128}, // Non-power-of-2
		{256, 256, 256},
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%dx%d", size.m, size.n, size.k), func(t *testing.T) {
			// Create test matrices
			a := make([]float32, size.m*size.k)
			b := make([]float32, size.k*size.n)
			c := make([]float32, size.m*size.n)
			cRef := make([]float32, size.m*size.n)

			// Initialize with random data
			for i := range a {
				a[i] = rand.Float32()
			}
			for i := range b {
				b[i] = rand.Float32()
			}

			alpha := float32(1.5)
			beta := float32(0.5)

			// Initialize C
			for i := range c {
				c[i] = rand.Float32()
				cRef[i] = c[i]
			}

			// Compute with cache-oblivious
			cog := NewCacheObliviousGEMM()
			cog.Compute(alpha, a, size.k, size.m, size.k, b, size.n, size.k, size.n, beta, c, size.n)

			// Compute reference
			ref := Reference{}
			ref.GEMM(false, false, size.m, size.n, size.k, alpha, a, size.k, b, size.n, beta, cRef, size.n)

			// Compare
			maxDiff := float32(0.0)
			for i := 0; i < size.m; i++ {
				for j := 0; j < size.n; j++ {
					idx := i*size.n + j
					diff := absFloat32(c[idx] - cRef[idx])
					if diff > maxDiff {
						maxDiff = diff
					}
				}
			}

			tolerance := float32(1e-5) * float32(size.k)
			if maxDiff > tolerance {
				t.Errorf("Cache-oblivious GEMM differs from reference: max diff %e > tolerance %e", maxDiff, tolerance)
			}
		})
	}
}

// TestCacheObliviousParallel tests the parallel implementation
func TestCacheObliviousParallel(t *testing.T) {
	m, n, k := 512, 512, 512

	// Create test matrices
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	cRef := make([]float32, m*n)

	// Initialize
	for i := range a {
		a[i] = rand.Float32()
	}
	for i := range b {
		b[i] = rand.Float32()
	}

	alpha := float32(1.0)
	beta := float32(0.0)

	// Compute with parallel cache-oblivious
	cogp := NewCacheObliviousGEMMParallel()
	cogp.Compute(alpha, a, k, m, k, b, n, k, n, beta, c, n)

	// Compute reference
	ref := Reference{}
	ref.GEMM(false, false, m, n, k, alpha, a, k, b, n, beta, cRef, n)

	// Compare
	maxDiff := float32(0.0)
	for i := range c {
		diff := absFloat32(c[i] - cRef[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	tolerance := float32(1e-4) * float32(k)
	if maxDiff > tolerance {
		t.Errorf("Parallel cache-oblivious differs from reference: max diff %e > tolerance %e", maxDiff, tolerance)
	}
}

// BenchmarkCacheOblivious compares cache-oblivious with standard GEMM
func BenchmarkCacheOblivious(b *testing.B) {
	sizes := []int{128, 256, 512, 1024}

	for _, size := range sizes {
		// Standard GEMM
		b.Run(fmt.Sprintf("Standard_%d", size), func(b *testing.B) {
			da, _ := Malloc(size * size * 4)
			db, _ := Malloc(size * size * 4)
			dc, _ := Malloc(size * size * 4)
			defer Free(da)
			defer Free(db)
			defer Free(dc)

			// Initialize
			a := da.Float32()
			bMat := db.Float32()
			for i := range a {
				a[i] = 1.0
			}
			for i := range bMat {
				bMat[i] = 1.0
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMM(false, false, size, size, size, 1.0, da, size, db, size, 0.0, dc, size)
				Synchronize()
			}

			reportGFLOPS(b, size, size, size)
		})

		// Cache-oblivious serial
		b.Run(fmt.Sprintf("CacheOblivious_%d", size), func(b *testing.B) {
			a := make([]float32, size*size)
			bMat := make([]float32, size*size)
			c := make([]float32, size*size)

			for i := range a {
				a[i] = 1.0
			}
			for i := range bMat {
				bMat[i] = 1.0
			}

			cog := NewCacheObliviousGEMM()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cog.Compute(1.0, a, size, size, size, bMat, size, size, size, 0.0, c, size)
			}

			reportGFLOPS(b, size, size, size)
		})

		// Cache-oblivious parallel
		b.Run(fmt.Sprintf("CacheObliviousParallel_%d", size), func(b *testing.B) {
			a := make([]float32, size*size)
			bMat := make([]float32, size*size)
			c := make([]float32, size*size)

			for i := range a {
				a[i] = 1.0
			}
			for i := range bMat {
				bMat[i] = 1.0
			}

			cogp := NewCacheObliviousGEMMParallel()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cogp.Compute(1.0, a, size, size, size, bMat, size, size, size, 0.0, c, size)
			}

			reportGFLOPS(b, size, size, size)
		})
	}
}

// BenchmarkCacheEfficiency measures cache misses (when perf counters available)
func BenchmarkCacheEfficiency(b *testing.B) {
	b.Skip("Performance counter benchmarks moved to dedicated test")

	size := 512
	
	b.Run("Standard", func(b *testing.B) {
		da, _ := Malloc(size * size * 4)
		db, _ := Malloc(size * size * 4)
		dc, _ := Malloc(size * size * 4)
		defer Free(da)
		defer Free(db)
		defer Free(dc)

		a := da.Float32()
		bMat := db.Float32()
		for i := range a {
			a[i] = 1.0
		}
		for i := range bMat {
			bMat[i] = 1.0
		}

		b.ResetTimer()
		
		BenchmarkWithCounters(b, "StandardGEMM", func() {
			GEMM(false, false, size, size, size, 1.0, da, size, db, size, 0.0, dc, size)
			Synchronize()
		})
		
		reportGFLOPS(b, size, size, size)
	})

	b.Run("CacheOblivious", func(b *testing.B) {
		a := make([]float32, size*size)
		bMat := make([]float32, size*size)
		c := make([]float32, size*size)

		for i := range a {
			a[i] = 1.0
		}
		for i := range bMat {
			bMat[i] = 1.0
		}

		cog := NewCacheObliviousGEMMParallel()

		b.ResetTimer()
		
		BenchmarkWithCounters(b, "CacheObliviousGEMM", func() {
			cog.Compute(1.0, a, size, size, size, bMat, size, size, size, 0.0, c, size)
		})
		
		reportGFLOPS(b, size, size, size)
	})
}

func reportGFLOPS(b *testing.B, m, n, k int) {
	flops := 2 * int64(m) * int64(n) * int64(k)
	seconds := b.Elapsed().Seconds() / float64(b.N)
	gflops := float64(flops) / (seconds * 1e9)
	b.ReportMetric(gflops, "GFLOPS")

	// Memory traffic
	bytes := int64((m*k + k*n + m*n) * 4) // float32
	bandwidth := float64(bytes) / (seconds * 1e9)
	b.ReportMetric(bandwidth, "GB/s")
}

// Using absFloat32 from blas_level2_benchmark_test.go