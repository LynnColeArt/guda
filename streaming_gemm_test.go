package guda

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestStreamingGEMM tests correctness of the streaming implementation
func TestStreamingGEMM(t *testing.T) {
	sizes := []struct {
		m, n, k int
	}{
		{16, 16, 16},
		{64, 64, 64},
		{127, 129, 128},
		{512, 512, 512},
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

			// Compute with streaming GEMM
			StreamingGEMM_Float32(false, false, size.m, size.n, size.k, alpha, a, size.k, b, size.n, beta, c, size.n)

			// Compute reference
			ref := Reference{}
			ref.GEMM(false, false, size.m, size.n, size.k, alpha, a, size.k, b, size.n, beta, cRef, size.n)

			// Compare
			maxDiff := float32(0.0)
			for i := range c {
				diff := absFloat32(c[i] - cRef[i])
				if diff > maxDiff {
					maxDiff = diff
				}
			}

			tolerance := float32(1e-5) * float32(size.k)
			if maxDiff > tolerance {
				t.Errorf("Streaming GEMM differs from reference: max diff %e > tolerance %e", maxDiff, tolerance)
			}
		})
	}
}

// BenchmarkStreamingGEMM compares streaming with standard GEMM
func BenchmarkStreamingGEMM(b *testing.B) {
	sizes := []int{256, 512, 1024, 2048}

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

		// Streaming GEMM
		b.Run(fmt.Sprintf("Streaming_%d", size), func(b *testing.B) {
			a := make([]float32, size*size)
			bMat := make([]float32, size*size)
			c := make([]float32, size*size)

			for i := range a {
				a[i] = 1.0
			}
			for i := range bMat {
				bMat[i] = 1.0
			}

			sg := NewStreamingGEMM()

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sg.Compute(1.0, a, size, size, size, bMat, size, size, size, 0.0, c, size)
			}

			reportGFLOPS(b, size, size, size)
		})
	}
}

// BenchmarkStreamingMemoryBandwidth measures memory bandwidth utilization
func BenchmarkStreamingMemoryBandwidth(b *testing.B) {
	size := 1024
	
	// Theoretical memory traffic for GEMM
	// Read: A (m×k) + B (k×n) 
	// Write: C (m×n)
	// Total: (m×k + k×n + m×n) × 4 bytes
	
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
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			GEMM(false, false, size, size, size, 1.0, da, size, db, size, 0.0, dc, size)
			Synchronize()
		}
		
		seconds := b.Elapsed().Seconds() / float64(b.N)
		memoryBytes := int64(3 * size * size * 4) // Theoretical minimum
		bandwidth := float64(memoryBytes) / (seconds * 1e9)
		b.ReportMetric(bandwidth, "GB/s_theoretical")
		
		// Actual memory traffic is higher due to cache misses
		// Estimate ~3x for poor locality
		actualBandwidth := bandwidth * 3
		b.ReportMetric(actualBandwidth, "GB/s_estimated")
	})
	
	b.Run("Streaming", func(b *testing.B) {
		a := make([]float32, size*size)
		bMat := make([]float32, size*size)
		c := make([]float32, size*size)

		for i := range a {
			a[i] = rand.Float32()
		}
		for i := range bMat {
			bMat[i] = rand.Float32()
		}

		sg := NewStreamingGEMM()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sg.Compute(1.0, a, size, size, size, bMat, size, size, size, 0.0, c, size)
		}
		
		seconds := b.Elapsed().Seconds() / float64(b.N)
		memoryBytes := int64(3 * size * size * 4) // Theoretical minimum
		bandwidth := float64(memoryBytes) / (seconds * 1e9)
		b.ReportMetric(bandwidth, "GB/s_theoretical")
		
		// Better locality should reduce actual traffic
		// Estimate ~1.5x for good tiling
		actualBandwidth := bandwidth * 1.5
		b.ReportMetric(actualBandwidth, "GB/s_estimated")
	})
}