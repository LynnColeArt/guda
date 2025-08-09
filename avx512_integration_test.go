package guda

import (
	"fmt"
	"testing"
)

// TestAVX512Integration tests that AVX-512 GEMM is properly integrated
func TestAVX512Integration(t *testing.T) {
	// Check if AVX-512 is available
	cpuInfo := GetCPUInfo()
	fmt.Printf("CPU Features: %s\n", cpuInfo)
	
	bestImpl := GetBestGemmImplementation()
	fmt.Printf("Best GEMM Implementation: %s\n", bestImpl)
	
	if bestImpl != "AVX512" {
		t.Skip("AVX-512 not available on this CPU")
	}
	
	// Test a small GEMM operation
	m, n, k := 64, 64, 64
	
	// Allocate matrices
	da, _ := Malloc(m * k * 4)
	db, _ := Malloc(k * n * 4)
	dc, _ := Malloc(m * n * 4)
	defer Free(da)
	defer Free(db)
	defer Free(dc)
	
	// Initialize matrices
	a := da.Float32()
	b := db.Float32()
	c := dc.Float32()
	
	// Fill A and B with simple pattern
	for i := 0; i < m*k; i++ {
		a[i] = float32(i%7 + 1)
	}
	for i := 0; i < k*n; i++ {
		b[i] = float32(i%5 + 1)
	}
	
	// Compute C = A * B using GUDA's GEMM
	fmt.Printf("\nCalling GEMM with m=%d, n=%d, k=%d\n", m, n, k)
	err := GEMM(false, false, m, n, k, 1.0, da, k, db, n, 0.0, dc, n)
	if err != nil {
		t.Fatalf("GEMM failed: %v", err)
	}
	
	// Verify using reference implementation
	ref := Reference{}
	cRef := make([]float32, m*n)
	ref.GEMM(false, false, m, n, k, 1.0, a, k, b, n, 0.0, cRef, n)
	
	// Compare results
	maxDiff := float32(0.0)
	var maxI, maxJ int
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*n + j
			diff := c[idx] - cRef[idx]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
				maxI = i
				maxJ = j
			}
		}
	}
	
	fmt.Printf("Maximum difference: %e at [%d,%d]\n", maxDiff, maxI, maxJ)
	if maxDiff > 0 {
		idx := maxI*n + maxJ
		fmt.Printf("GUDA: %f, Reference: %f\n", c[idx], cRef[idx])
		
		// Print a few values around the max difference
		fmt.Println("\nFirst few values:")
		for i := 0; i < min(4, m); i++ {
			for j := 0; j < min(4, n); j++ {
				idx := i*n + j
				fmt.Printf("[%d,%d] GUDA: %.1f, Ref: %.1f\n", i, j, c[idx], cRef[idx])
			}
		}
	}
	
	// Allow for some floating point error
	tolerance := float32(1e-3)
	if maxDiff > tolerance {
		t.Errorf("AVX-512 GEMM differs from reference: max diff %e > tolerance %e", maxDiff, tolerance)
	}
}

// BenchmarkGEMMAVX512 benchmarks GEMM with AVX-512
func BenchmarkGEMMAVX512(b *testing.B) {
	if GetBestGemmImplementation() != "AVX512" {
		b.Skip("AVX-512 not available")
	}
	
	sizes := []int{128, 256, 512, 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			m, n, k := size, size, size
			
			// Allocate matrices
			da, _ := Malloc(m * k * 4)
			db, _ := Malloc(k * n * 4)
			dc, _ := Malloc(m * n * 4)
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
			
			// Warm up
			GEMM(false, false, m, n, k, 1.0, da, k, db, n, 0.0, dc, n)
			Synchronize()
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMM(false, false, m, n, k, 1.0, da, k, db, n, 0.0, dc, n)
				Synchronize()
			}
			
			// Report metrics
			flops := 2 * int64(m) * int64(n) * int64(k)
			seconds := b.Elapsed().Seconds() / float64(b.N)
			gflops := float64(flops) / (seconds * 1e9)
			b.ReportMetric(gflops, "GFLOPS")
			
			// Memory bandwidth
			bytes := int64(m*k+k*n+m*n) * 4 // float32
			bandwidth := float64(bytes) / (seconds * 1e9)
			b.ReportMetric(bandwidth, "GB/s")
		})
	}
}