package guda

import (
	"fmt"
	"math/rand"
	"runtime"
	"testing"
)

// TestGEMMWithArchTolerance tests GEMM with architecture-specific tolerances
func TestGEMMWithArchTolerance(t *testing.T) {
	sizes := []struct {
		m, n, k int
	}{
		{8, 8, 8},
		{16, 16, 16},
		{63, 65, 64},   // Non-aligned sizes
		{128, 128, 128},
	}
	
	// Get architecture-specific tolerance
	tol := GetOperationTolerance("gemm")
	t.Logf("Running on %s with tolerance: AbsTol=%e, RelTol=%e, ULPTol=%d",
		runtime.GOARCH, tol.AbsTol, tol.RelTol, tol.ULPTol)
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%dx%d", size.m, size.n, size.k), func(t *testing.T) {
			// Create test matrices
			a := make([]float32, size.m*size.k)
			b := make([]float32, size.k*size.n)
			c := make([]float32, size.m*size.n)
			cRef := make([]float32, size.m*size.n)
			
			// Initialize with random values
			for i := range a {
				a[i] = rand.Float32()*2 - 1 // [-1, 1]
			}
			for i := range b {
				b[i] = rand.Float32()*2 - 1
			}
			for i := range c {
				c[i] = rand.Float32()
				cRef[i] = c[i]
			}
			
			alpha := float32(1.5)
			beta := float32(0.5)
			
			// Run optimized GEMM
			da, _ := Malloc(len(a) * 4)
			db, _ := Malloc(len(b) * 4)
			dc, _ := Malloc(len(c) * 4)
			defer Free(da)
			defer Free(db)
			defer Free(dc)
			
			copy(da.Float32(), a)
			copy(db.Float32(), b)
			copy(dc.Float32(), c)
			
			GEMM(false, false, size.m, size.n, size.k, alpha, da, size.k, db, size.n, beta, dc, size.n)
			Synchronize()
			
			copy(c, dc.Float32())
			
			// Run reference
			ref := Reference{}
			ref.GEMM(false, false, size.m, size.n, size.k, alpha, a, size.k, b, size.n, beta, cRef, size.n)
			
			// Verify with architecture-specific tolerance
			result := VerifyFloat32Array(cRef, c, tol)
			
			if result.NumErrors > 0 {
				t.Logf("Verification result:\n%s", result.String())
				
				// Check if it would pass with relaxed tolerance
				if IsARM64() && result.IsAcceptable(tol) {
					t.Logf("PASS: Errors within ARM64 tolerance")
				} else if !result.IsAcceptable(tol) {
					t.Errorf("FAIL: Errors exceed architecture tolerance")
				}
			}
		})
	}
}

// BenchmarkGEMMArchComparison compares GEMM accuracy across simulated architectures
func BenchmarkGEMMArchComparison(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping architecture comparison in short mode")
	}
	
	size := 256
	
	// Create test data
	a := make([]float32, size*size)
	bMat := make([]float32, size*size)
	c := make([]float32, size*size)
	cRef := make([]float32, size*size)
	
	for i := range a {
		a[i] = rand.Float32()*2 - 1
	}
	for i := range bMat {
		bMat[i] = rand.Float32()*2 - 1
	}
	
	// Run reference implementation
	ref := Reference{}
	ref.GEMM(false, false, size, size, size, 1.0, a, size, bMat, size, 0.0, cRef, size)
	
	// Allocate device memory
	da, _ := Malloc(len(a) * 4)
	db, _ := Malloc(len(bMat) * 4)
	dc, _ := Malloc(len(c) * 4)
	defer Free(da)
	defer Free(db)
	defer Free(dc)
	
	copy(da.Float32(), a)
	copy(db.Float32(), bMat)
	
	// Run optimized version multiple times to check consistency
	b.Run("ConsistencyCheck", func(b *testing.B) {
		maxDiffs := make([]float32, b.N)
		maxULPs := make([]int, b.N)
		
		for i := 0; i < b.N; i++ {
			GEMM(false, false, size, size, size, 1.0, da, size, db, size, 0.0, dc, size)
			Synchronize()
			
			copy(c, dc.Float32())
			
			// Compare with reference
			maxDiff := float32(0.0)
			maxULP := 0
			for j := range c {
				diff := c[j] - cRef[j]
				if diff < 0 {
					diff = -diff
				}
				if diff > maxDiff {
					maxDiff = diff
				}
				
				ulp := int(Float32ULPDiff(c[j], cRef[j]))
				if ulp > maxULP {
					maxULP = ulp
				}
			}
			
			maxDiffs[i] = maxDiff
			maxULPs[i] = maxULP
		}
		
		// Report statistics
		b.Logf("Architecture: %s", runtime.GOARCH)
		b.Logf("Max absolute differences across %d runs:", b.N)
		
		avgDiff := float32(0.0)
		avgULP := 0
		for i := range maxDiffs {
			avgDiff += maxDiffs[i]
			avgULP += maxULPs[i]
		}
		avgDiff /= float32(b.N)
		avgULP /= b.N
		
		b.Logf("  Average max diff: %e", avgDiff)
		b.Logf("  Average max ULP: %d", avgULP)
		
		// Check against architecture tolerance
		tol := GetOperationTolerance("gemm")
		b.Logf("Architecture tolerance: AbsTol=%e, ULPTol=%d", tol.AbsTol, tol.ULPTol)
		
		if avgDiff > tol.AbsTol {
			b.Logf("WARNING: Average diff exceeds architecture tolerance")
		}
		if avgULP > tol.ULPTol {
			b.Logf("WARNING: Average ULP diff exceeds architecture tolerance")
		}
	})
}