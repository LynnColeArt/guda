package guda

import (
	"testing"
)

func TestFusedSimple(t *testing.T) {
	// Very simple 8x8 test
	m, n, k := 8, 8, 8
	
	// Allocate matrices
	d_a, _ := Malloc(m * k * 4)
	d_b, _ := Malloc(k * n * 4)
	d_c1, _ := Malloc(m * n * 4)
	d_c2, _ := Malloc(m * n * 4)
	d_bias, _ := Malloc(n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c1)
	defer Free(d_c2)
	defer Free(d_bias)
	
	// Simple test pattern
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	// Identity-like pattern
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			aData[i*k+j] = 1.0
		}
	}
	
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				bData[i*n+j] = 1.0
			} else {
				bData[i*n+j] = 0.0
			}
		}
	}
	
	for i := 0; i < n; i++ {
		biasData[i] = float32(i) * 0.1
	}
	
	Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
	Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
	Memcpy(d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	// Reference: separate operations
	GEMM(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c1, n)
	AddBiasReLU(d_c1, d_bias, m*n)
	Synchronize()
	
	// Test: fused operation
	FusedGEMMBiasReLU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c2, n, d_bias)
	Synchronize()
	
	// Compare
	c1 := d_c1.Float32()[:m*n]
	c2 := d_c2.Float32()[:m*n]
	
	t.Log("Expected (separate):")
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			t.Logf("  [%d,%d] = %.2f", i, j, c1[i*n+j])
		}
	}
	
	t.Log("\nActual (fused):")
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			t.Logf("  [%d,%d] = %.2f", i, j, c2[i*n+j])
		}
	}
	
	// Check match
	maxError := float32(0)
	for i := 0; i < m*n; i++ {
		err := abs32(c1[i] - c2[i])
		if err > maxError {
			maxError = err
		}
	}
	
	if maxError > 1e-4 {
		t.Errorf("Results don't match: max error=%e", maxError)
	}
}