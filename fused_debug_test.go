package guda

import (
	"testing"
	"time"
)

func TestFusedVsSeparateDebug(t *testing.T) {
	m, n, k := 512, 512, 512
	
	// Allocate matrices
	d_a, _ := Malloc(m * k * 4)
	d_b, _ := Malloc(k * n * 4)
	d_c, _ := Malloc(m * n * 4)
	d_c2, _ := Malloc(m * n * 4)
	d_bias, _ := Malloc(n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	defer Free(d_c2)
	defer Free(d_bias)
	
	// Initialize with test data
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%100) * 0.01
	}
	for i := range biasData {
		biasData[i] = float32(i) * 0.1
	}
	
	Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
	Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
	Memcpy(d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	// Time separate operations
	start := time.Now()
	GEMM(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n)
	AddBiasReLU(d_c, d_bias, m*n)
	Synchronize()
	separateTime := time.Since(start)
	
	// Time fused operation
	start = time.Now()
	FusedGEMMBiasReLU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c2, n, d_bias)
	Synchronize()
	fusedTime := time.Since(start)
	
	t.Logf("Separate operations: %v", separateTime)
	t.Logf("Fused operation: %v", fusedTime)
	t.Logf("Speedup: %.2fx", float64(separateTime)/float64(fusedTime))
	
	// Verify results match
	c1 := d_c.Float32()[:m*n]
	c2 := d_c2.Float32()[:m*n]
	
	maxError := float32(0)
	errorCount := 0
	for i := 0; i < m*n; i++ {
		err := abs32(c1[i] - c2[i])
		if err > maxError {
			maxError = err
		}
		if err > 1e-4 {
			errorCount++
			if errorCount <= 5 {
				t.Logf("Mismatch at %d: separate=%f, fused=%f", i, c1[i], c2[i])
			}
		}
	}
	
	if maxError > 1e-4 {
		t.Errorf("Results don't match: max error=%e, errors=%d/%d", maxError, errorCount, m*n)
	} else {
		t.Logf("Results match (max error: %e)", maxError)
	}
}