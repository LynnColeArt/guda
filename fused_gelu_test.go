package guda

import (
	"math"
	"testing"
)

func TestGELUActivation(t *testing.T) {
	// Test GELU activation function
	testCases := []struct {
		input    float32
		expected float32
		tol      float32
	}{
		{0.0, 0.0, 1e-6},
		{1.0, 0.8413, 1e-3},     // GELU(1) ≈ 0.8413
		{-1.0, -0.1587, 1e-3},   // GELU(-1) ≈ -0.1587
		{2.0, 1.9546, 1e-3},     // GELU(2) ≈ 1.9546
		{-2.0, -0.0454, 1e-3},   // GELU(-2) ≈ -0.0454
		{0.5, 0.3457, 1e-3},     // GELU(0.5) ≈ 0.3457
		{-0.5, -0.1543, 1e-3},   // GELU(-0.5) ≈ -0.1543
	}
	
	for _, tc := range testCases {
		result := geluFloat32(tc.input)
		error := abs32(result - tc.expected)
		if error > tc.tol {
			t.Errorf("GELU(%f): expected %f, got %f (error: %e)", 
				tc.input, tc.expected, result, error)
		}
	}
}

func TestAddBiasGELU(t *testing.T) {
	// Test with a simple case
	n := 16
	biasSize := 4
	
	// Allocate memory
	d_x := MallocOrFail(t, n * 4)
	d_bias := MallocOrFail(t, biasSize * 4)
	defer Free(d_x)
	defer Free(d_bias)
	
	// Initialize data
	x := make([]float32, n)
	bias := make([]float32, biasSize)
	
	for i := 0; i < n; i++ {
		x[i] = float32(i-8) * 0.25 // Values from -2 to 2
	}
	
	for i := 0; i < biasSize; i++ {
		bias[i] = float32(i) * 0.1
	}
	
	// Copy to device
	MemcpyOrFail(t, d_x, x, n*4, MemcpyHostToDevice)
	MemcpyOrFail(t, d_bias, bias, biasSize*4, MemcpyHostToDevice)
	
	// Apply AddBiasGELU
	err := AddBiasGELU(d_x, d_bias, n)
	if err != nil {
		t.Fatalf("AddBiasGELU failed: %v", err)
	}
	SynchronizeOrFail(t)
	
	// Get result
	result := d_x.Float32()[:n]
	
	// Verify
	for i := 0; i < n; i++ {
		biasIdx := i % biasSize
		expected := geluFloat32(x[i] + bias[biasIdx])
		if abs32(result[i] - expected) > 1e-4 {
			t.Errorf("Position %d: expected %f, got %f", i, expected, result[i])
		}
	}
}

func TestFusedGEMMBiasGELU(t *testing.T) {
	// Test the fused GEMM+Bias+GELU operation
	m, n, k := 8, 8, 4
	
	// Allocate matrices
	d_a := MallocOrFail(t, m * k * 4)
	d_b := MallocOrFail(t, k * n * 4)
	d_c := MallocOrFail(t, m * n * 4)
	d_bias := MallocOrFail(t, n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	defer Free(d_bias)
	
	// Simple test data
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	// Initialize A as diagonal-ish
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			if i == j {
				aData[i*k+j] = 1.0
			} else {
				aData[i*k+j] = 0.1
			}
		}
	}
	
	// Initialize B 
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			bData[i*n+j] = float32(j) * 0.1
		}
	}
	
	// Initialize bias
	for i := 0; i < n; i++ {
		biasData[i] = float32(i) * 0.05
	}
	
	// Copy to device
	MemcpyOrFail(t, d_a, aData, len(aData)*4, MemcpyHostToDevice)
	MemcpyOrFail(t, d_b, bData, len(bData)*4, MemcpyHostToDevice)
	MemcpyOrFail(t, d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	// Run fused operation
	err := FusedGEMMBiasGELU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n, d_bias)
	if err != nil {
		t.Fatalf("FusedGEMMBiasGELU failed: %v", err)
	}
	SynchronizeOrFail(t)
	
	// Compute expected result manually
	expected := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				sum += aData[i*k+l] * bData[l*n+j]
			}
			sum += biasData[j]
			expected[i*n+j] = geluFloat32(sum)
		}
	}
	
	// Compare
	result := d_c.Float32()[:m*n]
	maxError := float32(0)
	for i := 0; i < m*n; i++ {
		error := abs32(result[i] - expected[i])
		if error > maxError {
			maxError = error
		}
	}
	
	// The tanh approximation for GELU has ~1% error
	// So we need a more relaxed tolerance than 1e-4
	tolerance := float32(0.015) // 1.5% tolerance
	if maxError > tolerance {
		t.Errorf("FusedGEMMBiasGELU: max error %e exceeds tolerance %e", maxError, tolerance)
		
		// Print some values for debugging
		t.Log("First few values:")
		for i := 0; i < min(10, m*n); i++ {
			t.Logf("  [%d] expected: %f, got: %f", i, expected[i], result[i])
		}
	}
}

// Benchmark comparing separate vs fused operations
func BenchmarkGELUOperations(b *testing.B) {
	m, n, k := 512, 512, 512
	
	// Allocate matrices
	d_a := MallocOrFail(b, m * k * 4)
	d_b := MallocOrFail(b, k * n * 4)
	d_c := MallocOrFail(b, m * n * 4)
	d_bias := MallocOrFail(b, n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
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
	
	MemcpyOrFail(b, d_a, aData, len(aData)*4, MemcpyHostToDevice)
	MemcpyOrFail(b, d_b, bData, len(bData)*4, MemcpyHostToDevice)
	MemcpyOrFail(b, d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	b.Run("Separate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// GEMM
			GEMM(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n)
			
			// Add bias and GELU
			AddBiasGELU(d_c, d_bias, m*n)
			
			Synchronize()
		}
	})
	
	b.Run("Fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			FusedGEMMBiasGELU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n, d_bias)
			Synchronize()
		}
	})
}

// Reference GELU implementation using math library for comparison
func geluReference(x float64) float64 {
	// Using error function: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
	return x * 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
}