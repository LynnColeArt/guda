package guda

import (
	"math"
	"testing"
)

// TestNumericalParityFramework demonstrates how we would test against CUDA
func TestNumericalParityFramework(t *testing.T) {
	// Example: Test GEMM numerical parity
	t.Run("GEMM_Parity", func(t *testing.T) {
		m, n, k := 64, 64, 64
		alpha := float32(1.5)
		beta := float32(0.5)
		
		// Generate test data
		a := generateTestMatrix(m, k, "normal")
		b := generateTestMatrix(k, n, "normal")
		c := generateTestMatrix(m, n, "uniform")
		
		// Run GUDA computation
		d_a, _ := Malloc(m * k * 4)
		d_b, _ := Malloc(k * n * 4)
		d_c, _ := Malloc(m * n * 4)
		defer Free(d_a)
		defer Free(d_b)
		defer Free(d_c)
		
		Memcpy(d_a, a, len(a)*4, MemcpyHostToDevice)
		Memcpy(d_b, b, len(b)*4, MemcpyHostToDevice)
		Memcpy(d_c, c, len(c)*4, MemcpyHostToDevice)
		
		err := GEMM(false, false, m, n, k, alpha, d_a, k, d_b, n, beta, d_c, n)
		if err != nil {
			t.Fatalf("GEMM failed: %v", err)
		}
		
		gudaResult := d_c.Float32()[:m*n]
		
		// In real test, we would load CUDA result from file or compute it
		// For now, let's compute expected result manually
		expectedResult := computeGEMMReference(m, n, k, alpha, a, b, beta, c)
		
		// Compare results
		parity := NumericalParity{}
		parity.CompareSlices(expectedResult, gudaResult)
		
		// Check against tolerance
		tol := StandardTolerances["gemm"]
		if !parity.CheckTolerance(tol) {
			t.Errorf("GEMM numerical parity failed:\n"+
				"  Max abs error: %e (tolerance: %e)\n"+
				"  Max rel error: %e (tolerance: %e)\n"+
				"  Max ULP error: %d (tolerance: %d)\n"+
				"  Num errors: %d",
				parity.MaxAbsError, tol.AbsTol,
				parity.MaxRelError, tol.RelTol,
				parity.MaxULPError, tol.ULPTol,
				parity.NumErrors)
		} else {
			t.Logf("GEMM numerical parity PASSED:\n"+
				"  Max abs error: %e\n"+
				"  Max rel error: %e\n"+
				"  Max ULP error: %d",
				parity.MaxAbsError, parity.MaxRelError, parity.MaxULPError)
		}
	})
	
	// Test reduction operations
	t.Run("Reduction_Parity", func(t *testing.T) {
		n := 10000
		
		// Test different data patterns
		patterns := []string{"uniform", "normal", "edge_cases"}
		
		for _, pattern := range patterns {
			data := generateTestVector(n, pattern)
			
			// Test sum reduction
			d_data, _ := Malloc(n * 4)
			defer Free(d_data)
			Memcpy(d_data, data, n*4, MemcpyHostToDevice)
			
			gudaSum := d_data.Sum(n)
			
			// Reference sum (using Kahan summation for accuracy)
			expectedSum := kahanSum(data)
			
			parity := NumericalParity{}
			parity.CompareFloat32(expectedSum, gudaSum)
			
			tol := StandardTolerances["reduce_sum"]
			if !parity.CheckTolerance(tol) {
				t.Errorf("Sum reduction parity failed for %s data:\n"+
					"  Expected: %e, Got: %e\n"+
					"  Abs error: %e, Rel error: %e",
					pattern, expectedSum, gudaSum,
					parity.MaxAbsError, parity.MaxRelError)
			}
		}
	})
}

// generateTestMatrix creates test matrices with specific patterns
func generateTestMatrix(rows, cols int, pattern string) []float32 {
	data := make([]float32, rows*cols)
	
	switch pattern {
	case "normal":
		// Simulate normal distribution
		for i := range data {
			data[i] = float32(math.Sin(float64(i)) * 0.5)
		}
	case "uniform":
		// Uniform distribution [0, 1]
		for i := range data {
			data[i] = float32(i) / float32(len(data))
		}
	case "edge_cases":
		// Include special values
		specials := []float32{0, 1, -1, 0.5, -0.5, 1e-6, -1e-6, 1e6, -1e6}
		for i := range data {
			data[i] = specials[i%len(specials)]
		}
	}
	
	return data
}

func generateTestVector(n int, pattern string) []float32 {
	return generateTestMatrix(n, 1, pattern)
}

// computeGEMMReference computes GEMM using simple loops (for testing)
func computeGEMMReference(m, n, k int, alpha float32, a, b []float32, beta float32, c []float32) []float32 {
	result := make([]float32, m*n)
	
	// First scale C by beta
	for i := range result {
		result[i] = beta * c[i]
	}
	
	// Then add alpha * A * B
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for kk := 0; kk < k; kk++ {
				sum += a[i*k+kk] * b[kk*n+j]
			}
			result[i*n+j] += alpha * sum
		}
	}
	
	return result
}

// kahanSum computes sum with reduced numerical error
func kahanSum(data []float32) float32 {
	sum := float32(0)
	c := float32(0) // Error compensation
	
	for _, val := range data {
		y := val - c
		t := sum + y
		c = (t - sum) - y
		sum = t
	}
	
	return sum
}

// TestULPDifference verifies our ULP calculation
func TestULPDifference(t *testing.T) {
	tests := []struct {
		a, b     float32
		expected int32
	}{
		{1.0, 1.0, 0},
		{1.0, math.Float32frombits(math.Float32bits(1.0) + 1), 1},
		{1.0, math.Float32frombits(math.Float32bits(1.0) + 2), 2},
		{1.0, -1.0, math.MaxInt32}, // Crossing zero
		{0.0, 0.0, 0},
	}
	
	for _, test := range tests {
		ulp := ULPDiffFloat32(test.a, test.b)
		// For the crossing zero case, just check it's very large
		if test.expected > 1000000 {
			if ulp < 1000000 {
				t.Errorf("ULP(%v, %v) = %d, expected > 1000000", test.a, test.b, ulp)
			}
		} else if ulp != test.expected {
			t.Errorf("ULP(%v, %v) = %d, expected %d", test.a, test.b, ulp, test.expected)
		}
	}
}

// BenchmarkNumericalParity measures overhead of parity checking
func BenchmarkNumericalParity(b *testing.B) {
	n := 1000
	data1 := generateTestVector(n, "normal")
	data2 := make([]float32, n)
	copy(data2, data1)
	
	// Add small perturbations
	for i := range data2 {
		data2[i] += float32(i%10) * 1e-7
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parity := NumericalParity{}
		parity.CompareSlices(data1, data2)
	}
	
	b.ReportMetric(float64(n), "elements/op")
}