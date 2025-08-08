package guda

import (
	"math"
	"testing"
)

// TestMathematicalCorrectness verifies all operations produce correct results
func TestMathematicalCorrectness(t *testing.T) {
	t.Run("VectorAddition", testVectorAddition)
	t.Run("AXPY", testAXPYCorrectness)
	t.Run("DotProduct", testDotProduct)
	t.Run("GEMM", testGEMMCorrectness)
	t.Run("Float16Precision", testFloat16Precision)
}

func testVectorAddition(t *testing.T) {
	sizes := []int{1, 7, 63, 127, 256, 1000, 10000}
	
	for _, n := range sizes {
		// Create test vectors
		a := make([]float32, n)
		b := make([]float32, n)
		expected := make([]float32, n)
		
		// Fill with deterministic values
		for i := 0; i < n; i++ {
			a[i] = float32(i) * 0.1
			b[i] = float32(n-i) * 0.2
			expected[i] = a[i] + b[i]
		}
		
		// GPU computation
		d_a, _ := Malloc(n * 4)
		d_b, _ := Malloc(n * 4)
		d_c, _ := Malloc(n * 4)
		defer Free(d_a)
		defer Free(d_b)
		defer Free(d_c)
		
		Memcpy(d_a, a, n*4, MemcpyHostToDevice)
		Memcpy(d_b, b, n*4, MemcpyHostToDevice)
		
		Add(d_a, d_b, d_c, n)
		Synchronize()
		
		result := d_c.Float32()[:n]
		
		// Verify
		for i := 0; i < n; i++ {
			if !almostEqual(result[i], expected[i], 1e-6) {
				t.Errorf("Add[n=%d] at %d: expected %f, got %f", n, i, expected[i], result[i])
				break
			}
		}
	}
}

func testAXPYCorrectness(t *testing.T) {
	n := 1000
	alphas := []float32{0, 1, -1, 2.5, -3.7, 0.1}
	
	for _, alpha := range alphas {
		x := make([]float32, n)
		y := make([]float32, n)
		expected := make([]float32, n)
		
		// Fill with test data
		for i := 0; i < n; i++ {
			x[i] = float32(math.Sin(float64(i) * 0.01))
			y[i] = float32(math.Cos(float64(i) * 0.01))
			expected[i] = alpha*x[i] + y[i]
		}
		
		// GPU computation
		d_x, _ := Malloc(n * 4)
		d_y, _ := Malloc(n * 4)
		defer Free(d_x)
		defer Free(d_y)
		
		Memcpy(d_x, x, n*4, MemcpyHostToDevice)
		Memcpy(d_y, y, n*4, MemcpyHostToDevice)
		
		AXPY(alpha, d_x, d_y, n)
		Synchronize()
		
		result := d_y.Float32()[:n]
		
		// Verify
		for i := 0; i < n; i++ {
			if !almostEqual(result[i], expected[i], 1e-5) {
				t.Errorf("AXPY[alpha=%f] at %d: expected %f, got %f", alpha, i, expected[i], result[i])
				break
			}
		}
	}
}

func testDotProduct(t *testing.T) {
	n := 1000
	
	x := make([]float32, n)
	y := make([]float32, n)
	var expected float32
	
	// Known values
	for i := 0; i < n; i++ {
		x[i] = float32(i%10) * 0.1
		y[i] = float32((i+5)%10) * 0.1
		expected += x[i] * y[i]
	}
	
	// GPU computation
	d_x, _ := Malloc(n * 4)
	d_y, _ := Malloc(n * 4)
	defer Free(d_x)
	defer Free(d_y)
	
	Memcpy(d_x, x, n*4, MemcpyHostToDevice)
	Memcpy(d_y, y, n*4, MemcpyHostToDevice)
	
	result, err := DOT(d_x, d_y, n)
	if err != nil {
		t.Fatalf("DOT failed: %v", err)
	}
	
	// Verify
	if !almostEqual(result, expected, 1e-3) {
		t.Errorf("DOT: expected %f, got %f", expected, result)
	}
}

func testGEMMCorrectness(t *testing.T) {
	// Test cases: C = alpha*A*B + beta*C
	testCases := []struct {
		m, n, k      int
		alpha, beta  float32
		transA, transB bool
	}{
		{4, 3, 2, 1.0, 0.0, false, false},     // Basic multiply
		{3, 3, 3, 2.0, 1.0, false, false},     // With scaling
		{5, 4, 3, 1.0, 0.0, true, false},      // A transposed
		{4, 5, 3, 1.0, 0.0, false, true},      // B transposed
		{10, 10, 10, -1.5, 0.5, true, true},   // Both transposed
	}
	
	for _, tc := range testCases {
		// Create matrices with known values
		a := makeMatrix(tc.m, tc.k, func(i, j int) float32 {
			return float32(i*tc.k + j)
		})
		b := makeMatrix(tc.k, tc.n, func(i, j int) float32 {
			return float32((i+1)*(j+1))
		})
		c := makeMatrix(tc.m, tc.n, func(i, j int) float32 {
			return float32(i + j)
		})
		
		// Calculate expected result
		expected := computeGEMM(a, b, c, tc.m, tc.n, tc.k, 
			tc.alpha, tc.beta, tc.transA, tc.transB)
		
		// GPU computation
		d_a, _ := Malloc(tc.m * tc.k * 4)
		d_b, _ := Malloc(tc.k * tc.n * 4)
		d_c, _ := Malloc(tc.m * tc.n * 4)
		defer Free(d_a)
		defer Free(d_b)
		defer Free(d_c)
		
		Memcpy(d_a, a, len(a)*4, MemcpyHostToDevice)
		Memcpy(d_b, b, len(b)*4, MemcpyHostToDevice)
		Memcpy(d_c, c, len(c)*4, MemcpyHostToDevice)
		
		lda := tc.k
		ldb := tc.n
		if tc.transA {
			lda = tc.m
		}
		if tc.transB {
			ldb = tc.k
		}
		
		err := GEMM(tc.transA, tc.transB, tc.m, tc.n, tc.k,
			tc.alpha, d_a, lda, d_b, ldb, tc.beta, d_c, tc.n)
		if err != nil {
			t.Fatalf("GEMM failed: %v", err)
		}
		Synchronize()
		
		result := d_c.Float32()[:tc.m*tc.n]
		
		// Verify
		maxError := float32(0)
		for i := 0; i < tc.m*tc.n; i++ {
			error := float32(math.Abs(float64(result[i] - expected[i])))
			if error > maxError {
				maxError = error
			}
		}
		
		// Tolerance based on matrix size (accumulation error)
		tolerance := float32(tc.k) * 1e-5
		if maxError > tolerance {
			t.Errorf("GEMM[%dx%dx%d,α=%f,β=%f,tA=%v,tB=%v]: max error %f > tolerance %f",
				tc.m, tc.n, tc.k, tc.alpha, tc.beta, tc.transA, tc.transB, 
				maxError, tolerance)
		}
	}
}

func testFloat16Precision(t *testing.T) {
	n := 1000
	
	// Test values that are exactly representable in float16
	testValues := []float32{
		0, 1, -1, 0.5, -0.5, 2, -2, 
		1024, -1024, // Powers of 2
		0.00024414062, // Smallest normal float16
		65504, // Largest float16
	}
	
	for _, val := range testValues {
		// Create array
		input := make([]float32, n)
		for i := range input {
			input[i] = val
		}
		
		// Convert to float16 and back
		d_f32, _ := Malloc(n * 4)
		d_f16, _ := Malloc(n * 2)
		defer Free(d_f32)
		defer Free(d_f16)
		
		Memcpy(d_f32, input, n*4, MemcpyHostToDevice)
		
		// Convert to float16
		f32Slice := d_f32.Float32()[:n]
		f16Slice := d_f16.Float16()
		for i := 0; i < n; i++ {
			f16Slice.SetFloat32(i, f32Slice[i])
		}
		
		// Convert back
		result := make([]float32, n)
		for i := 0; i < n; i++ {
			result[i] = f16Slice.GetFloat32(i)
		}
		
		// Verify
		for i := 0; i < n; i++ {
			if val != 0 && math.Abs(float64(val)) < 65504 && math.Abs(float64(val)) > 0.00024414062 {
				// Normal range - should be exact for powers of 2
				if isPowerOfTwo(val) && result[i] != val {
					t.Errorf("Float16 conversion of %f failed: got %f", val, result[i])
					break
				}
			}
		}
	}
}

// Helper functions

func almostEqual(a, b, tolerance float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return false
	}
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		return a == b
	}
	return math.Abs(float64(a-b)) <= float64(tolerance)
}

func makeMatrix(rows, cols int, f func(i, j int) float32) []float32 {
	m := make([]float32, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i*cols+j] = f(i, j)
		}
	}
	return m
}

func computeGEMM(a, b, c []float32, m, n, k int, alpha, beta float32, transA, transB bool) []float32 {
	result := make([]float32, m*n)
	
	// Initialize with beta*C
	for i := 0; i < m*n; i++ {
		result[i] = beta * c[i]
	}
	
	// Add alpha*A*B
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				aIdx := i*k + l
				bIdx := l*n + j
				
				if transA {
					aIdx = l*m + i
				}
				if transB {
					bIdx = j*k + l
				}
				
				sum += a[aIdx] * b[bIdx]
			}
			result[i*n+j] += alpha * sum
		}
	}
	
	return result
}

func isPowerOfTwo(x float32) bool {
	if x <= 0 {
		return false
	}
	bits := math.Float32bits(x)
	mantissa := bits & 0x7FFFFF
	return mantissa == 0
}

// TestGEMMNumericalStability checks for numerical stability issues
func TestGEMMNumericalStability(t *testing.T) {
	// Test with poorly conditioned matrices
	n := 100
	
	// Create a matrix with large condition number
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	
	// Hilbert matrix (poorly conditioned)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*n+j] = 1.0 / float32(i+j+1)
			if i == j {
				b[i*n+j] = 1.0 // Identity
			} else {
				b[i*n+j] = 0.0
			}
		}
	}
	
	// GPU computation
	d_a, _ := Malloc(n * n * 4)
	d_b, _ := Malloc(n * n * 4)
	d_c, _ := Malloc(n * n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	
	Memcpy(d_a, a, n*n*4, MemcpyHostToDevice)
	Memcpy(d_b, b, n*n*4, MemcpyHostToDevice)
	
	err := GEMM(false, false, n, n, n, 1.0, d_a, n, d_b, n, 0.0, d_c, n)
	if err != nil {
		t.Fatalf("GEMM failed: %v", err)
	}
	Synchronize()
	
	result := d_c.Float32()[:n*n]
	
	// Check that result ≈ a (since b = I)
	maxRelError := float32(0)
	for i := 0; i < n*n; i++ {
		if a[i] != 0 {
			relError := math.Abs(float64((result[i] - a[i]) / a[i]))
			if float32(relError) > maxRelError {
				maxRelError = float32(relError)
			}
		}
	}
	
	// With Hilbert matrix, expect some error but not catastrophic
	if maxRelError > 0.01 {
		t.Errorf("GEMM numerical stability: relative error %f too large", maxRelError)
	}
}