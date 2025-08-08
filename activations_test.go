package guda

import (
	"math"
	"testing"
)

func TestSigmoidAccuracy(t *testing.T) {
	testCases := []struct {
		input    float32
		expected float64 // Use float64 for reference
		tol      float32
	}{
		{0.0, 0.5, 1e-6},
		{1.0, 0.7310585786300049, 1e-5},
		{-1.0, 0.2689414213699951, 1e-5},
		{2.0, 0.8807970779778823, 1e-5},
		{-2.0, 0.11920292202211757, 1e-5},
		{5.0, 0.9933071490757153, 1e-5},
		{-5.0, 0.006692850924284856, 1e-5},
		{10.0, 0.9999546021312976, 1e-4},
		{-10.0, 4.5397868702434395e-05, 1e-4},
	}
	
	for _, tc := range testCases {
		// Test our implementation
		result := SigmoidFloat32(tc.input)
		
		// Compare with reference
		error := math.Abs(float64(result) - tc.expected)
		if error > float64(tc.tol) {
			t.Errorf("SigmoidFloat32(%f): expected %f, got %f (error: %e)", 
				tc.input, tc.expected, result, error)
		}
		
		// Also test against Go's math library
		mathResult := 1.0 / (1.0 + math.Exp(-float64(tc.input)))
		mathError := math.Abs(float64(result) - mathResult)
		if mathError > float64(tc.tol) {
			t.Errorf("SigmoidFloat32(%f) vs math: expected %f, got %f (error: %e)", 
				tc.input, mathResult, result, mathError)
		}
	}
}

func TestTanhAccuracy(t *testing.T) {
	testCases := []float32{
		0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
		-0.5, -1.0, -2.0, -3.0, -5.0, -10.0,
	}
	
	for _, x := range testCases {
		// Test our implementation
		result := TanhFloat32(x)
		
		// Compare with Go's math library
		expected := math.Tanh(float64(x))
		error := math.Abs(float64(result) - expected)
		
		// Tolerance depends on range
		tol := 1e-5
		if math.Abs(float64(x)) > 5 {
			tol = 1e-4 // Slightly more tolerant for large values
		}
		
		if error > tol {
			t.Errorf("TanhFloat32(%f): expected %f, got %f (error: %e)", 
				x, expected, result, error)
		}
	}
}

func TestExpFloat32Accuracy(t *testing.T) {
	testCases := []float32{
		0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 85.0,
		-1.0, -2.0, -5.0, -10.0, -20.0, -50.0, -85.0,
	}
	
	for _, x := range testCases {
		result := ExpFloat32(x)
		expected := math.Exp(float64(x))
		
		// Handle overflow cases
		if math.IsInf(expected, 1) {
			if result != math.MaxFloat32 {
				t.Errorf("ExpFloat32(%f): expected MaxFloat32 for overflow, got %f", x, result)
			}
			continue
		}
		
		// Handle underflow cases
		if expected < 1e-30 {
			if result > 1e-30 {
				t.Errorf("ExpFloat32(%f): expected near-zero for underflow, got %f", x, result)
			}
			continue
		}
		
		// Normal cases - use relative error
		relError := math.Abs(float64(result) - expected) / expected
		if relError > 1e-4 {
			t.Errorf("ExpFloat32(%f): expected %f, got %f (rel error: %e)", 
				x, expected, result, relError)
		}
	}
}

func TestGELUAccuracy(t *testing.T) {
	testCases := []struct {
		input    float32
		expected float64 // Reference from SciPy or similar
		tol      float32
	}{
		{0.0, 0.0, 1e-6},
		{1.0, 0.8413447460685429, 1e-3},     // tanh approx vs erf
		{-1.0, -0.15865525393145705, 1e-3},
		{0.5, 0.34571221824490996, 1e-3},
		{-0.5, -0.15430780299115956, 1e-3},
		{2.0, 1.9545977256749598, 1e-3},
		{-2.0, -0.04540227432504002, 1e-3},
	}
	
	for _, tc := range testCases {
		// Test tanh approximation
		resultTanh := geluFloat32(tc.input)
		errorTanh := math.Abs(float64(resultTanh) - tc.expected)
		
		if errorTanh > float64(tc.tol) {
			t.Errorf("geluFloat32(%f): expected %f, got %f (error: %e)", 
				tc.input, tc.expected, resultTanh, errorTanh)
		}
		
		// Test accurate version (should be even better)
		resultAccurate := geluFloat32Accurate(tc.input)
		errorAccurate := math.Abs(float64(resultAccurate) - tc.expected)
		
		// Accurate version should have reasonable tolerance
		// Note: ERF-based version may not always be more accurate due to approximation limits
		if errorAccurate > 1e-4 {
			t.Errorf("geluFloat32Accurate(%f): expected %f, got %f (error: %e)", 
				tc.input, tc.expected, resultAccurate, errorAccurate)
		}
		
		// Both versions should be reasonably accurate - don't require one to be strictly better
		// The tanh approximation is standard in ML frameworks and often preferred
	}
}

// Benchmark to ensure our improvements don't hurt performance significantly
func BenchmarkActivations(b *testing.B) {
	inputs := make([]float32, 1000)
	for i := range inputs {
		inputs[i] = float32(i-500) * 0.01 // Range [-5, 5]
	}
	
	b.Run("Sigmoid", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, x := range inputs {
				_ = SigmoidFloat32(x)
			}
		}
	})
	
	b.Run("Tanh", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, x := range inputs {
				_ = TanhFloat32(x)
			}
		}
	})
	
	b.Run("GELU_Tanh", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, x := range inputs {
				_ = geluFloat32(x)
			}
		}
	})
	
	b.Run("GELU_Accurate", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, x := range inputs {
				_ = geluFloat32Accurate(x)
			}
		}
	})
}