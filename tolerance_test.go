package guda

import (
	"math"
	"testing"
)

func TestFloat32NearEqual(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		tol      ToleranceConfig
		expected bool
	}{
		// Exact equality
		{
			name:     "Exact_Equal",
			a:        1.0,
			b:        1.0,
			tol:      DefaultTolerance(),
			expected: true,
		},
		// Within absolute tolerance
		{
			name:     "Within_AbsTol",
			a:        1e-8,
			b:        2e-8,
			tol:      DefaultTolerance(),
			expected: true,
		},
		// Outside absolute tolerance
		{
			name:     "Outside_AbsTol",
			a:        1e-6,
			b:        2e-6,
			tol:      DefaultTolerance(),
			expected: false,
		},
		// Within relative tolerance
		{
			name:     "Within_RelTol",
			a:        1000.0,
			b:        1000.01,
			tol:      DefaultTolerance(),
			expected: true,
		},
		// Zero handling
		{
			name:     "Both_Zero",
			a:        0.0,
			b:        -0.0,
			tol:      DefaultTolerance(),
			expected: true,
		},
		// NaN handling
		{
			name:     "Both_NaN",
			a:        float32(math.NaN()),
			b:        float32(math.NaN()),
			tol:      DefaultTolerance(),
			expected: true,
		},
		{
			name: "NaN_Not_Checked",
			a:    float32(math.NaN()),
			b:    float32(math.NaN()),
			tol: ToleranceConfig{
				CheckNaN: false,
			},
			expected: false,
		},
		// Infinity handling
		{
			name:     "Both_PosInf",
			a:        float32(math.Inf(1)),
			b:        float32(math.Inf(1)),
			tol:      DefaultTolerance(),
			expected: true,
		},
		{
			name:     "Both_NegInf",
			a:        float32(math.Inf(-1)),
			b:        float32(math.Inf(-1)),
			tol:      DefaultTolerance(),
			expected: true,
		},
		{
			name:     "Mixed_Inf",
			a:        float32(math.Inf(1)),
			b:        float32(math.Inf(-1)),
			tol:      DefaultTolerance(),
			expected: false,
		},
		// ULP tolerance
		{
			name:     "Within_ULP",
			a:        1.0,
			b:        math.Float32frombits(math.Float32bits(1.0) + 2),
			tol:      DefaultTolerance(),
			expected: true,
		},
		{
			name:     "Outside_ULP",
			a:        1.0,
			b:        math.Float32frombits(math.Float32bits(1.0) + 5),
			tol:      DefaultTolerance(),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Float32NearEqual(tt.a, tt.b, tt.tol)
			if result != tt.expected {
				t.Errorf("Float32NearEqual(%v, %v) = %v, want %v",
					tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestFloat32ULPDiff(t *testing.T) {
	tests := []struct {
		name     string
		a, b     float32
		expected int
	}{
		{
			name:     "Same_Value",
			a:        1.0,
			b:        1.0,
			expected: 0,
		},
		{
			name:     "Adjacent_Values",
			a:        1.0,
			b:        math.Float32frombits(math.Float32bits(1.0) + 1),
			expected: 1,
		},
		{
			name:     "Two_ULPs",
			a:        1.0,
			b:        math.Float32frombits(math.Float32bits(1.0) + 2),
			expected: 2,
		},
		{
			name:     "Different_Signs",
			a:        1.0,
			b:        -1.0,
			expected: math.MaxInt32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Float32ULPDiff(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Float32ULPDiff(%v, %v) = %v, want %v",
					tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestVerifyFloat32Array(t *testing.T) {
	tests := []struct {
		name     string
		expected []float32
		actual   []float32
		tol      ToleranceConfig
		wantPass bool
	}{
		{
			name:     "All_Match",
			expected: []float32{1.0, 2.0, 3.0, 4.0},
			actual:   []float32{1.0, 2.0, 3.0, 4.0},
			tol:      DefaultTolerance(),
			wantPass: true,
		},
		{
			name:     "Within_Tolerance",
			expected: []float32{1.0, 2.0, 3.0, 4.0},
			actual:   []float32{1.00001, 2.00001, 3.00001, 4.00001},
			tol:      DefaultTolerance(),
			wantPass: true,
		},
		{
			name:     "Outside_Tolerance",
			expected: []float32{1.0, 2.0, 3.0, 4.0},
			actual:   []float32{1.1, 2.0, 3.0, 4.0},
			tol:      DefaultTolerance(),
			wantPass: false,
		},
		{
			name:     "Different_Lengths",
			expected: []float32{1.0, 2.0, 3.0},
			actual:   []float32{1.0, 2.0},
			tol:      DefaultTolerance(),
			wantPass: false,
		},
		{
			name:     "With_NaN",
			expected: []float32{1.0, float32(math.NaN()), 3.0},
			actual:   []float32{1.0, float32(math.NaN()), 3.0},
			tol:      DefaultTolerance(),
			wantPass: true,
		},
		{
			name:     "Accumulated_Error",
			expected: []float32{1000.0},
			actual:   []float32{1001.0},
			tol:      RelaxedTolerance(),
			wantPass: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := VerifyFloat32Array(tt.expected, tt.actual, tt.tol)
			passed := result.IsAcceptable(tt.tol)
			
			if passed != tt.wantPass {
				t.Errorf("VerifyFloat32Array: got pass=%v, want pass=%v\n%s",
					passed, tt.wantPass, result.String())
			}
			
			// Additional checks for specific cases
			if tt.name == "All_Match" && result.NumErrors != 0 {
				t.Errorf("Expected 0 errors, got %d", result.NumErrors)
			}
			
			if tt.name == "Different_Lengths" && result.NumErrors != len(tt.expected) {
				t.Errorf("Expected %d errors for different lengths, got %d",
					len(tt.expected), result.NumErrors)
			}
		})
	}
}

func TestKernelVerifier(t *testing.T) {
	// Test with ReLU kernel
	ref := Reference{}
	
	verifier := KernelVerifier{
		Name: "ReLU",
		Reference: func(x []float32) []float32 {
			result := make([]float32, len(x))
			copy(result, x)
			ref.ReLU(result)
			return result
		},
		Optimized: func(d DevicePtr) error {
			return ReLU(d, d.size/4)
		},
		Tolerance: DefaultTolerance(),
	}
	
	// Test data with positive and negative values
	input := []float32{-2.0, -1.0, 0.0, 1.0, 2.0}
	
	result, err := verifier.Verify(input)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}
	
	if !result.IsAcceptable(verifier.Tolerance) {
		t.Errorf("ReLU verification failed:\n%s", result.String())
	}
}

func TestBatchVerifier(t *testing.T) {
	ref := Reference{}
	
	// Create verifiers for multiple kernels
	verifiers := []KernelVerifier{
		{
			Name: "ReLU",
			Reference: func(x []float32) []float32 {
				result := make([]float32, len(x))
				copy(result, x)
				ref.ReLU(result)
				return result
			},
			Optimized: func(d DevicePtr) error {
				return ReLU(d, d.size/4)
			},
			Tolerance: DefaultTolerance(),
		},
		{
			Name: "Sigmoid",
			Reference: func(x []float32) []float32 {
				result := make([]float32, len(x))
				copy(result, x)
				ref.Sigmoid(result)
				return result
			},
			Optimized: func(d DevicePtr) error {
				return Sigmoid(d, d.size/4)
			},
			Tolerance: RelaxedTolerance(), // Sigmoid uses approximations
		},
	}
	
	// Create test cases
	testCases := []TestCase{
		{
			Name:  "Small_Values",
			Input: []float32{-0.1, 0.0, 0.1},
		},
		{
			Name:  "Normal_Values",
			Input: []float32{-2.0, -1.0, 0.0, 1.0, 2.0},
		},
		{
			Name:  "Large_Values",
			Input: []float32{-10.0, -5.0, 5.0, 10.0},
		},
	}
	
	batch := BatchVerifier{
		Verifiers: verifiers,
		TestCases: testCases,
	}
	
	results, err := batch.RunAll()
	if err != nil {
		t.Fatalf("Batch verification failed: %v", err)
	}
	
	// Check results
	for _, br := range results {
		t.Logf("%s", br.Summary())
		
		for _, tr := range br.Results {
			if tr.Error != nil {
				t.Errorf("%s/%s: error: %v", br.KernelName, tr.TestName, tr.Error)
			} else if !tr.Result.IsAcceptable(DefaultTolerance()) {
				t.Errorf("%s/%s: verification failed:\n%s",
					br.KernelName, tr.TestName, tr.Result.String())
			}
		}
	}
}

func TestTolerancePresets(t *testing.T) {
	// Test that preset tolerances have expected values
	tests := []struct {
		name   string
		tol    ToleranceConfig
		absMin float32
		absMax float32
		relMin float32
		relMax float32
		ulpMin int
		ulpMax int
	}{
		{
			name:   "Default",
			tol:    DefaultTolerance(),
			absMin: 1e-8,
			absMax: 1e-6,
			relMin: 1e-6,
			relMax: 1e-4,
			ulpMin: 2,
			ulpMax: 8,
		},
		{
			name:   "Strict",
			tol:    StrictTolerance(),
			absMin: 1e-10,
			absMax: 1e-8,
			relMin: 1e-8,
			relMax: 1e-6,
			ulpMin: 1,
			ulpMax: 2,
		},
		{
			name:   "Relaxed",
			tol:    RelaxedTolerance(),
			absMin: 1e-6,
			absMax: 1e-4,
			relMin: 1e-4,
			relMax: 1e-2,
			ulpMin: 8,
			ulpMax: 32,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.tol.AbsTol < tt.absMin || tt.tol.AbsTol > tt.absMax {
				t.Errorf("AbsTol %e not in range [%e, %e]",
					tt.tol.AbsTol, tt.absMin, tt.absMax)
			}
			if tt.tol.RelTol < tt.relMin || tt.tol.RelTol > tt.relMax {
				t.Errorf("RelTol %e not in range [%e, %e]",
					tt.tol.RelTol, tt.relMin, tt.relMax)
			}
			if tt.tol.ULPTol < tt.ulpMin || tt.tol.ULPTol > tt.ulpMax {
				t.Errorf("ULPTol %d not in range [%d, %d]",
					tt.tol.ULPTol, tt.ulpMin, tt.ulpMax)
			}
		})
	}
}