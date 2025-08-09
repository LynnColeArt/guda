// Example demonstrating tolerance-based verification in GUDA
package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Tolerance-Based Verification Example")
	fmt.Println("=========================================")
	
	// Initialize reference implementations
	ref := guda.Reference{}
	
	// Example 1: Simple operation verification
	fmt.Println("\n1. Verifying ReLU activation:")
	verifyReLU(ref)
	
	// Example 2: Accumulated error in reductions
	fmt.Println("\n2. Verifying Sum reduction with accumulated error:")
	verifyReduction(ref)
	
	// Example 3: Different tolerance levels
	fmt.Println("\n3. Demonstrating different tolerance levels:")
	demonstrateToleranceLevels()
	
	// Example 4: Batch verification of multiple kernels
	fmt.Println("\n4. Batch verification of neural network operations:")
	batchVerifyNeuralOps(ref)
}

func verifyReLU(ref guda.Reference) {
	// Create test data with edge cases
	testData := []float32{
		-1e-10,  // Very small negative
		-0.0,    // Negative zero
		0.0,     // Positive zero
		1e-10,   // Very small positive
		-1.0,    // Normal negative
		1.0,     // Normal positive
		float32(math.Inf(-1)), // Negative infinity
		float32(math.Inf(1)),  // Positive infinity
		float32(math.NaN()),   // NaN
	}
	
	// Run reference implementation
	expected := make([]float32, len(testData))
	copy(expected, testData)
	ref.ReLU(expected)
	
	// Run optimized implementation
	d_data, err := guda.Malloc(len(testData) * 4)
	if err != nil {
		log.Fatal(err)
	}
	defer guda.Free(d_data)
	
	copy(d_data.Float32(), testData)
	
	err = guda.ReLU(d_data, len(testData))
	if err != nil {
		log.Fatal(err)
	}
	
	err = guda.Synchronize()
	if err != nil {
		log.Fatal(err)
	}
	
	actual := make([]float32, len(testData))
	copy(actual, d_data.Float32())
	
	// Verify with strict tolerance
	result := guda.VerifyFloat32Array(expected, actual, guda.StrictTolerance())
	
	fmt.Printf("  Result: %s\n", result.String())
	fmt.Printf("  Test data included: very small values, zeros, infinities, and NaN\n")
}

func verifyReduction(ref guda.Reference) {
	// Create large array to accumulate rounding errors
	n := 10000
	rng := rand.New(rand.NewSource(42))
	
	testData := make([]float32, n)
	for i := range testData {
		// Use values that will accumulate error
		testData[i] = float32(rng.Float64()) * 0.1
	}
	
	// Reference sum (using float64 for higher precision)
	var sum64 float64
	for _, v := range testData {
		sum64 += float64(v)
	}
	expectedSum := float32(sum64)
	
	// Optimized sum
	d_data, err := guda.Malloc(n * 4)
	if err != nil {
		log.Fatal(err)
	}
	defer guda.Free(d_data)
	
	copy(d_data.Float32(), testData)
	
	actualSum, err := guda.Reduce(d_data, n, func(a, b float32) float32 { return a + b })
	if err != nil {
		log.Fatal(err)
	}
	
	err = guda.Synchronize()
	if err != nil {
		log.Fatal(err)
	}
	
	// Check with different tolerances
	fmt.Printf("  Array size: %d elements\n", n)
	fmt.Printf("  Expected sum: %.6f\n", expectedSum)
	fmt.Printf("  Actual sum:   %.6f\n", actualSum)
	fmt.Printf("  Absolute error: %.2e\n", math.Abs(float64(expectedSum-actualSum)))
	fmt.Printf("  Relative error: %.2e\n", math.Abs(float64(expectedSum-actualSum))/float64(expectedSum))
	
	// Verify with relaxed tolerance (appropriate for reductions)
	if guda.Float32NearEqual(expectedSum, actualSum, guda.RelaxedTolerance()) {
		fmt.Println("  ✓ PASS with relaxed tolerance")
	} else {
		fmt.Println("  ✗ FAIL even with relaxed tolerance")
	}
}

func demonstrateToleranceLevels() {
	a := float32(1.0)
	b := float32(1.0000001)
	
	fmt.Printf("  Comparing %.9f and %.9f:\n", a, b)
	fmt.Printf("  Absolute difference: %.2e\n", math.Abs(float64(a-b)))
	fmt.Printf("  Relative difference: %.2e\n", math.Abs(float64(a-b))/float64(a))
	fmt.Printf("  ULP difference: %d\n", guda.Float32ULPDiff(a, b))
	
	tolerances := []struct {
		name string
		tol  guda.ToleranceConfig
	}{
		{"Strict", guda.StrictTolerance()},
		{"Default", guda.DefaultTolerance()},
		{"Relaxed", guda.RelaxedTolerance()},
	}
	
	for _, tc := range tolerances {
		equal := guda.Float32NearEqual(a, b, tc.tol)
		fmt.Printf("  %s tolerance (abs=%.0e, rel=%.0e, ulp=%d): %v\n",
			tc.name, tc.tol.AbsTol, tc.tol.RelTol, tc.tol.ULPTol, equal)
	}
}

func batchVerifyNeuralOps(ref guda.Reference) {
	// Create test cases representing different scenarios
	testCases := []guda.TestCase{
		{
			Name:  "Zeros",
			Input: make([]float32, 10), // All zeros
		},
		{
			Name:  "Small_Gradients",
			Input: generateSmallValues(10, 1e-6),
		},
		{
			Name:  "Normal_Activations",
			Input: generateNormalValues(10, 0, 1),
		},
		{
			Name:  "Large_Logits",
			Input: generateNormalValues(10, 0, 10),
		},
	}
	
	// Create verifiers for common neural network operations
	verifiers := []guda.KernelVerifier{
		{
			Name: "ReLU",
			Reference: func(x []float32) []float32 {
				result := make([]float32, len(x))
				copy(result, x)
				ref.ReLU(result)
				return result
			},
			Optimized: func(d guda.DevicePtr) error {
				// DevicePtr stores size in bytes, divide by 4 for float32 count
				return guda.ReLU(d, d.Size()/4)
			},
			Tolerance: guda.StrictTolerance(), // ReLU should be exact
		},
		{
			Name: "Tanh",
			Reference: func(x []float32) []float32 {
				result := make([]float32, len(x))
				copy(result, x)
				ref.Tanh(result)
				return result
			},
			Optimized: func(d guda.DevicePtr) error {
				return guda.Tanh(d, d.Size()/4)
			},
			Tolerance: guda.DefaultTolerance(), // Tanh uses approximations
		},
		{
			Name: "Softmax",
			Reference: func(x []float32) []float32 {
				result := make([]float32, len(x))
				copy(result, x)
				ref.Softmax(result)
				return result
			},
			Optimized: func(d guda.DevicePtr) error {
				return guda.Softmax(d, d.Size()/4)
			},
			Tolerance: guda.RelaxedTolerance(), // Softmax accumulates error
		},
	}
	
	// Run batch verification
	batch := guda.BatchVerifier{
		Verifiers: verifiers,
		TestCases: testCases,
	}
	
	results, err := batch.RunAll()
	if err != nil {
		log.Fatal(err)
	}
	
	// Display results
	fmt.Println("\n  Summary:")
	for _, br := range results {
		fmt.Printf("  %s\n", br.Summary())
	}
	
	// Detailed results for any failures
	fmt.Println("\n  Detailed results:")
	for _, br := range results {
		for _, tr := range br.Results {
			if tr.Error != nil || tr.Result.NumErrors > 0 {
				fmt.Printf("  %s/%s: %s\n", br.KernelName, tr.TestName,
					tr.Result.String())
			}
		}
	}
}

// Helper functions

func generateSmallValues(n int, scale float32) []float32 {
	result := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := range result {
		result[i] = (float32(rng.Float64()) - 0.5) * 2 * scale
	}
	return result
}

func generateNormalValues(n int, mean, stddev float32) []float32 {
	result := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := range result {
		result[i] = float32(rng.NormFloat64())*stddev + mean
	}
	return result
}