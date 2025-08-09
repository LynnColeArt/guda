// Package guda tolerance-based verification for floating-point comparisons
package guda

import (
	"fmt"
	"math"
)

// ToleranceConfig defines tolerance parameters for floating-point comparison
type ToleranceConfig struct {
	// AbsTol is the absolute tolerance for values near zero
	AbsTol float32
	
	// RelTol is the relative tolerance as a fraction of the larger value
	RelTol float32
	
	// ULPTol is the maximum allowed difference in ULPs (Units in Last Place)
	ULPTol int
	
	// CheckNaN determines if NaN values should be considered equal
	CheckNaN bool
	
	// CheckInf determines if Inf values should be considered equal
	CheckInf bool
}

// DefaultTolerance returns default tolerance configuration
func DefaultTolerance() ToleranceConfig {
	return ToleranceConfig{
		AbsTol:   1e-7,
		RelTol:   1e-5,
		ULPTol:   4,
		CheckNaN: true,
		CheckInf: true,
	}
}

// StrictTolerance returns strict tolerance configuration for high precision
func StrictTolerance() ToleranceConfig {
	return ToleranceConfig{
		AbsTol:   1e-9,
		RelTol:   1e-7,
		ULPTol:   1,
		CheckNaN: true,
		CheckInf: true,
	}
}

// RelaxedTolerance returns relaxed tolerance for accumulated operations
func RelaxedTolerance() ToleranceConfig {
	return ToleranceConfig{
		AbsTol:   1e-5,
		RelTol:   1e-3,
		ULPTol:   16,
		CheckNaN: true,
		CheckInf: true,
	}
}

// Float32NearEqual checks if two float32 values are equal within tolerance
func Float32NearEqual(a, b float32, tol ToleranceConfig) bool {
	// Handle special cases
	if tol.CheckNaN && math.IsNaN(float64(a)) && math.IsNaN(float64(b)) {
		return true
	}
	
	if tol.CheckInf {
		if math.IsInf(float64(a), 1) && math.IsInf(float64(b), 1) {
			return true // Both +Inf
		}
		if math.IsInf(float64(a), -1) && math.IsInf(float64(b), -1) {
			return true // Both -Inf
		}
	}
	
	// Check if exactly equal (handles ±0)
	if a == b {
		return true
	}
	
	// Absolute difference
	diff := math.Abs(float64(a - b))
	
	// Check absolute tolerance
	if diff <= float64(tol.AbsTol) {
		return true
	}
	
	// Check relative tolerance
	larger := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if diff <= larger*float64(tol.RelTol) {
		return true
	}
	
	// Check ULP difference
	if tol.ULPTol > 0 {
		ulpDiff := Float32ULPDiff(a, b)
		if ulpDiff <= tol.ULPTol {
			return true
		}
	}
	
	return false
}

// Float32ULPDiff computes the difference in ULPs between two float32 values
func Float32ULPDiff(a, b float32) int {
	// Convert to bits
	aBits := math.Float32bits(a)
	bBits := math.Float32bits(b)
	
	// Check for different signs
	if (aBits^bBits)&0x80000000 != 0 {
		// Different signs, can't use simple subtraction
		// Return max int to indicate very different
		return math.MaxInt32
	}
	
	// Same sign, compute ULP difference
	var diff int
	if aBits > bBits {
		diff = int(aBits - bBits)
	} else {
		diff = int(bBits - aBits)
	}
	
	return diff
}

// VerifyFloat32Array compares two float32 arrays with tolerance
type VerificationResult struct {
	MaxAbsError float32
	MaxRelError float32
	MaxULPError int
	NumErrors   int
	TotalItems  int
	FirstError  int // Index of first error, -1 if none
}

// VerifyFloat32Array compares two float32 arrays and returns detailed results
func VerifyFloat32Array(expected, actual []float32, tol ToleranceConfig) VerificationResult {
	result := VerificationResult{
		TotalItems: len(expected),
		FirstError: -1,
	}
	
	if len(expected) != len(actual) {
		// Arrays have different lengths
		result.NumErrors = len(expected)
		return result
	}
	
	for i := range expected {
		if !Float32NearEqual(expected[i], actual[i], tol) {
			result.NumErrors++
			if result.FirstError == -1 {
				result.FirstError = i
			}
			
			// Update max errors
			absDiff := float32(math.Abs(float64(expected[i] - actual[i])))
			if absDiff > result.MaxAbsError {
				result.MaxAbsError = absDiff
			}
			
			// Relative error (avoid division by zero)
			if expected[i] != 0 {
				relDiff := absDiff / float32(math.Abs(float64(expected[i])))
				if relDiff > result.MaxRelError {
					result.MaxRelError = relDiff
				}
			}
			
			// ULP error
			ulpDiff := Float32ULPDiff(expected[i], actual[i])
			if ulpDiff > result.MaxULPError {
				result.MaxULPError = ulpDiff
			}
		}
	}
	
	return result
}

// IsAcceptable returns true if the verification result is within tolerance
func (r VerificationResult) IsAcceptable(tol ToleranceConfig) bool {
	return r.NumErrors == 0 ||
		(r.MaxAbsError <= tol.AbsTol && 
		 r.MaxRelError <= tol.RelTol && 
		 r.MaxULPError <= tol.ULPTol)
}

// String formats the verification result for display
func (r VerificationResult) String() string {
	if r.NumErrors == 0 {
		return "PASS: All values match within tolerance"
	}
	
	errorRate := float64(r.NumErrors) / float64(r.TotalItems) * 100
	return fmt.Sprintf("FAIL: %d/%d values differ (%.2f%%)\n"+
		"  Max absolute error: %e\n"+
		"  Max relative error: %e\n"+
		"  Max ULP difference: %d\n"+
		"  First error at index: %d",
		r.NumErrors, r.TotalItems, errorRate,
		r.MaxAbsError, r.MaxRelError, r.MaxULPError,
		r.FirstError)
}

// VerifyKernel runs a kernel with reference implementation and verifies results
type KernelVerifier struct {
	Name      string
	Reference func([]float32) []float32
	Optimized func(DevicePtr) error
	Tolerance ToleranceConfig
}

// Verify runs both implementations and compares results
func (kv KernelVerifier) Verify(input []float32) (VerificationResult, error) {
	// Run reference implementation
	expected := kv.Reference(append([]float32(nil), input...)) // Copy input
	
	// Run optimized implementation
	d_input, err := Malloc(len(input) * 4)
	if err != nil {
		return VerificationResult{}, NewMemoryError("Verify", "failed to allocate device memory", err)
	}
	defer Free(d_input)
	
	copy(d_input.Float32(), input)
	
	err = kv.Optimized(d_input)
	if err != nil {
		return VerificationResult{}, NewExecutionError("Verify", "optimized kernel failed", err)
	}
	
	// Synchronize to ensure completion
	err = Synchronize()
	if err != nil {
		return VerificationResult{}, NewExecutionError("Verify", "synchronization failed", err)
	}
	
	// Get results
	actual := make([]float32, len(input))
	copy(actual, d_input.Float32())
	
	// Compare
	return VerifyFloat32Array(expected, actual, kv.Tolerance), nil
}

// BatchVerifier runs multiple test cases and aggregates results
type BatchVerifier struct {
	Verifiers []KernelVerifier
	TestCases []TestCase
}

// TestCase represents a single test input
type TestCase struct {
	Name  string
	Input []float32
}

// RunAll executes all verifiers on all test cases
func (bv BatchVerifier) RunAll() ([]BatchResult, error) {
	results := make([]BatchResult, len(bv.Verifiers))
	
	for i, verifier := range bv.Verifiers {
		results[i].KernelName = verifier.Name
		results[i].Results = make([]TestResult, len(bv.TestCases))
		
		for j, testCase := range bv.TestCases {
			result, err := verifier.Verify(testCase.Input)
			results[i].Results[j] = TestResult{
				TestName: testCase.Name,
				Result:   result,
				Error:    err,
			}
		}
	}
	
	return results, nil
}

// BatchResult contains results for one kernel across all test cases
type BatchResult struct {
	KernelName string
	Results    []TestResult
}

// TestResult contains the result of one test case
type TestResult struct {
	TestName string
	Result   VerificationResult
	Error    error
}

// Summary returns a summary of the batch results
func (br BatchResult) Summary() string {
	passed := 0
	failed := 0
	errors := 0
	
	for _, r := range br.Results {
		if r.Error != nil {
			errors++
		} else if r.Result.NumErrors == 0 {
			passed++
		} else {
			failed++
		}
	}
	
	return fmt.Sprintf("%s: %d passed, %d failed, %d errors",
		br.KernelName, passed, failed, errors)
}