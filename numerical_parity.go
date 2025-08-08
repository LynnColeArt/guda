package guda

import (
	"math"
)

// NumericalParity provides tools for comparing floating point results
type NumericalParity struct {
	MaxAbsError float32
	MaxRelError float32
	MaxULPError int32
	NumErrors   int
}

// CompareFloat32 compares two float32 values and updates error statistics
func (np *NumericalParity) CompareFloat32(expected, actual float32) {
	absErr := AbsFloat32(expected - actual)
	if absErr > np.MaxAbsError {
		np.MaxAbsError = absErr
	}
	
	// Relative error (avoid division by zero)
	if expected != 0 {
		relErr := absErr / AbsFloat32(expected)
		if relErr > np.MaxRelError {
			np.MaxRelError = relErr
		}
	}
	
	// ULP error
	ulpErr := ULPDiffFloat32(expected, actual)
	if ulpErr > np.MaxULPError {
		np.MaxULPError = ulpErr
	}
	
	// Count errors above thresholds
	if absErr > 1e-6 || (expected != 0 && absErr/AbsFloat32(expected) > 1e-5) {
		np.NumErrors++
	}
}

// CompareSlices compares two slices of float32
func (np *NumericalParity) CompareSlices(expected, actual []float32) {
	n := len(expected)
	if len(actual) < n {
		n = len(actual)
	}
	
	for i := 0; i < n; i++ {
		np.CompareFloat32(expected[i], actual[i])
	}
}

// ULPDiffFloat32 computes the ULP (Units in Last Place) difference
func ULPDiffFloat32(a, b float32) int32 {
	if a == b {
		return 0
	}
	
	// Handle special cases
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return math.MaxInt32
	}
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		if a == b {
			return 0
		}
		return math.MaxInt32
	}
	
	// Convert to bits
	aBits := math.Float32bits(a)
	bBits := math.Float32bits(b)
	
	// Check if signs differ
	if (aBits>>31) != (bBits>>31) {
		// Signs differ, calculate distance through zero
		absA := float32(math.Abs(float64(a)))
		absB := float32(math.Abs(float64(b)))
		return ULPDiffFloat32(absA, 0) + ULPDiffFloat32(absB, 0)
	}
	
	// Same sign, compute difference
	var diff int32
	if aBits > bBits {
		diff = int32(aBits - bBits)
	} else {
		diff = int32(bBits - aBits)
	}
	
	return diff
}

// AbsFloat32 returns absolute value
func AbsFloat32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Tolerance levels for different operations
const (
	// Exact operations (add, multiply scalars)
	ToleranceExact = 0
	
	// High precision (GEMM, convolution)
	ToleranceHigh = 1e-6
	
	// Medium precision (reductions)
	ToleranceMedium = 1e-5
	
	// Low precision (transcendentals, approximations)
	ToleranceLow = 1e-4
	
	// ULP tolerances
	ULPExact  = 0
	ULPHigh   = 2
	ULPMedium = 10
	ULPLow    = 100
)

// OperationTolerance defines acceptable error for an operation
type OperationTolerance struct {
	Name    string
	AbsTol  float32
	RelTol  float32
	ULPTol  int32
}

// Standard tolerances for different operations
var StandardTolerances = map[string]OperationTolerance{
	"add":       {Name: "add", AbsTol: 0, RelTol: 0, ULPTol: ULPExact},
	"multiply":  {Name: "multiply", AbsTol: 0, RelTol: 0, ULPTol: ULPExact},
	"gemm":      {Name: "gemm", AbsTol: ToleranceHigh, RelTol: ToleranceHigh, ULPTol: ULPHigh},
	"conv2d":    {Name: "conv2d", AbsTol: ToleranceHigh, RelTol: ToleranceHigh, ULPTol: ULPHigh},
	"reduce_sum": {Name: "reduce_sum", AbsTol: ToleranceMedium, RelTol: ToleranceMedium, ULPTol: ULPMedium},
	"softmax":   {Name: "softmax", AbsTol: ToleranceMedium, RelTol: ToleranceMedium, ULPTol: ULPMedium},
	"gelu":      {Name: "gelu", AbsTol: ToleranceLow, RelTol: ToleranceLow, ULPTol: ULPLow},
	"exp":       {Name: "exp", AbsTol: ToleranceLow, RelTol: ToleranceLow, ULPTol: ULPLow},
}

// CheckTolerance returns true if the numerical differences are within tolerance
func (np *NumericalParity) CheckTolerance(tol OperationTolerance) bool {
	return np.MaxAbsError <= tol.AbsTol && 
	       np.MaxRelError <= tol.RelTol && 
	       np.MaxULPError <= tol.ULPTol
}