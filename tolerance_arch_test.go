package guda

import (
	"runtime"
	"testing"
)

func TestArchSpecificTolerance(t *testing.T) {
	// Test that we get appropriate tolerances for current architecture
	tol := GetOperationTolerance("gemm")
	
	t.Logf("Running on %s architecture", runtime.GOARCH)
	t.Logf("GEMM tolerance: AbsTol=%e, RelTol=%e, ULPTol=%d", 
		tol.AbsTol, tol.RelTol, tol.ULPTol)
	
	// Verify we get relaxed tolerance on ARM64
	if IsARM64() {
		if tol.ULPTol < 16 {
			t.Errorf("ARM64 should have ULPTol >= 16, got %d", tol.ULPTol)
		}
		if tol.RelTol < 1e-4 {
			t.Errorf("ARM64 should have RelTol >= 1e-4, got %e", tol.RelTol)
		}
	}
}

func TestToleranceMerging(t *testing.T) {
	base := ToleranceConfig{
		AbsTol:   1e-7,
		RelTol:   1e-6,
		ULPTol:   2,
		CheckNaN: true,
		CheckInf: false,
	}
	
	override := ToleranceConfig{
		RelTol: 1e-4,
		ULPTol: 16,
	}
	
	merged := mergeTolerances(base, override)
	
	// Check that non-overridden values are preserved
	if merged.AbsTol != base.AbsTol {
		t.Errorf("AbsTol should be preserved: got %e, want %e", merged.AbsTol, base.AbsTol)
	}
	
	// Check that overridden values are updated
	if merged.RelTol != override.RelTol {
		t.Errorf("RelTol should be overridden: got %e, want %e", merged.RelTol, override.RelTol)
	}
	
	if merged.ULPTol != override.ULPTol {
		t.Errorf("ULPTol should be overridden: got %d, want %d", merged.ULPTol, override.ULPTol)
	}
}

// TestFloatingPointDifferences demonstrates typical FP differences between architectures
func TestFloatingPointDifferences(t *testing.T) {
	// Example: Dot product computed differently
	a := []float32{1.1, 2.2, 3.3, 4.4, 5.5}
	b := []float32{6.6, 7.7, 8.8, 9.9, 10.1}
	
	// Sequential accumulation (typical x86)
	sum1 := float32(0.0)
	for i := range a {
		sum1 += a[i] * b[i]
	}
	
	// Pairwise accumulation (typical ARM NEON)
	sum2 := float32(0.0)
	for i := 0; i < len(a)-1; i += 2 {
		sum2 += a[i]*b[i] + a[i+1]*b[i+1]
	}
	if len(a)%2 == 1 {
		sum2 += a[len(a)-1] * b[len(a)-1]
	}
	
	// Check difference
	diff := sum1 - sum2
	if diff < 0 {
		diff = -diff
	}
	
	t.Logf("Sequential sum: %f", sum1)
	t.Logf("Pairwise sum: %f", sum2)
	t.Logf("Absolute difference: %e", diff)
	t.Logf("ULP difference: %d", Float32ULPDiff(sum1, sum2))
	
	// This shows why we need architecture-specific tolerances
	defaultTol := DefaultTolerance()
	archTol := GetOperationTolerance("reduce_sum")
	
	if !Float32NearEqual(sum1, sum2, archTol) {
		t.Logf("Values differ beyond architecture-specific tolerance")
	}
	
	t.Logf("Default tolerance would %s", 
		map[bool]string{true: "pass", false: "fail"}[Float32NearEqual(sum1, sum2, defaultTol)])
	t.Logf("Arch-specific tolerance would %s",
		map[bool]string{true: "pass", false: "fail"}[Float32NearEqual(sum1, sum2, archTol)])
}