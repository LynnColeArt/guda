package guda

import (
	"testing"
	"path/filepath"
	"os"
)

// TestWithCPUReference generates CPU reference vectors and compares GUDA against them
func TestWithCPUReference(t *testing.T) {
	// Generate CPU reference vectors
	err := GenerateCPUReference()
	if err != nil {
		t.Fatalf("Failed to generate CPU reference: %v", err)
	}

	// Path to test vectors
	vectorPath := "/media/lynn/big_drive/workspaces/Guda/cuda-analysis/cpu_test_vectors"
	
	// Check if test vectors exist
	if _, err := os.Stat(vectorPath); os.IsNotExist(err) {
		t.Fatal("CPU test vectors not found after generation")
	}
	
	// Test cases to check
	testCases := []struct {
		name       string
		m, n, k    int
		alpha, beta float32
	}{
		{"gemm_3x3x3_simple", 3, 3, 3, 1.0, 0.0},
		{"gemm_10x10x10", 10, 10, 10, 1.0, 0.0},
		{"gemm_10x10x10_alphabeta", 10, 10, 10, 1.5, 0.5},
		{"gemm_37x29x41_neg", 37, 29, 41, -1.5, 0.5},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Load test vector
			tv, err := LoadTestCase(vectorPath, tc.name)
			if err != nil {
				t.Fatalf("Could not load test vector: %v", err)
			}
			
			// Set dimensions
			tv.M, tv.N, tv.K = tc.m, tc.n, tc.k
			tv.Alpha, tv.Beta = tc.alpha, tc.beta
			
			// Run GUDA GEMM
			d_a, _ := Malloc(len(tv.A) * 4)
			d_b, _ := Malloc(len(tv.B) * 4)
			d_c, _ := Malloc(len(tv.CInput) * 4)
			defer Free(d_a)
			defer Free(d_b)
			defer Free(d_c)
			
			Memcpy(d_a, tv.A, len(tv.A)*4, MemcpyHostToDevice)
			Memcpy(d_b, tv.B, len(tv.B)*4, MemcpyHostToDevice)
			Memcpy(d_c, tv.CInput, len(tv.CInput)*4, MemcpyHostToDevice)
			
			err = GEMM(false, false, tc.m, tc.n, tc.k,
				tc.alpha, d_a, tc.k, d_b, tc.n,
				tc.beta, d_c, tc.n)
			if err != nil {
				t.Fatalf("GEMM failed: %v", err)
			}
			
			gudaResult := d_c.Float32()[:tc.m*tc.n]
			
			// Compare with CPU result
			parity := NumericalParity{}
			parity.CompareSlices(tv.CExpected, gudaResult)
			
			// Check tolerance - use slightly relaxed tolerance for CPU comparison
			tol := OperationTolerance{
				Name:   "cpu_reference",
				AbsTol: 1e-5,
				RelTol: 1e-4,
				ULPTol: 10,
			}
			
			if !parity.CheckTolerance(tol) {
				// Print detailed comparison for small matrices
				if tc.m <= 10 && tc.n <= 10 {
					t.Logf("Expected (CPU):")
					for i := 0; i < tc.m; i++ {
						for j := 0; j < tc.n; j++ {
							t.Logf("  [%d,%d] = %f", i, j, tv.CExpected[i*tc.n+j])
						}
					}
					t.Logf("Got (GUDA):")
					for i := 0; i < tc.m; i++ {
						for j := 0; j < tc.n; j++ {
							t.Logf("  [%d,%d] = %f", i, j, gudaResult[i*tc.n+j])
						}
					}
				}
				
				t.Errorf("GEMM numerical parity failed:\n"+
					"  Max abs error: %e (tolerance: %e)\n"+
					"  Max rel error: %e (tolerance: %e)\n"+
					"  Max ULP error: %d (tolerance: %d)",
					parity.MaxAbsError, tol.AbsTol,
					parity.MaxRelError, tol.RelTol,
					parity.MaxULPError, tol.ULPTol)
			} else {
				t.Logf("CPU parity PASSED: max abs=%e, max rel=%e, max ULP=%d",
					parity.MaxAbsError, parity.MaxRelError, parity.MaxULPError)
			}
		})
	}
	
	// Also test that results are readable
	if _, err := os.Stat(filepath.Join(vectorPath, "gemm_3x3x3_simple_result.txt")); err == nil {
		content, _ := os.ReadFile(filepath.Join(vectorPath, "gemm_3x3x3_simple_result.txt"))
		t.Logf("Sample CPU reference output:\n%s", string(content))
	}
}