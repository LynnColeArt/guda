package guda

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestVector represents a test case with inputs and expected output
type TestVector struct {
	Name       string
	M, N, K    int
	Alpha, Beta float32
	A, B, CInput []float32
	CExpected   []float32
}

// LoadTestVector loads a binary test vector file
func LoadTestVector(filename string) ([]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	// Read size header
	var size uint32
	err = binary.Read(file, binary.LittleEndian, &size)
	if err != nil {
		return nil, err
	}
	
	// Read data
	data := make([]float32, size)
	err = binary.Read(file, binary.LittleEndian, &data)
	if err != nil {
		return nil, err
	}
	
	return data, nil
}

// LoadTestCase loads all files for a test case
func LoadTestCase(basePath, name string) (*TestVector, error) {
	tv := &TestVector{Name: name}
	
	// Load A matrix
	A, err := LoadTestVector(filepath.Join(basePath, name+"_A.bin"))
	if err != nil {
		return nil, fmt.Errorf("loading A: %w", err)
	}
	tv.A = A
	
	// Load B matrix
	B, err := LoadTestVector(filepath.Join(basePath, name+"_B.bin"))
	if err != nil {
		return nil, fmt.Errorf("loading B: %w", err)
	}
	tv.B = B
	
	// Load input C matrix
	CInput, err := LoadTestVector(filepath.Join(basePath, name+"_C_input.bin"))
	if err != nil {
		return nil, fmt.Errorf("loading C input: %w", err)
	}
	tv.CInput = CInput
	
	// Load expected output
	CExpected, err := LoadTestVector(filepath.Join(basePath, name+"_C_output.bin"))
	if err != nil {
		return nil, fmt.Errorf("loading C output: %w", err)
	}
	tv.CExpected = CExpected
	
	// Derive dimensions from sizes
	// For now, we'll parse from the name or use a config file
	// This is a simplified version
	
	return tv, nil
}

// TestWithCUDAVectors runs GUDA against CUDA/ROCm reference vectors
func TestWithCUDAVectors(t *testing.T) {
	// Path to test vectors (update this path as needed)
	vectorPath := "/media/lynn/big_drive/workspaces/Guda/cuda-analysis/test_vectors"
	
	// Check if test vectors exist
	if _, err := os.Stat(vectorPath); os.IsNotExist(err) {
		t.Skip("Test vectors not found. Run generate_test_vectors on a CUDA/ROCm system first.")
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
		{"gemm_64x64x64", 64, 64, 64, 1.0, 0.0},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Load test vector
			tv, err := LoadTestCase(vectorPath, tc.name)
			if err != nil {
				t.Skipf("Could not load test vector: %v", err)
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
			
			// Compare with CUDA result
			parity := NumericalParity{}
			parity.CompareSlices(tv.CExpected, gudaResult)
			
			// Check tolerance
			tol := StandardTolerances["gemm"]
			if !parity.CheckTolerance(tol) {
				// Print detailed comparison for small matrices
				if tc.m <= 10 && tc.n <= 10 {
					t.Logf("Expected (CUDA):")
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
				t.Logf("CUDA parity PASSED: max abs=%e, max rel=%e, max ULP=%d",
					parity.MaxAbsError, parity.MaxRelError, parity.MaxULPError)
			}
		})
	}
}

// GenerateTestVectorInstructions prints instructions for generating test vectors
func GenerateTestVectorInstructions() {
	fmt.Println(`To generate CUDA/ROCm test vectors:

1. On a system with CUDA:
   cd /media/lynn/big_drive/workspaces/Guda/cuda-analysis
   nvcc -o generate_test_vectors generate_test_vectors.cu -lcublas
   ./generate_test_vectors

2. On a system with ROCm:
   cd /media/lynn/big_drive/workspaces/Guda/cuda-analysis
   hipcc -o generate_test_vectors generate_test_vectors.cu -lhipblas
   ./generate_test_vectors

The test vectors will be created in the test_vectors/ subdirectory.`)
}