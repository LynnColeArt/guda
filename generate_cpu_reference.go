package guda

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
)

// GenerateCPUReference creates reference test vectors using CPU computation
func GenerateCPUReference() error {
	// Create output directory
	outputDir := "/media/lynn/big_drive/workspaces/Guda/cuda-analysis/cpu_test_vectors"
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		return fmt.Errorf("creating output directory: %w", err)
	}
	
	// Test cases
	testCases := []struct {
		name        string
		m, n, k     int
		alpha, beta float32
	}{
		{"gemm_3x3x3_simple", 3, 3, 3, 1.0, 0.0},
		{"gemm_10x10x10", 10, 10, 10, 1.0, 0.0},
		{"gemm_10x10x10_alphabeta", 10, 10, 10, 1.5, 0.5},
		{"gemm_37x29x41_neg", 37, 29, 41, -1.5, 0.5},
	}
	
	for _, tc := range testCases {
		fmt.Printf("Generating %s...\n", tc.name)
		
		// Generate test matrices
		A := make([]float32, tc.m*tc.k)
		B := make([]float32, tc.k*tc.n)
		C := make([]float32, tc.m*tc.n)
		
		// Sequential pattern matching CUDA test
		for i := range A {
			A[i] = float32(i%100) * 0.01
		}
		for i := range B {
			B[i] = float32(i%50) * 0.02
		}
		for i := range C {
			C[i] = float32(i%10) * 0.1
		}
		
		// Save inputs
		saveVector(filepath.Join(outputDir, tc.name+"_A.bin"), A)
		saveVector(filepath.Join(outputDir, tc.name+"_B.bin"), B)
		saveVector(filepath.Join(outputDir, tc.name+"_C_input.bin"), C)
		
		// Compute C = alpha*A*B + beta*C
		result := make([]float32, tc.m*tc.n)
		copy(result, C)
		
		// Scale C by beta
		for i := range result {
			result[i] *= tc.beta
		}
		
		// Add alpha*A*B
		for i := 0; i < tc.m; i++ {
			for j := 0; j < tc.n; j++ {
				sum := float32(0)
				for k := 0; k < tc.k; k++ {
					sum += A[i*tc.k+k] * B[k*tc.n+j]
				}
				result[i*tc.n+j] += tc.alpha * sum
			}
		}
		
		// Save output
		saveVector(filepath.Join(outputDir, tc.name+"_C_output.bin"), result)
		
		// For small matrices, also save readable output
		if tc.m <= 10 && tc.n <= 10 {
			file, err := os.Create(filepath.Join(outputDir, tc.name+"_result.txt"))
			if err == nil {
				fmt.Fprintf(file, "CPU Reference for %s\n", tc.name)
				fmt.Fprintf(file, "Dimensions: %dx%dx%d, alpha=%g, beta=%g\n\n", 
					tc.m, tc.n, tc.k, tc.alpha, tc.beta)
				
				fmt.Fprintf(file, "Result C:\n")
				for i := 0; i < tc.m; i++ {
					for j := 0; j < tc.n; j++ {
						fmt.Fprintf(file, "%8.4f ", result[i*tc.n+j])
					}
					fmt.Fprintf(file, "\n")
				}
				file.Close()
			}
		}
		
		// Print first few values
		fmt.Printf("  First 5 values: ")
		for i := 0; i < 5 && i < len(result); i++ {
			fmt.Printf("%.4f ", result[i])
		}
		fmt.Println()
	}
	
	fmt.Printf("\nCPU reference vectors saved to %s\n", outputDir)
	return nil
}

func saveVector(filename string, data []float32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write size header
	size := uint32(len(data))
	err = binary.Write(file, binary.LittleEndian, size)
	if err != nil {
		return err
	}
	
	// Write data
	err = binary.Write(file, binary.LittleEndian, data)
	return err
}