package guda

import (
	"fmt"
	"testing"
)

func TestFusedGEMMBiasReLU8x8Debug(t *testing.T) {
	// Create a minimal 8x8 test case
	m, n, k := 8, 8, 2
	alpha := float32(1.0)
	lda, ldb, ldc := k, n, n
	
	// Simple test data
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)
	cExpected := make([]float32, m*n)
	bias := make([]float32, n)
	
	// Initialize A matrix (8x2)
	for i := 0; i < m; i++ {
		for j := 0; j < k; j++ {
			a[i*lda+j] = float32(i + 1) // Row i has value i+1
		}
	}
	
	// Initialize B matrix (2x8)
	for i := 0; i < k; i++ {
		for j := 0; j < n; j++ {
			b[i*ldb+j] = float32(j + 1) // Column j has value j+1
		}
	}
	
	// Initialize bias
	for i := 0; i < n; i++ {
		bias[i] = float32(i) * 0.1 // Small bias values
	}
	
	// Compute expected result manually
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			sum = alpha * sum + bias[j]
			if sum < 0 {
				sum = 0
			}
			cExpected[i*ldc+j] = sum
		}
	}
	
	// Print expected values for debugging
	t.Logf("Expected C matrix:")
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%6.2f ", cExpected[i*ldc+j])
		}
		fmt.Println()
	}
	
	// Test the naive implementation first
	cNaive := make([]float32, m*n)
	copy(cNaive, c)
	fusedGEMMBiasReLUNaive(m, n, k, alpha, a, lda, b, ldb, bias, cNaive, ldc)
	
	// Compare naive with expected
	maxError := float32(0)
	for i := 0; i < m*n; i++ {
		err := abs32(cNaive[i] - cExpected[i])
		if err > maxError {
			maxError = err
		}
	}
	
	if maxError > 1e-6 {
		t.Errorf("Naive implementation error: %e", maxError)
		t.Logf("Naive C matrix:")
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				fmt.Printf("%6.2f ", cNaive[i*ldc+j])
			}
			fmt.Println()
		}
	} else {
		t.Logf("Naive implementation correct (max error: %e)", maxError)
	}
	
	// Now test AVX2 if available
	if hasAVX2 {
		cAVX2 := make([]float32, m*n)
		copy(cAVX2, c)
		
		// Use the kernel function directly
		fusedGEMMBiasReLUKernel(m, n, k, alpha, a, lda, b, ldb, bias, cAVX2, ldc)
		
		// Compare AVX2 with expected
		maxError = 0
		var firstErrorIdx int = -1
		for i := 0; i < m*n; i++ {
			err := abs32(cAVX2[i] - cExpected[i])
			if err > maxError {
				maxError = err
				if firstErrorIdx == -1 && err > 1e-6 {
					firstErrorIdx = i
				}
			}
		}
		
		if maxError > 1e-6 {
			t.Errorf("AVX2 implementation error: %e", maxError)
			if firstErrorIdx >= 0 {
				row := firstErrorIdx / n
				col := firstErrorIdx % n
				t.Errorf("First error at [%d,%d]: expected %f, got %f", 
					row, col, cExpected[firstErrorIdx], cAVX2[firstErrorIdx])
			}
			
			t.Logf("AVX2 C matrix:")
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					fmt.Printf("%6.2f ", cAVX2[i*ldc+j])
					if abs32(cAVX2[i*ldc+j] - cExpected[i*ldc+j]) > 1e-6 {
						fmt.Printf("! ")
					} else {
						fmt.Printf("  ")
					}
				}
				fmt.Println()
			}
		} else {
			t.Logf("AVX2 implementation correct (max error: %e)", maxError)
		}
	} else {
		t.Skip("AVX2 not available")
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}