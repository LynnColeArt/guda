//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"fmt"
	"testing"
)

// TestAVX512Debug helps debug the AVX-512 kernel
func TestAVX512Debug(t *testing.T) {
	// Initialize CPU features explicitly
	SetCPUFeatures(true, true) // Force AVX-512 and AVX2 for testing
	
	// First check if AVX-512 is detected
	fmt.Printf("HasAVX512Support: %v\n", HasAVX512Support)
	
	// Simple 4x4 test case
	m, n, k := 4, 4, 2
	
	// A: 4x2
	// [1 2]
	// [3 4]
	// [5 6]
	// [7 8]
	a := []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	
	// B: 2x4
	// [1 2 3 4]
	// [5 6 7 8]
	b := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	
	// Expected C = A * B
	// [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
	// [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
	// [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
	// [7*1+8*5, 7*2+8*6, 7*3+8*7, 7*4+8*8] = [47, 62, 77, 92]
	
	// Test packing first
	fmt.Println("\nTesting A packing:")
	aPacked := make([]float32, 16*k) // MR=16, so we need 16*k
	PackAMatrixAVX512(aPacked, a, 2, m, k)
	fmt.Printf("A packed: %v\n", aPacked)
	
	fmt.Println("\nTesting B packing:")
	bPacked := make([]float32, k*4) // NR=4
	PackBMatrixAVX512(bPacked, b, 4, k, n)
	fmt.Printf("B packed: %v\n", bPacked)
	
	// Now test the full GEMM
	// Make C larger to see what the kernel is actually writing
	c := make([]float32, 32*32) // Much larger than needed
	
	// Fill with sentinel values
	for i := range c {
		c[i] = -999.0
	}
	
	GemmAVX512(false, false, m, n, k, 1.0, a, 2, b, 4, 0.0, c, 4)
	
	fmt.Println("\nResult C (first 64 values):")
	for i := 0; i < 4; i++ {
		fmt.Printf("Values %d-%d: ", i*16, i*16+15)
		for j := 0; j < 16; j++ {
			if c[i*16+j] != -999.0 {
				fmt.Printf("%.0f ", c[i*16+j])
			} else {
				fmt.Print("X ")
			}
		}
		fmt.Println()
	}
	
	// Print the actual 4x4 result in standard layout
	fmt.Println("\nActual 4x4 result:")
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			fmt.Printf("%3.0f ", c[i*4+j])
		}
		fmt.Println()
	}
	
	// Check one value
	if c[0] != 11 {
		t.Errorf("C[0,0] = %f, want 11", c[0])
		
		// Debug: compute manually with packed data
		fmt.Println("\nDebug computation:")
		fmt.Printf("A[0,0]=%f, A[0,1]=%f\n", aPacked[0], aPacked[16])
		fmt.Printf("B[0,0]=%f, B[1,0]=%f\n", bPacked[0], bPacked[1])
		fmt.Printf("Manual C[0,0] = %f*%f + %f*%f = %f\n", 
			aPacked[0], bPacked[0], aPacked[16], bPacked[1],
			aPacked[0]*bPacked[0] + aPacked[16]*bPacked[1])
	}
}