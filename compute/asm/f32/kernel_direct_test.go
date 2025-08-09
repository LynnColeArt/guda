//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"fmt"
	"testing"
	"unsafe"
)

// TestKernelDirect tests the AVX-512 kernel directly
func TestKernelDirect(t *testing.T) {
	SetCPUFeatures(true, true)
	
	// Simple 2x2 in 16x4 tiles
	kc := 2
	
	// Pack A: 16 rows, 2 cols
	// First column: 1,2,3,4,0,0,...
	// Second column: 5,6,7,8,0,0,...
	aPack := make([]float32, 16*kc)
	for k := 0; k < kc; k++ {
		for i := 0; i < 4; i++ {
			aPack[k*16+i] = float32(i + 1 + k*4)
		}
	}
	
	// Pack B: 2 rows, 4 cols  
	// Layout: col0 values, col1 values, col2 values, col3 values
	bPack := make([]float32, kc*4)
	// Column 0: [1, 2]
	bPack[0] = 1
	bPack[1] = 2
	// Column 1: [3, 4]
	bPack[2] = 3
	bPack[3] = 4
	// Column 2: [5, 6]
	bPack[4] = 5
	bPack[5] = 6
	// Column 3: [7, 8]
	bPack[6] = 7
	bPack[7] = 8
	
	// Result C: 16x4 (but we only care about top 4x4)
	c := make([]float32, 16*4)
	
	// Call kernel directly
	aPtr := unsafe.Pointer(&aPack[0])
	bPtr := unsafe.Pointer(&bPack[0])
	cPtr := unsafe.Pointer(&c[0])
	
	fmt.Println("Before kernel:")
	fmt.Printf("aPack: %v\n", aPack[:32])
	fmt.Printf("bPack: %v\n", bPack)
	
	sgemmKernel16x4AVX512(aPtr, bPtr, cPtr, int64(kc), int64(4*4)) // ldc=4 floats * 4 bytes
	
	fmt.Println("\nAfter kernel:")
	fmt.Println("C values:")
	for i := 0; i < 4; i++ {
		fmt.Printf("Row %d: ", i)
		for j := 0; j < 16; j++ {
			if c[i*16+j] != 0 {
				fmt.Printf("%.0f ", c[i*16+j])
			} else {
				fmt.Print("_ ")
			}
		}
		fmt.Println()
	}
	
	// Expected for first 4x4:
	// C[0,0] = 1*1 + 5*2 = 11
	// C[0,1] = 1*3 + 5*4 = 23
	// C[0,2] = 1*5 + 5*6 = 35
	// C[0,3] = 1*7 + 5*8 = 47
	expected00 := float32(11)
	if c[0] != expected00 {
		t.Errorf("C[0,0] = %f, expected %f", c[0], expected00)
	}
}