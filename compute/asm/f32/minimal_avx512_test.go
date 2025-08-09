//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"fmt"
	"testing"
	"unsafe"
)

// TestMinimalAVX512 tests the simplest possible case
func TestMinimalAVX512(t *testing.T) {
	SetCPUFeatures(true, true)
	
	// Just 2x2 multiply: C = A * B
	// A = [1 2]
	//     [3 4]
	// B = [5 6]
	//     [7 8]
	// C = [19 22]
	//     [43 50]
	
	kc := 2
	
	// Pack A (16x2, with only first 2 rows used)
	aPack := make([]float32, 16*kc)
	// Column 0 of A: 1, 3, 0, 0, ...
	aPack[0] = 1
	aPack[1] = 3
	// Column 1 of A: 2, 4, 0, 0, ...
	aPack[16] = 2
	aPack[17] = 4
	
	// Pack B (2x4, with only first 2 cols used)
	bPack := make([]float32, kc*4)
	// Column 0 of B: 5, 7
	bPack[0] = 5
	bPack[1] = 7
	// Column 1 of B: 6, 8
	bPack[2] = 6
	bPack[3] = 8
	
	// Result C
	c := make([]float32, 64)
	
	fmt.Println("Packed A (first 32):", aPack[:32])
	fmt.Println("Packed B:", bPack)
	
	// Call kernel
	sgemmKernel16x4AVX512(
		unsafe.Pointer(&aPack[0]),
		unsafe.Pointer(&bPack[0]),
		unsafe.Pointer(&c[0]),
		int64(kc),
		int64(16), // ldc = 4 floats * 4 bytes = 16 bytes
	)
	
	fmt.Println("\nResult (first 16):", c[:16])
	
	// With the current column-major storage in first row:
	// We expect: [19, 43, 0, 0, 22, 50, 0, 0, ...]
	//            C00 C10 C20 C30 C01 C11 C21 C31
	
	if c[0] != 19 {
		t.Errorf("C[0,0] = %f, expected 19", c[0])
	}
	if c[1] != 43 {
		t.Errorf("C[1,0] = %f, expected 43", c[1])
	}
	if c[4] != 22 {
		t.Errorf("C[0,1] = %f, expected 22", c[4])
	}
	if c[5] != 50 {
		t.Errorf("C[1,1] = %f, expected 50", c[5])
	}
}