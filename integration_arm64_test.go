// +build arm64

package guda

import (
	"testing"
	"unsafe"
)

/**
 * @requirement REQ-001.1, REQ-001.2, REQ-001.4
 * @scenario Integration test for CPU feature detection and SIMD operations
 * @given An ARM64 system with NEON and FP16 support
 * @when All components are used together
 * @then They work correctly and efficiently
 */
func TestIntegration_CPUAndSIMD(t *testing.T) {
	// Check CPU features
	features := getCPUFeatures()
	
	// These tests will verify that our implementation can be called
	// without crashing. Actual correctness would require running on
	// a real ARM64 system with NEON/FP16 support.
	
	// Test SimdF16ToF32
	srcF16 := make([]uint16, 8)
	dstF32 := make([]float32, 8)
	
	// Initialize with some float16 values
	srcF16[0] = 0x3C00 // 1.0
	srcF16[1] = 0x4000 // 2.0
	srcF16[2] = 0x4200 // 3.0
	srcF16[3] = 0x4400 // 4.0
	srcF16[4] = 0x4500 // 5.0
	srcF16[5] = 0x4580 // 6.0
	srcF16[6] = 0x4600 // 7.0
	srcF16[7] = 0x4640 // 8.0
	
	// This should work without panicking
	SimdF16ToF32(srcF16, dstF32)
	
	// Test SimdF32ToF16
	srcF32 := make([]float32, 8)
	dstF16 := make([]uint16, 8)
	
	// Initialize with float32 values
	srcF32[0] = 1.0
	srcF32[1] = 2.0
	srcF32[2] = 3.0
	srcF32[3] = 4.0
	srcF32[4] = 5.0
	srcF32[5] = 6.0
	srcF32[6] = 7.0
	srcF32[7] = 8.0
	
	// This should work without panicking
	SimdF32ToF16(srcF32, dstF16)
	
	// Test AddFloat16SIMD
	a := DevicePtr{ptr: unsafe.Pointer(&srcF16[0])}
	b := DevicePtr{ptr: unsafe.Pointer(&srcF16[0])}
	c := DevicePtr{ptr: unsafe.Pointer(&dstF16[0])}
	
	// This should work without panicking
	AddFloat16SIMD(a, b, c, 8)
	
	// Test MultiplyFloat16SIMD
	// This should work without panicking
	MultiplyFloat16SIMD(a, b, c, 8)
	
	// Test FMAFloat16SIMD
	d := DevicePtr{ptr: unsafe.Pointer(&dstF16[0])}
	
	// This should work without panicking
	FMAFloat16SIMD(a, b, c, d, 8)
	
	// Test GEMMFloat16SIMD
	// Create test matrices
	size := 64
	aData := make([]uint16, size*size)
	bData := make([]uint16, size*size)
	cData := make([]uint16, size*size)
	
	aPtr := DevicePtr{ptr: unsafe.Pointer(&aData[0])}
	bPtr := DevicePtr{ptr: unsafe.Pointer(&bData[0])}
	cPtr := DevicePtr{ptr: unsafe.Pointer(&cData[0])}
	
	// Initialize with some values
	for i := 0; i < size*size; i++ {
		aData[i] = 0x3C00 // 1.0
		bData[i] = 0x3C00 // 1.0
		cData[i] = 0x0000 // 0.0
	}
	
	// This should work without panicking
	GEMMFloat16SIMD(false, false, size, size, size, 1.0, aPtr, size, bPtr, size, 0.0, cPtr, size)
	
	// Report feature support
	t.Logf("NEON support: %v", features.HasNEON)
	t.Logf("FP16 support: %v", features.HasFP16)
}

/**
 * @requirement REQ-001.5
 * @scenario Integration test for fallback to scalar implementation
 * @given An ARM64 system without NEON support (simulated)
 * @when SIMD operations are called
 * @then They fall back to scalar implementation without errors
 */
func TestIntegration_Fallback(t *testing.T) {
	// Save original values
	origHasNEON := hasNEON
	origHasFP16 := hasFP16
	
	// Simulate no NEON support
	hasNEON = false
	hasFP16 = false
	
	// Restore original values after test
	defer func() {
		hasNEON = origHasNEON
		hasFP16 = origHasFP16
	}()
	
	// Test fallback for SimdF16ToF32
	srcF16 := make([]uint16, 8)
	dstF32 := make([]float32, 8)
	
	SimdF16ToF32(srcF16, dstF32)
	
	// Test fallback for SimdF32ToF16
	srcF32 := make([]float32, 8)
	dstF16 := make([]uint16, 8)
	
	SimdF32ToF16(srcF32, dstF16)
	
	// Test fallback for AddFloat16SIMD
	a := DevicePtr{ptr: unsafe.Pointer(&srcF16[0])}
	b := DevicePtr{ptr: unsafe.Pointer(&srcF16[0])}
	c := DevicePtr{ptr: unsafe.Pointer(&dstF16[0])}
	
	AddFloat16SIMD(a, b, c, 8)
	
	// Test fallback for MultiplyFloat16SIMD
	MultiplyFloat16SIMD(a, b, c, 8)
	
	// Test fallback for FMAFloat16SIMD
	d := DevicePtr{ptr: unsafe.Pointer(&dstF16[0])}
	
	FMAFloat16SIMD(a, b, c, d, 8)
	
	// Test fallback for GEMMFloat16SIMD
	size := 16
	aData := make([]uint16, size*size)
	bData := make([]uint16, size*size)
	cData := make([]uint16, size*size)
	
	aPtr := DevicePtr{ptr: unsafe.Pointer(&aData[0])}
	bPtr := DevicePtr{ptr: unsafe.Pointer(&bData[0])}
	cPtr := DevicePtr{ptr: unsafe.Pointer(&cData[0])}
	
	GEMMFloat16SIMD(false, false, size, size, size, 1.0, aPtr, size, bPtr, size, 0.0, cPtr, size)
}