// +build arm64

package guda

import (
	"testing"
	"unsafe"
)

// BenchmarkSimdF16ToF32NEON benchmarks the ARM64 NEON implementation for float16 to float32 conversion
func BenchmarkSimdF16ToF32NEON(b *testing.B) {
	// Create test data
	size := 1024
	src := make([]uint16, size)
	dst := make([]float32, size)
	
	// Initialize with some values
	for i := 0; i < size; i++ {
		src[i] = 0x3C00 // 1.0 in float16
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdF16ToF32NEON(unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), size)
	}
}

// BenchmarkSimdF32ToF16NEON benchmarks the ARM64 NEON implementation for float32 to float16 conversion
func BenchmarkSimdF32ToF16NEON(b *testing.B) {
	// Create test data
	size := 1024
	src := make([]float32, size)
	dst := make([]uint16, size)
	
	// Initialize with some values
	for i := 0; i < size; i++ {
		src[i] = 1.0
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdF32ToF16NEON(unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), size)
	}
}

// BenchmarkSimdAddFloat16NEON benchmarks the ARM64 NEON implementation for float16 vector addition
func BenchmarkSimdAddFloat16NEON(b *testing.B) {
	// Create test data
	size := 1024
	a := make([]uint16, size)
	bv := make([]uint16, size)
	c := make([]uint16, size)
	
	// Initialize with some values
	for i := 0; i < size; i++ {
		a[i] = 0x3C00 // 1.0 in float16
		bv[i] = 0x4000 // 2.0 in float16
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdAddFloat16NEON(unsafe.Pointer(&a[0]), unsafe.Pointer(&bv[0]), unsafe.Pointer(&c[0]), size)
	}
}

// BenchmarkSimdMulFloat16NEON benchmarks the ARM64 NEON implementation for float16 vector multiplication
func BenchmarkSimdMulFloat16NEON(b *testing.B) {
	// Create test data
	size := 1024
	a := make([]uint16, size)
	bv := make([]uint16, size)
	c := make([]uint16, size)
	
	// Initialize with some values
	for i := 0; i < size; i++ {
		a[i] = 0x3C00 // 1.0 in float16
		bv[i] = 0x4000 // 2.0 in float16
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdMulFloat16NEON(unsafe.Pointer(&a[0]), unsafe.Pointer(&bv[0]), unsafe.Pointer(&c[0]), size)
	}
}

// BenchmarkSimdFMAFloat16NEON benchmarks the ARM64 NEON implementation for float16 fused multiply-add
func BenchmarkSimdFMAFloat16NEON(b *testing.B) {
	// Create test data
	size := 1024
	a := make([]uint16, size)
	bv := make([]uint16, size)
	c := make([]uint16, size)
	d := make([]uint16, size)
	
	// Initialize with some values
	for i := 0; i < size; i++ {
		a[i] = 0x3C00 // 1.0 in float16
		bv[i] = 0x4000 // 2.0 in float16
		c[i] = 0x4200 // 3.0 in float16
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdFMAFloat16NEON(unsafe.Pointer(&a[0]), unsafe.Pointer(&bv[0]), unsafe.Pointer(&c[0]), unsafe.Pointer(&d[0]), size)
	}
}