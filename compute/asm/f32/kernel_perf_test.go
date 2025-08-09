//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"fmt"
	"testing"
	"unsafe"
)

// BenchmarkKernel16x4 benchmarks just the AVX-512 kernel
func BenchmarkKernel16x4(b *testing.B) {
	SetCPUFeatures(true, true)
	
	if !HasAVX512Support {
		b.Skip("AVX-512 not supported")
	}
	
	// Test different K sizes
	kSizes := []int{64, 128, 256, 512}
	
	for _, kc := range kSizes {
		b.Run(fmt.Sprintf("K%d", kc), func(b *testing.B) {
			// Prepare packed data
			aPack := make([]float32, 16*kc)
			bPack := make([]float32, kc*4)
			c := make([]float32, 16*4)
			
			// Initialize with data
			for i := range aPack {
				aPack[i] = 1.0
			}
			for i := range bPack {
				bPack[i] = 1.0
			}
			
			aPtr := unsafe.Pointer(&aPack[0])
			bPtr := unsafe.Pointer(&bPack[0])
			cPtr := unsafe.Pointer(&c[0])
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				sgemmKernel16x4AVX512(aPtr, bPtr, cPtr, int64(kc), 16)
			}
			
			// Calculate kernel GFLOPS
			flops := int64(2 * 16 * 4 * kc) // 2 ops per FMA
			gflops := float64(flops) * float64(b.N) / b.Elapsed().Seconds() / 1e9
			b.ReportMetric(gflops, "kernel-GFLOPS")
		})
	}
}

// BenchmarkPackingOverhead measures packing cost
func BenchmarkPackingOverhead(b *testing.B) {
	m, k := 128, 128
	
	b.Run("PackA", func(b *testing.B) {
		src := make([]float32, m*k)
		dst := make([]float32, ((m+15)/16)*16*k)
		
		for i := range src {
			src[i] = float32(i)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			PackAMatrixAVX512(dst, src, k, m, k)
		}
		
		bytes := int64(m * k * 4) // float32
		b.SetBytes(bytes)
	})
	
	b.Run("PackB", func(b *testing.B) {
		src := make([]float32, k*m)
		dst := make([]float32, ((m+3)/4)*4*k)
		
		for i := range src {
			src[i] = float32(i)
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			PackBMatrixAVX512(dst, src, m, k, m)
		}
		
		bytes := int64(k * m * 4)
		b.SetBytes(bytes)
	})
}