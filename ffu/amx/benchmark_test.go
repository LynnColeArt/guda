//go:build amd64
// +build amd64

package amx

import (
	"fmt"
	"testing"
	"time"
	"unsafe"
)

// Benchmark matrix sizes relevant to AI workloads
var benchmarkSizes = []struct {
	name string
	m, n, k int
}{
	{"Tiny_16x16x64", 16, 16, 64},        // Single tile
	{"Small_64x64x64", 64, 64, 64},       // 4x4 tiles
	{"Medium_256x256x256", 256, 256, 256}, // 16x16 tiles
	{"Large_512x512x512", 512, 512, 512},  // 32x32 tiles
	{"Transformer_768x768x768", 768, 768, 768}, // BERT hidden size
	{"Transformer_1024x1024x1024", 1024, 1024, 1024}, // GPT-2 medium
	{"Wide_256x1024x256", 256, 1024, 256}, // Attention patterns
	{"Tall_1024x256x256", 1024, 256, 256}, // FC layers
}

// BenchmarkAMXKernel tests the AMX kernel implementation
func BenchmarkAMXKernel(b *testing.B) {
	if !HasAMX() {
		SetAMXSupport(true, true, true)
	}
	
	for _, size := range benchmarkSizes {
		b.Run(size.name, func(b *testing.B) {
			kernel := NewAMXKernel()
			defer kernel.Release()
			
			M, N, K := size.m, size.n, size.k
			
			// Allocate aligned buffers
			A := makeAlignedBuffer(M*K, 64)
			B := makeAlignedBuffer(K*N, 64)
			C := make([]int32, M*N)
			
			// Initialize with realistic data
			initializeMatrix(A, M*K)
			initializeMatrix(B, K*N)
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				err := kernel.Int8GEMM(M, N, K, A, B, C, 1.0, 0.0)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			reportMetrics(b, M, N, K)
		})
	}
}

// BenchmarkAMXPacked tests performance with packed matrices
func BenchmarkAMXPacked(b *testing.B) {
	if !HasAMX() {
		SetAMXSupport(true, true, true)
	}
	
	for _, size := range benchmarkSizes {
		b.Run(size.name, func(b *testing.B) {
			kernel := NewAMXKernel()
			defer kernel.Release()
			
			M, N, K := size.m, size.n, size.k
			
			// Original matrices
			A := make([]int8, M*K)
			B := make([]int8, K*N)
			C := make([]int32, M*N)
			
			// Packed matrices (with padding)
			packedA := make([]int8, M*K+1024)
			packedB := make([]int8, K*N+1024)
			
			// Initialize
			for i := range A {
				A[i] = int8(i % 127)
			}
			for i := range B {
				B[i] = int8(i % 127)
			}
			
			// Pack matrices
			PackA(M, K, A, packedA)
			PackB(K, N, B, packedB)
			
			// Convert to byte slices
			aBytes := (*[1 << 30]byte)(unsafe.Pointer(&packedA[0]))[:len(packedA)]
			bBytes := (*[1 << 30]byte)(unsafe.Pointer(&packedB[0]))[:len(packedB)]
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				err := kernel.Int8GEMM(M, N, K, aBytes, bBytes, C, 1.0, 0.0)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			reportMetrics(b, M, N, K)
		})
	}
}

// BenchmarkReferenceVsAMX compares reference and AMX implementations
func BenchmarkReferenceVsAMX(b *testing.B) {
	sizes := []struct {
		name string
		m, n, k int
	}{
		{"64x64x64", 64, 64, 64},
		{"256x256x256", 256, 256, 256},
	}
	
	for _, size := range sizes {
		M, N, K := size.m, size.n, size.k
		
		// Prepare data
		A := makeAlignedBuffer(M*K, 64)
		B := makeAlignedBuffer(K*N, 64)
		C := make([]int32, M*N)
		
		initializeMatrix(A, M*K)
		initializeMatrix(B, K*N)
		
		// Benchmark reference implementation
		b.Run(fmt.Sprintf("Reference_%s", size.name), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				amxInt8GEMMReference(M, N, K, A, B, C)
			}
			reportMetrics(b, M, N, K)
		})
		
		// Benchmark AMX implementation
		b.Run(fmt.Sprintf("AMX_%s", size.name), func(b *testing.B) {
			if !HasAMX() {
				SetAMXSupport(true, true, true)
			}
			
			kernel := NewAMXKernel()
			defer kernel.Release()
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				err := kernel.Int8GEMM(M, N, K, A, B, C, 1.0, 0.0)
				if err != nil {
					b.Fatal(err)
				}
			}
			reportMetrics(b, M, N, K)
		})
	}
}

// BenchmarkMemoryBandwidth measures memory bandwidth utilization
func BenchmarkMemoryBandwidth(b *testing.B) {
	sizes := []int{64, 128, 256, 512, 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			M, N, K := size, size, size
			
			A := makeAlignedBuffer(M*K, 64)
			B := makeAlignedBuffer(K*N, 64)
			C := make([]int32, M*N)
			
			initializeMatrix(A, M*K)
			initializeMatrix(B, K*N)
			
			kernel := NewAMXKernel()
			defer kernel.Release()
			
			b.ResetTimer()
			
			start := time.Now()
			for i := 0; i < b.N; i++ {
				err := kernel.Int8GEMM(M, N, K, A, B, C, 1.0, 0.0)
				if err != nil {
					b.Fatal(err)
				}
			}
			elapsed := time.Since(start)
			
			// Calculate bandwidth
			bytesRead := int64(M*K + K*N) * int64(b.N)
			bytesWritten := int64(M*N*4) * int64(b.N)
			totalBytes := bytesRead + bytesWritten
			bandwidth := float64(totalBytes) / elapsed.Seconds() / 1e9
			
			b.ReportMetric(bandwidth, "GB/s")
			reportMetrics(b, M, N, K)
		})
	}
}

// Helper functions

func initializeMatrix(buf []byte, size int) {
	for i := 0; i < size; i++ {
		// Realistic distribution: mostly small values
		buf[i] = byte(i % 63)
	}
}

func reportMetrics(b *testing.B, M, N, K int) {
	ops := int64(2 * M * N * K) // Multiply-add = 2 ops
	bytes := int64(M*K + K*N + M*N*4) // A + B + C (INT32)
	
	b.SetBytes(bytes)
	
	// Report GOPS
	gops := float64(ops*int64(b.N)) / b.Elapsed().Seconds() / 1e9
	b.ReportMetric(gops, "GOPS")
	
	// Report arithmetic intensity
	flopsPerByte := float64(ops) / float64(bytes)
	b.ReportMetric(flopsPerByte, "FLOPS/byte")
}

