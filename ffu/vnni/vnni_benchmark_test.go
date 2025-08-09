package vnni

import (
	"testing"
	
	"github.com/LynnColeArt/guda/ffu"
)

// Scalar INT8 GEMM for comparison
func scalarInt8GEMM(m, n, k int, a, b []int8, c []int32) {
	// Clear C
	for i := range c {
		c[i] = 0
	}
	
	// Triple nested loop
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := int32(0)
			for kk := 0; kk < k; kk++ {
				sum += int32(a[i*k+kk]) * int32(b[kk*n+j])
			}
			c[i*n+j] = sum
		}
	}
}

// BenchmarkVNNIProgression shows performance progression
func BenchmarkVNNIProgression(b *testing.B) {
	M, N, K := 256, 256, 256
	
	A := make([]int8, M*K)
	B := make([]int8, K*N)
	C := make([]int32, M*N)
	
	// Initialize
	for i := range A {
		A[i] = int8(i % 127)
	}
	for i := range B {
		B[i] = int8(i % 127)
	}
	
	// 1. Scalar Go
	b.Run("1_Scalar_Go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			scalarInt8GEMM(M, N, K, A, B, C)
		}
		reportBenchMetrics(b, M, N, K)
	})
	
	// 2. Assembly Reference
	b.Run("2_Assembly_Ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			vnniInt8GEMMRef(M, N, K, A, B, C)
		}
		reportBenchMetrics(b, M, N, K)
	})
	
	// 3. VNNI FFU
	b.Run("3_VNNI_FFU", func(b *testing.B) {
		vnniFFU := NewVNNIFFU()
		workload := &ffu.VNNIWorkload{
			Operation: ffu.VNNIMatMul,
			M: M, N: N, K: K,
			A: A, B: B, C: C,
			Alpha: 1,
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			err := vnniFFU.Execute(workload)
			if err != nil {
				b.Fatal(err)
			}
		}
		reportBenchMetrics(b, M, N, K)
	})
	
	// 4. Theoretical VNNI Peak
	b.Run("4_Theoretical_300GOPS", func(b *testing.B) {
		ops := int64(2 * M * N * K)
		theoreticalTime := float64(ops) / 300e9 // 300 GOPS
		
		b.ReportMetric(300.0, "GOPS")
		b.ReportMetric(theoreticalTime*1e9, "ns/op")
	})
}

// BenchmarkVNNISizes tests different matrix sizes
func BenchmarkVNNISizes(b *testing.B) {
	sizes := []struct {
		name string
		m, n, k int
	}{
		{"Tiny_16x16x64", 16, 16, 64},
		{"Small_64x64x128", 64, 64, 128},
		{"Medium_256x256x256", 256, 256, 256},
		{"Large_512x512x512", 512, 512, 512},
		{"Wide_128x512x128", 128, 512, 128},
		{"Tall_512x128x128", 512, 128, 128},
		{"Transformer_768x768x768", 768, 768, 768},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			A := make([]int8, size.m*size.k)
			B := make([]int8, size.k*size.n)
			C := make([]int32, size.m*size.n)
			
			// Initialize
			for i := range A {
				A[i] = int8(i % 63)
			}
			for i := range B {
				B[i] = int8(i % 63)
			}
			
			vnniFFU := NewVNNIFFU()
			workload := &ffu.VNNIWorkload{
				Operation: ffu.VNNIMatMul,
				M: size.m, N: size.n, K: size.k,
				A: A, B: B, C: C,
				Alpha: 1,
			}
			
			// Adjust K to be divisible by 16 if needed
			if workload.K < 64 || workload.K%16 != 0 {
				b.Skipf("K=%d not suitable for VNNI", workload.K)
			}
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				err := vnniFFU.Execute(workload)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			reportBenchMetrics(b, size.m, size.n, size.k)
		})
	}
}

func reportBenchMetrics(b *testing.B, m, n, k int) {
	ops := int64(2 * m * n * k)
	bytes := int64(m*k + k*n + m*n*4)
	
	b.SetBytes(bytes)
	
	gops := float64(ops*int64(b.N)) / b.Elapsed().Seconds() / 1e9
	b.ReportMetric(gops, "GOPS")
	
	flopsPerByte := float64(ops) / float64(bytes)
	b.ReportMetric(flopsPerByte, "FLOPS/byte")
}