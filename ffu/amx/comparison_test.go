package amx

import (
	"fmt"
	"testing"
	"unsafe"
	
	"github.com/LynnColeArt/guda/ffu"
)

// BenchmarkProgressionToAMX shows the performance progression
func BenchmarkProgressionToAMX(b *testing.B) {
	M, N, K := 256, 256, 256
	
	// Prepare data
	A := make([]byte, M*K)
	B := make([]byte, K*N)
	C := make([]int32, M*N)
	
	for i := range A {
		A[i] = byte(i % 127)
	}
	for i := range B {
		B[i] = byte(i % 127)
	}
	
	// 1. Scalar implementation
	b.Run("1_Scalar_Go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			scalarInt8GEMM(M, N, K, A, B, C)
		}
		reportMetrics(b, M, N, K)
	})
	
	// 2. Assembly reference (optimized loops)
	b.Run("2_Assembly_Reference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			amxInt8GEMMReference(M, N, K, A, B, C)
		}
		reportMetrics(b, M, N, K)
	})
	
	// 3. AMX with FFU framework
	b.Run("3_AMX_FFU", func(b *testing.B) {
		SetAMXSupport(true, true, true)
		amxFFU := NewAMXFFU()
		
		// Create workload
		workload := &ffu.AMXWorkload{
			Operation: ffu.AMXMatMul,
			DataType:  ffu.AMXInt8,
			M:         M,
			N:         N,
			K:         K,
			A:         A,
			B:         B,
			C:         (*[1 << 30]byte)(unsafe.Pointer(&C[0]))[:M*N*4],
			ScaleA:    1.0,
			ScaleB:    1.0,
			ScaleC:    1.0,
		}
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			err := amxFFU.Execute(workload)
			if err != nil {
				b.Fatal(err)
			}
		}
		reportMetrics(b, M, N, K)
	})
	
	// 4. Theoretical AMX peak (for comparison)
	b.Run("4_Theoretical_AMX_2TOPS", func(b *testing.B) {
		// Just report theoretical peak
		ops := int64(2 * M * N * K)
		theoreticalTime := float64(ops) / 2e12 // 2 TOPS
		
		b.ReportMetric(2000.0, "GOPS")
		b.ReportMetric(theoreticalTime*1e9, "ns/op")
	})
}

// scalarInt8GEMM is a simple scalar implementation for comparison
func scalarInt8GEMM(M, N, K int, A, B []byte, C []int32) {
	// Clear C
	for i := range C {
		C[i] = 0
	}
	
	// Triple nested loop
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := int32(0)
			for k := 0; k < K; k++ {
				aVal := int32(int8(A[i*K+k]))
				bVal := int32(int8(B[k*N+j]))
				sum += aVal * bVal
			}
			C[i*N+j] = sum
		}
	}
}

// BenchmarkFFUOverhead measures the overhead of the FFU framework
func BenchmarkFFUOverhead(b *testing.B) {
	sizes := []int{64, 256, 1024}
	
	for _, size := range sizes {
		M, N, K := size, size, size
		
		// Prepare aligned data
		A := makeAlignedBuffer(M*K, 64)
		B := makeAlignedBuffer(K*N, 64)
		C := make([]int32, M*N)
		
		initializeMatrix(A, M*K)
		initializeMatrix(B, K*N)
		
		// Direct kernel call
		b.Run(fmt.Sprintf("Direct_%dx%d", size, size), func(b *testing.B) {
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
		
		// Through FFU framework
		b.Run(fmt.Sprintf("FFU_%dx%d", size, size), func(b *testing.B) {
			SetAMXSupport(true, true, true)
			
			registry := ffu.NewRegistry()
			amxFFU := NewAMXFFU()
			registry.Register(amxFFU)
			
			workload := &ffu.AMXWorkload{
				Operation: ffu.AMXMatMul,
				DataType:  ffu.AMXInt8,
				M:         M,
				N:         N,
				K:         K,
				A:         A,
				B:         B,
				C:         (*[1 << 30]byte)(unsafe.Pointer(&C[0]))[:M*N*4],
				ScaleA:    1.0,
				ScaleB:    1.0,
				ScaleC:    1.0,
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bestFFU, _ := registry.FindBest(workload)
				err := bestFFU.Execute(workload)
				if err != nil {
					b.Fatal(err)
				}
			}
			reportMetrics(b, M, N, K)
		})
	}
}

