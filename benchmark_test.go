package guda

import (
	"fmt"
	"testing"
)

// Benchmark memory bandwidth
func BenchmarkMemoryBandwidth(b *testing.B) {
	sizes := []int{
		1 << 10,  // 1KB
		L1CacheSize,  // 32KB (L1 cache)
		L2CacheSize,  // 256KB (L2 cache)
		L3CacheSize,  // 8MB (L3 cache)
		1 << 26,  // 64MB (RAM)
		1 << 28,  // 256MB (RAM)
	}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Copy_%s", formatBytes(size)), func(b *testing.B) {
			src, _ := Malloc(size)
			dst, _ := Malloc(size)
			defer Free(src)
			defer Free(dst)
			
			b.SetBytes(int64(size * 2)) // Read + Write
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				Memcpy(dst, src, size, MemcpyDeviceToDevice)
			}
		})
	}
}

// Benchmark AXPY at different sizes
func BenchmarkAXPY(b *testing.B) {
	sizes := []int{1024, 16384, 262144, 1048576, 16777216}
	
	for _, N := range sizes {
		b.Run(fmt.Sprintf("N_%d", N), func(b *testing.B) {
			d_X, _ := Malloc(N * 4)
			d_Y, _ := Malloc(N * 4)
			defer Free(d_X)
			defer Free(d_Y)
			
			alpha := float32(2.5)
			
			b.SetBytes(int64(3 * N * 4)) // Read X, Read Y, Write Y
			
			// Use performance counter integration
			IntegratePerfCounters(b, fmt.Sprintf("AXPY_%d", N), func() {
				AXPY(alpha, d_X, d_Y, N)
				Synchronize()
			})
			
			// Report GFLOPS per operation (correct calculation)
			flops := float64(2 * N) // multiply + add per operation
			timePerOp := b.Elapsed().Seconds() / float64(b.N)
			gflopsPerOp := flops / timePerOp / 1e9
			b.ReportMetric(gflopsPerOp, "GFLOPS(hot-cache)")
			
			// Report arithmetic intensity
			arithmeticIntensity := flops / float64(3*N*4) // FLOPS per byte
			b.ReportMetric(arithmeticIntensity, "FLOPS/byte")
		})
	}
}

// Benchmark DOT product
func BenchmarkDOT(b *testing.B) {
	sizes := []int{1024, 16384, 262144, 1048576}
	
	for _, N := range sizes {
		b.Run(fmt.Sprintf("N_%d", N), func(b *testing.B) {
			d_X, _ := Malloc(N * 4)
			d_Y, _ := Malloc(N * 4)
			defer Free(d_X)
			defer Free(d_Y)
			
			b.SetBytes(int64(2 * N * 4)) // Read X and Y
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				DOT(d_X, d_Y, N)
			}
			
			// Report GFLOPS per operation (correct calculation)
			flops := float64(2 * N) // multiply + add per operation
			timePerOp := b.Elapsed().Seconds() / float64(b.N)
			gflopsPerOp := flops / timePerOp / 1e9
			b.ReportMetric(gflopsPerOp, "GFLOPS(hot-cache)")
		})
	}
}

// Benchmark GEMM
func BenchmarkGEMM(b *testing.B) {
	sizes := []int{128, 256, 512, 1024, 2048, 4096}
	
	for _, N := range sizes {
		b.Run(fmt.Sprintf("N_%d", N), func(b *testing.B) {
			d_A, _ := Malloc(N * N * 4)
			d_B, _ := Malloc(N * N * 4)
			d_C, _ := Malloc(N * N * 4)
			defer Free(d_A)
			defer Free(d_B)
			defer Free(d_C)
			
			// Initialize with test data for consistent results
			InitTestData(d_A.Float32(), 1.0)
			InitTestData(d_B.Float32(), 1.0)
			
			b.SetBytes(int64(3 * N * N * 4)) // Simplified
			
			// Use performance counter integration
			IntegratePerfCounters(b, fmt.Sprintf("GEMM_%dx%dx%d", N, N, N), func() {
				GEMM(false, false, N, N, N, 1.0, d_A, N, d_B, N, 0.0, d_C, N)
				Synchronize()
			})
			
			// Report GFLOPS per operation (correct calculation)
			flops := float64(2 * N * N * N) // multiply-add per GEMM
			timePerOp := b.Elapsed().Seconds() / float64(b.N)
			gflopsPerOp := flops / timePerOp / 1e9
			
			
			b.ReportMetric(gflopsPerOp, "GFLOPS(hot-cache)")
			
			// Report efficiency vs realistic theoretical peak
			// AMD Ryzen 7 7700X: 8 cores × 4.5 GHz × 8 FP32 ops/cycle (AVX2) = ~288 GFLOPS theoretical
			// Practical peak for GEMM is usually 30-50% of theoretical = ~100 GFLOPS
			practicalPeak := 100.0
			efficiency := gflopsPerOp / practicalPeak * 100
			b.ReportMetric(efficiency, "efficiency_%")
			
			// Report arithmetic intensity
			bytes := float64(3 * N * N * 4)
			arithmeticIntensity := flops / bytes
			b.ReportMetric(arithmeticIntensity, "FLOPS/byte")
			
			// Roofline analysis
			peakBandwidth := 50.0 // GB/s typical DDR4
			memoryBound := peakBandwidth * arithmeticIntensity
			achievable := min(gflopsPerOp, memoryBound)
			if memoryBound < gflopsPerOp {
				b.ReportMetric(1.0, "memory_bound")
			} else {
				b.ReportMetric(0.0, "compute_bound")
			}
		})
	}
}

// Benchmark kernel launch overhead
func BenchmarkKernelLaunchOverhead(b *testing.B) {
	// Empty kernel to measure pure launch overhead
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		// Do nothing
	})
	
	gridSizes := []int{1, 10, 100, 1000}
	
	for _, gridSize := range gridSizes {
		b.Run(fmt.Sprintf("Grid_%d", gridSize), func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				Launch(kernel, Dim3{X: gridSize, Y: 1, Z: 1}, Dim3{X: DefaultBlockSize, Y: 1, Z: 1})
				Synchronize()
			}
			
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "launches/sec")
		})
	}
}

// Benchmark fusion benefits
func BenchmarkFusionSpeedup(b *testing.B) {
	N := 1 << 20 // 1M elements
	
	d_X, _ := Malloc(N * 4)
	d_Y, _ := Malloc(N * 4)
	d_Z, _ := Malloc(N * 4)
	defer Free(d_X)
	defer Free(d_Y)
	defer Free(d_Z)
	
	// Benchmark separate operations
	b.Run("Separate_3ops", func(b *testing.B) {
		b.SetBytes(int64(9 * N * 4)) // 3 ops, each 3N bytes
		
		for i := 0; i < b.N; i++ {
			Add(d_X, d_Y, d_Z, N)      // Z = X + Y
			Scale(2.0, d_Z, N)         // Z = 2 * Z
			ReLU(d_Z, N)               // Z = ReLU(Z)
			Synchronize()
		}
	})
	
	// Benchmark fused operation
	b.Run("Fused_3ops", func(b *testing.B) {
		b.SetBytes(int64(3 * N * 4)) // Read X,Y, Write Z
		
		kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
			idx := tid.Global()
			if idx < N {
				x := d_X.Float32()[idx]
				y := d_Y.Float32()[idx]
				z := 2.0 * (x + y)
				if z < 0 {
					z = 0
				}
				d_Z.Float32()[idx] = z
			}
		})
		
		for i := 0; i < b.N; i++ {
			Launch(kernel, 
				Dim3{X: (N + DefaultBlockSize - 1) / DefaultBlockSize, Y: 1, Z: 1},
				Dim3{X: DefaultBlockSize, Y: 1, Z: 1})
			Synchronize()
		}
	})
}

// Benchmark parallel scaling
func BenchmarkParallelScaling(b *testing.B) {
	N := 1 << 24 // 16M elements
	
	d_X, _ := Malloc(N * 4)
	d_Y, _ := Malloc(N * 4)
	defer Free(d_X)
	defer Free(d_Y)
	
	// Vary the grid size to control parallelism
	blockSize := DefaultBlockSize
	maxGridSize := (N + blockSize - 1) / blockSize
	
	gridSizes := []int{1, 2, 4, 8, 16, 32, 64, maxGridSize}
	
	for _, gridSize := range gridSizes {
		if gridSize > maxGridSize {
			break
		}
		
		b.Run(fmt.Sprintf("GridSize_%d", gridSize), func(b *testing.B) {
			actualWork := min(gridSize*blockSize, N)
			b.SetBytes(int64(3 * actualWork * 4))
			
			kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
				idx := tid.Global()
				if idx < actualWork {
					d_Y.Float32()[idx] = d_X.Float32()[idx] * 2.0
				}
			})
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				Launch(kernel,
					Dim3{X: gridSize, Y: 1, Z: 1},
					Dim3{X: blockSize, Y: 1, Z: 1})
				Synchronize()
			}
			
			// Report throughput
			throughput := float64(actualWork) * float64(b.N) / b.Elapsed().Seconds() / 1e6
			b.ReportMetric(throughput, "Melems/sec")
		})
	}
}

// Helper to format bytes
func formatBytes(bytes int) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%dB", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%d%cB", bytes/int(div), "KMGTPE"[exp])
}

// min returns the minimum of two values
func min(a, b interface{}) interface{} {
	switch a := a.(type) {
	case int:
		if b := b.(int); a < b {
			return a
		}
		return b
	case float64:
		if b := b.(float64); a < b {
			return a
		}
		return b
	default:
		panic("unsupported type")
	}
}