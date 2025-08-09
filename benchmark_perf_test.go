package guda

import (
	"fmt"
	"testing"
)

// BenchmarkGEMMWithPerfCounters demonstrates performance counter integration
func BenchmarkGEMMWithPerfCounters(b *testing.B) {
	sizes := []struct {
		name string
		m, n, k int
	}{
		{"Small_128", 128, 128, 128},
		{"Medium_512", 512, 512, 512},
		{"Large_1024", 1024, 1024, 1024},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Allocate matrices
			da, _ := Malloc(size.m * size.k * 4)
			db, _ := Malloc(size.k * size.n * 4)
			dc, _ := Malloc(size.m * size.n * 4)
			defer Free(da)
			defer Free(db)
			defer Free(dc)
			
			// Initialize with test data
			for i := range da.Float32() {
				da.Float32()[i] = 1.0
			}
			for i := range db.Float32() {
				db.Float32()[i] = 1.0
			}
			
			// Warm up
			GEMM(false, false, size.m, size.n, size.k, 1.0, da, size.k, db, size.n, 0.0, dc, size.n)
			Synchronize()
			
			// Benchmark with counters
			b.ResetTimer()
			
			// Run benchmark and measure
			BenchmarkWithCounters(b, fmt.Sprintf("GEMM_%dx%dx%d", size.m, size.n, size.k), func() {
				GEMM(false, false, size.m, size.n, size.k, 1.0, da, size.k, db, size.n, 0.0, dc, size.n)
				Synchronize()
			})
			
			// Calculate theoretical metrics
			flops := uint64(2) * uint64(size.m) * uint64(size.n) * uint64(size.k) // 2 ops per multiply-add
			bytes := uint64(size.m*size.k + size.k*size.n + size.m*size.n) * 4     // float32 = 4 bytes
			
			// Report GFLOPS
			flopsPerOp := float64(flops)
			seconds := b.Elapsed().Seconds() / float64(b.N)
			gflops := flopsPerOp / (seconds * 1e9)
			b.ReportMetric(gflops, "GFLOPS")
			
			// Report arithmetic intensity
			arithmeticIntensity := float64(flops) / float64(bytes)
			b.ReportMetric(arithmeticIntensity, "FLOPS/byte")
			
			// Report memory bandwidth (theoretical)
			bandwidth := float64(bytes) / (seconds * 1e9) // GB/s
			b.ReportMetric(bandwidth, "GB/s")
		})
	}
}

// BenchmarkReductionWithPerfCounters measures reduction operations
func BenchmarkReductionWithPerfCounters(b *testing.B) {
	sizes := []int{1024, 16384, 262144, 1048576} // 1K, 16K, 256K, 1M elements
	
	for _, n := range sizes {
		b.Run(fmt.Sprintf("Sum_%d", n), func(b *testing.B) {
			// Allocate data
			d_data, _ := Malloc(n * 4)
			defer Free(d_data)
			
			// Initialize
			for i := range d_data.Float32() {
				d_data.Float32()[i] = 1.0
			}
			
			// Warm up
			Reduce(d_data, n, func(a, b float32) float32 { return a + b })
			Synchronize()
			
			b.ResetTimer()
			
			// Benchmark
			for i := 0; i < b.N; i++ {
				result, _ := Reduce(d_data, n, func(a, b float32) float32 { return a + b })
				Synchronize()
				_ = result
			}
			
			// Report metrics
			bytes := uint64(n) * 4 // Reading n float32s
			seconds := b.Elapsed().Seconds() / float64(b.N)
			bandwidth := float64(bytes) / (seconds * 1e9)
			b.ReportMetric(bandwidth, "GB/s")
			b.ReportMetric(float64(n)/seconds/1e6, "Melements/s")
			
			// Expected cache behavior
			if n*4 <= L1CacheSize {
				b.ReportMetric(0, "expected_L1_hits")
			} else if n*4 <= L2CacheSize {
				b.ReportMetric(0, "expected_L2_hits")
			} else if n*4 <= L3CacheSize {
				b.ReportMetric(0, "expected_L3_hits")
			} else {
				b.ReportMetric(0, "expected_memory_bound")
			}
		})
	}
}

// BenchmarkKernelLaunchWithCounters measures kernel launch overhead
func BenchmarkKernelLaunchWithCounters(b *testing.B) {
	// Empty kernel to measure pure overhead
	emptyKernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		// Do nothing
	})
	
	grids := []Dim3{
		{X: 1, Y: 1, Z: 1},       // Single block
		{X: 10, Y: 1, Z: 1},      // 10 blocks
		{X: 100, Y: 1, Z: 1},     // 100 blocks
		{X: 1000, Y: 1, Z: 1},    // 1000 blocks
	}
	
	for _, grid := range grids {
		b.Run(fmt.Sprintf("Grid_%dx%dx%d", grid.X, grid.Y, grid.Z), func(b *testing.B) {
			block := Dim3{X: DefaultBlockSize, Y: 1, Z: 1}
			
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				Launch(emptyKernel, grid, block)
				Synchronize()
			}
			
			// Report launch overhead
			launchTime := b.Elapsed().Seconds() / float64(b.N)
			b.ReportMetric(launchTime*1e6, "μs/launch")
			
			// Report thread efficiency
			totalThreads := grid.X * grid.Y * grid.Z * block.X * block.Y * block.Z
			b.ReportMetric(float64(totalThreads), "threads")
		})
	}
}

// DemonstrateRooflineAnalysis shows roofline model analysis
func DemonstrateRooflineAnalysis() {
	fmt.Println("Performance Counter Analysis Example")
	fmt.Println("===================================")
	
	// Simulate GEMM operation
	m, n, k := 1024, 1024, 1024
	
	// Calculate theoretical limits
	peakGFLOPS := float64(GetDevice().NumCores) * 4.0 * 2.0 * 8.0 // cores * GHz * 2 FMA * 8 float32/vec
	peakBandwidth := 50.0 // GB/s typical DDR4
	
	// Calculate operation metrics
	flops := 2 * m * n * k
	bytes := (m*k + k*n + m*n) * 4
	arithmeticIntensity := float64(flops) / float64(bytes)
	
	fmt.Printf("\nOperation: GEMM %dx%dx%d\n", m, n, k)
	fmt.Printf("FLOPs: %d\n", flops)
	fmt.Printf("Memory: %d bytes\n", bytes)
	fmt.Printf("Arithmetic Intensity: %.2f FLOPS/byte\n", arithmeticIntensity)
	
	// Roofline analysis
	computeBound := peakGFLOPS
	memoryBound := peakBandwidth * arithmeticIntensity
	achievable := minFloat64(computeBound, memoryBound)
	
	fmt.Printf("\nRoofline Analysis:\n")
	fmt.Printf("Peak GFLOPS: %.0f\n", peakGFLOPS)
	fmt.Printf("Peak Bandwidth: %.0f GB/s\n", peakBandwidth)
	fmt.Printf("Compute Bound: %.0f GFLOPS\n", computeBound)
	fmt.Printf("Memory Bound: %.0f GFLOPS\n", memoryBound)
	fmt.Printf("Achievable: %.0f GFLOPS\n", achievable)
	
	if memoryBound < computeBound {
		fmt.Println("Status: Memory Bound")
		fmt.Printf("Need %.0f GB/s bandwidth to be compute bound\n", 
			computeBound/arithmeticIntensity)
	} else {
		fmt.Println("Status: Compute Bound")
		fmt.Printf("Operating at %.1f%% of peak compute\n", 
			(achievable/peakGFLOPS)*100)
	}
}

// minFloat64 helper
func minFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}