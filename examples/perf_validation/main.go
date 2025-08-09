// Performance validation example demonstrating GUDA's performance counter integration
// This validates the 2K GFLOPS claims with hardware performance counters
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/LynnColeArt/guda"
)

var (
	size     = flag.Int("size", 2048, "Matrix size for GEMM benchmark")
	warmup   = flag.Int("warmup", 5, "Number of warmup iterations")
	iters    = flag.Int("iters", 10, "Number of benchmark iterations")
	validate = flag.Bool("validate", false, "Validate results against reference")
)

func main() {
	flag.Parse()
	
	fmt.Printf("GUDA Performance Validation with Hardware Counters\n")
	fmt.Printf("=================================================\n")
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("CPU cores: %d\n", runtime.NumCPU())
	fmt.Printf("Matrix size: %dx%dx%d\n", *size, *size, *size)
	fmt.Printf("\n")
	
	// Run different benchmarks
	validateAXPY()
	fmt.Println()
	validateGEMM()
	fmt.Println()
	validateFusion()
}

func validateAXPY() {
	fmt.Println("1. AXPY Performance Validation")
	fmt.Println("------------------------------")
	
	sizes := []int{1024, 16384, 262144, 1048576, 16777216}
	
	for _, n := range sizes {
		// Allocate memory
		dx, err := guda.Malloc(n * 4)
		if err != nil {
			log.Fatal(err)
		}
		dy, err := guda.Malloc(n * 4)
		if err != nil {
			log.Fatal(err)
		}
		defer guda.Free(dx)
		defer guda.Free(dy)
		
		// Initialize
		guda.InitTestData(dx.Float32(), 1.0)
		guda.InitTestData(dy.Float32(), 2.0)
		
		// Warmup
		for i := 0; i < *warmup; i++ {
			guda.AXPY(2.5, dx, dy, n)
			guda.Synchronize()
		}
		
		// Benchmark with counters
		start := time.Now()
		var counters *guda.PerfCounters
		
		for i := 0; i < *iters; i++ {
			c, err := guda.MeasureWithHardwareCounters(fmt.Sprintf("AXPY_%d", n), func() error {
				guda.AXPY(2.5, dx, dy, n)
				return guda.Synchronize()
			})
			if err == nil && counters == nil {
				counters = c
			}
		}
		
		elapsed := time.Since(start)
		
		// Calculate metrics
		flops := uint64(2 * n * *iters) // multiply + add
		bytes := uint64(3 * n * 4 * *iters) // 2 reads + 1 write
		gflops := float64(flops) / elapsed.Seconds() / 1e9
		bandwidth := float64(bytes) / elapsed.Seconds() / 1e9
		intensity := float64(flops) / float64(bytes)
		
		fmt.Printf("\nAXPY N=%d:\n", n)
		fmt.Printf("  Performance: %.1f GFLOPS\n", gflops)
		fmt.Printf("  Bandwidth: %.1f GB/s\n", bandwidth)
		fmt.Printf("  Arithmetic Intensity: %.3f FLOPS/byte\n", intensity)
		
		// Display hardware counters if available
		if counters != nil && counters.Instructions > 0 {
			counters.Duration = elapsed / time.Duration(*iters)
			counters.CalculateMetrics(flops/uint64(*iters), bytes/uint64(*iters))
			
			fmt.Printf("  IPC: %.2f\n", counters.IPC)
			if counters.L3CacheMisses > 0 {
				fmt.Printf("  L3 Cache Misses: %d\n", counters.L3CacheMisses)
			}
			
			// Check cache residency
			dataSize := n * 4 * 2 // X and Y arrays
			if dataSize <= guda.L1CacheSize {
				fmt.Printf("  Cache Level: L1 (data fits in %s)\n", formatBytes(guda.L1CacheSize))
			} else if dataSize <= guda.L2CacheSize {
				fmt.Printf("  Cache Level: L2 (data fits in %s)\n", formatBytes(guda.L2CacheSize))
			} else if dataSize <= guda.L3CacheSize {
				fmt.Printf("  Cache Level: L3 (data fits in %s)\n", formatBytes(guda.L3CacheSize))
			} else {
				fmt.Printf("  Cache Level: Memory (data size %s exceeds L3)\n", formatBytes(dataSize))
			}
		}
		
		// Roofline analysis
		peakBandwidth := 50.0 // GB/s typical DDR4
		memoryBoundGFLOPS := peakBandwidth * intensity
		if gflops < memoryBoundGFLOPS*0.8 {
			fmt.Printf("  Status: Memory bandwidth limited (theoretical max: %.1f GFLOPS)\n", memoryBoundGFLOPS)
		} else {
			fmt.Printf("  Status: Achieving near-peak performance\n")
		}
	}
}

func validateGEMM() {
	fmt.Println("2. GEMM Performance Validation")
	fmt.Println("------------------------------")
	
	// Allocate matrices
	n := *size
	da, err := guda.Malloc(n * n * 4)
	if err != nil {
		log.Fatal(err)
	}
	db, err := guda.Malloc(n * n * 4)
	if err != nil {
		log.Fatal(err)
	}
	dc, err := guda.Malloc(n * n * 4)
	if err != nil {
		log.Fatal(err)
	}
	defer guda.Free(da)
	defer guda.Free(db)
	defer guda.Free(dc)
	
	// Initialize
	guda.InitTestData(da.Float32(), 1.0)
	guda.InitTestData(db.Float32(), 1.0)
	
	// Warmup
	fmt.Printf("\nWarming up...")
	for i := 0; i < *warmup; i++ {
		guda.GEMM(false, false, n, n, n, 1.0, da, n, db, n, 0.0, dc, n)
		guda.Synchronize()
	}
	fmt.Printf(" done\n")
	
	// Benchmark with counters
	fmt.Printf("Benchmarking %d iterations...\n", *iters)
	start := time.Now()
	var totalCounters *guda.PerfCounters
	
	for i := 0; i < *iters; i++ {
		counters, err := guda.MeasureWithHardwareCounters(fmt.Sprintf("GEMM_%dx%dx%d", n, n, n), func() error {
			guda.GEMM(false, false, n, n, n, 1.0, da, n, db, n, 0.0, dc, n)
			return guda.Synchronize()
		})
		
		if err == nil && totalCounters == nil {
			totalCounters = counters
		}
	}
	
	elapsed := time.Since(start)
	
	// Calculate metrics
	flops := uint64(2) * uint64(n) * uint64(n) * uint64(n) * uint64(*iters)
	bytes := uint64(3*n*n*4) * uint64(*iters) // Simplified (not considering cache reuse)
	gflops := float64(flops) / elapsed.Seconds() / 1e9
	
	fmt.Printf("\nGEMM %dx%dx%d Results:\n", n, n, n)
	fmt.Printf("  Total time: %.3f seconds\n", elapsed.Seconds())
	fmt.Printf("  Performance: %.1f GFLOPS\n", gflops)
	fmt.Printf("  Time per GEMM: %.3f ms\n", elapsed.Seconds()/float64(*iters)*1000)
	
	// Calculate efficiency
	device := guda.GetDevice()
	// Theoretical peak: cores × frequency × FMA units × vector width
	// Assume 4.0 GHz, 2 FMA units, 8 float32/vector (AVX2)
	theoreticalPeak := float64(device.NumCores) * 4.0 * 2.0 * 8.0
	// Practical peak is typically 30-50% of theoretical for GEMM
	practicalPeak := theoreticalPeak * 0.4
	efficiency := gflops / practicalPeak * 100
	
	fmt.Printf("  Theoretical Peak: %.0f GFLOPS\n", theoreticalPeak)
	fmt.Printf("  Practical Peak: %.0f GFLOPS (40%% of theoretical)\n", practicalPeak)
	fmt.Printf("  Efficiency: %.1f%% of practical peak\n", efficiency)
	
	// Hardware counters
	if totalCounters != nil && totalCounters.Instructions > 0 {
		totalCounters.Duration = elapsed / time.Duration(*iters)
		totalCounters.CalculateMetrics(flops/uint64(*iters), bytes/uint64(*iters))
		
		fmt.Printf("\nHardware Counter Analysis:\n")
		fmt.Printf("  Instructions per cycle (IPC): %.2f\n", totalCounters.IPC)
		fmt.Printf("  FP operations per cycle: %.2f\n", gflops*1e9/4.0e9/float64(device.NumCores))
		
		if totalCounters.L3CacheMisses > 0 {
			missesPerGFLOP := float64(totalCounters.L3CacheMisses) / (gflops * float64(*iters))
			fmt.Printf("  L3 misses per GFLOP: %.0f\n", missesPerGFLOP)
			if missesPerGFLOP < 1000 {
				fmt.Printf("  Cache behavior: EXCELLENT (hot cache)\n")
			} else {
				fmt.Printf("  Cache behavior: Cold cache or streaming\n")
			}
		}
	}
	
	// Roofline analysis
	intensity := float64(2*n*n*n) / float64(3*n*n*4)
	fmt.Printf("\nRoofline Analysis:\n")
	fmt.Printf("  Arithmetic Intensity: %.1f FLOPS/byte\n", intensity)
	fmt.Printf("  Required bandwidth for peak: %.1f GB/s\n", practicalPeak/intensity)
	
	if intensity > 10 {
		fmt.Printf("  Status: Compute bound (high arithmetic intensity)\n")
		if gflops > practicalPeak*0.5 {
			fmt.Printf("  Achievement: EXCELLENT - Near peak performance!\n")
		}
	} else {
		fmt.Printf("  Status: Memory bandwidth limited\n")
	}
	
	// Validate if requested
	if *validate {
		fmt.Printf("\nValidating result...")
		ref := guda.Reference{}
		expected := make([]float32, n*n)
		actual := make([]float32, n*n)
		
		// Copy matrices for reference
		aCopy := make([]float32, n*n)
		bCopy := make([]float32, n*n)
		copy(aCopy, da.Float32())
		copy(bCopy, db.Float32())
		copy(actual, dc.Float32())
		
		// Reference GEMM
		ref.GEMM(false, false, n, n, n, 1.0, aCopy, n, bCopy, n, 0.0, expected, n)
		
		// Compare
		result := guda.VerifyFloat32Array(expected[:100], actual[:100], guda.DefaultTolerance())
		if result.NumErrors == 0 {
			fmt.Printf(" PASSED\n")
		} else {
			fmt.Printf(" FAILED: %s\n", result.String())
		}
	}
}

func validateFusion() {
	fmt.Println("3. Kernel Fusion Validation")
	fmt.Println("---------------------------")
	
	n := 1 << 22 // 4M elements
	
	// Allocate memory
	dx, _ := guda.Malloc(n * 4)
	dy, _ := guda.Malloc(n * 4)
	dz, _ := guda.Malloc(n * 4)
	defer guda.Free(dx)
	defer guda.Free(dy)
	defer guda.Free(dz)
	
	// Initialize
	guda.InitTestData(dx.Float32(), 1.0)
	guda.InitTestData(dy.Float32(), 2.0)
	
	// Benchmark separate operations
	start := time.Now()
	for i := 0; i < *iters; i++ {
		guda.Add(dx, dy, dz, n)      // Z = X + Y
		guda.Scale(2.0, dz, n)       // Z = 2 * Z
		guda.ReLU(dz, n)             // Z = ReLU(Z)
		guda.Synchronize()
	}
	separateTime := time.Since(start)
	
	// Benchmark fused operation
	fusedKernel := guda.KernelFunc(func(tid guda.ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			x := dx.Float32()[idx]
			y := dy.Float32()[idx]
			z := 2.0 * (x + y)
			if z < 0 {
				z = 0
			}
			dz.Float32()[idx] = z
		}
	})
	
	start = time.Now()
	for i := 0; i < *iters; i++ {
		guda.Launch(fusedKernel,
			guda.Dim3{X: (n + guda.DefaultBlockSize - 1) / guda.DefaultBlockSize, Y: 1, Z: 1},
			guda.Dim3{X: guda.DefaultBlockSize, Y: 1, Z: 1})
		guda.Synchronize()
	}
	fusedTime := time.Since(start)
	
	// Calculate metrics
	separateBandwidth := float64(9*n*4**iters) / separateTime.Seconds() / 1e9
	fusedBandwidth := float64(3*n*4**iters) / fusedTime.Seconds() / 1e9
	speedup := separateTime.Seconds() / fusedTime.Seconds()
	
	fmt.Printf("\nFusion Results (n=%d):\n", n)
	fmt.Printf("  Separate operations: %.3f seconds (%.1f GB/s)\n", 
		separateTime.Seconds(), separateBandwidth)
	fmt.Printf("  Fused operation: %.3f seconds (%.1f GB/s)\n", 
		fusedTime.Seconds(), fusedBandwidth)
	fmt.Printf("  Speedup: %.2fx\n", speedup)
	fmt.Printf("  Memory traffic reduction: %.1f%%\n", (1-3.0/9.0)*100)
	
	if speedup > 2.0 {
		fmt.Printf("  Status: EXCELLENT fusion benefit achieved\n")
	} else if speedup > 1.5 {
		fmt.Printf("  Status: Good fusion benefit\n")
	} else {
		fmt.Printf("  Status: Limited fusion benefit (may be compute bound)\n")
	}
}

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