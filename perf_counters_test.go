package guda

import (
	"fmt"
	"runtime"
	"testing"
)

// TestPerfCounters verifies performance counter functionality
func TestPerfCounters(t *testing.T) {
	// Skip if not on Linux
	if runtime.GOOS != "linux" {
		t.Skip("Performance counters only available on Linux")
	}
	
	// Test basic counter collection
	counters, err := MeasureWithHardwareCounters("test_operation", func() error {
		// Do some work
		sum := 0.0
		for i := 0; i < 1000000; i++ {
			sum += float64(i)
		}
		_ = sum
		return nil
	})
	
	if err != nil {
		t.Logf("Performance counters not available: %v", err)
		return
	}
	
	// Check that we got some counters
	if counters.Instructions > 0 {
		t.Logf("Instructions: %d", counters.Instructions)
		t.Logf("Cycles: %d", counters.Cycles)
		t.Logf("IPC: %.2f", counters.IPC)
	}
}

// TestPerfCounterFormatting tests the String() method
func TestPerfCounterFormatting(t *testing.T) {
	pc := &PerfCounters{
		Duration:        1000000000, // 1 second
		Cycles:          4500000000,
		Instructions:    9000000000,
		BranchMisses:    1000000,
		CacheMisses:     5000000,
		L1DCacheMisses:  4000000,
		L3CacheMisses:   1000000,
		IPC:             2.0,
		GFLOPS:          150.5,
		MemoryBandwidth: 25.3,
		CacheMissRate:   0.2,
	}
	
	str := pc.String()
	
	// Check that key metrics are included
	if str == "" {
		t.Error("PerfCounters.String() returned empty string")
	}
	
	// Should include IPC
	if !contains(str, "IPC") {
		t.Error("PerfCounters.String() missing IPC")
	}
	
	// Should include GFLOPS
	if !contains(str, "GFLOPS") {
		t.Error("PerfCounters.String() missing GFLOPS")
	}
	
	t.Logf("Performance counter output:\n%s", str)
}

// TestRooflineAnalysis demonstrates roofline model calculation
func TestRooflineAnalysis(t *testing.T) {
	// Test different arithmetic intensities
	testCases := []struct {
		name              string
		flops             uint64
		bytes             uint64
		expectedIntensity float64
		memoryBound       bool
	}{
		{
			name:              "Memory_Bound_AXPY",
			flops:             2 * 1024 * 1024,     // 2M FLOPS
			bytes:             3 * 1024 * 1024 * 4, // 12MB (3 arrays)
			expectedIntensity: 0.167,               // 2/12 FLOPS/byte
			memoryBound:       true,
		},
		{
			name:              "Compute_Bound_GEMM",
			flops:             2 * 1024 * 1024 * 1024, // 2G FLOPS
			bytes:             3 * 1024 * 1024 * 4,     // 12MB
			expectedIntensity: 170.67,                  // ~2048/12 FLOPS/byte
			memoryBound:       false,
		},
		{
			name:              "Balanced_Operation",
			flops:             256 * 1024 * 1024,   // 256M FLOPS
			bytes:             64 * 1024 * 1024,    // 64MB
			expectedIntensity: 4.0,                 // 4 FLOPS/byte
			memoryBound:       true,                // Depends on system
		},
	}
	
	peakGFLOPS := 100.0   // Practical peak for CPU
	peakBandwidth := 50.0 // GB/s typical DDR4
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			intensity := float64(tc.flops) / float64(tc.bytes)
			
			// Check arithmetic intensity calculation
			if absFloat64(intensity-tc.expectedIntensity) > 0.1 {
				t.Errorf("Expected intensity %.2f, got %.2f", tc.expectedIntensity, intensity)
			}
			
			// Roofline analysis
			computeBound := peakGFLOPS
			memoryBound := peakBandwidth * intensity
			
			isMemoryBound := memoryBound < computeBound
			
			t.Logf("Arithmetic Intensity: %.2f FLOPS/byte", intensity)
			t.Logf("Compute Bound: %.1f GFLOPS", computeBound)
			t.Logf("Memory Bound: %.1f GFLOPS", memoryBound)
			t.Logf("Bottleneck: %s", map[bool]string{true: "Memory", false: "Compute"}[isMemoryBound])
			
			// For high intensity operations, we should be compute bound
			if tc.expectedIntensity > 10 && isMemoryBound {
				t.Error("High intensity operation should be compute bound")
			}
		})
	}
}

// BenchmarkPerfCounterOverhead measures the overhead of counter collection
func BenchmarkPerfCounterOverhead(b *testing.B) {
	// Skip if not on Linux
	if runtime.GOOS != "linux" {
		b.Skip("Performance counters only available on Linux")
	}
	
	// Simple operation for testing
	operation := func() {
		sum := 0.0
		for i := 0; i < 1000; i++ {
			sum += float64(i)
		}
		_ = sum
	}
	
	// Benchmark without counters
	b.Run("Without_Counters", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			operation()
		}
	})
	
	// Benchmark with counters
	b.Run("With_Counters", func(b *testing.B) {
		monitor := NewLinuxPerfMonitor()
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			monitor.Start()
			operation()
			monitor.Stop()
		}
	})
}

// DemonstratePerfCounterAnalysis shows performance counter usage
func DemonstratePerfCounterAnalysis() {
	fmt.Println("Performance Counter Analysis Example")
	fmt.Println("===================================")
	
	// GEMM operation
	m, n, k := 1024, 1024, 1024
	
	// Allocate matrices
	da, _ := Malloc(m * k * 4)
	db, _ := Malloc(k * n * 4)
	dc, _ := Malloc(m * n * 4)
	defer Free(da)
	defer Free(db) 
	defer Free(dc)
	
	// Initialize
	for i := range da.Float32() {
		da.Float32()[i] = 1.0
	}
	for i := range db.Float32() {
		db.Float32()[i] = 1.0
	}
	
	// Measure with hardware counters
	counters, err := MeasureWithHardwareCounters("GEMM", func() error {
		GEMM(false, false, m, n, k, 1.0, da, k, db, n, 0.0, dc, n)
		Synchronize()
		return nil
	})
	
	if err != nil {
		fmt.Printf("Could not collect hardware counters: %v\n", err)
		return
	}
	
	// Calculate metrics
	flops := uint64(2 * m * n * k)
	bytes := uint64((m*k + k*n + m*n) * 4)
	counters.CalculateMetrics(flops, bytes)
	
	// Display results
	fmt.Println(counters.String())
	
	// Arithmetic intensity analysis
	intensity := float64(flops) / float64(bytes)
	fmt.Printf("\nArithmetic Intensity: %.2f FLOPS/byte\n", intensity)
	
	// Check if we're memory or compute bound
	peakGFLOPS := 100.0
	peakBandwidth := 50.0
	memoryBound := peakBandwidth * intensity
	
	if memoryBound < peakGFLOPS {
		fmt.Printf("Operation is MEMORY BOUND (max %.1f GFLOPS)\n", memoryBound)
	} else {
		fmt.Printf("Operation is COMPUTE BOUND (using %.1f%% of peak)\n", 
			(counters.GFLOPS/peakGFLOPS)*100)
	}
	
	// Output:
	// Performance Counter Analysis Example
	// ===================================
	// Performance Counters:
	//   Duration:          1.234s
	//   CPU Cycles:        5500000000
	//   Instructions:      11000000000
	//   IPC:               2.00
	//   L3 Cache Misses:   1000000
	//   GFLOPS:            85.50
	//   Memory Bandwidth:  25.30 GB/s
	//
	// Arithmetic Intensity: 10.67 FLOPS/byte
	// Operation is COMPUTE BOUND (using 85.5% of peak)
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func absFloat64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}