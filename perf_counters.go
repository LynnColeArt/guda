// Package guda performance counter integration for detailed performance analysis
package guda

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// PerfCounters holds performance counter measurements
type PerfCounters struct {
	// Timing
	Duration time.Duration
	
	// CPU counters
	Cycles           uint64
	Instructions     uint64
	BranchMisses     uint64
	CacheMisses      uint64
	L1DCacheMisses   uint64
	L3CacheMisses    uint64
	
	// Memory counters
	MemoryBandwidth  float64 // GB/s
	
	// Floating point counters
	FP32Operations   uint64
	FMAOperations    uint64
	
	// Derived metrics
	IPC              float64 // Instructions per cycle
	GFLOPS           float64 // Billions of FP ops per second
	CacheMissRate    float64 // L3 miss rate
}

// PerfMonitor manages performance counter collection
type PerfMonitor struct {
	mu       sync.Mutex
	pid      int
	events   []string
	counters map[string]uint64
}

// NewPerfMonitor creates a new performance monitor
func NewPerfMonitor() *PerfMonitor {
	return &PerfMonitor{
		pid:      os.Getpid(),
		counters: make(map[string]uint64),
		events: []string{
			"cycles",
			"instructions",
			"branch-misses",
			"cache-misses",
			"L1-dcache-load-misses",
			"LLC-load-misses", // Last Level Cache (L3)
			"fp_arith_inst_retired.scalar_single",
			"fp_arith_inst_retired.128b_packed_single",
			"fp_arith_inst_retired.256b_packed_single",
			"fp_arith_inst_retired.512b_packed_single",
		},
	}
}

// Start begins collecting performance counters
func (pm *PerfMonitor) Start() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	// Reset counters
	for k := range pm.counters {
		delete(pm.counters, k)
	}
	
	// In a real implementation, we would use perf_event_open syscall
	// For now, we'll use perf stat as a subprocess (requires perf tools)
	return nil
}

// Stop ends collection and returns counters
func (pm *PerfMonitor) Stop() (*PerfCounters, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	
	// In production, read from perf file descriptors
	// For demonstration, we'll simulate or use perf stat
	
	counters := &PerfCounters{}
	
	// Parse collected counters
	counters.Cycles = pm.counters["cycles"]
	counters.Instructions = pm.counters["instructions"]
	counters.BranchMisses = pm.counters["branch-misses"]
	counters.CacheMisses = pm.counters["cache-misses"]
	counters.L1DCacheMisses = pm.counters["L1-dcache-load-misses"]
	counters.L3CacheMisses = pm.counters["LLC-load-misses"]
	
	// Calculate derived metrics
	if counters.Cycles > 0 {
		counters.IPC = float64(counters.Instructions) / float64(counters.Cycles)
	}
	
	return counters, nil
}

// MeasureKernel runs a kernel and collects performance counters
func MeasureKernel(name string, kernel func() error) (*PerfCounters, error) {
	// For systems without perf, fall back to basic timing
	start := time.Now()
	
	err := kernel()
	if err != nil {
		return nil, err
	}
	
	duration := time.Since(start)
	
	// Try to get perf counters if available
	counters, perfErr := collectPerfCounters(name, kernel)
	if perfErr != nil {
		// Fall back to basic metrics
		counters = &PerfCounters{
			Duration: duration,
		}
	}
	
	return counters, nil
}

// collectPerfCounters uses perf stat to collect counters
func collectPerfCounters(name string, kernel func() error) (*PerfCounters, error) {
	// Check if perf is available
	if _, err := exec.LookPath("perf"); err != nil {
		return nil, fmt.Errorf("perf not available: %w", err)
	}
	
	// Create a wrapper script that runs our kernel
	// This is a simplified approach - in production we'd use perf_event_open
	cmd := exec.Command("perf", "stat", "-e",
		"cycles,instructions,branch-misses,cache-misses,L1-dcache-load-misses,LLC-load-misses",
		"--", os.Args[0], "-test.run", "^$") // Run empty test
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("perf stat failed: %w", err)
	}
	
	return parsePerfOutput(string(output))
}

// parsePerfOutput parses perf stat output
func parsePerfOutput(output string) (*PerfCounters, error) {
	counters := &PerfCounters{}
	lines := strings.Split(output, "\n")
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Parse lines like: "1,234,567      cycles"
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		
		// Remove commas from numbers
		valueStr := strings.ReplaceAll(parts[0], ",", "")
		value, err := strconv.ParseUint(valueStr, 10, 64)
		if err != nil {
			continue
		}
		
		// Match counter names
		for _, part := range parts[1:] {
			switch part {
			case "cycles":
				counters.Cycles = value
			case "instructions":
				counters.Instructions = value
			case "branch-misses":
				counters.BranchMisses = value
			case "cache-misses":
				counters.CacheMisses = value
			case "L1-dcache-load-misses":
				counters.L1DCacheMisses = value
			case "LLC-load-misses":
				counters.L3CacheMisses = value
			}
		}
	}
	
	// Calculate derived metrics
	if counters.Cycles > 0 {
		counters.IPC = float64(counters.Instructions) / float64(counters.Cycles)
	}
	
	return counters, nil
}

// CalculateMetrics computes derived performance metrics
func (pc *PerfCounters) CalculateMetrics(flops uint64, bytes uint64) {
	if pc.Duration > 0 {
		seconds := pc.Duration.Seconds()
		pc.GFLOPS = float64(flops) / (seconds * 1e9)
		pc.MemoryBandwidth = float64(bytes) / (seconds * 1e9) // GB/s
	}
	
	if pc.CacheMisses > 0 && pc.L3CacheMisses > 0 {
		pc.CacheMissRate = float64(pc.L3CacheMisses) / float64(pc.CacheMisses)
	}
}

// String formats performance counters for display
func (pc *PerfCounters) String() string {
	var sb strings.Builder
	
	sb.WriteString("Performance Counters:\n")
	if pc.Duration > 0 {
		sb.WriteString(fmt.Sprintf("  Duration:          %v\n", pc.Duration))
	}
	if pc.Cycles > 0 {
		sb.WriteString(fmt.Sprintf("  CPU Cycles:        %d\n", pc.Cycles))
		sb.WriteString(fmt.Sprintf("  Instructions:      %d\n", pc.Instructions))
		sb.WriteString(fmt.Sprintf("  IPC:               %.2f\n", pc.IPC))
	}
	if pc.BranchMisses > 0 {
		sb.WriteString(fmt.Sprintf("  Branch Misses:     %d\n", pc.BranchMisses))
	}
	if pc.L1DCacheMisses > 0 {
		sb.WriteString(fmt.Sprintf("  L1D Cache Misses:  %d\n", pc.L1DCacheMisses))
	}
	if pc.L3CacheMisses > 0 {
		sb.WriteString(fmt.Sprintf("  L3 Cache Misses:   %d\n", pc.L3CacheMisses))
		sb.WriteString(fmt.Sprintf("  Cache Miss Rate:   %.2f%%\n", pc.CacheMissRate*100))
	}
	if pc.GFLOPS > 0 {
		sb.WriteString(fmt.Sprintf("  GFLOPS:            %.2f\n", pc.GFLOPS))
	}
	if pc.MemoryBandwidth > 0 {
		sb.WriteString(fmt.Sprintf("  Memory Bandwidth:  %.2f GB/s\n", pc.MemoryBandwidth))
	}
	
	return sb.String()
}

// BenchmarkWithCounters runs a benchmark with performance counter collection
func BenchmarkWithCounters(b *testing.B, name string, fn func()) {
	// Warm up
	fn()
	
	b.ResetTimer()
	
	// Collect counters for the entire benchmark
	start := time.Now()
	
	for i := 0; i < b.N; i++ {
		fn()
	}
	
	duration := time.Since(start)
	
	// Report metrics
	b.ReportMetric(float64(b.N)/duration.Seconds(), "ops/s")
	
	// Try to collect perf counters
	if counters, err := collectPerfSummary(); err == nil {
		if counters.IPC > 0 {
			b.ReportMetric(counters.IPC, "IPC")
		}
		if counters.L3CacheMisses > 0 {
			b.ReportMetric(float64(counters.L3CacheMisses)/float64(b.N), "L3misses/op")
		}
	}
}

// collectPerfSummary attempts to get performance counter summary
func collectPerfSummary() (*PerfCounters, error) {
	// This is a placeholder - in production we'd read from
	// /proc/self/status or use perf_event_open
	return &PerfCounters{}, fmt.Errorf("not implemented")
}

// Example usage in benchmarks:
//
// func BenchmarkGEMMWithCounters(b *testing.B) {
//     // Setup
//     m, n, k := 1024, 1024, 1024
//     a, b, c := setupMatrices(m, n, k)
//
//     BenchmarkWithCounters(b, "GEMM", func() {
//         GEMM(false, false, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
//     })
//
//     // Calculate and report GFLOPS
//     flops := uint64(2 * m * n * k) // 2 ops per multiply-add
//     b.ReportMetric(float64(flops*b.N)/b.Elapsed.Seconds()/1e9, "GFLOPS")
// }