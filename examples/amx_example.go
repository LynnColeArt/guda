// +build ignore

package main

import (
	"fmt"
	"log"
	"time"
	"unsafe"

	"github.com/LynnColeArt/guda/ffu"
	"github.com/LynnColeArt/guda/ffu/amx"
)

func main() {
	// Create AMX FFU
	amxFFU := amx.NewAMXFFU()
	
	// Check if AMX is available
	if !amxFFU.IsAvailable() {
		fmt.Println("AMX is not available on this CPU")
		fmt.Println("AMX requires Intel Sapphire Rapids or newer processors")
		fmt.Println("\nRunning simulation mode...")
		
		// For demonstration, enable AMX support
		// In real usage, this would be detected via CPUID
		amx.SetAMXSupport(true, true, true)
		amxFFU = amx.NewAMXFFU()
	}
	
	// Create a registry and register the AMX FFU
	registry := ffu.NewRegistry()
	if err := registry.Register(amxFFU); err != nil {
		log.Fatalf("Failed to register AMX FFU: %v", err)
	}
	
	// Create an INT8 matrix multiplication workload
	// A: 256×256, B: 256×256, C: 256×256
	M, N, K := 256, 256, 256
	
	// Allocate aligned memory for matrices
	A := makeAlignedBuffer(M*K, 16)
	B := makeAlignedBuffer(K*N, 16)
	C := makeAlignedBuffer(M*N*4, 16) // INT32 output
	
	// Initialize matrices with test data
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 127)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte((i + 64) % 127)
	}
	
	// Create workload
	workload := &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		ScaleA:    1.0,
		ScaleB:    1.0,
		ScaleC:    1.0,
	}
	
	// Validate workload
	if err := workload.Validate(); err != nil {
		log.Fatalf("Invalid workload: %v", err)
	}
	
	// Find best FFU for this workload
	bestFFU, cost := registry.FindBest(workload)
	if bestFFU == nil {
		log.Fatal("No suitable FFU found for workload")
	}
	
	fmt.Printf("Selected FFU: %s\n", bestFFU.Name())
	fmt.Printf("Estimated duration: %v\n", cost.Duration)
	fmt.Printf("Estimated throughput: %.2f GB/s\n", float64(cost.MemoryBandwidth)/1e9)
	fmt.Printf("Confidence: %.0f%%\n", cost.Confidence*100)
	
	// Execute the workload
	start := time.Now()
	if err := bestFFU.Execute(workload); err != nil {
		log.Fatalf("Execution failed: %v", err)
	}
	elapsed := time.Since(start)
	
	// Calculate performance
	ops := workload.Operations()
	actualTOPS := float64(ops) / elapsed.Seconds() / 1e12
	
	fmt.Printf("\nExecution completed in: %v\n", elapsed)
	fmt.Printf("Operations: %d\n", ops)
	fmt.Printf("Performance: %.2f TOPS\n", actualTOPS)
	
	// Show some results (spot check)
	cInt32 := (*[1 << 30]int32)(unsafe.Pointer(&C[0]))[:M*N]
	fmt.Printf("\nSample results (C[0:5]): %v\n", cInt32[0:5])
	
	// Display metrics
	metrics := bestFFU.Metrics()
	fmt.Printf("\nFFU Metrics:\n")
	fmt.Printf("  Workloads executed: %d\n", metrics.WorkloadCount)
	fmt.Printf("  Bytes processed: %.2f GB\n", float64(metrics.BytesProcessed)/1e9)
	
	// Show AMX capabilities
	if amxCap := amxFFU.Capability(); amxCap != nil {
		fmt.Printf("\nAMX Capabilities:\n")
		fmt.Printf("  INT8 support: %v\n", amxCap.SupportsInt8)
		fmt.Printf("  BF16 support: %v\n", amxCap.SupportsBF16)
		fmt.Printf("  Peak INT8: %.1f TOPS\n", amxCap.PeakInt8TOPS)
		fmt.Printf("  Peak BF16: %.1f TFLOPS\n", amxCap.PeakBF16TFLOPS)
		fmt.Printf("  Tile config: %d tiles, %d×%d max\n",
			amxCap.NumTiles, amxCap.MaxTileRows, amxCap.MaxTileCols)
	}
}

// Helper to create aligned buffers
func makeAlignedBuffer(size int, align int) []byte {
	buf := make([]byte, size+align)
	ptr := uintptr(unsafe.Pointer(&buf[0]))
	offset := (align - int(ptr%uintptr(align))) % align
	return buf[offset : offset+size]
}

// For testing - expose the internal function
func SetAMXSupport(tile, int8, bf16 bool) {
	// This would call the package-internal function
	// In real code, this would be handled by build tags
}