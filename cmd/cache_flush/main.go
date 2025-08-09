package main

import (
	"fmt"
	"runtime"
	"time"
)

func main() {
	fmt.Println("Cache flush utility: Simulating cold cache by allocating large memory...")
	
	// Get available memory
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// Allocate 2x the typical L3 cache size (safe approach)
	// Most modern CPUs have 8-32MB L3 cache
	allocSize := 64 * 1024 * 1024 // 64MB - enough to flush most L3 caches
	
	fmt.Printf("Allocating %d MB to flush CPU caches...\n", allocSize/(1024*1024))
	
	// Allocate and touch memory to ensure it's not just virtually allocated
	data := make([]byte, allocSize)
	
	// Touch every cache line (64 bytes) to ensure physical allocation
	touchSize := 64
	start := time.Now()
	
	for i := 0; i < len(data); i += touchSize {
		data[i] = byte(i % 256)
	}
	
	// Do a second pass with different pattern to ensure cache replacement
	for i := 0; i < len(data); i += touchSize {
		data[i] = byte((i * 7) % 256)
	}
	
	elapsed := time.Since(start)
	fmt.Printf("Cache flush completed in %v\n", elapsed)
	fmt.Println("Caches should now be mostly cold. Running benchmarks...")
	
	// Force a GC to clean up before benchmarks
	runtime.GC()
}