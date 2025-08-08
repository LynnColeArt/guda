package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA SIMD Test")
	fmt.Println("==============")

	// Test AXPY operation which uses gonum's SIMD
	const N = 10_000_000
	alpha := float32(2.5)

	// Allocate
	d_X, _ := guda.Malloc(N * 4)
	d_Y, _ := guda.Malloc(N * 4)
	defer guda.Free(d_X)
	defer guda.Free(d_Y)

	// Initialize
	xSlice := d_X.Float32()
	ySlice := d_Y.Float32()
	for i := 0; i < N; i++ {
		xSlice[i] = rand.Float32()
		ySlice[i] = rand.Float32()
	}

	// Benchmark AXPY (should use SIMD)
	start := time.Now()
	err := guda.AXPY(alpha, d_X, d_Y, N)
	if err != nil {
		panic(err)
	}
	guda.Synchronize()
	axpyTime := time.Since(start)

	// Calculate performance
	flops := int64(2 * N) // multiply + add
	gflops := float64(flops) / axpyTime.Seconds() / 1e9
	bandwidth := float64(3*N*4) / axpyTime.Seconds() / 1e9

	fmt.Printf("AXPY Time: %v\n", axpyTime)
	fmt.Printf("Performance: %.2f GFLOPS\n", gflops)
	fmt.Printf("Memory Bandwidth: %.2f GB/s\n", bandwidth)

	// Verify a few values
	fmt.Println("\nVerifying first 5 values...")
	for i := 0; i < 5; i++ {
		fmt.Printf("Y[%d] = %.6f\n", i, ySlice[i])
	}
}