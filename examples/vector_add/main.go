package main

import (
	"fmt"
	"math/rand"
	"time"

	guda "github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Vector Addition Example")
	fmt.Println("============================")

	// Problem size
	const N = 1_000_000

	// Allocate host memory
	h_A := make([]float32, N)
	h_B := make([]float32, N)
	h_C := make([]float32, N)
	h_C_ref := make([]float32, N)

	// Initialize host arrays
	for i := 0; i < N; i++ {
		h_A[i] = rand.Float32()
		h_B[i] = rand.Float32()
	}

	// CPU reference implementation
	start := time.Now()
	for i := 0; i < N; i++ {
		h_C_ref[i] = h_A[i] + h_B[i]
	}
	cpuTime := time.Since(start)
	fmt.Printf("CPU Time: %v\n", cpuTime)

	// GUDA implementation
	fmt.Println("\nUsing GUDA...")

	// Allocate device memory
	d_A, err := guda.Malloc(N * 4) // 4 bytes per float32
	if err != nil {
		panic(err)
	}
	defer guda.Free(d_A)

	d_B, err := guda.Malloc(N * 4)
	if err != nil {
		panic(err)
	}
	defer guda.Free(d_B)

	d_C, err := guda.Malloc(N * 4)
	if err != nil {
		panic(err)
	}
	defer guda.Free(d_C)

	// Copy data to device
	err = guda.Memcpy(d_A, h_A, N*4, guda.MemcpyHostToDevice)
	if err != nil {
		panic(err)
	}

	err = guda.Memcpy(d_B, h_B, N*4, guda.MemcpyHostToDevice)
	if err != nil {
		panic(err)
	}

	// Launch kernel
	blockSize := 256
	gridSize := (N + blockSize - 1) / blockSize

	start = time.Now()
	
	kernel := guda.KernelFunc(func(tid guda.ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < N {
			aSlice := d_A.Float32()
			bSlice := d_B.Float32()
			cSlice := d_C.Float32()
			cSlice[idx] = aSlice[idx] + bSlice[idx]
		}
	})

	err = guda.Launch(kernel,
		guda.Dim3{X: gridSize, Y: 1, Z: 1},
		guda.Dim3{X: blockSize, Y: 1, Z: 1})
	if err != nil {
		panic(err)
	}

	// Synchronize
	err = guda.Synchronize()
	if err != nil {
		panic(err)
	}

	gudaTime := time.Since(start)
	fmt.Printf("GUDA Time: %v\n", gudaTime)
	fmt.Printf("Speedup: %.2fx\n", float64(cpuTime)/float64(gudaTime))

	// Copy result back
	err = guda.Memcpy(h_C, d_C, N*4, guda.MemcpyDeviceToHost)
	if err != nil {
		panic(err)
	}

	// Verify results
	maxError := float32(0.0)
	for i := 0; i < N; i++ {
		diff := h_C[i] - h_C_ref[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxError {
			maxError = diff
		}
	}

	fmt.Printf("\nMax error: %e\n", maxError)
	if maxError < 1e-5 {
		fmt.Println("Test PASSED!")
	} else {
		fmt.Println("Test FAILED!")
	}

	// Memory bandwidth calculation
	bytesTransferred := int64(3 * N * 4) // Read A, B, Write C
	bandwidth := float64(bytesTransferred) / gudaTime.Seconds() / 1e9
	fmt.Printf("\nMemory Bandwidth: %.2f GB/s\n", bandwidth)
}