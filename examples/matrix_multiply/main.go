package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Matrix Multiplication Example")
	fmt.Println("==================================")

	// Matrix dimensions
	const M = 512
	const N = 512
	const K = 512

	// Allocate host memory
	h_A := make([]float32, M*K)
	h_B := make([]float32, K*N)
	h_C := make([]float32, M*N)
	h_C_ref := make([]float32, M*N)

	// Initialize matrices
	for i := 0; i < M*K; i++ {
		h_A[i] = rand.Float32()
	}
	for i := 0; i < K*N; i++ {
		h_B[i] = rand.Float32()
	}

	// CPU reference implementation
	fmt.Println("Running CPU reference...")
	start := time.Now()
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			for k := 0; k < K; k++ {
				sum += h_A[i*K+k] * h_B[k*N+j]
			}
			h_C_ref[i*N+j] = sum
		}
	}
	cpuTime := time.Since(start)
	fmt.Printf("CPU Time: %v\n", cpuTime)

	// Calculate CPU GFLOPS
	ops := int64(2 * M * N * K) // multiply-add
	cpuGflops := float64(ops) / cpuTime.Seconds() / 1e9
	fmt.Printf("CPU Performance: %.2f GFLOPS\n", cpuGflops)

	// GUDA implementation using optimized GEMM
	fmt.Println("\nUsing GUDA GEMM...")

	// Allocate device memory
	d_A, _ := guda.Malloc(M * K * 4)
	defer guda.Free(d_A)
	
	d_B, _ := guda.Malloc(K * N * 4)
	defer guda.Free(d_B)
	
	d_C, _ := guda.Malloc(M * N * 4)
	defer guda.Free(d_C)

	// Copy data to device
	guda.Memcpy(d_A, h_A, M*K*4, guda.MemcpyHostToDevice)
	guda.Memcpy(d_B, h_B, K*N*4, guda.MemcpyHostToDevice)

	// Launch GEMM
	start = time.Now()
	
	err := guda.GEMM(
		false, false,  // no transpose
		M, N, K,
		1.0,          // alpha
		d_A, K,       // A matrix, leading dimension K
		d_B, N,       // B matrix, leading dimension N
		0.0,          // beta
		d_C, N,       // C matrix, leading dimension N
	)
	if err != nil {
		panic(err)
	}

	guda.Synchronize()
	gemmTime := time.Since(start)
	fmt.Printf("GUDA GEMM Time: %v\n", gemmTime)
	
	// Calculate GUDA GFLOPS
	gudaGflops := float64(ops) / gemmTime.Seconds() / 1e9
	fmt.Printf("GUDA Performance: %.2f GFLOPS\n", gudaGflops)
	fmt.Printf("Speedup: %.2fx\n", float64(cpuTime)/float64(gemmTime))

	// Copy result back
	guda.Memcpy(h_C, d_C, M*N*4, guda.MemcpyDeviceToHost)

	// Verify results
	maxError := float32(0.0)
	for i := 0; i < M*N; i++ {
		diff := h_C[i] - h_C_ref[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxError {
			maxError = diff
		}
	}

	fmt.Printf("\nMax error: %e\n", maxError)
	if maxError < 1e-3 { // Slightly higher tolerance for GEMM
		fmt.Println("Test PASSED!")
	} else {
		fmt.Println("Test FAILED!")
	}

	// Memory bandwidth calculation
	// GEMM reads each element of A and B K times, writes C once
	bytesTransferred := int64(M*K*4 + K*N*4 + M*N*4)
	bandwidth := float64(bytesTransferred) / gemmTime.Seconds() / 1e9
	fmt.Printf("\nEffective Memory Bandwidth: %.2f GB/s\n", bandwidth)
	
	// Compute intensity
	computeIntensity := float64(ops) / float64(bytesTransferred)
	fmt.Printf("Compute Intensity: %.2f FLOP/byte\n", computeIntensity)
}