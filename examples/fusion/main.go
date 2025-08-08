package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Kernel Fusion Example")
	fmt.Println("==========================")
	fmt.Println("Demonstrating: y = ReLU(alpha*x + bias)")
	fmt.Println()

	const N = 10_000_000

	// Allocate host memory
	h_X := make([]float32, N)
	h_Bias := make([]float32, N)
	h_Y_separate := make([]float32, N)
	h_Y_fused := make([]float32, N)

	// Initialize data
	alpha := float32(2.0)
	for i := 0; i < N; i++ {
		h_X[i] = rand.Float32()*2 - 1 // [-1, 1]
		h_Bias[i] = rand.Float32() * 0.1
	}

	// Allocate device memory
	d_X, _ := guda.Malloc(N * 4)
	defer guda.Free(d_X)
	
	d_Bias, _ := guda.Malloc(N * 4)
	defer guda.Free(d_Bias)
	
	d_Y, _ := guda.Malloc(N * 4)
	defer guda.Free(d_Y)

	// Copy data to device
	guda.Memcpy(d_X, h_X, N*4, guda.MemcpyHostToDevice)
	guda.Memcpy(d_Bias, h_Bias, N*4, guda.MemcpyHostToDevice)

	// Method 1: Separate operations (3 passes through memory)
	fmt.Println("Method 1: Separate Operations")
	fmt.Println("-----------------------------")
	
	// Reset Y
	guda.Memcpy(d_Y, h_X, N*4, guda.MemcpyHostToDevice)
	
	start := time.Now()
	
	// Pass 1: Scale
	guda.Scale(alpha, d_Y, N)
	guda.Synchronize()
	
	// Pass 2: Add bias
	guda.AXPY(1.0, d_Bias, d_Y, N)
	guda.Synchronize()
	
	// Pass 3: ReLU
	guda.ReLU(d_Y, N)
	guda.Synchronize()
	
	separateTime := time.Since(start)
	fmt.Printf("Time: %v\n", separateTime)
	
	// Copy result
	guda.Memcpy(h_Y_separate, d_Y, N*4, guda.MemcpyDeviceToHost)

	// Method 2: Fused operation (1 pass through memory)
	fmt.Println("\nMethod 2: Fused Operation")
	fmt.Println("-------------------------")
	
	// Reset Y
	guda.Memcpy(d_Y, h_X, N*4, guda.MemcpyHostToDevice)
	
	start = time.Now()
	
	// Single fused kernel
	err := guda.NewFusedKernel().
		MulScalar(alpha).
		Add(1.0).
		ReLU().
		Execute(d_Y, []guda.DevicePtr{d_Bias}, N)
	
	if err != nil {
		panic(err)
	}
	
	guda.Synchronize()
	fusedTime := time.Since(start)
	fmt.Printf("Time: %v\n", fusedTime)
	fmt.Printf("Speedup: %.2fx\n", float64(separateTime)/float64(fusedTime))
	
	// Copy result
	guda.Memcpy(h_Y_fused, d_Y, N*4, guda.MemcpyDeviceToHost)

	// Verify results match
	maxError := float32(0.0)
	for i := 0; i < N; i++ {
		diff := h_Y_separate[i] - h_Y_fused[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxError {
			maxError = diff
		}
	}
	
	fmt.Printf("\nMax difference: %e\n", maxError)
	if maxError < 1e-5 {
		fmt.Println("Results match - Test PASSED!")
	} else {
		fmt.Println("Results don't match - Test FAILED!")
	}

	// Memory bandwidth analysis
	fmt.Println("\nMemory Bandwidth Analysis")
	fmt.Println("-------------------------")
	
	// Separate: 3 reads + 3 writes = 6 * N * 4 bytes
	separateBytes := int64(6 * N * 4)
	separateBandwidth := float64(separateBytes) / separateTime.Seconds() / 1e9
	fmt.Printf("Separate operations: %.2f GB/s\n", separateBandwidth)
	
	// Fused: 2 reads (X, bias) + 1 write = 3 * N * 4 bytes
	fusedBytes := int64(3 * N * 4)
	fusedBandwidth := float64(fusedBytes) / fusedTime.Seconds() / 1e9
	fmt.Printf("Fused operation: %.2f GB/s\n", fusedBandwidth)
	
	fmt.Printf("\nMemory traffic reduction: %.1fx\n", 
		float64(separateBytes)/float64(fusedBytes))
	
	// Show the power of fusion
	fmt.Println("\nKey Insight:")
	fmt.Println("Fusion reduces memory traffic from 6N to 3N elements")
	fmt.Println("This is why kernel fusion is critical for CPU performance!")
}