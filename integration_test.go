package guda

import (
	"math"
	"testing"
	"time"
)

// TestNeuralNetworkLayer simulates a real neural network layer
func TestNeuralNetworkLayer(t *testing.T) {
	// Simulate a small neural network layer:
	// output = ReLU(input @ weights + bias)
	
	batchSize := 64
	inputDim := 512
	outputDim := 256
	
	// Allocate memory
	input, _ := Malloc(batchSize * inputDim * 4)
	weights, _ := Malloc(inputDim * outputDim * 4)
	bias, _ := Malloc(outputDim * 4)
	output, _ := Malloc(batchSize * outputDim * 4)
	defer Free(input)
	defer Free(weights)
	defer Free(bias)
	defer Free(output)
	
	// Initialize with some values
	inputSlice := input.Float32()
	weightsSlice := weights.Float32()
	biasSlice := bias.Float32()
	
	for i := range inputSlice {
		inputSlice[i] = float32(i%10) * 0.1
	}
	for i := range weightsSlice {
		weightsSlice[i] = float32(i%5) * 0.01
	}
	for i := range biasSlice {
		biasSlice[i] = -0.5 + float32(i%10)*0.1
	}
	
	// Perform the operation
	// 1. Matrix multiply
	err := GEMM(false, false, batchSize, outputDim, inputDim,
		1.0, input, inputDim, weights, outputDim,
		0.0, output, outputDim)
	if err != nil {
		t.Fatalf("GEMM failed: %v", err)
	}
	
	// 2. Add bias and ReLU (fused)
	err = AddBiasReLU(output, bias, batchSize*outputDim)
	if err != nil {
		t.Fatalf("AddBiasReLU failed: %v", err)
	}
	
	// Verify some outputs are non-zero
	outputSlice := output.Float32()
	nonZeroCount := 0
	for _, v := range outputSlice {
		if v > 0 {
			nonZeroCount++
		}
	}
	
	if nonZeroCount == 0 {
		t.Error("All outputs are zero, ReLU might not be working")
	}
	
	t.Logf("Neural network layer test passed: %d/%d outputs are non-zero",
		nonZeroCount, len(outputSlice))
}

// TestMemoryBandwidthLimit tests if we're hitting memory bandwidth limits
func TestMemoryBandwidthLimit(t *testing.T) {
	sizes := []int{
		1 << 20,  // 1MB
		1 << 24,  // 16MB
		1 << 26,  // 64MB
	}
	
	for _, size := range sizes {
		N := size / 4 // number of float32s
		
		// Allocate
		a, _ := Malloc(size)
		b, _ := Malloc(size)
		c, _ := Malloc(size)
		defer Free(a)
		defer Free(b)
		defer Free(c)
		
		// Warm up
		Add(a, b, c, N)
		Synchronize()
		
		// Time the operation
		start := nanotime()
		Add(a, b, c, N)
		Synchronize()
		elapsed := nanotime() - start
		
		// Calculate bandwidth
		bytesTransferred := int64(3 * size) // Read a, b, Write c
		bandwidth := float64(bytesTransferred) / float64(elapsed) * 1e9 / 1e9 // GB/s
		
		t.Logf("Size: %d MB, Bandwidth: %.2f GB/s", size/(1<<20), bandwidth)
		
		// Check if reasonable (at least 1 GB/s)
		if bandwidth < 1.0 {
			t.Errorf("Bandwidth too low: %.2f GB/s", bandwidth)
		}
	}
}

// TestParallelScaling tests if performance scales with parallelism
func TestParallelScaling(t *testing.T) {
	N := 10_000_000
	
	// Allocate
	x, _ := Malloc(N * 4)
	y, _ := Malloc(N * 4)
	defer Free(x)
	defer Free(y)
	
	// Test with different grid sizes
	blockSize := 256
	gridSizes := []int{1, 2, 4, 8, 16}
	
	var basetime int64
	
	for _, gridSize := range gridSizes {
		kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
			idx := tid.Global()
			if idx < N {
				y.Float32()[idx] = x.Float32()[idx] * 2.0
			}
		})
		
		// Time it
		start := nanotime()
		Launch(kernel, Dim3{X: gridSize, Y: 1, Z: 1}, Dim3{X: blockSize, Y: 1, Z: 1})
		Synchronize()
		elapsed := nanotime() - start
		
		if gridSize == 1 {
			basetime = elapsed
		}
		
		speedup := float64(basetime) / float64(elapsed)
		t.Logf("Grid size %2d: %.2fx speedup", gridSize, speedup)
	}
}

// TestCUDACompatibility tests CUDA-like code patterns
func TestCUDACompatibility(t *testing.T) {
	// This is how CUDA code would look
	N := 1000
	
	// Allocate device memory
	d_A, err := Malloc(N * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer Free(d_A)
	
	d_B, err := Malloc(N * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer Free(d_B)
	
	d_C, err := Malloc(N * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer Free(d_C)
	
	// Initialize on host
	h_A := make([]float32, N)
	h_B := make([]float32, N)
	for i := 0; i < N; i++ {
		h_A[i] = float32(i)
		h_B[i] = float32(i * 2)
	}
	
	// Copy to device
	Memcpy(d_A, h_A, N*4, MemcpyHostToDevice)
	Memcpy(d_B, h_B, N*4, MemcpyHostToDevice)
	
	// Launch kernel
	Add(d_A, d_B, d_C, N)
	Synchronize()
	
	// Copy back
	h_C := make([]float32, N)
	Memcpy(h_C, d_C, N*4, MemcpyDeviceToHost)
	
	// Verify
	for i := 0; i < N; i++ {
		expected := h_A[i] + h_B[i]
		if math.Abs(float64(h_C[i]-expected)) > 1e-5 {
			t.Errorf("Mismatch at %d: expected %f, got %f", i, expected, h_C[i])
			break
		}
	}
}

// nanotime returns current time in nanoseconds
func nanotime() int64 {
	return time.Now().UnixNano()
}