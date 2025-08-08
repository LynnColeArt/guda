package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA vs Native Go Comparison")
	fmt.Println("============================")
	fmt.Println()

	sizes := []int{1_000, 10_000, 100_000, 1_000_000, 10_000_000}
	
	for _, N := range sizes {
		fmt.Printf("Size: %d elements\n", N)
		fmt.Println("-" + strings.Repeat("-", len(fmt.Sprintf("%d", N))))
		
		// Test 1: Vector Addition
		testVectorAdd(N)
		
		// Test 2: SAXPY
		testSAXPY(N)
		
		// Test 3: Fused Operations
		testFusion(N)
		
		fmt.Println()
	}
}

func testVectorAdd(N int) {
	// Prepare data
	a := make([]float32, N)
	b := make([]float32, N)
	c := make([]float32, N)
	
	for i := 0; i < N; i++ {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}
	
	// Native Go (sequential)
	start := time.Now()
	for i := 0; i < N; i++ {
		c[i] = a[i] + b[i]
	}
	goTime := time.Since(start)
	
	// Native Go (parallel)
	start = time.Now()
	parallelVectorAdd(a, b, c)
	goParallelTime := time.Since(start)
	
	// GUDA
	d_A, _ := guda.Malloc(N * 4)
	d_B, _ := guda.Malloc(N * 4)
	d_C, _ := guda.Malloc(N * 4)
	defer guda.Free(d_A)
	defer guda.Free(d_B)
	defer guda.Free(d_C)
	
	guda.Memcpy(d_A, a, N*4, guda.MemcpyHostToDevice)
	guda.Memcpy(d_B, b, N*4, guda.MemcpyHostToDevice)
	
	start = time.Now()
	guda.Add(d_A, d_B, d_C, N)
	guda.Synchronize()
	gudaTime := time.Since(start)
	
	fmt.Printf("  Vector Add: Go=%.2fms, Go||=%.2fms, GUDA=%.2fms (Speedup: %.2fx vs seq, %.2fx vs ||)\n",
		goTime.Seconds()*1000,
		goParallelTime.Seconds()*1000,
		gudaTime.Seconds()*1000,
		goTime.Seconds()/gudaTime.Seconds(),
		goParallelTime.Seconds()/gudaTime.Seconds())
}

func testSAXPY(N int) {
	// Prepare data
	x := make([]float32, N)
	y := make([]float32, N)
	alpha := float32(2.5)
	
	for i := 0; i < N; i++ {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	
	// Native Go
	yCopy := make([]float32, N)
	copy(yCopy, y)
	
	start := time.Now()
	for i := 0; i < N; i++ {
		yCopy[i] += alpha * x[i]
	}
	goTime := time.Since(start)
	
	// GUDA
	d_X, _ := guda.Malloc(N * 4)
	d_Y, _ := guda.Malloc(N * 4)
	defer guda.Free(d_X)
	defer guda.Free(d_Y)
	
	guda.Memcpy(d_X, x, N*4, guda.MemcpyHostToDevice)
	guda.Memcpy(d_Y, y, N*4, guda.MemcpyHostToDevice)
	
	start = time.Now()
	guda.AXPY(alpha, d_X, d_Y, N)
	guda.Synchronize()
	gudaTime := time.Since(start)
	
	bandwidth := float64(3*N*4) / gudaTime.Seconds() / 1e9
	
	fmt.Printf("  SAXPY: Go=%.2fms, GUDA=%.2fms (Speedup: %.2fx, BW: %.2f GB/s)\n",
		goTime.Seconds()*1000,
		gudaTime.Seconds()*1000,
		goTime.Seconds()/gudaTime.Seconds(),
		bandwidth)
}

func testFusion(N int) {
	// Test: y = ReLU(2*x + bias)
	x := make([]float32, N)
	bias := make([]float32, N)
	
	for i := 0; i < N; i++ {
		x[i] = rand.Float32()*2 - 1
		bias[i] = rand.Float32() * 0.1
	}
	
	// Native Go (separate operations)
	y1 := make([]float32, N)
	copy(y1, x)
	
	start := time.Now()
	// Scale
	for i := 0; i < N; i++ {
		y1[i] *= 2.0
	}
	// Add bias
	for i := 0; i < N; i++ {
		y1[i] += bias[i]
	}
	// ReLU
	for i := 0; i < N; i++ {
		if y1[i] < 0 {
			y1[i] = 0
		}
	}
	goSeparateTime := time.Since(start)
	
	// Native Go (fused)
	y2 := make([]float32, N)
	start = time.Now()
	for i := 0; i < N; i++ {
		val := 2.0*x[i] + bias[i]
		if val < 0 {
			val = 0
		}
		y2[i] = val
	}
	goFusedTime := time.Since(start)
	
	// GUDA fused
	d_X, _ := guda.Malloc(N * 4)
	d_Bias, _ := guda.Malloc(N * 4)
	defer guda.Free(d_X)
	defer guda.Free(d_Bias)
	
	guda.Memcpy(d_X, x, N*4, guda.MemcpyHostToDevice)
	guda.Memcpy(d_Bias, bias, N*4, guda.MemcpyHostToDevice)
	
	start = time.Now()
	guda.NewFusedKernel().
		MulScalar(2.0).
		Add(1.0).
		ReLU().
		Execute(d_X, []guda.DevicePtr{d_Bias}, N)
	guda.Synchronize()
	gudaTime := time.Since(start)
	
	fmt.Printf("  Fusion: Go-sep=%.2fms, Go-fused=%.2fms, GUDA=%.2fms (Speedup: %.2fx vs fused)\n",
		goSeparateTime.Seconds()*1000,
		goFusedTime.Seconds()*1000,
		gudaTime.Seconds()*1000,
		goFusedTime.Seconds()/gudaTime.Seconds())
}

func parallelVectorAdd(a, b, c []float32) {
	n := len(a)
	numCPU := runtime.NumCPU()
	chunkSize := n / numCPU
	
	var wg sync.WaitGroup
	wg.Add(numCPU)
	
	for i := 0; i < numCPU; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == numCPU-1 {
			end = n
		}
		
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				c[j] = a[j] + b[j]
			}
		}(start, end)
	}
	
	wg.Wait()
}

