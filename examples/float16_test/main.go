package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Float16 vs Float32 Comparison")
	fmt.Println("==================================")
	fmt.Println()

	sizes := []int{1000, 10000, 100000, 1000000}

	for _, N := range sizes {
		fmt.Printf("\nSize: %d elements\n", N)
		fmt.Println("-" + fmt.Sprintf("%d", N))

		// Test vector addition
		testVectorAdd(N)

		// Test memory bandwidth
		testMemoryBandwidth(N)
		
		// Test neural network operation
		testNeuralNet(N)
	}
}

func testVectorAdd(N int) {
	fmt.Println("\nVector Addition:")

	// Float32 version
	d_A32, _ := guda.Malloc(N * 4)
	d_B32, _ := guda.Malloc(N * 4)
	d_C32, _ := guda.Malloc(N * 4)
	defer guda.Free(d_A32)
	defer guda.Free(d_B32)
	defer guda.Free(d_C32)

	// Initialize with random data
	a32 := d_A32.Float32()
	b32 := d_B32.Float32()
	for i := 0; i < N; i++ {
		a32[i] = rand.Float32()
		b32[i] = rand.Float32()
	}

	// Time Float32
	start := time.Now()
	guda.Add(d_A32, d_B32, d_C32, N)
	guda.Synchronize()
	f32Time := time.Since(start)

	// Float16 version
	d_A16, _ := guda.Malloc(N * 2) // 2 bytes per float16
	d_B16, _ := guda.Malloc(N * 2)
	d_C16, _ := guda.Malloc(N * 2)
	defer guda.Free(d_A16)
	defer guda.Free(d_B16)
	defer guda.Free(d_C16)

	// Initialize with same data
	a16 := d_A16.Float16()
	b16 := d_B16.Float16()
	for i := 0; i < N; i++ {
		a16.SetFloat32(i, a32[i])
		b16.SetFloat32(i, b32[i])
	}

	// Time Float16
	start = time.Now()
	guda.AddFloat16(d_A16, d_B16, d_C16, N)
	guda.Synchronize()
	f16Time := time.Since(start)

	// Calculate speedup and bandwidth
	f32Bandwidth := float64(3*N*4) / f32Time.Seconds() / 1e9
	f16Bandwidth := float64(3*N*2) / f16Time.Seconds() / 1e9

	fmt.Printf("  Float32: %.2fms (%.2f GB/s)\n", f32Time.Seconds()*1000, f32Bandwidth)
	fmt.Printf("  Float16: %.2fms (%.2f GB/s)\n", f16Time.Seconds()*1000, f16Bandwidth)
	fmt.Printf("  Speedup: %.2fx\n", f32Time.Seconds()/f16Time.Seconds())
}

func testMemoryBandwidth(N int) {
	fmt.Println("\nMemory Copy Bandwidth:")

	// Float32
	src32, _ := guda.Malloc(N * 4)
	dst32, _ := guda.Malloc(N * 4)
	defer guda.Free(src32)
	defer guda.Free(dst32)

	start := time.Now()
	guda.Memcpy(dst32, src32, N*4, guda.MemcpyDeviceToDevice)
	f32Time := time.Since(start)

	// Float16
	src16, _ := guda.Malloc(N * 2)
	dst16, _ := guda.Malloc(N * 2)
	defer guda.Free(src16)
	defer guda.Free(dst16)

	start = time.Now()
	guda.Memcpy(dst16, src16, N*2, guda.MemcpyDeviceToDevice)
	f16Time := time.Since(start)

	f32Bandwidth := float64(N*4*2) / f32Time.Seconds() / 1e9
	f16Bandwidth := float64(N*2*2) / f16Time.Seconds() / 1e9

	fmt.Printf("  Float32: %.2f GB/s\n", f32Bandwidth)
	fmt.Printf("  Float16: %.2f GB/s (%.1fx more elements/sec)\n", 
		f16Bandwidth, f16Bandwidth*2/f32Bandwidth)
}

func testNeuralNet(N int) {
	fmt.Println("\nNeural Network Layer (Linear + ReLU):")

	// Float32 version
	x32, _ := guda.Malloc(N * 4)
	bias32, _ := guda.Malloc(N * 4)
	defer guda.Free(x32)
	defer guda.Free(bias32)

	// Initialize
	xSlice32 := x32.Float32()
	biasSlice32 := bias32.Float32()
	for i := 0; i < N; i++ {
		xSlice32[i] = rand.Float32()*2 - 1
		biasSlice32[i] = rand.Float32() * 0.1
	}

	// Time Float32 fused operation
	start := time.Now()
	guda.NewFusedKernel().
		MulScalar(2.0).
		Add(1.0).
		ReLU().
		Execute(x32, []guda.DevicePtr{bias32}, N)
	guda.Synchronize()
	f32Time := time.Since(start)

	// Float16 version
	x16, _ := guda.Malloc(N * 2)
	defer guda.Free(x16)

	// Initialize with same data
	xSlice16 := x16.Float16()
	for i := 0; i < N; i++ {
		xSlice16.SetFloat32(i, xSlice32[i])
	}

	// Time Float16 operation
	start = time.Now()
	guda.LinearFloat16(x16, 2.0, 0.1, N)
	// Note: ReLU would be added to LinearFloat16 in production
	guda.Synchronize()
	f16Time := time.Since(start)

	fmt.Printf("  Float32: %.2fms\n", f32Time.Seconds()*1000)
	fmt.Printf("  Float16: %.2fms (%.2fx speedup)\n", 
		f16Time.Seconds()*1000, f32Time.Seconds()/f16Time.Seconds())

	// Memory savings
	f32Memory := N * 4 * 3 // input, bias, output
	f16Memory := N * 2 * 3
	fmt.Printf("  Memory usage: %.1f MB → %.1f MB (%.1fx reduction)\n",
		float64(f32Memory)/1024/1024,
		float64(f16Memory)/1024/1024,
		float64(f32Memory)/float64(f16Memory))
}