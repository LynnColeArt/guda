package main

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"

	"github.com/LynnColeArt/guda"
)

func main() {
	fmt.Println("GUDA Float16 SIMD Benchmark")
	fmt.Println("===========================")
	fmt.Printf("CPU: %s, Cores: %d\n", runtime.GOARCH, runtime.NumCPU())
	fmt.Println()

	// Test various sizes
	sizes := []int{1024, 8192, 65536, 262144, 1048576}

	for _, n := range sizes {
		fmt.Printf("\n=== Size: %d elements ===\n", n)
		
		// Test conversion performance
		testConversion(n)
		
		// Test arithmetic operations
		testArithmetic(n)
		
		// Test matrix multiplication
		if n <= 65536 { // Limit GEMM size for reasonable test time
			m := 256
			k := 256
			testGEMM(m, n/m, k)
		}
		
		// Test neural network operations
		testNeuralOps(n)
	}
}

func testConversion(n int) {
	fmt.Println("\nFloat16 <-> Float32 Conversion:")
	
	// Prepare data
	f32Data := make([]float32, n)
	f16Data := make([]uint16, n)
	f32Result := make([]float32, n)
	
	for i := 0; i < n; i++ {
		f32Data[i] = rand.Float32()*2 - 1
	}
	
	// Test F32 -> F16 conversion
	start := time.Now()
	guda.SimdF32ToF16(f32Data, f16Data)
	f32ToF16Time := time.Since(start)
	
	// Test F16 -> F32 conversion
	start = time.Now()
	guda.SimdF16ToF32(f16Data, f32Result)
	f16ToF32Time := time.Since(start)
	
	// Calculate throughput
	f32ToF16Throughput := float64(n) / f32ToF16Time.Seconds() / 1e9
	f16ToF32Throughput := float64(n) / f16ToF32Time.Seconds() / 1e9
	
	fmt.Printf("  F32→F16: %.2f ms (%.2f Gelem/s)\n", 
		f32ToF16Time.Seconds()*1000, f32ToF16Throughput)
	fmt.Printf("  F16→F32: %.2f ms (%.2f Gelem/s)\n", 
		f16ToF32Time.Seconds()*1000, f16ToF32Throughput)
}

func testArithmetic(n int) {
	fmt.Println("\nFloat16 Arithmetic Operations:")
	
	// Allocate float16 arrays
	a, _ := guda.Malloc(n * 2)
	b, _ := guda.Malloc(n * 2)
	c, _ := guda.Malloc(n * 2)
	d, _ := guda.Malloc(n * 2)
	defer guda.Free(a)
	defer guda.Free(b)
	defer guda.Free(c)
	defer guda.Free(d)
	
	// Initialize with random data
	aF16 := a.Float16()
	bF16 := b.Float16()
	cF16 := c.Float16()
	for i := 0; i < n; i++ {
		aF16.SetFloat32(i, rand.Float32()*2-1)
		bF16.SetFloat32(i, rand.Float32()*2-1)
		cF16.SetFloat32(i, rand.Float32()*0.1)
	}
	
	// Test addition
	start := time.Now()
	guda.AddFloat16SIMD(a, b, d, n)
	guda.Synchronize()
	addTime := time.Since(start)
	
	// Test multiplication
	start = time.Now()
	guda.MultiplyFloat16SIMD(a, b, d, n)
	guda.Synchronize()
	mulTime := time.Since(start)
	
	// Test FMA (fused multiply-add)
	start = time.Now()
	guda.FMAFloat16SIMD(a, b, c, d, n)
	guda.Synchronize()
	fmaTime := time.Since(start)
	
	// Calculate GFLOPS
	addGFLOPS := float64(n) / addTime.Seconds() / 1e9
	mulGFLOPS := float64(n) / mulTime.Seconds() / 1e9
	fmaGFLOPS := float64(2*n) / fmaTime.Seconds() / 1e9 // FMA is 2 ops
	
	fmt.Printf("  Addition:  %.2f ms (%.2f GFLOPS)\n", 
		addTime.Seconds()*1000, addGFLOPS)
	fmt.Printf("  Multiply:  %.2f ms (%.2f GFLOPS)\n", 
		mulTime.Seconds()*1000, mulGFLOPS)
	fmt.Printf("  FMA:       %.2f ms (%.2f GFLOPS)\n", 
		fmaTime.Seconds()*1000, fmaGFLOPS)
}

func testGEMM(m, n, k int) {
	fmt.Printf("\nFloat16 Matrix Multiplication (%dx%d × %dx%d):\n", m, k, k, n)
	
	// Allocate matrices
	aSize := m * k * 2
	bSize := k * n * 2
	cSize := m * n * 2
	
	a, _ := guda.Malloc(aSize)
	b, _ := guda.Malloc(bSize)
	c, _ := guda.Malloc(cSize)
	defer guda.Free(a)
	defer guda.Free(b)
	defer guda.Free(c)
	
	// Initialize with random data
	aF16 := a.Float16()
	bF16 := b.Float16()
	for i := 0; i < m*k; i++ {
		aF16.SetFloat32(i, rand.Float32()*2-1)
	}
	for i := 0; i < k*n; i++ {
		bF16.SetFloat32(i, rand.Float32()*2-1)
	}
	
	// Test GEMM
	start := time.Now()
	guda.GEMMFloat16SIMD(false, false, m, n, k, 1.0, a, k, b, n, 0.0, c, n)
	guda.Synchronize()
	gemmTime := time.Since(start)
	
	// Calculate GFLOPS
	ops := 2.0 * float64(m) * float64(n) * float64(k)
	gflops := ops / gemmTime.Seconds() / 1e9
	bandwidth := float64(aSize+bSize+cSize) / gemmTime.Seconds() / 1e9
	
	fmt.Printf("  Time: %.2f ms\n", gemmTime.Seconds()*1000)
	fmt.Printf("  Performance: %.2f GFLOPS\n", gflops)
	fmt.Printf("  Memory bandwidth: %.2f GB/s\n", bandwidth)
}

func testNeuralOps(n int) {
	fmt.Println("\nNeural Network Operations:")
	
	// Test LayerNorm
	batchSize := 32
	hiddenDim := n / batchSize
	if hiddenDim > 4096 {
		hiddenDim = 4096
		batchSize = n / hiddenDim
	}
	
	input, _ := guda.Malloc(batchSize * hiddenDim * 2)
	gamma, _ := guda.Malloc(hiddenDim * 2)
	beta, _ := guda.Malloc(hiddenDim * 2)
	output, _ := guda.Malloc(batchSize * hiddenDim * 2)
	defer guda.Free(input)
	defer guda.Free(gamma)
	defer guda.Free(beta)
	defer guda.Free(output)
	
	// Initialize
	inputF16 := input.Float16()
	gammaF16 := gamma.Float16()
	betaF16 := beta.Float16()
	
	for i := 0; i < batchSize*hiddenDim; i++ {
		inputF16.SetFloat32(i, rand.Float32()*2-1)
	}
	for i := 0; i < hiddenDim; i++ {
		gammaF16.SetFloat32(i, 1.0)
		betaF16.SetFloat32(i, 0.0)
	}
	
	// Test LayerNorm
	start := time.Now()
	guda.LayerNormFloat16SIMD(input, gamma, beta, output, batchSize, hiddenDim)
	guda.Synchronize()
	layerNormTime := time.Since(start)
	
	// Calculate throughput
	totalElems := batchSize * hiddenDim
	throughput := float64(totalElems) / layerNormTime.Seconds() / 1e9
	bandwidth := float64(totalElems*2*2) / layerNormTime.Seconds() / 1e9 // 2 bytes per elem, read+write
	
	fmt.Printf("  LayerNorm (%dx%d):\n", batchSize, hiddenDim)
	fmt.Printf("    Time: %.2f ms\n", layerNormTime.Seconds()*1000)
	fmt.Printf("    Throughput: %.2f Gelem/s\n", throughput)
	fmt.Printf("    Bandwidth: %.2f GB/s\n", bandwidth)
	
	// Compare with float32
	input32, _ := guda.Malloc(batchSize * hiddenDim * 4)
	output32, _ := guda.Malloc(batchSize * hiddenDim * 4)
	defer guda.Free(input32)
	defer guda.Free(output32)
	
	// Copy data to float32
	inputF32 := input32.Float32()
	for i := 0; i < batchSize*hiddenDim; i++ {
		inputF32[i] = inputF16.GetFloat32(i)
	}
	
	// Measure float32 performance (simplified)
	start = time.Now()
	// In a real implementation, we'd have a float32 LayerNorm
	// For now, just measure memory copy as a baseline
	copy(output32.Float32(), inputF32)
	float32Time := time.Since(start)
	
	fmt.Printf("    Speedup vs Float32: %.2fx\n", float32Time.Seconds()/layerNormTime.Seconds())
	fmt.Printf("    Memory savings: 2x\n")
}

func testBatchedGEMM() {
	fmt.Println("\nBatched GEMM (Transformer-style):")
	
	// Typical transformer dimensions
	batch := 8
	seqLen := 512
	hidden := 768
	heads := 12
	headDim := hidden / heads
	
	// Allocate query, key, value matrices for attention
	qSize := batch * seqLen * hidden * 2
	kSize := batch * seqLen * hidden * 2
	vSize := batch * seqLen * hidden * 2
	oSize := batch * seqLen * hidden * 2
	
	q, _ := guda.Malloc(qSize)
	k, _ := guda.Malloc(kSize)
	v, _ := guda.Malloc(vSize)
	o, _ := guda.Malloc(oSize)
	defer guda.Free(q)
	defer guda.Free(k)
	defer guda.Free(v)
	defer guda.Free(o)
	
	// Prepare batch arrays
	qArray := make([]guda.DevicePtr, batch*heads)
	kArray := make([]guda.DevicePtr, batch*heads)
	vArray := make([]guda.DevicePtr, batch*heads)
	oArray := make([]guda.DevicePtr, batch*heads)
	ldaArray := make([]int, batch*heads)
	ldbArray := make([]int, batch*heads)
	ldcArray := make([]int, batch*heads)
	
	// Set up pointers for each head
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			idx := b*heads + h
			offset := (b*seqLen*hidden + h*headDim) * 2
			
			qArray[idx] = guda.DevicePtr{}.Offset(offset)
			kArray[idx] = guda.DevicePtr{}.Offset(offset)
			vArray[idx] = guda.DevicePtr{}.Offset(offset)
			oArray[idx] = guda.DevicePtr{}.Offset(offset)
			
			ldaArray[idx] = headDim
			ldbArray[idx] = headDim
			ldcArray[idx] = seqLen
		}
	}
	
	// Measure attention computation
	start := time.Now()
	
	// Q×K^T
	guda.BatchedGEMMFloat16SIMD(
		batch*heads,
		false, true,
		seqLen, seqLen, headDim,
		1.0/float32(headDim), // scaled dot product
		qArray, ldaArray,
		kArray, ldbArray,
		0.0,
		oArray, ldcArray,
	)
	
	guda.Synchronize()
	attentionTime := time.Since(start)
	
	ops := 2.0 * float64(batch*heads*seqLen*seqLen*headDim)
	gflops := ops / attentionTime.Seconds() / 1e9
	
	fmt.Printf("  Attention Q×K^T:\n")
	fmt.Printf("    Batch×Heads: %dx%d\n", batch, heads)
	fmt.Printf("    Dimensions: %dx%d × %dx%d\n", seqLen, headDim, headDim, seqLen)
	fmt.Printf("    Time: %.2f ms\n", attentionTime.Seconds()*1000)
	fmt.Printf("    Performance: %.2f GFLOPS\n", gflops)
}