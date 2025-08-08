package guda

import (
	"testing"
)

func BenchmarkFusedGEMMBiasReLU(b *testing.B) {
	sizes := []struct {
		name string
		m, n, k int
	}{
		{"Small_32x32x32", 32, 32, 32},
		{"Medium_128x128x128", 128, 128, 128},
		{"Large_512x512x512", 512, 512, 512},
		{"Transformer_768x768x768", 768, 768, 768},
	}
	
	for _, size := range sizes {
		b.Run(size.name+"_Naive", func(b *testing.B) {
			benchmarkFusedGEMMBiasReLUNaive(b, size.m, size.n, size.k)
		})
		
		b.Run(size.name+"_Optimized", func(b *testing.B) {
			benchmarkFusedGEMMBiasReLUOptimized(b, size.m, size.n, size.k)
		})
	}
}

func benchmarkFusedGEMMBiasReLUNaive(b *testing.B, m, n, k int) {
	// Allocate matrices
	d_a, _ := Malloc(m * k * 4)
	d_b, _ := Malloc(k * n * 4)
	d_c, _ := Malloc(m * n * 4)
	d_bias, _ := Malloc(n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	defer Free(d_bias)
	
	// Initialize with test data
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%100) * 0.01
	}
	for i := range biasData {
		biasData[i] = float32(i) * 0.1
	}
	
	Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
	Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
	Memcpy(d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	// Force to use naive implementation
	oldHasAVX2 := hasAVX2
	hasAVX2 = false
	defer func() { hasAVX2 = oldHasAVX2 }()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedGEMMBiasReLU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n, d_bias)
		Synchronize()
	}
}

func benchmarkFusedGEMMBiasReLUOptimized(b *testing.B, m, n, k int) {
	// Allocate matrices
	d_a, _ := Malloc(m * k * 4)
	d_b, _ := Malloc(k * n * 4)
	d_c, _ := Malloc(m * n * 4)
	d_bias, _ := Malloc(n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	defer Free(d_bias)
	
	// Initialize with test data
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%100) * 0.01
	}
	for i := range biasData {
		biasData[i] = float32(i) * 0.1
	}
	
	Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
	Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
	Memcpy(d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	// Use optimized implementation with AVX2
	if !hasAVX2 {
		b.Skip("AVX2 not available")
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedGEMMBiasReLU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n, d_bias)
		Synchronize()
	}
}

// Compare fused vs separate operations
func BenchmarkSeparateOperations(b *testing.B) {
	m, n, k := 512, 512, 512
	
	// Allocate matrices
	d_a, _ := Malloc(m * k * 4)
	d_b, _ := Malloc(k * n * 4)
	d_c, _ := Malloc(m * n * 4)
	d_bias, _ := Malloc(n * 4)
	defer Free(d_a)
	defer Free(d_b)
	defer Free(d_c)
	defer Free(d_bias)
	
	// Initialize with test data
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	biasData := make([]float32, n)
	
	for i := range aData {
		aData[i] = float32(i%100) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%100) * 0.01
	}
	for i := range biasData {
		biasData[i] = float32(i) * 0.1
	}
	
	Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
	Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
	Memcpy(d_bias, biasData, len(biasData)*4, MemcpyHostToDevice)
	
	b.Run("Separate", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// GEMM
			GEMM(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n)
			
			// Add bias
			AddBiasReLU(d_c, d_bias, m*n)
			
			Synchronize()
		}
	})
	
	b.Run("Fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			FusedGEMMBiasReLU(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n, d_bias)
			Synchronize()
		}
	})
}