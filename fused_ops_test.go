// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package guda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func TestFusedGEMMBiasReLU(t *testing.T) {
	// Test small matrices
	testCases := []struct {
		m, n, k int
		alpha   float32
	}{
		{8, 8, 8, 1.0},
		{16, 16, 16, 2.0},
		{32, 32, 32, 1.0},
		{64, 64, 64, 1.5},
	}
	
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%dx%dx%d", tc.m, tc.n, tc.k), func(t *testing.T) {
			// Create test matrices
			a := make([]float32, tc.m*tc.k)
			b := make([]float32, tc.k*tc.n)
			bias := make([]float32, tc.n)
			c := make([]float32, tc.m*tc.n)
			expected := make([]float32, tc.m*tc.n)
			
			// Initialize with random data
			rng := rand.New(rand.NewSource(42))
			for i := range a {
				a[i] = float32(rng.NormFloat64())
			}
			for i := range b {
				b[i] = float32(rng.NormFloat64())
			}
			for i := range bias {
				bias[i] = float32(rng.NormFloat64())
			}
			
			// Compute expected result (unfused)
			computeExpectedGEMMBiasReLU(tc.m, tc.n, tc.k, tc.alpha, a, tc.k, b, tc.n, bias, expected, tc.n)
			
			// Allocate device memory
			aPtr, _ := Malloc(len(a) * 4)
			bPtr, _ := Malloc(len(b) * 4)
			cPtr, _ := Malloc(len(c) * 4)
			biasPtr, _ := Malloc(len(bias) * 4)
			defer Free(aPtr)
			defer Free(bPtr)
			defer Free(cPtr)
			defer Free(biasPtr)
			
			// Copy data to device
			Memcpy(aPtr, a, len(a)*4, MemcpyHostToDevice)
			Memcpy(bPtr, b, len(b)*4, MemcpyHostToDevice)
			Memcpy(biasPtr, bias, len(bias)*4, MemcpyHostToDevice)
			
			// Compute fused result
			err := FusedGEMMBiasReLU(false, false, tc.m, tc.n, tc.k, tc.alpha, aPtr, tc.k, bPtr, tc.n, 0, cPtr, tc.n, biasPtr)
			if err != nil {
				t.Fatalf("FusedGEMMBiasReLU failed: %v", err)
			}
			
			// Copy result back
			Memcpy(c, cPtr, len(c)*4, MemcpyDeviceToHost)
			
			// Compare results
			// Use relative tolerance for float32 precision
			for i := range c {
				diff := math.Abs(float64(c[i] - expected[i]))
				relErr := diff / math.Abs(float64(expected[i]))
				// Use absolute tolerance for small values, relative for large
				if diff > 1e-4 && relErr > 1e-5 {
					t.Errorf("Mismatch at index %d: got %f, expected %f (diff=%e, rel=%e)", 
						i, c[i], expected[i], diff, relErr)
				}
			}
		})
	}
}

func computeExpectedGEMMBiasReLU(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, bias []float32, c []float32, ldc int) {
	// Traditional 3-pass implementation
	// Pass 1: GEMM
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			c[i*ldc+j] = alpha * sum
		}
	}
	
	// Pass 2: Add bias
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i*ldc+j] += bias[j]
		}
	}
	
	// Pass 3: ReLU
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if c[i*ldc+j] < 0 {
				c[i*ldc+j] = 0
			}
		}
	}
}

func BenchmarkFusedGEMMBiasReLUOps(b *testing.B) {
	sizes := []int{128, 256, 512, 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Fused_%d", size), func(b *testing.B) {
			m, n, k := size, size, size
			alpha := float32(1.0)
			
			a := make([]float32, m*k)
			bMat := make([]float32, k*n)
			bias := make([]float32, n)
			c := make([]float32, m*n)
			
			// Initialize
			for i := range a {
				a[i] = float32(i%100) * 0.01
			}
			for i := range bMat {
				bMat[i] = float32(i%100) * 0.01
			}
			for i := range bias {
				bias[i] = float32(i%10) * 0.1
			}
			
			// Allocate device memory
			aPtr, _ := Malloc(len(a) * 4)
			bPtr, _ := Malloc(len(bMat) * 4)
			cPtr, _ := Malloc(len(c) * 4)
			biasPtr, _ := Malloc(len(bias) * 4)
			defer Free(aPtr)
			defer Free(bPtr)
			defer Free(cPtr)
			defer Free(biasPtr)
			
			// Copy data to device
			Memcpy(aPtr, a, len(a)*4, MemcpyHostToDevice)
			Memcpy(bPtr, bMat, len(bMat)*4, MemcpyHostToDevice)
			Memcpy(biasPtr, bias, len(bias)*4, MemcpyHostToDevice)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FusedGEMMBiasReLU(false, false, m, n, k, alpha, aPtr, k, bPtr, n, 0, cPtr, n, biasPtr)
			}
			b.SetBytes(int64(m*k + k*n + n + m*n) * 4) // float32 = 4 bytes
		})
		
		b.Run(fmt.Sprintf("Unfused_%d", size), func(b *testing.B) {
			m, n, k := size, size, size
			alpha := float32(1.0)
			
			a := make([]float32, m*k)
			bMat := make([]float32, k*n)
			bias := make([]float32, n)
			c := make([]float32, m*n)
			
			// Initialize
			for i := range a {
				a[i] = float32(i%100) * 0.01
			}
			for i := range bMat {
				bMat[i] = float32(i%100) * 0.01
			}
			for i := range bias {
				bias[i] = float32(i%10) * 0.1
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				computeExpectedGEMMBiasReLU(m, n, k, alpha, a, k, bMat, n, bias, c, n)
			}
			b.SetBytes(int64(m*k + k*n + n + m*n) * 4)
		})
	}
}

// Benchmark memory bandwidth reduction
func BenchmarkMemoryBandwidthFused(b *testing.B) {
	// This benchmark shows the memory bandwidth savings
	size := 1024
	m, n, k := size, size, size
	
	a := make([]float32, m*k)
	bMat := make([]float32, k*n)
	bias := make([]float32, n)
	c := make([]float32, m*n)
	temp := make([]float32, m*n) // Extra buffer for unfused
	
	b.Run("FusedBandwidth", func(b *testing.B) {
		b.ResetTimer()
		bytes := 0
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// For simplicity, use the naive implementation directly
			fusedGEMMBiasReLUOptimized(m, n, k, 1.0, a, k, bMat, n, bias, c, n)
			// Fused: Read A, B, bias once. Write C once.
			bytes += (m*k + k*n + n + m*n) * 4
		}
		elapsed := time.Since(start).Seconds()
		b.ReportMetric(float64(bytes)/elapsed/1e9, "GB/s")
	})
	
	b.Run("UnfusedBandwidth", func(b *testing.B) {
		b.ResetTimer()
		bytes := 0
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// Pass 1: GEMM (read A, B, write temp)
			computeGEMMNaive(m, n, k, 1.0, a, k, bMat, n, temp, n)
			bytes += (m*k + k*n + m*n) * 4
			
			// Pass 2: Add bias (read temp, bias, write c)
			addBias(m, n, temp, n, bias, c, n)
			bytes += (m*n + n + m*n) * 4
			
			// Pass 3: ReLU (read c, write c)
			applyReLU(m, n, c, n)
			bytes += (m*n + m*n) * 4
		}
		elapsed := time.Since(start).Seconds()
		b.ReportMetric(float64(bytes)/elapsed/1e9, "GB/s")
	})
}

// Helper functions for unfused benchmark
func computeGEMMNaive(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			c[i*ldc+j] = alpha * sum
		}
	}
}

func addBias(m, n int, input []float32, ldi int, bias []float32, output []float32, ldo int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			output[i*ldo+j] = input[i*ldi+j] + bias[j]
		}
	}
}

func applyReLU(m, n int, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if c[i*ldc+j] < 0 {
				c[i*ldc+j] = 0
			}
		}
	}
}