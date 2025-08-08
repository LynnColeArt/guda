// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import (
	"fmt"
	"testing"
)

// Benchmark AVX2 vs SSE implementations
func BenchmarkAxpyComparison(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		x := make([]float32, size)
		y := make([]float32, size)
		alpha := float32(2.5)
		
		// Initialize with test data
		for i := range x {
			x[i] = float32(i)
			y[i] = float32(i * 2)
		}
		
		b.Run(fmt.Sprintf("SSE/size=%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				axpyUnitarySSE(alpha, x, y)
			}
		})
		
		if hasAVX2 {
			b.Run(fmt.Sprintf("AVX2/size=%d", size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					axpyUnitaryAVX2(alpha, x, y)
				}
			})
		}
		
		if hasAVX2 && hasFMA {
			b.Run(fmt.Sprintf("FMA/size=%d", size), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					axpyUnitaryFMA(alpha, x, y)
				}
			})
		}
	}
}

func BenchmarkDotComparison(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		x := make([]float32, size)
		y := make([]float32, size)
		
		// Initialize with test data
		for i := range x {
			x[i] = float32(i)
			y[i] = float32(i * 2)
		}
		
		b.Run(fmt.Sprintf("SSE/size=%d", size), func(b *testing.B) {
			var sum float32
			for i := 0; i < b.N; i++ {
				sum = dotUnitarySSE(x, y)
			}
			_ = sum
		})
		
		if hasAVX2 {
			b.Run(fmt.Sprintf("AVX2/size=%d", size), func(b *testing.B) {
				var sum float32
				for i := 0; i < b.N; i++ {
					sum = dotUnitaryAVX2(x, y)
				}
				_ = sum
			})
		}
	}
}

// Test correctness
func TestAxpyAVX2Correctness(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}
	
	sizes := []int{1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000}
	alpha := float32(2.5)
	
	for _, size := range sizes {
		x := make([]float32, size)
		ySSE := make([]float32, size)
		yAVX2 := make([]float32, size)
		
		// Initialize with test data
		for i := range x {
			x[i] = float32(i) * 0.1
			ySSE[i] = float32(i) * 0.2
			yAVX2[i] = float32(i) * 0.2
		}
		
		// Run both implementations
		axpyUnitarySSE(alpha, x, ySSE)
		axpyUnitaryAVX2(alpha, x, yAVX2)
		
		// Compare results
		for i := range ySSE {
			if ySSE[i] != yAVX2[i] {
				t.Errorf("Size %d: Mismatch at index %d: SSE=%f, AVX2=%f", size, i, ySSE[i], yAVX2[i])
			}
		}
	}
}

func TestDotAVX2Correctness(t *testing.T) {
	if !hasAVX2 {
		t.Skip("AVX2 not available")
	}
	
	sizes := []int{1, 7, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000}
	
	for _, size := range sizes {
		x := make([]float32, size)
		y := make([]float32, size)
		
		// Initialize with test data
		for i := range x {
			x[i] = float32(i) * 0.1
			y[i] = float32(i) * 0.2
		}
		
		// Run both implementations
		sumSSE := dotUnitarySSE(x, y)
		sumAVX2 := dotUnitaryAVX2(x, y)
		
		// Compare results (allow for small floating point differences)
		diff := sumSSE - sumAVX2
		if diff < 0 {
			diff = -diff
		}
		tolerance := float32(size) * 1e-5 // Scale tolerance with size
		if diff > tolerance {
			t.Errorf("Size %d: Mismatch: SSE=%f, AVX2=%f, diff=%f", size, sumSSE, sumAVX2, diff)
		}
	}
}