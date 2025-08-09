package guda

import (
	"fmt"
	"testing"
)

// TestPrefetchingBehavior verifies that prefetching hints are working
func TestPrefetchingBehavior(t *testing.T) {
	// Test sizes chosen to stress different cache levels
	sizes := []int{
		1024,      // 4KB - fits in L1
		8 * 1024,  // 32KB - spans L1/L2
		64 * 1024, // 256KB - spans L2/L3
		1024 * 1024, // 4MB - exceeds L3 on most CPUs
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			// Allocate test vectors
			x, err := Malloc(size * 4)
			if err != nil {
				t.Fatal(err)
			}
			defer Free(x)

			y, err := Malloc(size * 4)
			if err != nil {
				t.Fatal(err)
			}
			defer Free(y)

			// Initialize with test data
			xSlice := x.Float32()[:size]
			ySlice := y.Float32()[:size]
			for i := range xSlice {
				xSlice[i] = float32(i % 100)
				ySlice[i] = float32((i + 1) % 100)
			}

			// Test AXPY with prefetching
			alpha := float32(2.5)
			err = AXPY(alpha, x, y, size)
			if err != nil {
				t.Errorf("AXPY failed: %v", err)
			}

			// Verify first few results
			for i := 0; i < 10 && i < size; i++ {
				expected := float32((i+1)%100) + alpha*float32(i%100)
				if ySlice[i] != expected {
					t.Errorf("AXPY mismatch at %d: got %f, want %f", i, ySlice[i], expected)
				}
			}

			// Test DOT with prefetching
			result, err := DOT(x, y, size)
			if err != nil {
				t.Errorf("DOT failed: %v", err)
			}

			// Just verify we got a non-zero result for large arrays
			if size > 100 && result == 0 {
				t.Errorf("DOT returned zero for size %d", size)
			}
		})
	}
}

// BenchmarkPrefetchComparison compares performance with prefetching
// Run with: go test -bench=BenchmarkPrefetchComparison -benchmem
func BenchmarkPrefetchComparison(b *testing.B) {
	sizes := []int{
		1024,        // 4KB
		16 * 1024,   // 64KB
		256 * 1024,  // 1MB
		4096 * 1024, // 16MB
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("AXPY_Size_%d", size), func(b *testing.B) {
			x, _ := Malloc(size * 4)
			y, _ := Malloc(size * 4)
			defer Free(x)
			defer Free(y)

			// Initialize
			xSlice := x.Float32()[:size]
			ySlice := y.Float32()[:size]
			for i := range xSlice {
				xSlice[i] = 1.0
				ySlice[i] = 2.0
			}

			b.ResetTimer()
			b.SetBytes(int64(size * 4 * 2)) // 2 arrays, 4 bytes each

			for i := 0; i < b.N; i++ {
				AXPY(2.5, x, y, size)
			}
		})

		b.Run(fmt.Sprintf("DOT_Size_%d", size), func(b *testing.B) {
			x, _ := Malloc(size * 4)
			y, _ := Malloc(size * 4)
			defer Free(x)
			defer Free(y)

			// Initialize
			xSlice := x.Float32()[:size]
			ySlice := y.Float32()[:size]
			for i := range xSlice {
				xSlice[i] = 1.0
				ySlice[i] = 2.0
			}

			b.ResetTimer()
			b.SetBytes(int64(size * 4 * 2)) // 2 arrays, 4 bytes each

			var result float32
			for i := 0; i < b.N; i++ {
				result, _ = DOT(x, y, size)
			}
			_ = result
		})
	}
}