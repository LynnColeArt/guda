package guda

import (
	"fmt"
	"math"
	"testing"
)

func TestGenerateFloat32(t *testing.T) {
	// Test deterministic generation
	data1 := GenerateFloat32(100, 12345)
	data2 := GenerateFloat32(100, 12345)
	
	if !SlicesAlmostEqual(data1, data2, 0) {
		t.Error("GenerateFloat32 is not deterministic")
	}
	
	// Test different seeds produce different data
	data3 := GenerateFloat32(100, 54321)
	if SlicesAlmostEqual(data1, data3, 0) {
		t.Error("Different seeds should produce different data")
	}
	
	// Test range [0, 1)
	for i, v := range data1 {
		if v < 0 || v >= 1 {
			t.Errorf("Value %d out of range [0, 1): %f", i, v)
		}
	}
}

func TestGenerateFloat32Range(t *testing.T) {
	min := float32(-5.0)
	max := float32(10.0)
	data := GenerateFloat32Range(1000, 42, min, max)
	
	for i, v := range data {
		if v < min || v >= max {
			t.Errorf("Value %d out of range [%f, %f): %f", i, min, max, v)
		}
	}
}

func TestGenerateFloat32EdgeCases(t *testing.T) {
	edges := GenerateFloat32EdgeCases()
	
	// Check we have the expected special values
	hasZero := false
	hasNaN := false
	hasInf := false
	
	for _, v := range edges {
		if v == 0.0 {
			hasZero = true
		}
		if math.IsNaN(float64(v)) {
			hasNaN = true
		}
		if math.IsInf(float64(v), 0) {
			hasInf = true
		}
	}
	
	if !hasZero {
		t.Error("Edge cases should include zero")
	}
	if !hasNaN {
		t.Error("Edge cases should include NaN")
	}
	if !hasInf {
		t.Error("Edge cases should include infinity")
	}
}

func TestGenerateIdentityMatrix(t *testing.T) {
	size := 4
	identity := GenerateIdentityMatrix(size)
	
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			expected := float32(0.0)
			if i == j {
				expected = 1.0
			}
			actual := identity[i*size+j]
			if actual != expected {
				t.Errorf("Identity[%d,%d] = %f, expected %f", i, j, actual, expected)
			}
		}
	}
}

func TestAlmostEqual(t *testing.T) {
	tests := []struct {
		a, b      float32
		tolerance float32
		expected  bool
		name      string
	}{
		{1.0, 1.0, 0.0, true, "exact equal"},
		{1.0, 1.0001, 0.001, true, "within tolerance"},
		{1.0, 1.01, 0.001, false, "outside tolerance"},
		{float32(math.NaN()), float32(math.NaN()), 0.0, true, "NaN equals NaN"},
		{float32(math.Inf(1)), float32(math.Inf(1)), 0.0, true, "positive inf"},
		{float32(math.Inf(-1)), float32(math.Inf(-1)), 0.0, true, "negative inf"},
		{float32(math.Inf(1)), float32(math.Inf(-1)), 0.0, false, "different inf"},
	}
	
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := AlmostEqual(tc.a, tc.b, tc.tolerance)
			if result != tc.expected {
				t.Errorf("AlmostEqual(%f, %f, %f) = %v, expected %v",
					tc.a, tc.b, tc.tolerance, result, tc.expected)
			}
		})
	}
}

func TestTestDataSizes(t *testing.T) {
	sizes := TestDataSizes()
	
	// Verify sizes are in ascending order
	for i := 1; i < len(sizes); i++ {
		if sizes[i] <= sizes[i-1] {
			t.Errorf("Sizes should be ascending: %d <= %d", sizes[i], sizes[i-1])
		}
	}
	
	// Verify we have appropriate cache-sized entries
	hasL1 := false
	hasL2 := false
	hasL3 := false
	
	for _, size := range sizes {
		bytes := size * 4 // float32
		if bytes >= 32*1024 && bytes <= 64*1024 {
			hasL1 = true
		}
		if bytes >= 256*1024 && bytes <= 512*1024 {
			hasL2 = true
		}
		if bytes >= 1024*1024 && bytes <= 8*1024*1024 {
			hasL3 = true
		}
	}
	
	if !hasL1 || !hasL2 || !hasL3 {
		t.Error("Missing cache-appropriate test sizes")
	}
}

func BenchmarkGenerateFloat32(b *testing.B) {
	sizes := []int{1024, 64 * 1024, 1024 * 1024}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			for i := 0; i < b.N; i++ {
				_ = GenerateFloat32(size, uint64(i))
			}
		})
	}
}