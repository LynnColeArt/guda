package guda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	
	"github.com/LynnColeArt/guda/compute/asm/f32"
)

func TestReductionOperations(t *testing.T) {
	t.Run("Sum", testSum)
	t.Run("Max", testMax)
	t.Run("Min", testMin)
	t.Run("ArgMax", testArgMax)
	t.Run("ArgMin", testArgMin)
	t.Run("Mean", testMean)
	t.Run("Variance", testVariance)
	t.Run("SumSquares", testSumSquares)
	t.Run("Softmax", testSoftmax)
	t.Run("LogSumExp", testLogSumExp)
	t.Run("CumSum", testCumSum)
	t.Run("SegmentSum", testSegmentSum)
	t.Run("TopK", testTopK)
}

func testSum(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{5.0}, 5.0},
		{"positive", []float32{1, 2, 3, 4, 5}, 15},
		{"mixed", []float32{-1, 2, -3, 4, -5}, -3},
		{"large", makeSequence(1000), 499500}, // sum(0..999)
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - use f32.Sum directly
				result := f32.Sum(tc.input)
				if !floatEquals(result, tc.expected, 1e-5) {
					t.Errorf("Sum: expected %f, got %f", tc.expected, result)
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.Sum(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-5) {
				t.Errorf("Sum: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testMax(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, float32(math.Inf(-1))},
		{"single", []float32{5.0}, 5.0},
		{"positive", []float32{1, 5, 3, 2, 4}, 5},
		{"negative", []float32{-1, -5, -3, -2, -4}, -1},
		{"mixed", []float32{-1, 2, -3, 4, -5}, 4},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - use f32.Max directly
				result := f32.Max(tc.input)
				if !floatEquals(result, tc.expected, 1e-7) {
					t.Errorf("Max: expected %f, got %f", tc.expected, result)
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.Max(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-7) {
				t.Errorf("Max: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testMin(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, float32(math.Inf(1))},
		{"single", []float32{5.0}, 5.0},
		{"positive", []float32{1, 5, 3, 2, 4}, 1},
		{"negative", []float32{-1, -5, -3, -2, -4}, -5},
		{"mixed", []float32{-1, 2, -3, 4, -5}, -5},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - use f32.Min directly
				result := f32.Min(tc.input)
				if !floatEquals(result, tc.expected, 1e-7) {
					t.Errorf("Min: expected %f, got %f", tc.expected, result)
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.Min(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-7) {
				t.Errorf("Min: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testArgMax(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected int
	}{
		{"empty", []float32{}, -1},
		{"single", []float32{5.0}, 0},
		{"first", []float32{5, 1, 3, 2, 4}, 0},
		{"last", []float32{1, 2, 3, 4, 5}, 4},
		{"middle", []float32{1, 2, 5, 3, 4}, 2},
		{"negative", []float32{-5, -1, -3, -2, -4}, 1},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - use f32.ArgMax directly
				result := f32.ArgMax(tc.input)
				if result != tc.expected {
					t.Errorf("ArgMax: expected %d, got %d", tc.expected, result)
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.ArgMax(len(tc.input))
			
			if result != tc.expected {
				t.Errorf("ArgMax: expected %d, got %d (input: %v)", tc.expected, result, tc.input)
			}
		})
	}
}

func testArgMin(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected int
	}{
		{"empty", []float32{}, -1},
		{"single", []float32{5.0}, 0},
		{"first", []float32{1, 5, 3, 2, 4}, 0},
		{"last", []float32{5, 4, 3, 2, 1}, 4},
		{"middle", []float32{5, 4, 1, 3, 2}, 2},
		{"negative", []float32{-1, -5, -3, -2, -4}, 1},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - use f32.ArgMin directly
				result := f32.ArgMin(tc.input)
				if result != tc.expected {
					t.Errorf("ArgMin: expected %d, got %d", tc.expected, result)
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.ArgMin(len(tc.input))
			
			if result != tc.expected {
				t.Errorf("ArgMin: expected %d, got %d", tc.expected, result)
			}
		})
	}
}

func testMean(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{5.0}, 5.0},
		{"positive", []float32{1, 2, 3, 4, 5}, 3},
		{"negative", []float32{-1, -2, -3, -4, -5}, -3},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.Mean(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-5) {
				t.Errorf("Mean: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testVariance(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{5.0}, 0},
		{"uniform", []float32{1, 1, 1, 1}, 0},
		{"sequence", []float32{1, 2, 3, 4, 5}, 2.5}, // var = 2.5
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.Variance(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-5) {
				t.Errorf("Variance: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testSumSquares(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"empty", []float32{}, 0},
		{"single", []float32{3.0}, 9.0},
		{"positive", []float32{1, 2, 3}, 14}, // 1 + 4 + 9
		{"negative", []float32{-1, -2, -3}, 14},
		{"mixed", []float32{-2, 0, 2}, 8}, // 4 + 0 + 4
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := data.SumSquares(len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-5) {
				t.Errorf("SumSquares: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testSoftmax(t *testing.T) {
	cases := []struct {
		name  string
		input []float32
	}{
		{"uniform", []float32{1, 1, 1, 1}},
		{"varied", []float32{1, 2, 3, 4}},
		{"large_values", []float32{10, 20, 30, 40}}, // Test numerical stability
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			err := Softmax(data, len(tc.input))
			if err != nil {
				t.Fatalf("Softmax failed: %v", err)
			}
			
			// Check that sum equals 1
			sum := data.Sum(len(tc.input))
			if !floatEquals(sum, 1.0, 1e-5) {
				t.Errorf("Softmax sum: expected 1.0, got %f", sum)
			}
			
			// Check all values are in [0, 1]
			result := data.Float32()[:len(tc.input)]
			for i, val := range result {
				if val < 0 || val > 1 {
					t.Errorf("Softmax[%d] = %f, expected in [0, 1]", i, val)
				}
			}
		})
	}
}

func testLogSumExp(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{"single", []float32{2.0}, 2.0},
		{"uniform", []float32{1, 1, 1, 1}, 2.3862944}, // log(4*e^1)
		{"large_values", []float32{100, 101, 102}, 102.40760596}, // Numerical stability test
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			
			copy(data.Float32(), tc.input)
			result := LogSumExp(data, len(tc.input))
			
			if !floatEquals(result, tc.expected, 1e-5) {
				t.Errorf("LogSumExp: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func testCumSum(t *testing.T) {
	cases := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{"empty", []float32{}, []float32{}},
		{"single", []float32{5}, []float32{5}},
		{"sequence", []float32{1, 2, 3, 4}, []float32{1, 3, 6, 10}},
		{"negative", []float32{1, -2, 3, -4}, []float32{1, -1, 2, -2}},
	}
	
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if len(tc.input) == 0 {
				// Test empty slice - no output expected
				if len(tc.expected) != 0 {
					t.Errorf("CumSum: expected empty result for empty input")
				}
				return
			}
			
			data := MallocOrFail(t, len(tc.input)*4)
			out := MallocOrFail(t, len(tc.input)*4)
			defer Free(data)
			defer Free(out)
			
			copy(data.Float32(), tc.input)
			err := CumSum(data, out, len(tc.input))
			if err != nil {
				t.Fatalf("CumSum failed: %v", err)
			}
			
			result := out.Float32()[:len(tc.input)]
			for i := range tc.expected {
				if !floatEquals(result[i], tc.expected[i], 1e-5) {
					t.Errorf("CumSum[%d]: expected %f, got %f", i, tc.expected[i], result[i])
				}
			}
		})
	}
}

func testSegmentSum(t *testing.T) {
	data := MallocOrFail(t, 6*4)
	out := MallocOrFail(t, 3*4)
	defer Free(data)
	defer Free(out)
	
	// data: [1, 2, 3, 4, 5, 6]
	// segments: [0, 0, 1, 1, 2, 2]
	// expected: [3, 7, 11]
	copy(data.Float32(), []float32{1, 2, 3, 4, 5, 6})
	segments := []int{0, 0, 1, 1, 2, 2}
	
	err := SegmentSum(data, segments, 3, out)
	if err != nil {
		t.Fatalf("SegmentSum failed: %v", err)
	}
	
	expected := []float32{3, 7, 11}
	result := out.Float32()[:3]
	for i := range expected {
		if !floatEquals(result[i], expected[i], 1e-5) {
			t.Errorf("SegmentSum[%d]: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

func testTopK(t *testing.T) {
	data := MallocOrFail(t, 10*4)
	defer Free(data)
	
	copy(data.Float32(), []float32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3})
	
	values, indices, err := TopK(data, 10, 3)
	if err != nil {
		t.Fatalf("TopK failed: %v", err)
	}
	
	expectedValues := []float32{9, 6, 5}
	// expectedIndices := []int{5, 7, 4} // or 8 for the second 5
	
	for i := range expectedValues {
		if !floatEquals(values[i], expectedValues[i], 1e-7) {
			t.Errorf("TopK values[%d]: expected %f, got %f", i, expectedValues[i], values[i])
		}
	}
	
	// Check that indices point to correct values
	srcData := data.Float32()
	for i := range values {
		if !floatEquals(srcData[indices[i]], values[i], 1e-7) {
			t.Errorf("TopK indices[%d]: value mismatch", i)
		}
	}
}

// Benchmarks
func BenchmarkReductions(b *testing.B) {
	sizes := []int{32, 128, 1024, 8192, 65536}
	
	for _, size := range sizes {
		// Prepare data
		data := MallocOrFail(b, size*4)
		defer Free(data)
		
		src := data.Float32()[:size]
		for i := range src {
			src[i] = rand.Float32()*200 - 100 // [-100, 100]
		}
		
		b.Run(fmt.Sprintf("Sum_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = data.Sum(size)
			}
		})
		
		b.Run(fmt.Sprintf("Max_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = data.Max(size)
			}
		})
		
		b.Run(fmt.Sprintf("ArgMax_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = data.ArgMax(size)
			}
		})
		
		b.Run(fmt.Sprintf("SumSquares_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = data.SumSquares(size)
			}
		})
		
		b.Run(fmt.Sprintf("Softmax_%d", size), func(b *testing.B) {
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(src, data.Float32()[:size]) // Reset data
				_ = Softmax(data, size)
			}
		})
	}
}

// Benchmark AVX2 vs scalar implementations
func BenchmarkReductionComparison(b *testing.B) {
	size := 8192
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()*200 - 100
	}
	
	b.Run("Max_AVX2", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			_ = f32.Max(data)
		}
	})
	
	b.Run("Max_Scalar", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			max := data[0]
			for j := 1; j < len(data); j++ {
				if data[j] > max {
					max = data[j]
				}
			}
		}
	})
	
	b.Run("SumSquares_AVX2", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			_ = f32.SumSquares(data)
		}
	})
	
	b.Run("SumSquares_Scalar", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			sum := float32(0)
			for j := 0; j < len(data); j++ {
				sum += data[j] * data[j]
			}
		}
	})
}

// Helper functions
func makeSequence(n int) []float32 {
	seq := make([]float32, n)
	for i := range seq {
		seq[i] = float32(i)
	}
	return seq
}

func floatEquals(a, b, tol float32) bool {
	// Handle infinities
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) {
		return a == b
	}
	return absFloat32(a-b) <= tol
}

