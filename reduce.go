package guda

import (
	"math"
	
	"github.com/LynnColeArt/guda/compute/asm/f32"
)

// Reduction operations for high-performance computation on GPU-like data
// These are essential building blocks for ML operations like softmax, attention, and pooling

// Sum computes the sum of all elements in x
func (d DevicePtr) Sum(n int) float32 {
	return f32.Sum(d.Float32()[:n])
}

// Max returns the maximum value in x
func (d DevicePtr) Max(n int) float32 {
	if n == 0 {
		return float32(math.Inf(-1))
	}
	return f32.Max(d.Float32()[:n])
}

// Min returns the minimum value in x
func (d DevicePtr) Min(n int) float32 {
	if n == 0 {
		return float32(math.Inf(1))
	}
	return f32.Min(d.Float32()[:n])
}

// ArgMax returns the index of the maximum value in x
func (d DevicePtr) ArgMax(n int) int {
	if n == 0 {
		return -1
	}
	return f32.ArgMax(d.Float32()[:n])
}

// ArgMin returns the index of the minimum value in x
func (d DevicePtr) ArgMin(n int) int {
	if n == 0 {
		return -1
	}
	return f32.ArgMin(d.Float32()[:n])
}

// SumSquares computes the sum of squares of all elements
// Useful for L2 norm computation
func (d DevicePtr) SumSquares(n int) float32 {
	if n == 0 {
		return 0
	}
	return f32.SumSquares(d.Float32()[:n])
}

// Mean computes the arithmetic mean of all elements
func (d DevicePtr) Mean(n int) float32 {
	if n == 0 {
		return 0
	}
	return d.Sum(n) / float32(n)
}

// Variance computes the variance of all elements
// Uses Welford's online algorithm for numerical stability
func (d DevicePtr) Variance(n int) float32 {
	if n < 2 {
		return 0
	}
	
	x := d.Float32()[:n]
	mean := float32(0)
	m2 := float32(0)
	
	// Welford's algorithm for numerical stability
	for i := 0; i < n; i++ {
		delta := x[i] - mean
		mean += delta / float32(i+1)
		delta2 := x[i] - mean
		m2 += delta * delta2
	}
	
	return m2 / float32(n-1)
}

// Product computes the product of all elements
func (d DevicePtr) Product(n int) float32 {
	x := d.Float32()[:n]
	if n == 0 {
		return 1
	}
	
	// TODO: Replace with AVX2 optimized version
	prod := float32(1)
	for i := 0; i < n; i++ {
		prod *= x[i]
	}
	return prod
}

// ReduceSum performs a sum reduction along the specified axis
// For 2D tensors: axis=0 sums columns, axis=1 sums rows
func ReduceSum(x DevicePtr, shape []int, axis int, out DevicePtr) error {
	if len(shape) != 2 {
		return ErrInvalidShape
	}
	
	m, n := shape[0], shape[1]
	xData := x.Float32()
	outData := out.Float32()
	
	switch axis {
	case 0: // Sum columns (result is 1×n)
		for j := 0; j < n; j++ {
			sum := float32(0)
			for i := 0; i < m; i++ {
				sum += xData[i*n+j]
			}
			outData[j] = sum
		}
	case 1: // Sum rows (result is m×1)
		for i := 0; i < m; i++ {
			sum := float32(0)
			for j := 0; j < n; j++ {
				sum += xData[i*n+j]
			}
			outData[i] = sum
		}
	default:
		return ErrInvalidAxis
	}
	
	return nil
}

// ReduceMax performs a max reduction along the specified axis
func ReduceMax(x DevicePtr, shape []int, axis int, out DevicePtr) error {
	if len(shape) != 2 {
		return ErrInvalidShape
	}
	
	m, n := shape[0], shape[1]
	xData := x.Float32()
	outData := out.Float32()
	
	switch axis {
	case 0: // Max over columns (result is 1×n)
		for j := 0; j < n; j++ {
			max := xData[j]
			for i := 1; i < m; i++ {
				val := xData[i*n+j]
				if val > max {
					max = val
				}
			}
			outData[j] = max
		}
	case 1: // Max over rows (result is m×1)
		for i := 0; i < m; i++ {
			max := xData[i*n]
			for j := 1; j < n; j++ {
				val := xData[i*n+j]
				if val > max {
					max = val
				}
			}
			outData[i] = max
		}
	default:
		return ErrInvalidAxis
	}
	
	return nil
}

// Softmax computes the softmax function: exp(x) / sum(exp(x))
// Essential for attention mechanisms and classification
func Softmax(x DevicePtr, n int) error {
	if n == 0 {
		return nil
	}
	f32.Softmax(x.Float32()[:n])
	return nil
}

// LogSumExp computes log(sum(exp(x))) in a numerically stable way
// This is a key operation in many ML algorithms
func LogSumExp(x DevicePtr, n int) float32 {
	if n == 0 {
		return float32(math.Inf(-1))
	}
	return f32.LogSumExp(x.Float32()[:n])
}

// CumSum computes the cumulative sum of elements
// out[i] = sum(x[0:i+1])
func CumSum(x, out DevicePtr, n int) error {
	xData := x.Float32()[:n]
	outData := out.Float32()[:n]
	
	if n == 0 {
		return nil
	}
	
	// TODO: Parallelize with prefix sum algorithm
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += xData[i]
		outData[i] = sum
	}
	
	return nil
}

// SegmentSum computes the sum of segments defined by segment_ids
// Essential for graph neural networks and attention mechanisms
func SegmentSum(data DevicePtr, segmentIds []int, nSegments int, out DevicePtr) error {
	n := len(segmentIds)
	if n == 0 {
		return nil
	}
	
	dataSlice := data.Float32()[:n]
	outSlice := out.Float32()[:nSegments]
	
	// Initialize output to zero
	for i := 0; i < nSegments; i++ {
		outSlice[i] = 0
	}
	
	// Sum each segment
	for i := 0; i < n; i++ {
		segId := segmentIds[i]
		if segId < 0 || segId >= nSegments {
			return ErrInvalidIndex
		}
		outSlice[segId] += dataSlice[i]
	}
	
	return nil
}

// TopK finds the k largest elements and their indices
// Returns values and indices in descending order
func TopK(x DevicePtr, n, k int) (values []float32, indices []int, err error) {
	if k > n {
		k = n
	}
	if k <= 0 {
		return nil, nil, ErrInvalidParameter
	}
	
	data := x.Float32()[:n]
	
	// For now, use a simple approach - full sort
	// TODO: Implement efficient partial sort with AVX2
	type pair struct {
		val float32
		idx int
	}
	
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{data[i], i}
	}
	
	// Partial sort - only need top k
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < n; j++ {
			if pairs[j].val > pairs[maxIdx].val {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}
	
	values = make([]float32, k)
	indices = make([]int, k)
	for i := 0; i < k; i++ {
		values[i] = pairs[i].val
		indices[i] = pairs[i].idx
	}
	
	return values, indices, nil
}