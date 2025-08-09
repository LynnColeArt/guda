// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !noasm && !gccgo && !safe
// +build amd64,!noasm,!gccgo,!safe

package f32

import (
	"math"
	"runtime"
	"golang.org/x/sys/cpu"
)

// CPU feature detection - use appropriate variables for the architecture
var (
	hasAVX2 = (runtime.GOARCH == "amd64" || runtime.GOARCH == "386") && cpu.X86.HasAVX2
	hasFMA  = (runtime.GOARCH == "amd64" || runtime.GOARCH == "386") && cpu.X86.HasFMA
)

// Assembly implementations for AVX2
func MaxAVX2(x []float32) float32
func MinAVX2(x []float32) float32
func ArgMaxAVX2(x []float32) int
func ArgMinAVX2(x []float32) int
func SumSquaresAVX2(x []float32) float32

// Max returns the maximum value in x
func Max(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(-1))
	}
	if hasAVX2 {
		return MaxAVX2(x)
	}
	return maxGeneric(x)
}

// Min returns the minimum value in x
func Min(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(1))
	}
	if hasAVX2 {
		return MinAVX2(x)
	}
	return minGeneric(x)
}

// ArgMax returns the index of the maximum value in x
func ArgMax(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	if hasAVX2 {
		return ArgMaxAVX2(x)
	}
	return argMaxGeneric(x)
}

// ArgMin returns the index of the minimum value in x
func ArgMin(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	if hasAVX2 {
		return ArgMinAVX2(x)
	}
	return argMinGeneric(x)
}

// SumSquares computes the sum of squares of all elements
func SumSquares(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	if hasAVX2 && hasFMA {
		return SumSquaresAVX2(x)
	}
	return sumSquaresGeneric(x)
}

// Generic fallback implementations
func maxGeneric(x []float32) float32 {
	max := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	return max
}

func minGeneric(x []float32) float32 {
	min := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
		}
	}
	return min
}

func argMaxGeneric(x []float32) int {
	maxIdx := 0
	maxVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxVal {
			maxVal = x[i]
			maxIdx = i
		}
	}
	return maxIdx
}

func argMinGeneric(x []float32) int {
	minIdx := 0
	minVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < minVal {
			minVal = x[i]
			minIdx = i
		}
	}
	return minIdx
}

func sumSquaresGeneric(x []float32) float32 {
	sum := float32(0)
	for i := 0; i < len(x); i++ {
		sum += x[i] * x[i]
	}
	return sum
}

// Softmax computes softmax in-place: x[i] = exp(x[i]) / sum(exp(x))
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	
	// Find max for numerical stability
	max := Max(x)
	
	// Compute exp(x - max) and sum using fast exp approximation
	sum := float32(0)
	for i := range x {
		x[i] = fastExp(x[i] - max)
		sum += x[i]
	}
	
	// Normalize
	invSum := 1.0 / sum
	ScalUnitary(invSum, x)
}

// fastExp computes a fast approximation of exp(x) for float32
// Optimized for softmax where inputs are typically in [-10, 10] range after max subtraction
func fastExp(x float32) float32 {
	// For softmax, after subtracting max, most values will be <= 0
	// and the largest will be 0. So we can skip the upper bound check.
	if x < -87.3 { 
		return 0
	}
	
	// Range reduction: exp(x) = 2^k * exp(r) where x = k*ln(2) + r
	const ln2 = 0.6931472 // ln(2) as float32
	const invLn2 = 1.4426951 // 1/ln(2) as float32
	
	// Fast computation of k using round-to-nearest
	k := int32(x*invLn2 + 0.5)
	if x < 0 {
		k = int32(x*invLn2 - 0.5)
	}
	r := x - float32(k)*ln2
	
	// Compute exp(r) using Horner's method for better performance
	// exp(r) ≈ 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
	const c1 = 1.0
	const c2 = 0.5
	const c3 = 0.16666667
	const c4 = 0.041666668
	const c5 = 0.008333334
	
	exp_r := 1.0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
	
	// Fast 2^k computation using bit manipulation
	// For softmax, k is typically in a small range
	bits := uint32((k + 127) << 23)
	scale := math.Float32frombits(bits)
	return exp_r * scale
}

// LogSoftmax computes log(softmax(x)) in a numerically stable way
func LogSoftmax(x []float32) {
	if len(x) == 0 {
		return
	}
	
	// Find max for numerical stability
	max := Max(x)
	
	// Compute log(sum(exp(x - max)))
	sum := float32(0)
	for i := range x {
		sum += float32(math.Exp(float64(x[i] - max)))
	}
	logSum := float32(math.Log(float64(sum)))
	
	// Compute log(softmax) = x - max - log(sum(exp(x - max)))
	offset := max + logSum
	for i := range x {
		x[i] -= offset
	}
}

// LogSumExp computes log(sum(exp(x))) in a numerically stable way
func LogSumExp(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(-1))
	}
	
	// Find max for numerical stability
	max := Max(x)
	
	// Handle case where max is -Inf
	if math.IsInf(float64(max), -1) {
		return max
	}
	
	// Compute sum(exp(x - max))
	sum := float32(0)
	for i := range x {
		sum += float32(math.Exp(float64(x[i] - max)))
	}
	
	return max + float32(math.Log(float64(sum)))
}

// Used by assembly implementation fallback
func logSumExpGo(x []float32) float32 {
	return LogSumExp(x)
}

// CumSum computes cumulative sum: out[i] = sum(x[0:i+1])
func CumSum(x, out []float32) {
	if len(x) == 0 {
		return
	}
	
	// TODO: Implement parallel prefix sum for large arrays
	sum := float32(0)
	for i := range x {
		sum += x[i]
		out[i] = sum
	}
}

// SegmentSum computes sum of segments defined by segment IDs
func SegmentSum(data []float32, segmentIds []int, nSegments int, out []float32) {
	// Initialize output to zero
	for i := range out[:nSegments] {
		out[i] = 0
	}
	
	// Sum each segment
	for i, val := range data {
		if i < len(segmentIds) {
			segId := segmentIds[i]
			if segId >= 0 && segId < nSegments {
				out[segId] += val
			}
		}
	}
}