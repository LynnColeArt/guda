// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && !noasm && !gccgo && !safe
// +build arm64,!noasm,!gccgo,!safe

package f32

import (
	"math"
)

// ARM64 NEON implementations for reduction operations

// MaxNEON returns the maximum value in x using NEON instructions
func MaxNEON(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(-1))
	}
	
	// For now, use generic implementation
	// Later we could replace this with actual NEON assembly
	max := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	return max
}

// MinNEON returns the minimum value in x using NEON instructions
func MinNEON(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(1))
	}
	
	// For now, use generic implementation
	// Later we could replace this with actual NEON assembly
	min := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
		}
	}
	return min
}

// ArgMaxNEON returns the index of the maximum value in x using NEON instructions
func ArgMaxNEON(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	
	// For now, use generic implementation
	// Later we could replace this with actual NEON assembly
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

// ArgMinNEON returns the index of the minimum value in x using NEON instructions
func ArgMinNEON(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	
	// For now, use generic implementation
	// Later we could replace this with actual NEON assembly
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

// SumSquaresNEON computes the sum of squares of all elements using NEON instructions
func SumSquaresNEON(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	
	// For now, use generic implementation
	// Later we could replace this with actual NEON assembly
	sum := float32(0)
	for i := 0; i < len(x); i++ {
		sum += x[i] * x[i]
	}
	return sum
}

// Max returns the maximum value in x
func Max(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(-1))
	}
	// Use NEON implementation
	return MaxNEON(x)
}

// Min returns the minimum value in x
func Min(x []float32) float32 {
	if len(x) == 0 {
		return float32(math.Inf(1))
	}
	// Use NEON implementation
	return MinNEON(x)
}

// ArgMax returns the index of the maximum value in x
func ArgMax(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	// Use NEON implementation
	return ArgMaxNEON(x)
}

// ArgMin returns the index of the minimum value in x
func ArgMin(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	// Use NEON implementation
	return ArgMinNEON(x)
}

// SumSquares computes the sum of squares of all elements
func SumSquares(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	// Use NEON implementation
	return SumSquaresNEON(x)
}

// Softmax computes softmax in-place: x[i] = exp(x[i]) / sum(exp(x))
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	
	// Find max for numerical stability
	max := Max(x)
	
	// Compute exp(x - max) and sum using math package
	sum := float32(0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	
	// Normalize
	invSum := 1.0 / sum
	for i := range x {
		x[i] *= invSum
	}
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

// CumSum computes cumulative sum: out[i] = sum(x[0:i+1])
func CumSum(x, out []float32) {
	if len(x) == 0 {
		return
	}
	
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