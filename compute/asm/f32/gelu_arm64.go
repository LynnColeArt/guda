// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64 && !noasm && !gccgo && !safe
// +build arm64,!noasm,!gccgo,!safe

package f32

import (
	"math"
)

// GELUNeon computes GELU activation using NEON instructions
func GELUNeon(x []float32) {
	// For now, use math package implementation
	// TODO: Replace with actual NEON assembly
	for i := range x {
		// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
		x[i] = 0.5 * x[i] * (1 + float32(math.Erf(float64(x[i]/float32(math.Sqrt(2))))))
	}
}

// GeluAVX2 computes GELU activation (AVX2 name for compatibility)
func GeluAVX2(x []float32) {
	GELUNeon(x)
}

// GELU computes GELU activation
func GELU(x []float32) {
	GELUNeon(x)
}