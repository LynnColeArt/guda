// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !noasm && !appengine && !gccgo
// +build amd64,!noasm,!appengine,!gccgo

package f32

// FusedGEMMBiasReLU8x8 computes an 8x8 block of C = ReLU(A*B + bias)
// This function processes data in AVX2 registers, keeping everything in cache
//
//go:noescape
func FusedGEMMBiasReLU8x8(k int, alpha float32, a []float32, lda int, b []float32, ldb int, bias []float32, c []float32, ldc int)

// FusedGEMMBiasReLU4x4 computes a 4x4 block for edge cases
//
//go:noescape
func FusedGEMMBiasReLU4x4(k int, alpha float32, a []float32, lda int, b []float32, ldb int, bias []float32, c []float32, ldc int)