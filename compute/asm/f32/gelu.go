// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !noasm && !gccgo && !safe
// +build amd64,!noasm,!gccgo,!safe

package f32

// GeluAVX2 computes GELU activation in-place using AVX2
//go:noescape
func GeluAVX2(x []float32)