// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !gccgo && !safe
// +build !noasm,!gccgo,!safe

package f32

// Sgemm4x4 computes C = alpha*A*B + beta*C for 4x4 matrices
// Fully unrolled for maximum performance
func Sgemm4x4(alpha float32, a []float32, b []float32, beta float32, c []float32) {
	// Load A matrix (row-major)
	a00, a01, a02, a03 := a[0], a[1], a[2], a[3]
	a10, a11, a12, a13 := a[4], a[5], a[6], a[7]
	a20, a21, a22, a23 := a[8], a[9], a[10], a[11]
	a30, a31, a32, a33 := a[12], a[13], a[14], a[15]
	
	// Load B matrix (row-major)
	b00, b01, b02, b03 := b[0], b[1], b[2], b[3]
	b10, b11, b12, b13 := b[4], b[5], b[6], b[7]
	b20, b21, b22, b23 := b[8], b[9], b[10], b[11]
	b30, b31, b32, b33 := b[12], b[13], b[14], b[15]
	
	// Compute C = A*B
	c00 := a00*b00 + a01*b10 + a02*b20 + a03*b30
	c01 := a00*b01 + a01*b11 + a02*b21 + a03*b31
	c02 := a00*b02 + a01*b12 + a02*b22 + a03*b32
	c03 := a00*b03 + a01*b13 + a02*b23 + a03*b33
	
	c10 := a10*b00 + a11*b10 + a12*b20 + a13*b30
	c11 := a10*b01 + a11*b11 + a12*b21 + a13*b31
	c12 := a10*b02 + a11*b12 + a12*b22 + a13*b32
	c13 := a10*b03 + a11*b13 + a12*b23 + a13*b33
	
	c20 := a20*b00 + a21*b10 + a22*b20 + a23*b30
	c21 := a20*b01 + a21*b11 + a22*b21 + a23*b31
	c22 := a20*b02 + a21*b12 + a22*b22 + a23*b32
	c23 := a20*b03 + a21*b13 + a22*b23 + a23*b33
	
	c30 := a30*b00 + a31*b10 + a32*b20 + a33*b30
	c31 := a30*b01 + a31*b11 + a32*b21 + a33*b31
	c32 := a30*b02 + a31*b12 + a32*b22 + a33*b32
	c33 := a30*b03 + a31*b13 + a32*b23 + a33*b33
	
	// Store result: C = alpha*result + beta*C
	if beta == 0 {
		c[0] = alpha * c00
		c[1] = alpha * c01
		c[2] = alpha * c02
		c[3] = alpha * c03
		c[4] = alpha * c10
		c[5] = alpha * c11
		c[6] = alpha * c12
		c[7] = alpha * c13
		c[8] = alpha * c20
		c[9] = alpha * c21
		c[10] = alpha * c22
		c[11] = alpha * c23
		c[12] = alpha * c30
		c[13] = alpha * c31
		c[14] = alpha * c32
		c[15] = alpha * c33
	} else {
		c[0] = alpha*c00 + beta*c[0]
		c[1] = alpha*c01 + beta*c[1]
		c[2] = alpha*c02 + beta*c[2]
		c[3] = alpha*c03 + beta*c[3]
		c[4] = alpha*c10 + beta*c[4]
		c[5] = alpha*c11 + beta*c[5]
		c[6] = alpha*c12 + beta*c[6]
		c[7] = alpha*c13 + beta*c[7]
		c[8] = alpha*c20 + beta*c[8]
		c[9] = alpha*c21 + beta*c[9]
		c[10] = alpha*c22 + beta*c[10]
		c[11] = alpha*c23 + beta*c[11]
		c[12] = alpha*c30 + beta*c[12]
		c[13] = alpha*c31 + beta*c[13]
		c[14] = alpha*c32 + beta*c[14]
		c[15] = alpha*c33 + beta*c[15]
	}
}

// Sgemm8x8 computes C = alpha*A*B + beta*C for 8x8 matrices
// Optimized with 2x2 unrolling for better performance
func Sgemm8x8(alpha float32, a []float32, b []float32, beta float32, c []float32) {
	// Process 2x2 blocks for better cache usage and ILP
	for i := 0; i < 8; i += 2 {
		for j := 0; j < 8; j += 2 {
			// Accumulate 2x2 block of C
			var c00, c01, c10, c11 float32
			
			// Unroll k loop by 2 for better performance
			for k := 0; k < 8; k += 2 {
				// Load 2x2 blocks from A and B
				a00, a01 := a[i*8+k], a[i*8+k+1]
				a10, a11 := a[(i+1)*8+k], a[(i+1)*8+k+1]
				
				b00, b01 := b[k*8+j], b[k*8+j+1]
				b10, b11 := b[(k+1)*8+j], b[(k+1)*8+j+1]
				
				// Compute products
				c00 += a00*b00 + a01*b10
				c01 += a00*b01 + a01*b11
				c10 += a10*b00 + a11*b10
				c11 += a10*b01 + a11*b11
			}
			
			// Store results
			idx00 := i*8 + j
			idx01 := i*8 + j + 1
			idx10 := (i+1)*8 + j
			idx11 := (i+1)*8 + j + 1
			
			if beta == 0 {
				c[idx00] = alpha * c00
				c[idx01] = alpha * c01
				c[idx10] = alpha * c10
				c[idx11] = alpha * c11
			} else {
				c[idx00] = alpha*c00 + beta*c[idx00]
				c[idx01] = alpha*c01 + beta*c[idx01]
				c[idx10] = alpha*c10 + beta*c[idx10]
				c[idx11] = alpha*c11 + beta*c[idx11]
			}
		}
	}
}

// Sgemm16x16 computes C = alpha*A*B + beta*C for 16x16 matrices
// Uses 4x4 blocking with full unrolling for each block
func Sgemm16x16(alpha float32, a []float32, b []float32, beta float32, c []float32) {
	// Process 4x4 blocks
	for ii := 0; ii < 16; ii += 4 {
		for jj := 0; jj < 16; jj += 4 {
			// Initialize 4x4 block of C
			var c00, c01, c02, c03 float32
			var c10, c11, c12, c13 float32
			var c20, c21, c22, c23 float32
			var c30, c31, c32, c33 float32
			
			// Compute 4x4 block
			for k := 0; k < 16; k++ {
				// Load column k from A block
				a0k := a[(ii+0)*16+k]
				a1k := a[(ii+1)*16+k]
				a2k := a[(ii+2)*16+k]
				a3k := a[(ii+3)*16+k]
				
				// Load row k from B block
				bk0 := b[k*16+(jj+0)]
				bk1 := b[k*16+(jj+1)]
				bk2 := b[k*16+(jj+2)]
				bk3 := b[k*16+(jj+3)]
				
				// Update C block
				c00 += a0k * bk0
				c01 += a0k * bk1
				c02 += a0k * bk2
				c03 += a0k * bk3
				
				c10 += a1k * bk0
				c11 += a1k * bk1
				c12 += a1k * bk2
				c13 += a1k * bk3
				
				c20 += a2k * bk0
				c21 += a2k * bk1
				c22 += a2k * bk2
				c23 += a2k * bk3
				
				c30 += a3k * bk0
				c31 += a3k * bk1
				c32 += a3k * bk2
				c33 += a3k * bk3
			}
			
			// Store 4x4 block with alpha and beta
			base := ii*16 + jj
			if beta == 0 {
				c[base+0] = alpha * c00
				c[base+1] = alpha * c01
				c[base+2] = alpha * c02
				c[base+3] = alpha * c03
				
				c[base+16] = alpha * c10
				c[base+17] = alpha * c11
				c[base+18] = alpha * c12
				c[base+19] = alpha * c13
				
				c[base+32] = alpha * c20
				c[base+33] = alpha * c21
				c[base+34] = alpha * c22
				c[base+35] = alpha * c23
				
				c[base+48] = alpha * c30
				c[base+49] = alpha * c31
				c[base+50] = alpha * c32
				c[base+51] = alpha * c33
			} else {
				c[base+0] = alpha*c00 + beta*c[base+0]
				c[base+1] = alpha*c01 + beta*c[base+1]
				c[base+2] = alpha*c02 + beta*c[base+2]
				c[base+3] = alpha*c03 + beta*c[base+3]
				
				c[base+16] = alpha*c10 + beta*c[base+16]
				c[base+17] = alpha*c11 + beta*c[base+17]
				c[base+18] = alpha*c12 + beta*c[base+18]
				c[base+19] = alpha*c13 + beta*c[base+19]
				
				c[base+32] = alpha*c20 + beta*c[base+32]
				c[base+33] = alpha*c21 + beta*c[base+33]
				c[base+34] = alpha*c22 + beta*c[base+34]
				c[base+35] = alpha*c23 + beta*c[base+35]
				
				c[base+48] = alpha*c30 + beta*c[base+48]
				c[base+49] = alpha*c31 + beta*c[base+49]
				c[base+50] = alpha*c32 + beta*c[base+50]
				c[base+51] = alpha*c33 + beta*c[base+51]
			}
		}
	}
}