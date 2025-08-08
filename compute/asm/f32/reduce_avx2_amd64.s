// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !gccgo && !safe
// +build !noasm,!gccgo,!safe

#include "textflag.h"

// MaxAVX2 finds the maximum value in a float32 slice using AVX2
// func MaxAVX2(x []float32) float32
TEXT ·MaxAVX2(SB), NOSPLIT, $0-28
	MOVQ x_base+0(FP), SI  // SI = &x[0]
	MOVQ x_len+8(FP), CX   // CX = len(x)
	
	// Handle empty slice
	TESTQ CX, CX
	JE    done_empty
	
	// Initialize max with first element
	VMOVSS (SI), X0        // X0 = x[0]
	VBROADCASTSS X0, Y0    // Y0 = [x[0], x[0], ..., x[0]]
	
	// Handle small slices (< 8 elements)
	CMPQ CX, $8
	JL   scalar_loop
	
	// Process 8 elements at a time
	MOVQ CX, DX
	SHRQ $3, DX            // DX = len(x) / 8
	ANDQ $7, CX            // CX = len(x) % 8
	
vector_loop:
	VMOVUPS (SI), Y1       // Load 8 floats
	VMAXPS Y1, Y0, Y0      // Y0 = max(Y0, Y1)
	ADDQ $32, SI           // Advance pointer by 8 floats
	DECQ DX
	JNZ  vector_loop
	
	// Horizontal max of Y0
	VEXTRACTF128 $1, Y0, X1
	VMAXPS X1, X0, X0      // X0 = max(lower 4, upper 4)
	VPSRLDQ $8, X0, X1     // Shift right by 8 bytes
	VMAXPS X1, X0, X0      // X0 = max(X0[0:2], X0[2:4])
	VPSRLDQ $4, X0, X1     // Shift right by 4 bytes
	VMAXSS X1, X0, X0      // X0 = max(X0[0], X0[1])
	
scalar_loop:
	TESTQ CX, CX
	JE    done
	VMOVSS (SI), X1
	VMAXSS X1, X0, X0
	ADDQ $4, SI
	DECQ CX
	JMP  scalar_loop
	
done:
	VMOVSS X0, ret+24(FP)
	RET
	
done_empty:
	// Return -Inf for empty slice
	MOVL $0xFF800000, AX    // -Inf in float32
	MOVL AX, ret+24(FP)
	RET

// MinAVX2 finds the minimum value in a float32 slice using AVX2
// func MinAVX2(x []float32) float32
TEXT ·MinAVX2(SB), NOSPLIT, $0-28
	MOVQ x_base+0(FP), SI  // SI = &x[0]
	MOVQ x_len+8(FP), CX   // CX = len(x)
	
	// Handle empty slice
	TESTQ CX, CX
	JE    min_done_empty
	
	// Initialize min with first element
	VMOVSS (SI), X0        // X0 = x[0]
	VBROADCASTSS X0, Y0    // Y0 = [x[0], x[0], ..., x[0]]
	
	// Handle small slices (< 8 elements)
	CMPQ CX, $8
	JL   min_scalar_loop
	
	// Process 8 elements at a time
	MOVQ CX, DX
	SHRQ $3, DX            // DX = len(x) / 8
	ANDQ $7, CX            // CX = len(x) % 8
	
min_vector_loop:
	VMOVUPS (SI), Y1       // Load 8 floats
	VMINPS Y1, Y0, Y0      // Y0 = min(Y0, Y1)
	ADDQ $32, SI           // Advance pointer by 8 floats
	DECQ DX
	JNZ  min_vector_loop
	
	// Horizontal min of Y0
	VEXTRACTF128 $1, Y0, X1
	VMINPS X1, X0, X0      // X0 = min(lower 4, upper 4)
	VPSRLDQ $8, X0, X1     // Shift right by 8 bytes
	VMINPS X1, X0, X0      // X0 = min(X0[0:2], X0[2:4])
	VPSRLDQ $4, X0, X1     // Shift right by 4 bytes
	VMINSS X1, X0, X0      // X0 = min(X0[0], X0[1])
	
min_scalar_loop:
	TESTQ CX, CX
	JE    min_done
	VMOVSS (SI), X1
	VMINSS X1, X0, X0
	ADDQ $4, SI
	DECQ CX
	JMP  min_scalar_loop
	
min_done:
	VMOVSS X0, ret+24(FP)
	RET
	
min_done_empty:
	// Return +Inf for empty slice
	MOVL $0x7F800000, AX    // +Inf in float32
	MOVL AX, ret+24(FP)
	RET

// ArgMaxAVX2 finds the index of the maximum value in a float32 slice
// func ArgMaxAVX2(x []float32) int
TEXT ·ArgMaxAVX2(SB), NOSPLIT, $0-32
	MOVQ x_base+0(FP), SI  // SI = &x[0]
	MOVQ x_len+8(FP), CX   // CX = len(x)
	
	// Handle empty slice
	TESTQ CX, CX
	JE    argmax_done_empty
	
	// Initialize with first element
	XORQ AX, AX            // AX = maxIdx = 0
	VMOVSS (SI), X0        // X0 = maxVal = x[0]
	
	// Simple scalar loop for now
	// TODO: Vectorize with index tracking
	MOVQ $1, DX            // DX = i = 1
	
argmax_loop:
	CMPQ DX, CX
	JGE  argmax_done
	
	VMOVSS (SI)(DX*4), X1  // X1 = x[i]
	VCOMISS X0, X1         // Compare maxVal with x[i]
	JBE  argmax_no_update  // If maxVal >= x[i], skip update
	
	VMOVSS X1, X0, X0      // maxVal = x[i]
	MOVQ DX, AX            // maxIdx = i
	
argmax_no_update:
	INCQ DX
	JMP  argmax_loop
	
argmax_done:
	MOVQ AX, ret+24(FP)
	RET
	
argmax_done_empty:
	MOVQ $-1, ret+24(FP)   // Return -1 for empty slice
	RET

// SumSquaresAVX2 computes the sum of squares using FMA instructions
// func SumSquaresAVX2(x []float32) float32
TEXT ·SumSquaresAVX2(SB), NOSPLIT, $0-28
	MOVQ x_base+0(FP), SI  // SI = &x[0]
	MOVQ x_len+8(FP), CX   // CX = len(x)
	
	// Initialize sum to zero
	VXORPS Y0, Y0, Y0      // Y0 = [0, 0, ..., 0]
	
	// Handle small slices (< 8 elements)
	CMPQ CX, $8
	JL   sumsq_scalar_loop
	
	// Process 8 elements at a time
	MOVQ CX, DX
	SHRQ $3, DX            // DX = len(x) / 8
	ANDQ $7, CX            // CX = len(x) % 8
	
sumsq_vector_loop:
	VMOVUPS (SI), Y1       // Load 8 floats
	VFMADD231PS Y1, Y1, Y0 // Y0 += Y1 * Y1 (FMA)
	ADDQ $32, SI           // Advance pointer by 8 floats
	DECQ DX
	JNZ  sumsq_vector_loop
	
	// Horizontal sum of Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0      // X0 = lower 4 + upper 4
	VHADDPS X0, X0, X0     // Horizontal add
	VHADDPS X0, X0, X0     // Horizontal add again
	
	// Handle remaining elements
sumsq_scalar_loop:
	TESTQ CX, CX
	JE    sumsq_done
	VMOVSS (SI), X1
	VFMADD231SS X1, X1, X0 // X0 += X1 * X1
	ADDQ $4, SI
	DECQ CX
	JMP  sumsq_scalar_loop
	
sumsq_done:
	VMOVSS X0, ret+24(FP)
	RET

// ArgMinAVX2 finds the index of the minimum value in a float32 slice
// func ArgMinAVX2(x []float32) int
TEXT ·ArgMinAVX2(SB), NOSPLIT, $0-32
	MOVQ x_base+0(FP), SI  // SI = &x[0]
	MOVQ x_len+8(FP), CX   // CX = len(x)
	
	// Handle empty slice
	TESTQ CX, CX
	JE    argmin_done_empty
	
	// Initialize with first element
	XORQ AX, AX            // AX = minIdx = 0
	VMOVSS (SI), X0        // X0 = minVal = x[0]
	
	// Simple scalar loop for now
	// TODO: Vectorize with index tracking
	MOVQ $1, DX            // DX = i = 1
	
argmin_loop:
	CMPQ DX, CX
	JGE  argmin_done
	
	VMOVSS (SI)(DX*4), X1  // X1 = x[i]
	VCOMISS X1, X0         // Compare x[i] with minVal
	JBE  argmin_no_update  // If x[i] >= minVal, skip update
	
	VMOVSS X1, X0, X0      // minVal = x[i]
	MOVQ DX, AX            // minIdx = i
	
argmin_no_update:
	INCQ DX
	JMP  argmin_loop
	
argmin_done:
	MOVQ AX, ret+24(FP)
	RET
	
argmin_done_empty:
	MOVQ $-1, ret+24(FP)   // Return -1 for empty slice
	RET
