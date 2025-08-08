// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !appengine && !gccgo
// +build !noasm,!appengine,!gccgo

#include "textflag.h"

// FusedGEMMBiasReLU8x8 computes C = ReLU(A*B + bias) for 8x8 blocks
// This is the heart of our memory wall breakthrough!
//
// func FusedGEMMBiasReLU8x8(k int, alpha float32, a []float32, lda int, 
//                          b []float32, ldb int, bias []float32, 
//                          c []float32, ldc int)
TEXT ·FusedGEMMBiasReLU8x8(SB), NOSPLIT, $0
    // Load parameters
    MOVQ    k+0(FP), CX           // k dimension
    MOVSS   alpha+8(FP), X15      // alpha scalar
    MOVQ    a_base+16(FP), SI     // a pointer
    MOVQ    lda+40(FP), R8        // lda
    MOVQ    b_base+48(FP), DI     // b pointer  
    MOVQ    ldb+72(FP), R9        // ldb
    MOVQ    bias_base+80(FP), R10 // bias pointer
    MOVQ    c_base+104(FP), R11   // c pointer
    MOVQ    ldc+128(FP), R12      // ldc
    
    // Convert strides to bytes
    SHLQ    $2, R8               // lda *= 4 (float32 size)
    SHLQ    $2, R9               // ldb *= 4
    SHLQ    $2, R12              // ldc *= 4
    
    // Broadcast alpha to all lanes
    VBROADCASTSS X15, Y15
    
    // Zero accumulators for 8x8 block
    // We use 8 YMM registers to hold the 8x8 result
    VXORPS  Y0, Y0, Y0   // Row 0
    VXORPS  Y1, Y1, Y1   // Row 1
    VXORPS  Y2, Y2, Y2   // Row 2
    VXORPS  Y3, Y3, Y3   // Row 3
    VXORPS  Y4, Y4, Y4   // Row 4
    VXORPS  Y5, Y5, Y5   // Row 5
    VXORPS  Y6, Y6, Y6   // Row 6
    VXORPS  Y7, Y7, Y7   // Row 7
    
    // Main K loop - this is where the magic happens
    // We keep everything in registers!
k_loop:
    TESTQ   CX, CX
    JE      k_done
    
    // For this iteration k, we need:
    // - Column k from 8 rows of A: a[0,k], a[1,k], ..., a[7,k]
    // - Row k from B: b[k,0], b[k,1], ..., b[k,7]
    
    // Save current k value
    MOVQ    CX, R15
    SUBQ    $1, R15              // R15 = remaining k - 1 = current k index
    
    // Load column k from A (8 values from different rows)
    // These are NOT contiguous - they are lda elements apart
    MOVQ    SI, R13              // Save current position
    MOVSS   (R13), X8            // a[0,k]
    ADDQ    R8, R13              // Next row
    MOVSS   (R13), X9            // a[1,k]
    VINSERTPS $0x10, X9, X8, X8
    ADDQ    R8, R13
    MOVSS   (R13), X9            // a[2,k]
    VINSERTPS $0x20, X9, X8, X8
    ADDQ    R8, R13
    MOVSS   (R13), X9            // a[3,k]
    VINSERTPS $0x30, X9, X8, X8
    ADDQ    R8, R13
    MOVSS   (R13), X9            // a[4,k]
    ADDQ    R8, R13
    MOVSS   (R13), X10           // a[5,k]
    VINSERTPS $0x10, X10, X9, X9
    ADDQ    R8, R13
    MOVSS   (R13), X10           // a[6,k]
    VINSERTPS $0x20, X10, X9, X9
    ADDQ    R8, R13
    MOVSS   (R13), X10           // a[7,k]
    VINSERTPS $0x30, X10, X9, X9
    VINSERTF128 $1, X9, Y8, Y8   // Y8 = [a[0,k]...a[7,k]]
    
    // Now process row k of B with the column from A
    VBROADCASTSS (DI), Y9        // b[k,0]
    VFMADD231PS Y8, Y9, Y0       // c[0:8, 0] += a[0:8, k] * b[k, 0]
    
    VBROADCASTSS 4(DI), Y9       // b[k,1]
    VFMADD231PS Y8, Y9, Y1       // c[0:8, 1] += a[0:8, k] * b[k, 1]
    
    VBROADCASTSS 8(DI), Y9       // b[k,2]
    VFMADD231PS Y8, Y9, Y2       // c[0:8, 2] += a[0:8, k] * b[k, 2]
    
    VBROADCASTSS 12(DI), Y9      // b[k,3]
    VFMADD231PS Y8, Y9, Y3       // c[0:8, 3] += a[0:8, k] * b[k, 3]
    
    VBROADCASTSS 16(DI), Y9      // b[k,4]
    VFMADD231PS Y8, Y9, Y4       // c[0:8, 4] += a[0:8, k] * b[k, 4]
    
    VBROADCASTSS 20(DI), Y9      // b[k,5]
    VFMADD231PS Y8, Y9, Y5       // c[0:8, 5] += a[0:8, k] * b[k, 5]
    
    VBROADCASTSS 24(DI), Y9      // b[k,6]
    VFMADD231PS Y8, Y9, Y6       // c[0:8, 6] += a[0:8, k] * b[k, 6]
    
    VBROADCASTSS 28(DI), Y9      // b[k,7]
    VFMADD231PS Y8, Y9, Y7       // c[0:8, 7] += a[0:8, k] * b[k, 7]
    
    // Advance pointers to next k
    ADDQ    $4, SI               // a += 1 float (next column)
    ADDQ    R9, DI               // b += ldb (next row)
    DECQ    CX                  // k--
    JMP     k_loop

k_done:
    // Scale by alpha
    VMULPS  Y15, Y0, Y0
    VMULPS  Y15, Y1, Y1
    VMULPS  Y15, Y2, Y2
    VMULPS  Y15, Y3, Y3
    VMULPS  Y15, Y4, Y4
    VMULPS  Y15, Y5, Y5
    VMULPS  Y15, Y6, Y6
    VMULPS  Y15, Y7, Y7
    
    // Load bias and add - FUSED!
    VBROADCASTSS (R10), Y8
    VADDPS  Y8, Y0, Y0
    
    VBROADCASTSS 4(R10), Y8
    VADDPS  Y8, Y1, Y1
    
    VBROADCASTSS 8(R10), Y8
    VADDPS  Y8, Y2, Y2
    
    VBROADCASTSS 12(R10), Y8
    VADDPS  Y8, Y3, Y3
    
    VBROADCASTSS 16(R10), Y8
    VADDPS  Y8, Y4, Y4
    
    VBROADCASTSS 20(R10), Y8
    VADDPS  Y8, Y5, Y5
    
    VBROADCASTSS 24(R10), Y8
    VADDPS  Y8, Y6, Y6
    
    VBROADCASTSS 28(R10), Y8
    VADDPS  Y8, Y7, Y7
    
    // ReLU activation - FUSED!
    VXORPS  Y8, Y8, Y8          // Zero vector
    VMAXPS  Y8, Y0, Y0          // max(0, c[0:8, 0])
    VMAXPS  Y8, Y1, Y1          // max(0, c[0:8, 1])
    VMAXPS  Y8, Y2, Y2          // max(0, c[0:8, 2])
    VMAXPS  Y8, Y3, Y3          // max(0, c[0:8, 3])
    VMAXPS  Y8, Y4, Y4          // max(0, c[0:8, 4])
    VMAXPS  Y8, Y5, Y5          // max(0, c[0:8, 5])
    VMAXPS  Y8, Y6, Y6          // max(0, c[0:8, 6])
    VMAXPS  Y8, Y7, Y7          // max(0, c[0:8, 7])
    
    // Store results respecting row-major layout
    // Each YMM register contains one column of the 8x8 result
    // We need to transpose and store
    
    // For simplicity, store each element individually
    // TODO: Optimize with transpose instructions
    
    // Column 0
    MOVQ    R11, R13             // Save C pointer
    VEXTRACTPS $0, X0, (R13)     // c[0,0]
    ADDQ    R12, R13             // Next row
    VEXTRACTPS $1, X0, (R13)     // c[1,0]
    ADDQ    R12, R13
    VEXTRACTPS $2, X0, (R13)     // c[2,0]
    ADDQ    R12, R13
    VEXTRACTPS $3, X0, (R13)     // c[3,0]
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y0, X8
    VEXTRACTPS $0, X8, (R13)     // c[4,0]
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)     // c[5,0]
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)     // c[6,0]
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)     // c[7,0]
    
    // Column 1
    MOVQ    R11, R13
    ADDQ    $4, R13              // Move to column 1
    VEXTRACTPS $0, X1, (R13)     // c[0,1]
    ADDQ    R12, R13
    VEXTRACTPS $1, X1, (R13)     // c[1,1]
    ADDQ    R12, R13
    VEXTRACTPS $2, X1, (R13)     // c[2,1]
    ADDQ    R12, R13
    VEXTRACTPS $3, X1, (R13)     // c[3,1]
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y1, X8
    VEXTRACTPS $0, X8, (R13)     // c[4,1]
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)     // c[5,1]
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)     // c[6,1]
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)     // c[7,1]
    
    // Continue for remaining columns
    // Column 2
    MOVQ    R11, R13
    ADDQ    $8, R13
    VEXTRACTPS $0, X2, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X2, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X2, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X2, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y2, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    // Column 3
    MOVQ    R11, R13
    ADDQ    $12, R13
    VEXTRACTPS $0, X3, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X3, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X3, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X3, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y3, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    // Column 4
    MOVQ    R11, R13
    ADDQ    $16, R13
    VEXTRACTPS $0, X4, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X4, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X4, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X4, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y4, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    // Column 5
    MOVQ    R11, R13
    ADDQ    $20, R13
    VEXTRACTPS $0, X5, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X5, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X5, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X5, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y5, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    // Column 6
    MOVQ    R11, R13
    ADDQ    $24, R13
    VEXTRACTPS $0, X6, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X6, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X6, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X6, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y6, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    // Column 7
    MOVQ    R11, R13
    ADDQ    $28, R13
    VEXTRACTPS $0, X7, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X7, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X7, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X7, (R13)
    ADDQ    R12, R13
    VEXTRACTF128 $1, Y7, X8
    VEXTRACTPS $0, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $1, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $2, X8, (R13)
    ADDQ    R12, R13
    VEXTRACTPS $3, X8, (R13)
    
    VZEROUPPER
    RET

// FusedGEMMBiasReLU4x4 for smaller matrices
// Uses XMM registers for 4x4 blocks
TEXT ·FusedGEMMBiasReLU4x4(SB), NOSPLIT, $0
    // TODO: Implement 4x4 kernel for edge cases
    RET
