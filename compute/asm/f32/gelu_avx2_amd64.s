// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !appengine && !gccgo
// +build !noasm,!appengine,!gccgo

#include "textflag.h"

// GeluAVX2 computes GELU activation for a slice of float32
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// func GeluAVX2(x []float32)
TEXT ·GeluAVX2(SB), NOSPLIT, $32-24
    MOVQ    x_base+0(FP), SI    // x pointer
    MOVQ    x_len+8(FP), CX     // length
    
    // Store constants on stack
    MOVL    $0x3f4c422a, 0(SP)  // 0.7978845608 (sqrt(2/π))
    MOVL    $0x3d372713, 4(SP)  // 0.044715
    MOVL    $0x3f000000, 8(SP)  // 0.5
    MOVL    $0x3f800000, 12(SP) // 1.0
    MOVL    $0x41d80000, 16(SP) // 27.0
    MOVL    $0x41100000, 20(SP) // 9.0
    MOVL    $0x40400000, 24(SP) // 3.0
    MOVL    $0xc0400000, 28(SP) // -3.0
    
    // Load and broadcast constants
    VBROADCASTSS 0(SP), Y15     // sqrt(2/π)
    VBROADCASTSS 4(SP), Y14     // 0.044715
    VBROADCASTSS 8(SP), Y13     // 0.5
    VBROADCASTSS 12(SP), Y12    // 1.0
    VBROADCASTSS 16(SP), Y11    // 27.0
    VBROADCASTSS 20(SP), Y10    // 9.0
    VBROADCASTSS 24(SP), Y9     // 3.0
    VBROADCASTSS 28(SP), Y8     // -3.0
    
    // Process 8 values at a time
    SHRQ    $3, CX              // CX = n / 8
    JZ      tail                // Skip if less than 8 elements
    
loop8:
    VMOVUPS (SI), Y0            // Load 8 x values
    
    // Compute x³
    VMULPS  Y0, Y0, Y1          // Y1 = x²
    VMULPS  Y1, Y0, Y2          // Y2 = x³
    
    // Compute 0.044715 * x³
    VMULPS  Y14, Y2, Y3         // Y3 = 0.044715 * x³
    
    // Compute x + 0.044715 * x³
    VADDPS  Y0, Y3, Y4          // Y4 = x + 0.044715 * x³
    
    // Compute arg = sqrt(2/π) * (x + 0.044715 * x³)
    VMULPS  Y15, Y4, Y5         // Y5 = arg
    
    // Clamp arg to [-3, 3]
    VMAXPS  Y8, Y5, Y5          // Y5 = max(-3, arg)
    VMINPS  Y9, Y5, Y5          // Y5 = min(3, max(-3, arg))
    
    // Compute tanh using rational approximation
    // tanh(x) ≈ x * (27 + x²) / (27 + 9x²)
    VMULPS  Y5, Y5, Y1          // Y1 = arg²
    VMULPS  Y10, Y1, Y2         // Y2 = 9*arg²
    VADDPS  Y11, Y1, Y3         // Y3 = 27 + arg²
    VADDPS  Y11, Y2, Y4         // Y4 = 27 + 9*arg²
    
    VMULPS  Y5, Y3, Y5          // Y5 = arg * (27 + arg²)
    VDIVPS  Y4, Y5, Y5          // Y5 = tanh(arg)
    
    // Compute 1 + tanh(arg)
    VADDPS  Y12, Y5, Y5         // Y5 = 1 + tanh(arg)
    
    // Compute 0.5 * x * (1 + tanh(arg))
    VMULPS  Y13, Y0, Y0         // Y0 = 0.5 * x
    VMULPS  Y5, Y0, Y0          // Y0 = 0.5 * x * (1 + tanh(arg))
    
    // Store result
    VMOVUPS Y0, (SI)
    
    ADDQ    $32, SI             // Advance pointer by 8 floats
    DECQ    CX
    JNZ     loop8
    
tail:
    // Handle remaining elements
    MOVQ    x_len+8(FP), CX
    ANDQ    $7, CX              // CX = n % 8
    JZ      done
    
    // Load constants for scalar operations
    MOVSS   0(SP), X15          // sqrt(2/π)
    MOVSS   4(SP), X14          // 0.044715
    MOVSS   8(SP), X13          // 0.5
    MOVSS   12(SP), X12         // 1.0
    MOVSS   16(SP), X11         // 27.0
    MOVSS   20(SP), X10         // 9.0
    MOVSS   24(SP), X9          // 3.0
    MOVSS   28(SP), X8          // -3.0
    
tail_loop:
    // Process one element at a time
    MOVSS   (SI), X0
    
    // Compute x³
    MOVSS   X0, X1
    MULSS   X0, X1              // X1 = x²
    MULSS   X0, X1              // X1 = x³
    
    // Compute 0.044715 * x³
    MULSS   X14, X1             // X1 = 0.044715 * x³
    
    // Compute x + 0.044715 * x³
    ADDSS   X0, X1              // X1 = x + 0.044715 * x³
    
    // Compute arg = sqrt(2/π) * (x + 0.044715 * x³)
    MULSS   X15, X1             // X1 = arg
    
    // Clamp arg to [-3, 3]
    MAXSS   X8, X1              // max(-3, arg)
    MINSS   X9, X1              // min(3, max(-3, arg))
    
    // Compute tanh
    MOVSS   X1, X2
    MULSS   X1, X2              // X2 = arg²
    
    MOVSS   X11, X3             // X3 = 27
    MOVSS   X2, X4
    ADDSS   X3, X4              // X4 = 27 + arg²
    
    MOVSS   X10, X5             // X5 = 9
    MULSS   X2, X5              // X5 = 9*arg²
    ADDSS   X3, X5              // X5 = 27 + 9*arg²
    
    MULSS   X4, X1              // X1 = arg * (27 + arg²)
    DIVSS   X5, X1              // X1 = tanh(arg)
    
    // Compute 1 + tanh(arg)
    ADDSS   X12, X1             // X1 = 1 + tanh(arg)
    
    // Compute 0.5 * x * (1 + tanh(arg))
    MULSS   X13, X0             // X0 = 0.5 * x
    MULSS   X1, X0              // X0 = 0.5 * x * (1 + tanh(arg))
    
    MOVSS   X0, (SI)
    ADDQ    $4, SI
    DECQ    CX
    JNZ     tail_loop
    
done:
    VZEROUPPER
    RET
