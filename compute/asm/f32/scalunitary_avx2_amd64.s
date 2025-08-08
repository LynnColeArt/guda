// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// AVX2 optimized version of ScalUnitary for Float32
// x[i] *= alpha
//
// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define X_PTR SI
#define IDX AX
#define LEN CX
#define TAIL BX
#define ALPHA Y0
#define ALPHA_2 Y1

// func ScalUnitaryAVX2(alpha float32, x []float32)
TEXT ·ScalUnitaryAVX2(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), X_PTR   // X_PTR := &x
	MOVQ    x_len+16(FP), LEN     // LEN = len(x)
	CMPQ    LEN, $0               // if LEN == 0 { return }
	JE      end
	
	XORQ    IDX, IDX
	
	// Broadcast alpha to all lanes of YMM register
	VBROADCASTSS alpha+0(FP), ALPHA    // ALPHA = { alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha }
	VMOVUPS      ALPHA, ALPHA_2        // ALPHA_2 = ALPHA for pipelining
	
	// Check if x is aligned to 32-byte boundary
	MOVQ    X_PTR, TAIL
	ANDQ    $31, TAIL
	JZ      no_trim
	
	// Align to 32-byte boundary
align_loop:
	MOVSS   (X_PTR)(IDX*4), X1    // X1 = x[i]
	MULSS   alpha+0(FP), X1       // X1 *= alpha
	MOVSS   X1, (X_PTR)(IDX*4)    // x[i] = X1
	INCQ    IDX
	DECQ    LEN
	JZ      end
	MOVQ    X_PTR, TAIL
	LEAQ    (TAIL)(IDX*4), TAIL
	ANDQ    $31, TAIL
	JNZ     align_loop

no_trim:
	MOVQ    LEN, TAIL
	ANDQ    $31, TAIL             // TAIL = LEN % 32
	SHRQ    $5, LEN               // LEN = LEN / 32
	JZ      tail_start

loop:  // Main loop: process 32 floats per iteration
	// Load x[i:i+32] and multiply by alpha
	VMULPS  (X_PTR)(IDX*4), ALPHA, Y2       // Y2 = alpha * x[i:i+8]
	VMULPS  32(X_PTR)(IDX*4), ALPHA_2, Y3   // Y3 = alpha * x[i+8:i+16]
	VMULPS  64(X_PTR)(IDX*4), ALPHA, Y4     // Y4 = alpha * x[i+16:i+24]
	VMULPS  96(X_PTR)(IDX*4), ALPHA_2, Y5   // Y5 = alpha * x[i+24:i+32]
	
	// Store results back
	VMOVUPS Y2, (X_PTR)(IDX*4)              // x[i:i+8] = Y2
	VMOVUPS Y3, 32(X_PTR)(IDX*4)            // x[i+8:i+16] = Y3
	VMOVUPS Y4, 64(X_PTR)(IDX*4)            // x[i+16:i+24] = Y4
	VMOVUPS Y5, 96(X_PTR)(IDX*4)            // x[i+24:i+32] = Y5
	
	ADDQ    $32, IDX
	DECQ    LEN
	JNZ     loop

tail_start:
	CMPQ    TAIL, $16
	JL      tail8_start
	
	// Process 16 floats
	VMULPS  (X_PTR)(IDX*4), ALPHA, Y2       // Y2 = alpha * x[i:i+8]
	VMULPS  32(X_PTR)(IDX*4), ALPHA_2, Y3   // Y3 = alpha * x[i+8:i+16]
	VMOVUPS Y2, (X_PTR)(IDX*4)              // x[i:i+8] = Y2
	VMOVUPS Y3, 32(X_PTR)(IDX*4)            // x[i+8:i+16] = Y3
	ADDQ    $16, IDX
	SUBQ    $16, TAIL

tail8_start:
	CMPQ    TAIL, $8
	JL      tail4_start
	
	// Process 8 floats
	VMULPS  (X_PTR)(IDX*4), ALPHA, Y2       // Y2 = alpha * x[i:i+8]
	VMOVUPS Y2, (X_PTR)(IDX*4)              // x[i:i+8] = Y2
	ADDQ    $8, IDX
	SUBQ    $8, TAIL

tail4_start:
	CMPQ    TAIL, $4
	JL      tail_start_scalar
	
	// Process 4 floats using SSE
	MOVUPS  (X_PTR)(IDX*4), X1              // X1 = x[i:i+4]
	MULPS   X0, X1                          // X1 *= alpha (X0 has alpha broadcasted)
	MOVUPS  X1, (X_PTR)(IDX*4)              // x[i:i+4] = X1
	ADDQ    $4, IDX
	SUBQ    $4, TAIL

tail_start_scalar:
	CMPQ    TAIL, $0
	JE      end

tail_loop:  // Process remaining elements one at a time
	MOVSS   (X_PTR)(IDX*4), X1    // X1 = x[i]
	MULSS   alpha+0(FP), X1       // X1 *= alpha
	MOVSS   X1, (X_PTR)(IDX*4)    // x[i] = X1
	INCQ    IDX
	DECQ    TAIL
	JNZ     tail_loop

end:
	RET
