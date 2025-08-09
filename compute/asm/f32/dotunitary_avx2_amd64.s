// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// AVX2/FMA optimized version of DotUnitary for Float32
// sum = Σ(x[i] * y[i])
//
// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define X_PTR SI
#define Y_PTR DI
#define IDX AX
#define LEN CX
#define TAIL BX

// Accumulator registers
#define ACC0 Y0
#define ACC1 Y1
#define ACC2 Y2
#define ACC3 Y3

// func DotUnitaryAVX2(x, y []float32) float32
TEXT ·dotUnitaryAVX2(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR   // X_PTR := &x
	MOVQ    y_base+24(FP), Y_PTR  // Y_PTR := &y
	MOVQ    x_len+8(FP), LEN      // LEN = min(len(x), len(y))
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	
	XORPS   X0, X0               // result = 0
	CMPQ    LEN, $0              // if LEN == 0 { return 0 }
	JE      end
	
	XORQ    IDX, IDX
	
	// Zero out accumulator registers
	VXORPS  ACC0, ACC0, ACC0
	VXORPS  ACC1, ACC1, ACC1
	VXORPS  ACC2, ACC2, ACC2
	VXORPS  ACC3, ACC3, ACC3
	
	MOVQ    LEN, TAIL
	ANDQ    $31, TAIL            // TAIL = LEN % 32
	SHRQ    $5, LEN              // LEN = LEN / 32
	JZ      tail_start
	
loop:  // Main loop: process 32 floats per iteration using 4 accumulators
	// Prefetch next iteration's data (8 cache lines ahead)
	PREFETCHT0 512(X_PTR)(IDX*4)      // Prefetch x[i+128:i+144]
	PREFETCHT0 512(Y_PTR)(IDX*4)      // Prefetch y[i+128:i+144]
	
	// Load x and y values
	VMOVUPS (X_PTR)(IDX*4), Y4
	VMOVUPS 32(X_PTR)(IDX*4), Y5
	VMOVUPS 64(X_PTR)(IDX*4), Y6
	VMOVUPS 96(X_PTR)(IDX*4), Y7
	
	// Multiply-accumulate: ACC = ACC + x * y
	VFMADD231PS (Y_PTR)(IDX*4), Y4, ACC0       // ACC0 += x[i:i+8] * y[i:i+8]
	VFMADD231PS 32(Y_PTR)(IDX*4), Y5, ACC1     // ACC1 += x[i+8:i+16] * y[i+8:i+16]
	VFMADD231PS 64(Y_PTR)(IDX*4), Y6, ACC2     // ACC2 += x[i+16:i+24] * y[i+16:i+24]
	VFMADD231PS 96(Y_PTR)(IDX*4), Y7, ACC3     // ACC3 += x[i+24:i+32] * y[i+24:i+32]
	
	ADDQ    $32, IDX
	DECQ    LEN
	JNZ     loop

	// Reduce 4 YMM accumulators to 1
	VADDPS  ACC1, ACC0, ACC0     // ACC0 = ACC0 + ACC1
	VADDPS  ACC3, ACC2, ACC2     // ACC2 = ACC2 + ACC3
	VADDPS  ACC2, ACC0, ACC0     // ACC0 = ACC0 + ACC2

tail_start:
	CMPQ    TAIL, $16
	JL      tail8_start
	
	// Process 16 floats
	VMOVUPS (X_PTR)(IDX*4), Y4
	VMOVUPS 32(X_PTR)(IDX*4), Y5
	VFMADD231PS (Y_PTR)(IDX*4), Y4, ACC0       // ACC0 += x[i:i+8] * y[i:i+8]
	VFMADD231PS 32(Y_PTR)(IDX*4), Y5, ACC0     // ACC0 += x[i+8:i+16] * y[i+8:i+16]
	ADDQ    $16, IDX
	SUBQ    $16, TAIL

tail8_start:
	CMPQ    TAIL, $8
	JL      tail4_start
	
	// Process 8 floats
	VMOVUPS (X_PTR)(IDX*4), Y4
	VFMADD231PS (Y_PTR)(IDX*4), Y4, ACC0       // ACC0 += x[i:i+8] * y[i:i+8]
	ADDQ    $8, IDX
	SUBQ    $8, TAIL

tail4_start:
	// Horizontal sum of YMM to XMM
	VEXTRACTF128 $1, ACC0, X1    // X1 = upper 128 bits of ACC0
	VADDPS  X1, X0, X0           // X0 = lower + upper halves
	
	CMPQ    TAIL, $4
	JL      tail_start_scalar
	
	// Process 4 floats
	MOVUPS  (X_PTR)(IDX*4), X1   // X1 = x[i:i+4]
	MOVUPS  (Y_PTR)(IDX*4), X2   // X2 = y[i:i+4]
	MULPS   X1, X2               // X2 = X1 * X2
	ADDPS   X2, X0               // X0 += X2
	ADDQ    $4, IDX
	SUBQ    $4, TAIL

tail_start_scalar:
	// Horizontal sum of XMM register
	MOVHLPS X0, X1               // X1 = X0[2:3]
	ADDPS   X1, X0               // X0[0:1] = X0[0:1] + X0[2:3]
	MOVSS   X0, X1
	SHUFPS  $0x1, X0, X0         // X0[0] = X0[1]
	ADDSS   X1, X0               // X0 = sum of all 4 floats
	
	CMPQ    TAIL, $0
	JE      end

tail_loop:  // Process remaining elements one at a time
	MOVSS   (X_PTR)(IDX*4), X1   // X1 = x[i]
	MULSS   (Y_PTR)(IDX*4), X1   // X1 = x[i] * y[i]
	ADDSS   X1, X0               // X0 += X1
	INCQ    IDX
	DECQ    TAIL
	JNZ     tail_loop

end:
	MOVSS   X0, ret+48(FP)       // return X0
	RET
