// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// FMA3 optimized version of AxpyUnitary for Float32
// y[i] += alpha * x[i] using fused multiply-add
//
// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define X_PTR SI
#define Y_PTR DI
#define DST_PTR DI
#define IDX AX
#define LEN CX
#define TAIL BX
#define ALPHA Y0
#define ALPHA_2 Y1

// func AxpyUnitaryFMA(alpha float32, x, y []float32)
TEXT ·axpyUnitaryFMA(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), X_PTR  // X_PTR := &x
	MOVQ    y_base+32(FP), Y_PTR // Y_PTR := &y
	MOVQ    x_len+16(FP), LEN    // LEN = min( len(x), len(y) )
	CMPQ    y_len+40(FP), LEN
	CMOVQLE y_len+40(FP), LEN
	CMPQ    LEN, $0              // if LEN == 0 { return }
	JE      end
	XORQ    IDX, IDX
	
	// Broadcast alpha to all lanes of YMM register (8 floats)
	VBROADCASTSS alpha+0(FP), ALPHA   // ALPHA := { alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha }
	VMOVUPS      ALPHA, ALPHA_2       // ALPHA_2 := ALPHA for pipelining
	
	MOVQ    Y_PTR, TAIL          // Check memory alignment
	ANDQ    $31, TAIL            // TAIL = &y % 32 (for 32-byte AVX2 alignment)
	JZ      no_trim              // if TAIL == 0 { goto no_trim }

	// Align on 32-byte boundary using scalar operations
align_loop:
	MOVSS (X_PTR)(IDX*4), X2     // X2 := x[i]
	MOVSS (Y_PTR)(IDX*4), X3     // X3 := y[i]
	VFMADD231SS alpha+0(FP), X2, X3  // X3 = X3 + alpha * X2
	MOVSS X3, (DST_PTR)(IDX*4)   // y[i] = X3
	INCQ  IDX                    // i++
	DECQ  LEN                    // LEN--
	JZ    end                    // if LEN == 0 { return }
	MOVQ  Y_PTR, TAIL
	LEAQ  (TAIL)(IDX*4), TAIL
	ANDQ  $31, TAIL
	JNZ   align_loop

no_trim:
	MOVQ LEN, TAIL
	ANDQ $31, TAIL   // TAIL := n % 32 (32 floats with AVX2)
	SHRQ $5, LEN     // LEN = floor( n / 32 )
	JZ   tail_start  // if LEN == 0 { goto tail_start }

loop:  // Main loop: process 32 floats per iteration using FMA
	// Load y[i:i+32] into 4 YMM registers
	VMOVUPS (Y_PTR)(IDX*4), Y2        // Y2 = y[i:i+8]
	VMOVUPS 32(Y_PTR)(IDX*4), Y3      // Y3 = y[i+8:i+16]
	VMOVUPS 64(Y_PTR)(IDX*4), Y4      // Y4 = y[i+16:i+24]
	VMOVUPS 96(Y_PTR)(IDX*4), Y5      // Y5 = y[i+24:i+32]
	
	// Load x[i:i+32] and perform fused multiply-add: y = y + alpha * x
	VFMADD231PS (X_PTR)(IDX*4), ALPHA, Y2      // Y2 = Y2 + ALPHA * x[i:i+8]
	VFMADD231PS 32(X_PTR)(IDX*4), ALPHA_2, Y3  // Y3 = Y3 + ALPHA_2 * x[i+8:i+16]
	VFMADD231PS 64(X_PTR)(IDX*4), ALPHA, Y4    // Y4 = Y4 + ALPHA * x[i+16:i+24]
	VFMADD231PS 96(X_PTR)(IDX*4), ALPHA_2, Y5  // Y5 = Y5 + ALPHA_2 * x[i+24:i+32]
	
	// Store results back to y
	VMOVUPS Y2, (DST_PTR)(IDX*4)       // y[i:i+8] = Y2
	VMOVUPS Y3, 32(DST_PTR)(IDX*4)     // y[i+8:i+16] = Y3
	VMOVUPS Y4, 64(DST_PTR)(IDX*4)     // y[i+16:i+24] = Y4
	VMOVUPS Y5, 96(DST_PTR)(IDX*4)     // y[i+24:i+32] = Y5
	
	ADDQ    $32, IDX             // IDX += 32
	DECQ    LEN
	JNZ     loop                 // } while --LEN > 0

tail_start:
	CMPQ TAIL, $16               // if TAIL >= 16
	JL   tail8_start

	// Process 16 floats
	VMOVUPS (Y_PTR)(IDX*4), Y2        // Y2 = y[i:i+8]
	VMOVUPS 32(Y_PTR)(IDX*4), Y3      // Y3 = y[i+8:i+16]
	VFMADD231PS (X_PTR)(IDX*4), ALPHA, Y2      // Y2 = Y2 + ALPHA * x[i:i+8]
	VFMADD231PS 32(X_PTR)(IDX*4), ALPHA_2, Y3  // Y3 = Y3 + ALPHA_2 * x[i+8:i+16]
	VMOVUPS Y2, (DST_PTR)(IDX*4)       // y[i:i+8] = Y2
	VMOVUPS Y3, 32(DST_PTR)(IDX*4)     // y[i+8:i+16] = Y3
	ADDQ $16, IDX
	SUBQ $16, TAIL

tail8_start:
	CMPQ TAIL, $8                // if TAIL >= 8
	JL   tail4_start

	// Process 8 floats
	VMOVUPS (Y_PTR)(IDX*4), Y2        // Y2 = y[i:i+8]
	VFMADD231PS (X_PTR)(IDX*4), ALPHA, Y2      // Y2 = Y2 + ALPHA * x[i:i+8]
	VMOVUPS Y2, (DST_PTR)(IDX*4)       // y[i:i+8] = Y2
	ADDQ $8, IDX
	SUBQ $8, TAIL

tail4_start:
	CMPQ TAIL, $4                // if TAIL >= 4
	JL   tail_start_scalar

	// Process 4 floats using SSE FMA
	MOVSS   alpha+0(FP), X0           // X0 = alpha
	SHUFPS  $0, X0, X0                // X0 = { alpha, alpha, alpha, alpha }
	MOVUPS (Y_PTR)(IDX*4), X2          // X2 = y[i:i+4]
	MOVUPS (X_PTR)(IDX*4), X3          // X3 = x[i:i+4]
	VFMADD231PS X3, X0, X2             // X2 = X2 + X0 * X3
	MOVUPS X2, (DST_PTR)(IDX*4)        // y[i:i+4] = X2
	ADDQ $4, IDX
	SUBQ $4, TAIL

tail_start_scalar:
	CMPQ TAIL, $0                // if TAIL == 0 { return }
	JE   end

tail_loop:  // Process remaining elements one at a time
	MOVSS (X_PTR)(IDX*4), X2     // X2 := x[i]
	MOVSS (Y_PTR)(IDX*4), X3     // X3 := y[i]
	VFMADD231SS alpha+0(FP), X2, X3  // X3 = X3 + alpha * X2
	MOVSS X3, (DST_PTR)(IDX*4)   // y[i] = X3
	INCQ  IDX                    // i++
	DECQ  TAIL
	JNZ   tail_loop              // } while --TAIL > 0

end:
	RET
