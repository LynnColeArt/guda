//go:build amd64
// +build amd64

#include "textflag.h"

// VNNI INT8 GEMM kernel using AVX512-VNNI instructions
// This provides INT8 matrix multiplication similar to AMX but using AVX512

// func vnniInt8GEMM_16x16(m, n, k int, a, b []int8, c []int32)
TEXT ·vnniInt8GEMM_16x16(SB), NOSPLIT, $0-104
	MOVQ m+0(FP), AX      // M dimension
	MOVQ n+8(FP), BX      // N dimension  
	MOVQ k+16(FP), CX     // K dimension
	MOVQ a_base+24(FP), SI  // A matrix
	MOVQ b_base+48(FP), DI  // B matrix
	MOVQ c_base+72(FP), R8  // C matrix
	
	// For now, implement reference version
	// Real VNNI would use VPDPBUSD instruction
	
	// Clear C matrix
	MOVQ AX, R9        // i = M
clear_outer:
	MOVQ BX, R10       // j = N
clear_inner:
	MOVL $0, (R8)
	ADDQ $4, R8        // C is INT32
	DECQ R10
	JNZ clear_inner
	DECQ R9
	JNZ clear_outer
	
	// Reset C pointer
	MOVQ c_base+72(FP), R8
	
	// Triple nested loop for reference
	MOVQ AX, R9        // i = M
loop_i:
	MOVQ BX, R10       // j = N
loop_j:
	XORL R11, R11      // sum = 0
	MOVQ CX, R12       // k = K
	
	MOVQ SI, R13       // A[i,0]
	MOVQ DI, R14       // B[0,j]
	
loop_k:
	MOVBLSX (R13), AX   // Load A[i,k]
	MOVBLSX (R14), DX   // Load B[k,j]
	IMULL DX, AX        // A*B
	ADDL AX, R11        // sum += A*B
	
	INCQ R13            // Next A element
	ADDQ BX, R14        // Next B row
	
	DECQ R12
	JNZ loop_k
	
	MOVL R11, (R8)      // Store result
	ADDQ $4, R8         // Next C element
	
	INCQ DI             // Next B column
	DECQ R10
	JNZ loop_j
	
	ADDQ CX, SI         // Next A row
	SUBQ BX, DI         // Reset B pointer
	
	DECQ R9
	JNZ loop_i
	
	RET

// func hasVNNI() bool
TEXT ·hasVNNI(SB), NOSPLIT, $0-1
	MOVL $7, AX
	MOVL $0, CX
	CPUID
	
	// Check ECX bit 11 for AVX512-VNNI
	MOVL CX, AX
	SHRL $11, AX
	ANDL $1, AX
	MOVB AX, ret+0(FP)
	RET