//go:build amd64 && !cgo
// +build amd64,!cgo

#include "textflag.h"

// Reference implementation without VNNI instructions
// This ensures our tests pass while we work on the real VNNI

// func vnniInt8GEMMRef(m, n, k int, a, b []int8, c []int32)
TEXT ·vnniInt8GEMMRef(SB), NOSPLIT, $0-104
	MOVQ m+0(FP), AX      // M dimension
	MOVQ n+8(FP), BX      // N dimension  
	MOVQ k+16(FP), CX     // K dimension
	MOVQ a_base+24(FP), SI  // A matrix
	MOVQ b_base+48(FP), DI  // B matrix
	MOVQ c_base+72(FP), R8  // C matrix
	
	// Clear C matrix
	MOVQ AX, R9           // i = M
	MOVQ BX, R10          // Save N
clear_loop:
	MOVQ R10, R11         // j = N
clear_inner:
	MOVL $0, (R8)
	ADDQ $4, R8           // C is INT32
	DECQ R11
	JNZ clear_inner
	DECQ R9
	JNZ clear_loop
	
	// Reset C pointer
	MOVQ c_base+72(FP), R8
	
	// Triple nested loop
	MOVQ AX, R9           // i = M
loop_i:
	MOVQ R10, R11         // j = N
loop_j:
	XORL R12, R12         // sum = 0
	MOVQ CX, R13          // k = K
	
	// Calculate pointers
	MOVQ SI, R14          // A[i,0]
	MOVQ DI, R15          // B[0,j]
	
loop_k:
	MOVBLSX (R14), AX     // Load A[i,k] as signed byte
	MOVBLSX (R15), DX     // Load B[k,j] as signed byte
	IMULL DX, AX          // AX = A[i,k] * B[k,j]
	ADDL AX, R12          // sum += AX
	
	INCQ R14              // A advance by 1
	ADDQ R10, R15         // B advance by N
	
	DECQ R13
	JNZ loop_k
	
	// Store result
	MOVL R12, (R8)
	ADDQ $4, R8           // Next C element
	
	INCQ DI               // Next column of B
	DECQ R11
	JNZ loop_j
	
	// Next row of A
	ADDQ CX, SI           // A advance by K
	SUBQ R10, DI          // Reset B pointer
	
	DECQ R9
	JNZ loop_i
	
	RET
