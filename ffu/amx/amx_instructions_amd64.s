//go:build amd64
// +build amd64

#include "textflag.h"

// AMX instruction encodings
// Since the assembler doesn't support AMX yet, we encode manually

// func amxInt8GEMMReference(m, n, k int, a, b []byte, c []int32)
// Reference implementation in assembly for correctness testing
TEXT ·amxInt8GEMMReference(SB), NOSPLIT, $0-104
	MOVQ m+0(FP), CX        // M dimension
	MOVQ n+8(FP), DX        // N dimension  
	MOVQ k+16(FP), BX       // K dimension
	MOVQ a_base+24(FP), SI  // A matrix base
	MOVQ b_base+48(FP), DI  // B matrix base
	MOVQ c_base+72(FP), R8  // C matrix base
	
	// Clear C matrix first
	MOVQ CX, R9             // i = M
	MOVQ DX, R10            // Save N
clear_loop:
	MOVQ R10, R11           // j = N
clear_inner:
	MOVL $0, (R8)
	ADDQ $4, R8             // C is INT32
	DECQ R11
	JNZ clear_inner
	DECQ R9
	JNZ clear_loop
	
	// Reset C pointer
	MOVQ c_base+72(FP), R8
	
	// Triple nested loop: C[i,j] += A[i,k] * B[k,j]
	MOVQ CX, R9             // i = M
loop_i:
	MOVQ R10, R11           // j = N
loop_j:
	XORL R12, R12           // sum = 0
	MOVQ BX, R13            // k = K
	
	// Calculate pointers
	MOVQ SI, R14            // A[i,0]
	MOVQ DI, R15            // B[0,j]
	
loop_k:
	MOVBLSX (R14), AX       // Load A[i,k] as signed byte
	MOVBLSX (R15), CX       // Load B[k,j] as signed byte
	IMULL CX, AX            // AX = A[i,k] * B[k,j]
	ADDL AX, R12            // sum += AX
	
	INCQ R14                // A advance by 1 (next k)
	ADDQ R10, R15           // B advance by N (next k)
	
	DECQ R13
	JNZ loop_k
	
	// Store result
	MOVL R12, (R8)
	ADDQ $4, R8             // Next C element
	
	INCQ DI                 // Next column of B
	DECQ R11
	JNZ loop_j
	
	// Next row of A
	ADDQ BX, SI             // A advance by K
	SUBQ R10, DI            // Reset B pointer
	
	DECQ R9
	JNZ loop_i
	
	RET
