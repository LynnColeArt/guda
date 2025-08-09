//go:build amd64 && !noasm
// +build amd64,!noasm

#include "textflag.h"

// func sgemmKernel16x4AVX512(a, b, c unsafe.Pointer, kc int64, ldc int64)
// Computes C[16x4] += A[16xkc] * B[kcx4]
// where:
//   a points to packed A (16-row blocks)
//   b points to packed B (4-column panels)
//   c points to output C in row-major layout
//   kc is the number of K iterations
//   ldc is the row stride of C in bytes

TEXT ·sgemmKernel16x4AVX512(SB), NOSPLIT, $0-40
    MOVQ a+0(FP), DI      // DI = A_pack
    MOVQ b+8(FP), SI      // SI = B_pack
    MOVQ c+16(FP), DX     // DX = C
    MOVQ kc+24(FP), R8    // R8 = KC counter
    MOVQ ldc+32(FP), CX   // CX = ldc in bytes

    // Zero accumulators for the 16x4 tile
    VXORPS Z0, Z0, Z0     // C[0:16, 0]
    VXORPS Z1, Z1, Z1     // C[0:16, 1]
    VXORPS Z2, Z2, Z2     // C[0:16, 2]
    VXORPS Z3, Z3, Z3     // C[0:16, 3]

    // Check if KC is zero
    TESTQ R8, R8
    JZ    store_c

    // Calculate offsets for B columns
    MOVQ kc+24(FP), R11   // R11 = original KC
    MOVQ R11, R10
    SHLQ $2, R10          // R10 = KC*4 (bytes per column)
    
    // R12 = 3*KC*4 for column 3
    MOVQ R10, R12
    ADDQ R10, R12
    ADDQ R10, R12

    // Check for unroll by 2
    MOVQ R8, R9
    SHRQ $1, R9          // R9 = KC/2
    JZ   k_remainder

k_loop_unroll2:
    // ---- K iteration 1 ----
    VMOVAPS 0(DI), Z9
    
    VBROADCASTSS 0(SI), Z8
    VFMADD231PS Z9, Z8, Z0
    
    VBROADCASTSS 0(SI)(R10*1), Z8
    VFMADD231PS Z9, Z8, Z1
    
    VBROADCASTSS 0(SI)(R10*2), Z8
    VFMADD231PS Z9, Z8, Z2
    
    VBROADCASTSS 0(SI)(R12*1), Z8
    VFMADD231PS Z9, Z8, Z3
    
    // ---- K iteration 2 ----
    VMOVAPS 64(DI), Z11
    
    VBROADCASTSS 4(SI), Z10
    VFMADD231PS Z11, Z10, Z0
    
    VBROADCASTSS 4(SI)(R10*1), Z10
    VFMADD231PS Z11, Z10, Z1
    
    VBROADCASTSS 4(SI)(R10*2), Z10
    VFMADD231PS Z11, Z10, Z2
    
    VBROADCASTSS 4(SI)(R12*1), Z10
    VFMADD231PS Z11, Z10, Z3
    
    // Advance pointers
    ADDQ $128, DI         // A += 32 floats
    ADDQ $8, SI           // B += 2 floats
    
    DECQ R9
    JNZ k_loop_unroll2

k_remainder:
    // Check if there's a remainder
    MOVQ R8, R9
    ANDQ $1, R9
    JZ   store_c
    
    // Single K iteration
    VMOVAPS 0(DI), Z9
    
    VBROADCASTSS 0(SI), Z8
    VFMADD231PS Z9, Z8, Z0
    
    VBROADCASTSS 0(SI)(R10*1), Z8
    VFMADD231PS Z9, Z8, Z1
    
    VBROADCASTSS 0(SI)(R10*2), Z8
    VFMADD231PS Z9, Z8, Z2
    
    VBROADCASTSS 0(SI)(R12*1), Z8
    VFMADD231PS Z9, Z8, Z3

store_c:
    // Now we need to store the 16x4 result in row-major layout
    // Z0-Z3 contain columns, but we need to store as rows
    
    // We'll use a more efficient approach with shuffles
    // First, let's handle storing using scatter operations if available,
    // or fall back to a transpose approach
    
    // Save non-volatile registers we'll use
    PUSHQ R12
    PUSHQ R13
    PUSHQ R14
    PUSHQ R15
    
    // Store 4x4 blocks at a time (4 blocks total)
    // Each 4x4 block will be transposed from column-major to row-major
    
    // Block 0: rows 0-3
    VEXTRACTF32X4 $0, Z0, X4   // X4 = C[0:4, 0]
    VEXTRACTF32X4 $0, Z1, X5   // X5 = C[0:4, 1]
    VEXTRACTF32X4 $0, Z2, X6   // X6 = C[0:4, 2]
    VEXTRACTF32X4 $0, Z3, X7   // X7 = C[0:4, 3]
    
    // Transpose 4x4 using unpack operations
    VUNPCKLPS X5, X4, X8       // X8 = [C00, C01, C10, C11]
    VUNPCKHPS X5, X4, X9       // X9 = [C20, C21, C30, C31]
    VUNPCKLPS X7, X6, X10      // X10 = [C02, C03, C12, C13]
    VUNPCKHPS X7, X6, X11      // X11 = [C22, C23, C32, C33]
    
    VUNPCKLPD X10, X8, X12     // X12 = [C00, C01, C02, C03] = row 0
    VUNPCKHPD X10, X8, X13     // X13 = [C10, C11, C12, C13] = row 1
    VUNPCKLPD X11, X9, X14     // X14 = [C20, C21, C22, C23] = row 2
    VUNPCKHPD X11, X9, X15     // X15 = [C30, C31, C32, C33] = row 3
    
    // Store rows
    VMOVUPS X12, 0(DX)         // row 0
    ADDQ CX, DX
    VMOVUPS X13, 0(DX)         // row 1
    ADDQ CX, DX
    VMOVUPS X14, 0(DX)         // row 2
    ADDQ CX, DX
    VMOVUPS X15, 0(DX)         // row 3
    ADDQ CX, DX
    
    // Block 1: rows 4-7
    VEXTRACTF32X4 $1, Z0, X4
    VEXTRACTF32X4 $1, Z1, X5
    VEXTRACTF32X4 $1, Z2, X6
    VEXTRACTF32X4 $1, Z3, X7
    
    VUNPCKLPS X5, X4, X8
    VUNPCKHPS X5, X4, X9
    VUNPCKLPS X7, X6, X10
    VUNPCKHPS X7, X6, X11
    
    VUNPCKLPD X10, X8, X12
    VUNPCKHPD X10, X8, X13
    VUNPCKLPD X11, X9, X14
    VUNPCKHPD X11, X9, X15
    
    VMOVUPS X12, 0(DX)
    ADDQ CX, DX
    VMOVUPS X13, 0(DX)
    ADDQ CX, DX
    VMOVUPS X14, 0(DX)
    ADDQ CX, DX
    VMOVUPS X15, 0(DX)
    ADDQ CX, DX
    
    // Block 2: rows 8-11
    VEXTRACTF32X4 $2, Z0, X4
    VEXTRACTF32X4 $2, Z1, X5
    VEXTRACTF32X4 $2, Z2, X6
    VEXTRACTF32X4 $2, Z3, X7
    
    VUNPCKLPS X5, X4, X8
    VUNPCKHPS X5, X4, X9
    VUNPCKLPS X7, X6, X10
    VUNPCKHPS X7, X6, X11
    
    VUNPCKLPD X10, X8, X12
    VUNPCKHPD X10, X8, X13
    VUNPCKLPD X11, X9, X14
    VUNPCKHPD X11, X9, X15
    
    VMOVUPS X12, 0(DX)
    ADDQ CX, DX
    VMOVUPS X13, 0(DX)
    ADDQ CX, DX
    VMOVUPS X14, 0(DX)
    ADDQ CX, DX
    VMOVUPS X15, 0(DX)
    ADDQ CX, DX
    
    // Block 3: rows 12-15
    VEXTRACTF32X4 $3, Z0, X4
    VEXTRACTF32X4 $3, Z1, X5
    VEXTRACTF32X4 $3, Z2, X6
    VEXTRACTF32X4 $3, Z3, X7
    
    VUNPCKLPS X5, X4, X8
    VUNPCKHPS X5, X4, X9
    VUNPCKLPS X7, X6, X10
    VUNPCKHPS X7, X6, X11
    
    VUNPCKLPD X10, X8, X12
    VUNPCKHPD X10, X8, X13
    VUNPCKLPD X11, X9, X14
    VUNPCKHPD X11, X9, X15
    
    VMOVUPS X12, 0(DX)
    ADDQ CX, DX
    VMOVUPS X13, 0(DX)
    ADDQ CX, DX
    VMOVUPS X14, 0(DX)
    ADDQ CX, DX
    VMOVUPS X15, 0(DX)
    
    // Restore registers
    POPQ R15
    POPQ R14
    POPQ R13
    POPQ R12
    
    RET

done: