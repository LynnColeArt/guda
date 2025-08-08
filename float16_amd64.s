// +build amd64

#include "textflag.h"

// func simdF16ToF32AVX2(src, dst unsafe.Pointer, n int)
// Converts float16 to float32 using F16C VCVTPH2PS instruction
TEXT ·simdF16ToF32AVX2(SB), NOSPLIT, $0-24
    MOVQ src+0(FP), SI  // source pointer
    MOVQ dst+8(FP), DI  // destination pointer
    MOVQ n+16(FP), CX   // count
    
    // Process 8 elements at a time
    MOVQ CX, AX
    SHRQ $3, AX         // AX = n / 8
    JZ tail             // If less than 8 elements, go to tail
    
loop8:
    VMOVDQU (SI), X0           // Load 8 float16 values (128 bits)
    VCVTPH2PS X0, Y0          // Convert to 8 float32 values (256 bits)
    VMOVUPS Y0, (DI)          // Store 8 float32 values
    
    ADDQ $16, SI              // Advance src by 16 bytes (8 float16s)
    ADDQ $32, DI              // Advance dst by 32 bytes (8 float32s)
    DECQ AX
    JNZ loop8
    
tail:
    ANDQ $7, CX               // CX = n % 8
    JZ done
    
tailloop:
    MOVW (SI), AX             // Load one float16
    MOVW AX, -2(SP)           // Store to stack
    VCVTPH2PS -2(SP), X0      // Convert single value
    VMOVSS X0, (DI)           // Store float32
    
    ADDQ $2, SI
    ADDQ $4, DI
    DECQ CX
    JNZ tailloop
    
done:
    VZEROUPPER
    RET

// func simdF32ToF16AVX2(src, dst unsafe.Pointer, n int)
// Converts float32 to float16 using F16C VCVTPS2PH instruction
TEXT ·simdF32ToF16AVX2(SB), NOSPLIT, $0-24
    MOVQ src+0(FP), SI  // source pointer
    MOVQ dst+8(FP), DI  // destination pointer
    MOVQ n+16(FP), CX   // count
    
    // Rounding mode: 0 = nearest even
    MOVQ $0, AX
    
    // Process 8 elements at a time
    MOVQ CX, BX
    SHRQ $3, BX         // BX = n / 8
    JZ tail2            // If less than 8 elements, go to tail
    
loop8_2:
    VMOVUPS (SI), Y0          // Load 8 float32 values (256 bits)
    VCVTPS2PH $0, Y0, X0      // Convert to 8 float16 values with rounding
    VMOVDQU X0, (DI)          // Store 8 float16 values
    
    ADDQ $32, SI              // Advance src by 32 bytes (8 float32s)
    ADDQ $16, DI              // Advance dst by 16 bytes (8 float16s)
    DECQ BX
    JNZ loop8_2
    
tail2:
    ANDQ $7, CX               // CX = n % 8
    JZ done2
    
tailloop2:
    VMOVSS (SI), X0           // Load one float32
    VCVTPS2PH $0, X0, X0      // Convert to float16
    VMOVD X0, AX              // Move to general register
    MOVW AX, (DI)             // Store float16
    
    ADDQ $4, SI
    ADDQ $2, DI
    DECQ CX
    JNZ tailloop2
    
done2:
    VZEROUPPER
    RET

// func simdAddFloat16AVX2(a, b, c unsafe.Pointer, n int)
// Adds two float16 arrays using F16C
TEXT ·simdAddFloat16AVX2(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), SI    // source A pointer
    MOVQ b+8(FP), DI    // source B pointer
    MOVQ c+16(FP), DX   // destination pointer
    MOVQ n+24(FP), CX   // count
    
    // Process 8 elements at a time
    MOVQ CX, AX
    SHRQ $3, AX         // AX = n / 8
    JZ tail3            // If less than 8 elements, go to tail
    
loop8_3:
    VMOVDQU (SI), X0          // Load 8 float16 from A
    VMOVDQU (DI), X1          // Load 8 float16 from B
    
    VCVTPH2PS X0, Y0          // Convert A to float32
    VCVTPH2PS X1, Y1          // Convert B to float32
    
    VADDPS Y0, Y1, Y2         // Add float32 values
    
    VCVTPS2PH $0, Y2, X2      // Convert result back to float16
    VMOVDQU X2, (DX)          // Store result
    
    ADDQ $16, SI              // Advance pointers by 16 bytes
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ loop8_3
    
tail3:
    ANDQ $7, CX               // CX = n % 8
    JZ done3
    
    // For tail, use scalar operations
tailloop3:
    MOVW (SI), AX             // Load A[i]
    MOVW (DI), BX             // Load B[i]
    
    // Convert and add using scalar F16C
    MOVW AX, -4(SP)
    MOVW BX, -2(SP)
    VCVTPH2PS -4(SP), X0
    VCVTPH2PS -2(SP), X1
    VADDSS X0, X1, X2
    VCVTPS2PH $0, X2, X2
    VMOVD X2, AX
    MOVW AX, (DX)
    
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $2, DX
    DECQ CX
    JNZ tailloop3
    
done3:
    VZEROUPPER
    RET

// func simdMulFloat16AVX2(a, b, c unsafe.Pointer, n int)
// Multiplies two float16 arrays using F16C
TEXT ·simdMulFloat16AVX2(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), SI    // source A pointer
    MOVQ b+8(FP), DI    // source B pointer
    MOVQ c+16(FP), DX   // destination pointer
    MOVQ n+24(FP), CX   // count
    
    // Process 8 elements at a time
    MOVQ CX, AX
    SHRQ $3, AX         // AX = n / 8
    JZ tail4            // If less than 8 elements, go to tail
    
loop8_4:
    VMOVDQU (SI), X0          // Load 8 float16 from A
    VMOVDQU (DI), X1          // Load 8 float16 from B
    
    VCVTPH2PS X0, Y0          // Convert A to float32
    VCVTPH2PS X1, Y1          // Convert B to float32
    
    VMULPS Y0, Y1, Y2         // Multiply float32 values
    
    VCVTPS2PH $0, Y2, X2      // Convert result back to float16
    VMOVDQU X2, (DX)          // Store result
    
    ADDQ $16, SI              // Advance pointers by 16 bytes
    ADDQ $16, DI
    ADDQ $16, DX
    DECQ AX
    JNZ loop8_4
    
tail4:
    ANDQ $7, CX               // CX = n % 8
    JZ done4
    
tailloop4:
    MOVW (SI), AX             // Load A[i]
    MOVW (DI), BX             // Load B[i]
    
    // Convert and multiply using scalar F16C
    MOVW AX, -4(SP)
    MOVW BX, -2(SP)
    VCVTPH2PS -4(SP), X0
    VCVTPH2PS -2(SP), X1
    VMULSS X0, X1, X2
    VCVTPS2PH $0, X2, X2
    VMOVD X2, AX
    MOVW AX, (DX)
    
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $2, DX
    DECQ CX
    JNZ tailloop4
    
done4:
    VZEROUPPER
    RET

// func simdFMAFloat16AVX2(a, b, c, d unsafe.Pointer, n int)
// Performs d = a*b + c using F16C and FMA
TEXT ·simdFMAFloat16AVX2(SB), NOSPLIT, $0-40
    MOVQ a+0(FP), SI    // source A pointer
    MOVQ b+8(FP), DI    // source B pointer
    MOVQ c+16(FP), R8   // source C pointer
    MOVQ d+24(FP), DX   // destination pointer
    MOVQ n+32(FP), CX   // count
    
    // Process 8 elements at a time
    MOVQ CX, AX
    SHRQ $3, AX         // AX = n / 8
    JZ tail5            // If less than 8 elements, go to tail
    
loop8_5:
    VMOVDQU (SI), X0          // Load 8 float16 from A
    VMOVDQU (DI), X1          // Load 8 float16 from B
    VMOVDQU (R8), X2          // Load 8 float16 from C
    
    VCVTPH2PS X0, Y0          // Convert A to float32
    VCVTPH2PS X1, Y1          // Convert B to float32
    VCVTPH2PS X2, Y2          // Convert C to float32
    
    VFMADD231PS Y1, Y0, Y2    // Y2 = Y0*Y1 + Y2
    
    VCVTPS2PH $0, Y2, X2      // Convert result back to float16
    VMOVDQU X2, (DX)          // Store result
    
    ADDQ $16, SI              // Advance pointers by 16 bytes
    ADDQ $16, DI
    ADDQ $16, R8
    ADDQ $16, DX
    DECQ AX
    JNZ loop8_5
    
tail5:
    ANDQ $7, CX               // CX = n % 8
    JZ done5
    
tailloop5:
    MOVW (SI), AX             // Load A[i]
    MOVW (DI), BX             // Load B[i]
    MOVW (R8), R9             // Load C[i]
    
    // Convert and FMA using scalar F16C
    MOVW AX, -6(SP)
    MOVW BX, -4(SP)
    MOVW R9, -2(SP)
    VCVTPH2PS -6(SP), X0
    VCVTPH2PS -4(SP), X1
    VCVTPH2PS -2(SP), X2
    VFMADD231SS X1, X0, X2    // X2 = X0*X1 + X2
    VCVTPS2PH $0, X2, X2
    VMOVD X2, AX
    MOVW AX, (DX)
    
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $2, R8
    ADDQ $2, DX
    DECQ CX
    JNZ tailloop5
    
done5:
    VZEROUPPER
    RET
