# Gonum Assembly Optimization Audit

## Overview
This audit examines gonum's x86/x64 assembly implementations to identify optimization opportunities.

## Key Findings Summary
1. Float32 operations have less SIMD optimization than Float64
2. Many operations lack AVX2/FMA variants
3. Some assembly appears to be hand-rolled without modern instructions
4. Float64 has AVX2/FMA versions (2024), Float32 does not

## Float32 vs Float64 Comparison

### AXPY Operation (y = alpha*x + y)

#### Float32 (axpyunitary_amd64.s)
- **SIMD Level**: SSE only (128-bit, 4 floats)
- **Instructions**: MULPS + ADDPS (no FMA)
- **Unrolling**: 16x (64 floats per iteration)
- **Alignment**: 16-byte alignment handling
- **No AVX2 version exists**

#### Float64 (axpyunitary_amd64.s + axpyunitary_avx2_amd64.s)
- **SSE2 version**: MULPD + ADDPD (128-bit, 2 doubles)
- **AVX2 version**: VMULPD + VADDPD (256-bit, 4 doubles)
- **Unrolling**: 16x in AVX2 (64 doubles per iteration)
- **Alignment**: 32-byte alignment for AVX2
- **Has modern AVX2 implementation (2024)**

### DOT Product

#### Float32 (dotunitary_amd64.s)
- **SIMD Level**: SSE only
- **Instructions**: MULPS for multiply, ADDPS for accumulation
- **Unrolling**: 16x with 2 accumulators
- **Horizontal sum**: Uses HADDPS (slower instruction)
- **No AVX2 version**

## Optimization Opportunities

### 1. Add AVX2 Support for Float32
- Float64 has AVX2 versions, Float32 doesn't
- Would double throughput (4→8 floats per instruction)
- Example improvement for AXPY:
  ```asm
  ; Current SSE (4 floats)
  MOVUPS (SI)(AX*4), X2
  MULPS  X0, X2
  ADDPS  (DI)(AX*4), X2
  
  ; Proposed AVX2 (8 floats)
  VMOVUPS (SI)(AX*4), Y2
  VMULPS  Y0, Y2, Y2    ; or VFMADD231PS for FMA
  VADDPS  (DI)(AX*4), Y2, Y2
  ```

### 2. Use FMA Instructions
- Both Float32 and Float64 could benefit from FMA
- Reduces latency and increases throughput
- Example:
  ```asm
  ; Current (2 instructions)
  VMULPS Y0, Y2, Y2
  VADDPS (DI)(AX*4), Y2, Y2
  
  ; With FMA (1 instruction)
  VFMADD231PS (DI)(AX*4), Y0, Y2
  ```

### 3. Better DOT Product Reduction
- Current uses HADDPS which has high latency
- Better approach:
  ```asm
  ; Vertical sum first
  VADDPS Y2, Y3, Y2
  VADDPS Y4, Y5, Y4
  VADDPS Y2, Y4, Y2
  ; Then horizontal
  VEXTRACTF128 $1, Y2, X3
  VADDPS X3, X2, X2
  ; Final reduction
  ```

### 4. Alignment Optimization
- Current code does scalar operations for alignment
- Could use masked loads/stores (AVX-512) or unaligned loads
- Modern CPUs handle unaligned loads well

### 5. Missing Float32 Operations
Operations that have Float64 ASM but not Float32:
- Scale operations (only has Dscal, no Sscal assembly)
- Many Level 2 BLAS operations
- Norm calculations
- **Statistics**: Float64 has 33 ASM files, Float32 only has 10

### 6. GEMM Kernel Analysis
- Float64 has AVX2/FMA optimized 4x4 and 8x8 kernels (2024)
- Float32 has NO assembly GEMM kernels at all
- This explains why Float32 GEMM might be slower

## File Count Comparison
- **Float32**: 10 assembly files
- **Float64**: 33 assembly files (3.3x more!)

Missing Float32 assembly implementations:
- gemm_kernel (critical for matrix multiply)
- cumsum/cumprod
- l1norm/l2norm
- div operations
- many more...

## Recommendations

### Priority 1: Add Float32 AVX2 GEMM Kernels
This would have the biggest impact on ML workloads:
```asm
// 8x8 Float32 kernel with AVX2/FMA
TEXT ·sgemm_kernel_8x8_avx2(SB), NOSPLIT, $0
    // Process 8x8 floats per iteration
    // Use YMM registers (8 floats each)
    // Use VFMADD231PS for multiply-add
```

### Priority 2: Port existing Float64 AVX2 code to Float32
Many Float64 operations have AVX2 versions that could be adapted:
- axpyunitary_avx2 → saxpyunitary_avx2
- dotunitary_avx2 → sdotunitary_avx2
- scalunitary_avx2 → sscalunitary_avx2

### Priority 3: Optimize Reduction Operations
Current Float32 dot product uses HADDPS (high latency).
Better approach with AVX2:
```asm
// Accumulate in 8 YMM registers
VMULPS (SI), Y0, Y8
VMULPS 32(SI), Y1, Y9
...
// Tree reduction
VADDPS Y8, Y9, Y8
VADDPS Y10, Y11, Y10
VADDPS Y8, Y10, Y8
...
```

## Potential Performance Gains
Based on the improvements we saw with Float32 BLAS operations:
- **AXPY**: Could go from 8-32 GFLOPS → 50-80 GFLOPS with AVX2
- **DOT**: Could go from 28-30 GFLOPS → 60-80 GFLOPS
- **GEMM**: Could see 30-50% improvement with proper kernels

## Implementation Strategy
1. Start with high-impact operations (GEMM, AXPY, DOT)
2. Use Go's generate tool to create Float32 versions from Float64
3. Test thoroughly with different alignments and sizes
4. Benchmark against current implementation