# AVX-512 Implementation Guide for GUDA

## Overview

This document captures the detailed implementation plan for AVX-512 support in GUDA, based on Mini's expert guidance. It serves as the authoritative reference for implementing high-performance AVX-512 kernels.

## Target and Constraints

### Hardware Considerations
- **AVX-512 on Zen 4 (AMD 7700X)**: Executes via dual 256-bit pipes (no frequency cliff like old Intel)
- **Throughput target**: 16 floats per vector × 2 (FMA) = 32 FLOPs per zmm FMA
- **Intel server (Ice/Sapphire Rapids)**: Watch for AVX-512 downclock

### Performance Goals
- **Per-core theoretical**: ~288-432 GFLOPS (4.5 GHz × ~64-96 FLOPs/cycle)
- **8-core sustained target**: ~230-300 GFLOPS (50-70% efficiency)
- **Expected improvement**: 1.2-1.4× over AVX2 on Zen 4

## Microkernel Design: sgemm_16×4_avx512

### Tile Shape
- **mr = 16** (rows of C, matches zmm register width)
- **nr = 4** (columns of C)
- Uses 4 ZMM accumulators (zmm0-zmm3) for the full 16×4 C tile

### Register Allocation
```
Accumulators (C tile):
- zmm0: C[0:16, j+0]
- zmm1: C[0:16, j+1]
- zmm2: C[0:16, j+2]
- zmm3: C[0:16, j+3]

Streaming operands:
- zmm8:  B[k, j+0:3] (broadcasts)
- zmm9:  A[0:16, k]
- zmm10: B[k+1, j+0:3] (unroll×2)
- zmm11: A[0:16, k+1]

General purpose:
- rdi: A_pack pointer
- rsi: B_pack pointer
- rdx: C base pointer
- rcx: ldc (in bytes)
- r8:  KC counter
- r9:  C row stride scratch
- r10: prefetch address scratch

Masks:
- k1: tail-M mask (if M<16)
- k2: tail-N mask (if N<4)
```

## Memory Layouts

### A Packing (mr×KC)
```c
// Pack A into 16-row blocks, contiguous in memory
for (k = 0; k < KC; k++) {
    for (i = 0; i < 16; i++) {
        A_pack[k*16 + i] = A[row+i][K0+k];
    }
}
// Each zmm load gets A[i:i+16, k] in one instruction
```

### B Packing (KC×nr)
```c
// Pack B into column panels
for (j = 0; j < 4; j++) {
    for (k = 0; k < KC; k++) {
        B_pack[j*KC + k] = B[K0+k][col+j];
    }
}
// Allows vbroadcastss from each column stream
```

## Inner Loop Implementation

### Kernel Prologue
```asm
; Inputs: rdi=A_pack, rsi=B_pack, rdx=C, rcx=ldc_bytes, r8=KC
; Zero accumulators
vxorps zmm0, zmm0, zmm0
vxorps zmm1, zmm1, zmm1
vxorps zmm2, zmm2, zmm2
vxorps zmm3, zmm3, zmm3
```

### Inner K-Loop (Unroll×2)
```asm
.K_loop:
    ; ----- k step -----
    vmovaps     zmm9,  [rdi]                    ; A[0:16,k]
    vbroadcastss zmm8, dword [rsi + 0*KC*4]     ; B[k,j+0]
    vfmadd231ps zmm0, zmm9, zmm8
    
    vbroadcastss zmm8, dword [rsi + 1*KC*4]     ; B[k,j+1]
    vfmadd231ps zmm1, zmm9, zmm8
    
    vbroadcastss zmm8, dword [rsi + 2*KC*4]     ; B[k,j+2]
    vfmadd231ps zmm2, zmm9, zmm8
    
    vbroadcastss zmm8, dword [rsi + 3*KC*4]     ; B[k,j+3]
    vfmadd231ps zmm3, zmm9, zmm8
    
    ; Prefetch next iteration
    prefetcht0  [rdi + 128]      ; A: 2 iterations ahead
    prefetcht0  [rsi + 8]        ; B column 0
    prefetcht0  [rsi + KC*4 + 8] ; B column 1
    prefetcht0  [rsi + 2*KC*4 + 8]
    prefetcht0  [rsi + 3*KC*4 + 8]
    
    add  rdi, 64    ; Next A column (16 floats)
    add  rsi, 4     ; Next k for all B columns
    dec  r8
    jz   .K_done
    
    ; ----- k+1 step (unrolled) -----
    vmovaps     zmm11, [rdi]
    vbroadcastss zmm10, dword [rsi + 0*KC*4]
    vfmadd231ps zmm0, zmm11, zmm10
    vbroadcastss zmm10, dword [rsi + 1*KC*4]
    vfmadd231ps zmm1, zmm11, zmm10
    vbroadcastss zmm10, dword [rsi + 2*KC*4]
    vfmadd231ps zmm2, zmm11, zmm10
    vbroadcastss zmm10, dword [rsi + 3*KC*4]
    vfmadd231ps zmm3, zmm11, zmm10
    
    ; Prefetch
    prefetcht0  [rdi + 128]
    prefetcht0  [rsi + 8]
    prefetcht0  [rsi + KC*4 + 8]
    prefetcht0  [rsi + 2*KC*4 + 8]
    prefetcht0  [rsi + 3*KC*4 + 8]
    
    add  rdi, 64
    add  rsi, 4
    dec  r8
    jnz  .K_loop
.K_done:
```

### Epilogue (Store C)
```asm
; Store full 16×4 tile (if no tails)
vmovups [rdx + 0*rcx], zmm0
vmovups [rdx + 1*rcx], zmm1
vmovups [rdx + 2*rcx], zmm2
vmovups [rdx + 3*rcx], zmm3
```

## Blocking Strategy

### Cache Blocking Parameters
- **KC**: 256-384 (fits A block + B panel in L2)
- **MC**: 128-256 (rows per core, A tiles stay in L2)
- **NC**: 256-512 (B panel fits L3, reused across cores)

### Outer Loop Structure
```
for (J = 0; J < N; J += NC) {
    for (K = 0; K < K; K += KC) {
        pack_B(KC × nr panels for J:J+NC)
        for (I = 0; I < M; I += MC) {
            pack_A(mr × KC panels for I:I+MC)
            for (j = J; j < J+NC; j += nr) {
                for (i = I; i < I+MC; i += mr) {
                    sgemm_16x4_avx512(A_pack[i], B_pack[j], &C[i,j], KC, ldc)
                }
            }
        }
    }
}
```

## Tail Handling

### M Tails (rows < 16)
- Use AVX-512 mask register k1
- `kmovw k1, (1<<M)-1` to create mask
- Use masked loads/stores: `vmovups zmm{k1}{z}`

### N Tails (columns < 4)
- Dispatch to separate kernels:
  - 16×2 kernel for N%4 == 2
  - 16×1 kernel for N%4 == 1
  - 16×3 could use main kernel with k2 mask

## Performance Optimization Notes

### Prefetch Tuning
- **PF_A**: Start with 128 bytes (2 iterations × 64 bytes)
- **PF_B**: Start with 8 bytes (2 iterations × 4 bytes)
- Adjust based on L2 latency (~12-14 cycles on Zen 4)

### Alignment
- Pack buffers 64-byte aligned
- Consider padding KC to multiple of 16 for better alignment

### Scheduling Considerations
- FMA latency: 4 cycles on Zen 4
- Load-to-use latency: 4-5 cycles
- Interleave loads and FMAs to hide latency
- 4 accumulators allow full latency hiding

## 16×2 Variant (for small N)

### Register Map
```
Accumulators:
- zmm0: C[0:16, j+0]
- zmm1: C[0:16, j+1]

Same A loading pattern
B broadcasts from 2 columns only
Better for N tails and small matrices
```

## Validation and Testing

### Correctness
- Compare against AVX2 implementation
- Compare against reference scalar implementation
- Use tolerance-based verification (1e-5 for float32)

### Performance Targets
- Kernel-only (hot, packed): 1.2-1.4× AVX2 performance
- End-to-end: Account for packing overhead
- Monitor with performance counters

### Performance Counters to Track
- Retired FP32 FMAs
- IPC (aim for >3 in hot loop)
- L1D/L2/LLC miss rates
- Load/store ratio
- CPU frequency (check for downclocking)

## Implementation Checklist

1. [ ] Implement sgemm_16x4_avx512 kernel
2. [ ] Implement A/B packing routines
3. [ ] Add CPU feature detection (AVX512F)
4. [ ] Create dispatcher logic
5. [ ] Implement 16×2 variant
6. [ ] Add tail handling
7. [ ] Integrate with existing GEMM framework
8. [ ] Add performance counters
9. [ ] Benchmark and validate
10. [ ] Auto-tune blocking parameters

## Notes from Mini

- Zen 4 doesn't downclock with AVX-512 (unlike Intel)
- The dual 256-bit execution doesn't hurt much due to better ILP
- Focus on kernel first, then optimize packing
- Consider non-temporal stores for very large C matrices
- Keep packing layout consistent across ISA variants