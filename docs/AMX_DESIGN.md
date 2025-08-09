# Intel AMX (Advanced Matrix Extensions) Integration Design

## Overview

Intel AMX is a game-changer for AI inference on CPU. It provides dedicated matrix multiplication units that can achieve up to 2 TOPS (trillion operations per second) for INT8 operations.

## What is AMX?

AMX introduces:
- **Tile Registers**: 8 tile registers, each up to 1KB (e.g., 16×64 bytes)
- **Matrix Operations**: Hardware-accelerated tile matrix multiply
- **Data Types**: INT8, BF16, and soon FP16
- **Peak Performance**: ~2 TOPS INT8, ~1 TFLOPS BF16

## Why AMX for GUDA?

1. **AI Inference**: Perfect for quantized neural networks
2. **Massive Speedup**: 8-16x faster than AVX-512 for INT8
3. **Power Efficient**: Better TOPS/watt than SIMD
4. **FFU Integration**: Natural fit for our framework

## AMX Programming Model

```
1. Configure tiles (TILECONFIG)
2. Load data into tiles (TILELOADD)
3. Compute (TDPBSSD for INT8)
4. Store results (TILESTORED)
5. Release tiles (TILERELEASE)
```

## Design for GUDA

### 1. AMX FFU Implementation

```go
type AMXFFU struct {
    available bool
    tileConfig TileConfig
    metrics atomic.Value
}

type AMXWorkload struct {
    Operation string // "matmul_int8", "matmul_bf16"
    M, N, K   int
    A, B, C   []byte // For INT8
    Alpha     int32  // Scaling factor
}
```

### 2. Tile Configuration Strategy

For INT8 GEMM, optimal tile usage:
- Tile 0-1: A matrix (16×64 bytes = 16×64 INT8)
- Tile 2-3: B matrix (64×16 bytes = 64×16 INT8)
- Tile 4-5: C accumulator (16×16 INT32)

This gives us 16×16 output per AMX operation.

### 3. Kernel Design

```asm
// Pseudo-code for AMX INT8 GEMM kernel
amx_gemm_16x16:
    TILECONFIG      // Configure 16x64, 64x16, 16x16 tiles
    
    // Main loop over K dimension
    mov rcx, K/64   // Process 64 K elements at a time
loop_k:
    TILELOADD tmm0, [A]      // Load 16x64 of A
    TILELOADD tmm2, [B]      // Load 64x16 of B
    TDPBSSD   tmm4, tmm0, tmm2  // C += A * B (INT8->INT32)
    
    add A, 64*16    // Advance A by 64 columns
    add B, 64*16    // Advance B by 64 rows
    dec rcx
    jnz loop_k
    
    TILESTORED [C], tmm4     // Store 16x16 result
    TILERELEASE
```

### 4. Integration Points

- **Detection**: CPUID check for AMX-TILE, AMX-INT8, AMX-BF16
- **Workload Routing**: Route quantized GEMM to AMX
- **Memory Layout**: Pack matrices for tile-friendly access
- **Scaling**: Post-process INT32 results to INT8/FP32

## Performance Expectations

| Operation | Size | AVX-512 VNNI | AMX | Speedup |
|-----------|------|--------------|-----|---------|
| INT8 GEMM | 256×256×256 | 125 GOPS | 1.8 TOPS | 14x |
| INT8 GEMM | 1024×1024×1024 | 140 GOPS | 2.0 TOPS | 14x |
| BF16 GEMM | 256×256×256 | 70 GFLOPS | 0.9 TFLOPS | 13x |

## Implementation Plan

### Phase 1: Basic AMX FFU (Today)
- [ ] AMX detection via CPUID
- [ ] Basic tile configuration
- [ ] Simple INT8 matmul kernel
- [ ] Correctness tests

### Phase 2: Optimization (Tomorrow)
- [ ] Packing routines for optimal layout
- [ ] Multi-tile kernel for larger matrices
- [ ] BF16 support
- [ ] Performance benchmarks

### Phase 3: Integration (Next)
- [ ] Quantization helpers (FP32→INT8)
- [ ] Dequantization (INT32→FP32)
- [ ] Scale/bias fusion
- [ ] Real AI workload tests

## Challenges

1. **Tile Register Pressure**: Only 8 tiles, need careful management
2. **Data Layout**: Matrices must be packed correctly
3. **Kernel Size**: 16×16 minimum, need blocking for larger
4. **Context Switching**: OS must save/restore tile state

## Success Criteria

- [ ] Detect AMX on Sapphire Rapids and newer
- [ ] 10x speedup over AVX-512 for INT8 GEMM
- [ ] Seamless integration with FFU framework
- [ ] Support common AI inference patterns

## Code Structure

```
ffu/
└── amx/
    ├── amx.go          # AMX FFU implementation
    ├── detect_amd64.go # CPUID detection
    ├── tile_config.go  # Tile configuration
    ├── pack.go         # Matrix packing routines
    ├── amx_amd64.s     # Assembly kernels
    └── amx_test.go     # Tests and benchmarks
```

## The Payoff

With AMX integrated:
- Quantized transformer inference at 100+ tokens/sec on CPU
- INT8 GEMM at 2 TOPS (vs 140 GOPS with AVX-512)
- Another proof point for heterogeneous compute
- Path to on-CPU AI without GPU

This is the future of CPU AI acceleration!