# GUDA Fused Operations Design

Based on CUDA's proven patterns, here's our approach for CPU/SIMD fused operations.

## Core Principle: One Memory Pass
Just like CUDA, we eliminate intermediate memory writes:
```
// Bad (3 memory passes):
GEMM(A, B, C)      // Write C to memory
AddBias(C, bias)   // Read C, write C
ReLU(C)           // Read C, write C

// Good (1 memory pass):
GEMMBiasReLU(A, B, C, bias)  // Everything in registers
```

## Priority Fused Operations

### 1. AddBiasReLU (Most Common)
```asm
; After GEMM tile computation, result in YMM0-YMM3
VBROADCASTSS bias_val, YMM4     ; Broadcast bias to all lanes
VADDPS YMM4, YMM0, YMM0         ; Add bias
VMAXPS YMM_ZERO, YMM0, YMM0     ; ReLU (max with 0)
; Repeat for YMM1-YMM3
```

### 2. AddBiasGELU (Transformers)
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

Approximation for SIMD:
```asm
; Fast GELU approximation using polynomial
; Constants pre-loaded
VMULPS X, X, X2              ; x²
VMULPS X2, X, X3             ; x³
VFMADD213PS X3, CONST_0_044715, X  ; x + 0.044715x³
VMULPS CONST_SQRT_2_PI, X, X       ; sqrt(2/π) * result
; Fast tanh approximation here
; Final multiply by 0.5 * original_x
```

### 3. LayerNorm (Critical for Transformers)
```
LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

Strategy:
- Use Welford's algorithm for stable mean/variance
- Fuse the normalization with scale/shift
- Process in chunks that fit in registers

### 4. AddBiasSiLU/Swish (Modern Networks)
SiLU(x) = x * sigmoid(x)

Can share sigmoid computation infrastructure with GELU.

## Implementation Structure

### Level 1: Basic Fused Ops
```c
// C interface
void sgemm_bias_relu_avx2(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    const float* bias,
    float alpha, float beta
);
```

### Level 2: Configurable Epilogue
```c
typedef enum {
    EPILOGUE_NONE,
    EPILOGUE_RELU,
    EPILOGUE_GELU,
    EPILOGUE_SILU,
    EPILOGUE_TANH
} epilogue_t;

void sgemm_bias_act_avx2(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    const float* bias,
    epilogue_t activation,
    float alpha, float beta
);
```

### Level 3: Graph-Based Fusion (Future)
Like cuDNN's graph API, but simpler:
```go
type FusionGraph struct {
    nodes []OpNode
    edges []DataFlow
}

func (g *FusionGraph) Compile() FusedKernel {
    // Analyze graph and generate optimal fused kernel
}
```

## Memory Access Patterns

### Cache Blocking for Fused Ops
```
MC = 256  // Fits in L2 cache
KC = 256  // Depth of GEMM
NC = 2048 // Multiple of SIMD width

for jc = 0 to N-1 step NC
  for pc = 0 to K-1 step KC
    B_packed = pack_B(B[pc:pc+KC, jc:jc+NC])
    for ic = 0 to M-1 step MC
      A_packed = pack_A(A[ic:ic+MC, pc:pc+KC])
      // Fused micro-kernel here
      gemm_bias_act_kernel(...)
```

### Register Blocking (AVX2)
- 16 YMM registers available
- Use 4x4 or 8x8 register tiles
- Reserve registers for:
  - Accumulation (C tile)
  - A values (broadcast)
  - B values (loaded)
  - Bias (broadcast)
  - Activation constants

## Performance Targets

Based on CUDA's improvements:
- Fused GEMM+Bias+ReLU: 1.2-1.5x faster than separate ops
- Memory bandwidth reduction: 66% (3 passes → 1 pass)
- Cache efficiency: Much better temporal locality

## Testing Strategy

1. **Correctness Tests**
   - Compare against unfused operations
   - Test edge cases (negative values for ReLU, large values for GELU)
   - Verify numerical stability

2. **Performance Tests**
   - Measure vs separate operations
   - Test various matrix sizes
   - Profile cache misses and bandwidth

3. **Integration Tests**
   - Full neural network layer forward pass
   - Transformer block computation
   - End-to-end inference

## Next Steps

1. Implement AddBiasReLU first (simplest, most common)
2. Add GELU approximation for transformers
3. Implement LayerNorm with stable numerics
4. Create benchmarks showing memory bandwidth savings
5. Consider AVX-512 variants for newer CPUs