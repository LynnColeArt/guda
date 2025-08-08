# Defensive Implementation Strategy for Fused Operations

## Lessons Learned from Previous Battles
1. **Dependencies lie** - That gonum Float64 conversion was hidden deep
2. **Silent corruption** - The 20-column offset only showed up in specific sizes
3. **Assembly is treacherous** - One wrong label or missing newline = chaos
4. **Performance fixes can break correctness** - Always verify numerics

## Implementation Strategy: Trust Nothing

### Phase 1: Build Incremental Verification
```go
// Start with babysteps that we can verify at each stage
func AddBiasReLU_Safe(x []float32, bias []float32) {
    // Stage 1: Just add bias (verify against naive)
    AddBias_Verified(x, bias)
    
    // Stage 2: Just ReLU (verify against naive)
    ReLU_Verified(x)
}

func AddBias_Verified(x []float32, bias []float32) {
    // Compute both ways
    result_optimized := AddBias_AVX2(x, bias)
    result_naive := AddBias_Naive(x, bias)
    
    // Check EVERY element
    for i := range result_optimized {
        if math.Abs(result_optimized[i] - result_naive[i]) > 1e-6 {
            panic(fmt.Sprintf("AddBias mismatch at %d: opt=%f, naive=%f", 
                i, result_optimized[i], result_naive[i]))
        }
    }
}
```

### Phase 2: Paranoid Testing Infrastructure
```go
// Test with pathological inputs designed to break things
func TestFusedOps_EdgeCases(t *testing.T) {
    testCases := []struct{
        name string
        size int
        bias float32
    }{
        // Sizes that broke gonum
        {"Matrix20ColumnBug", 20, 1.0},
        {"Matrix40ColumnBug", 40, 1.0},
        
        // Cache line boundaries  
        {"CacheLineMinusOne", 15, 1.0},  // 16 floats per cache line
        {"CacheLineExact", 16, 1.0},
        {"CacheLinePlusOne", 17, 1.0},
        
        // SIMD register boundaries
        {"AVX2MinusOne", 7, 1.0},   // 8 floats per YMM
        {"AVX2Exact", 8, 1.0},
        {"AVX2PlusOne", 9, 1.0},
        
        // Numerical edge cases
        {"Denormals", 100, 1e-38},
        {"NearOverflow", 100, 3.4e38},
        {"NegativeZero", 100, -0.0},
        {"NaN", 100, float32(math.NaN())},
    }
    
    for _, tc := range testCases {
        t.Run(tc.name, func(t *testing.T) {
            // Test exhaustively
        })
    }
}
```

### Phase 3: Canary Implementations
```go
// Before going full fusion, create "canary" versions
// that are SLIGHTLY optimized to catch issues early

// Level 0: Completely naive (known correct)
func TransformerLayer_Naive(x []float32) []float32

// Level 1: Just fuse bias+activation (small change)
func TransformerLayer_FusedActivation(x []float32) []float32

// Level 2: Fuse one GEMM+bias+act (medium change)  
func TransformerLayer_FusedGEMM(x []float32) []float32

// Level 3: Full fusion (big change)
func TransformerLayer_FullyFused(x []float32) []float32

// Test each level against previous
```

### Phase 4: Defensive Assembly
```asm
; Add guards and checks in assembly
CHECK_ALIGNMENT:
    MOV RAX, RDI
    AND RAX, 31          ; Check 32-byte alignment
    JNZ unaligned_path   ; Handle gracefully
    
CHECK_SIZE:
    CMP RCX, 0
    JE  early_exit       ; Don't crash on zero size
    
; Add canary values to detect overwrites
STORE_CANARY:
    MOV QWORD [RSP-8], 0xDEADBEEFDEADBEEF
    ; ... do work ...
CHECK_CANARY:
    CMP QWORD [RSP-8], 0xDEADBEEFDEADBEEF
    JNE stack_corruption_detected
```

### Phase 5: Bisection-Friendly Design
```go
// When things break, we need to bisect quickly
type FusionLevel int

const (
    NoFusion FusionLevel = iota
    BiasOnly
    BiasReLU  
    GEMMBias
    GEMMBiasReLU
    FullFusion
)

func TransformerLayer(x []float32, level FusionLevel) []float32 {
    switch level {
    case NoFusion:
        return transformerNaive(x)
    case BiasOnly:
        return transformerBiasOptimized(x)
    // ... etc
    }
}

// Now we can binary search when things break!
```

### Phase 6: Continuous Validation
```go
// In production, randomly validate results
var validateCounter uint32

func FusedOperation(x []float32) []float32 {
    // Every 1000th call, verify against naive
    if atomic.AddUint32(&validateCounter, 1) % 1000 == 0 {
        result_fused := fusedImpl(x)
        result_naive := naiveImpl(x)
        
        if !numericallyClose(result_fused, result_naive) {
            log.Printf("DIVERGENCE DETECTED: Fused ops drifting!")
            // Fall back to safe implementation
            return result_naive
        }
        return result_fused
    }
    
    return fusedImpl(x)
}
```

## The Debugging Toolkit

When (not if) things go wrong:

1. **Numerical diff tool**
   ```bash
   ./guda_debug --compare naive fused --input test.bin
   ```

2. **Assembly trace mode**
   ```go
   //go:build debug
   func tracedAVX2Operation() {
       // Logs every SIMD operation
   }
   ```

3. **Cache simulation**
   ```go
   func SimulateCacheAccess(addresses []uintptr) {
       // Verify our cache blocking assumptions
   }
   ```

4. **Automatic bisection**
   ```bash
   ./guda_bisect --good v1.0 --bad HEAD --test numerical_accuracy
   ```

## Remember: The Bug is Usually Where You're NOT Looking

- Fused ops expose alignment bugs in memory allocation
- Cache blocking reveals stride bugs in matrix storage  
- New code paths trigger old bugs in "stable" code
- Optimizations unmask undefined behavior

Stay paranoid, stay safe! 🛡️