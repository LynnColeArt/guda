# ARM64 Floating-Point Tolerance Guide

## Overview

This guide addresses floating-point tolerance issues on ARM64 architecture, particularly for contributors implementing NEON SIMD optimizations.

## Why ARM64 Needs Different Tolerances

ARM64 NEON and x86 AVX/SSE have subtle differences in floating-point behavior:

### 1. **FMA (Fused Multiply-Add) Differences**
- x86: `a * b + c` may round twice (after multiply, after add)
- ARM64 NEON: `FMLA` instruction rounds only once
- Result: Up to 1 ULP difference per operation

### 2. **Reduction Order**
- x86 typically uses sequential reduction
- ARM64 NEON often uses pairwise or tree reduction
- Different accumulation orders = different rounding

### 3. **Denormal Handling**
- ARM64 may flush denormals to zero more aggressively
- Can cause larger differences near underflow

### 4. **Compiler Optimizations**
- Different compilers may reorder operations differently
- `-ffast-math` has architecture-specific effects

## Current Tolerance Settings

### Default (x86/AMD64)
```go
GEMM: AbsTol=1e-6, RelTol=1e-5, ULPTol=4
```

### ARM64 Relaxed
```go
GEMM: AbsTol=1e-5, RelTol=1e-4, ULPTol=16
```

## Implementation Guidelines

### For GEMM Operations
```go
// Use architecture-aware tolerance
tol := GetOperationTolerance("gemm")
result := VerifyFloat32Array(expected, actual, tol)
```

### For Reduction Operations
```go
// Reductions accumulate more error
tol := GetOperationTolerance("reduce_sum")
```

## Testing Your ARM64 Implementation

1. **Run Architecture-Specific Tests**
```bash
go test -run TestGEMMWithArchTolerance -v
```

2. **Check Tolerance Usage**
```bash
go test -run TestArchSpecificTolerance -v
```

3. **Compare with Reference**
```bash
go test -run BenchmarkGEMMArchComparison -bench=. -v
```

## Common Issues and Solutions

### Issue: "FP difference is too high"
**Solution**: The code now uses relaxed tolerances for ARM64 automatically.

### Issue: Tests pass on x86 but fail on ARM64
**Solution**: Use `GetOperationTolerance()` instead of hardcoded tolerances.

### Issue: Inconsistent results between runs
**Possible causes**:
- Uninitialized memory
- Race conditions in parallel code
- Compiler optimization differences

## ARM64 NEON Best Practices

1. **Use FMA instructions when available**
   - More accurate (single rounding)
   - Better performance

2. **Be consistent with reduction patterns**
   - Document your reduction order
   - Use the same pattern as reference when possible

3. **Handle denormals explicitly**
   - Check CPU flush-to-zero settings
   - Consider using threshold checks

4. **Test with various data**
   - Small values (denormal range)
   - Large values (overflow range)  
   - Mixed magnitudes

## Debugging Tolerance Issues

```go
// Enable detailed comparison
result := VerifyFloat32Array(expected, actual, tol)
if result.NumErrors > 0 {
    fmt.Printf("Max abs error: %e\n", result.MaxAbsError)
    fmt.Printf("Max rel error: %e\n", result.MaxRelError)
    fmt.Printf("Max ULP error: %d\n", result.MaxULPError)
    fmt.Printf("First error at: %d\n", result.FirstError)
}
```

## Future Work

- [ ] Implement ARM64 NEON optimized kernels
- [ ] Add SVE/SVE2 support for newer ARM64
- [ ] Profile and optimize for specific ARM64 chips
- [ ] Create ARM64-specific benchmarks

## References

- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Floating-Point Arithmetic on ARM](https://developer.arm.com/documentation/100960/0100/Floating-point-arithmetic)
- [NEON Programmer's Guide](https://developer.arm.com/documentation/den0018/a/)

---

**Note to @vybecoder**: The tolerance system has been updated to automatically detect ARM64 and use appropriate relaxed tolerances. Your ARM64 implementations should now pass tests without manual tolerance adjustments.