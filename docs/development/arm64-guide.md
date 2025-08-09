# ARM64 Development Guide

This guide provides information for developers working on ARM64-specific features or optimizations in GUDA.

## Architecture Overview

GUDA's ARM64 support is built around NEON SIMD instructions, providing significant performance improvements for vector operations. The implementation automatically detects CPU features and selects the appropriate code paths.

## Key Files

- `cpu_arm64.go` - CPU feature detection
- `float16_simd_arm64.go` - Go interface to SIMD operations
- `float16_arm64.s` - Assembly implementation of SIMD operations
- `*_arm64_test.go` - ARM64-specific tests

## Testing ARM64 Features

To run ARM64-specific tests:

```bash
# Run all ARM64 tests
go test -v ./... -run "*ARM64*"

# Run integration tests
go test -v ./... -run "TestIntegration"

# Run SIMD tests
go test -v ./... -run "TestSimd"
```

## DevicePtr Considerations

When working with DevicePtr in ARM64 context:

1. Always check for nil pointers before accessing memory
2. Validate data length matches expected operation size
3. Handle edge cases where grid dimensions might be zero

## Numeric Tolerances

ARM64 processors might produce slightly different floating-point results compared to x86-64. When adding new numerical tests:

1. Use `NumericalParity` for comparing results
2. Set appropriate tolerances for ARM64 differences
3. Consider both absolute and relative error thresholds
4. Account for ULP (Units in Last Place) differences in floating-point comparisons

Example:
```go
tol := StandardTolerances["operation_name"]
if isARM64() {
    // Adjust tolerances for ARM64
    tol.AbsTol = adjustedValue
    tol.RelTol = adjustedValue
    tol.ULPTol = adjustedValue
}
if !parity.CheckTolerance(tol) {
    // Handle tolerance failure
}
```

For detailed information about numerical precision considerations in ARM64, see [ARM64 Numerical Precision Documentation](arm64-numerical-precision.md).

## Performance Profiling

To profile ARM64 performance:

```bash
# Run benchmarks
go test -bench=Benchmark*ARM64* -benchmem

# Profile specific functions
go test -bench=Benchmark*ARM64* -cpuprofile=cpu.prof
go tool pprof cpu.prof

# Memory profiling
go test -bench=Benchmark*ARM64* -memprofile=mem.prof
go tool pprof mem.prof
```

## Debugging Tips

1. Use test logging to understand numerical differences
2. Compare results with reference implementations
3. Check for different ULP behaviors between architectures
4. Verify assembly code is properly optimized for ARM64

## Contributing ARM64 Features

When adding new ARM64 features:

1. Implement core functionality in Go first with scalar fallbacks
2. Add optimized assembly implementations for performance-critical paths
3. Create comprehensive tests covering edge cases
4. Document new functions in README_ARM64.md
5. Verify compatibility with existing GUDA APIs