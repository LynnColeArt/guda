# GUDA Assimilation Status & Tasks

## What We've Accomplished

### 1. Performance Optimizations ✅
- Switched to native Float32 operations (20x speedup on AXPY/DOT)
- Identified Float32 is severely under-optimized in gonum
- Created comprehensive assembly audit

### 2. Gonum Assimilation ✅
- Copied BLAS implementation into `compute/`
- Copied assembly kernels (f32/f64)
- Removed complex number support (not needed for ML)
- Updated all imports to use local compute engine
- Tests passing with assimilated code

### 3. Bug Fixes ✅
- Fixed critical gonum matrix multiplication bug (20-column shift)
- Fixed AddBiasReLU broadcasting issue
- Achieved numerical correctness

## Outstanding Tasks

### High Priority - Performance
1. **Add Float32 AVX2 Assembly**
   - Port axpyunitary_avx2 from Float64 to Float32
   - Port dotunitary_avx2 from Float64 to Float32
   - Create Float32 GEMM kernels (8x8 and 4x4)
   - Add FMA instructions throughout

2. **ML-Specific Fused Operations**
   - AddBiasGELU (critical for transformers)
   - LayerNorm (with stable computation)
   - Softmax with numerical stability
   - RMSNorm (for modern models)

3. **Small Matrix GEMM Optimization**
   - Special kernels for common sizes (128, 256, 512)
   - Cache-aware tiling
   - Minimize overhead for small matrices

### Medium Priority - Functionality
4. **BLAS Level 2 Operations**
   - GEMV (matrix-vector multiply)
   - GER (rank-1 update)
   - Essential for RNNs and attention

5. **Reduction Patterns**
   - Sum (with AVX2 horizontal adds)
   - Max/ArgMax (for classifications)
   - Mean/Variance (for normalization)

6. **Convolution Operations**
   - Im2col approach initially
   - Direct convolution for common kernels
   - Depthwise separable support

### Low Priority - Infrastructure
7. **Error Handling**
   - Better error messages
   - Graceful degradation
   - Debug modes

8. **Profiling Tools**
   - Kernel timing
   - Memory bandwidth tracking
   - Operation counting

9. **ML Inference Example**
   - Load a small model
   - Run inference
   - Benchmark vs PyTorch

10. **Comparison Benchmarks**
    - vs NumPy
    - vs OpenBLAS
    - vs MKL-DNN

## Git History Preservation Plan

### Current Structure:
```
Guda/
├── .git/          (GUDA history)
├── gonum/
│   └── .git/      (gonum history - preserve this!)
└── compute/       (assimilated code)
```

### Proposed Approach:
1. Move gonum/.git to preserve history
2. Create unified repository with full history
3. Tag the assimilation point
4. Continue development with full context

This preserves:
- All gonum optimization history
- Our bug fixes and improvements
- The "before/after" assimilation point
- Full attribution and context