# GUDA Development Tasks

## 🔥 Current Sprint: Maximum Performance

### In Progress
- [ ] Set up unified git history

### Ready to Start

#### Float32 AVX2 Assembly (1-2 days each)
- [ ] Port axpyunitary_avx2.s from Float64 to Float32
- [ ] Port dotunitary_avx2.s from Float64 to Float32  
- [ ] Create sgemm_kernel_8x8_avx2.s
- [ ] Create sgemm_kernel_4x4_avx2.s
- [ ] Add FMA variants of all operations

#### ML Fused Operations (2-3 days)
- [ ] AddBiasGELU - Critical for GPT-style models
- [ ] LayerNorm - With Welford's algorithm for stability
- [ ] RMSNorm - Simpler, faster normalization
- [ ] SoftmaxCrossEntropy - Fused for training

#### Benchmarking Suite (1 day)
- [ ] Create comprehensive benchmark comparing:
  - Before assimilation (external gonum)
  - After assimilation (integrated compute)
  - With new AVX2 optimizations
  - Against PyTorch CPU operations

## 📋 Backlog

### BLAS Level 2 (1 week)
- [ ] GEMV - Matrix-vector multiply
- [ ] GER - Rank-1 update  
- [ ] SYMV - Symmetric matrix-vector
- [ ] TRMV - Triangular matrix-vector

### Reduction Operations (3-4 days)
- [ ] Tree reduction for Sum
- [ ] Parallel Max/ArgMax
- [ ] Stable Mean/Variance calculation
- [ ] L2 norm with overflow protection

### Convolution (1-2 weeks)
- [ ] Im2col implementation
- [ ] Direct 3x3 convolution
- [ ] Winograd for larger kernels
- [ ] Depthwise separable support

### Float16 Support (1 week)
- [ ] F16C conversion routines
- [ ] Mixed precision GEMM
- [ ] BFloat16 operations
- [ ] Automatic mixed precision

## 🎯 Success Metrics

### Performance Targets
- AXPY: 50+ GFLOPS (currently 30)
- DOT: 60+ GFLOPS (currently 28)
- GEMM: 200+ GFLOPS for large matrices
- Fused ops: 2-3x faster than separate operations

### Code Quality
- Zero allocations in hot paths
- Cache-friendly memory patterns
- NUMA awareness for large systems
- Graceful feature detection

## 💡 Ideas Parking Lot
- GPU compute shaders via Vulkan
- WebAssembly SIMD target
- ARM NEON optimizations
- TPU-style systolic array simulation
- Automatic kernel fusion compiler

---
*Last Updated: [DATE]*
*Next Review: [DATE+1 week]*