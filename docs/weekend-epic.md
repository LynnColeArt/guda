# Weekend Epic: GUDA Improvements

Based on peer review feedback and Mini's performance analysis insights.

## Track 0: Immediate Bug Fixes

1. ✅ **Fix broken manual links in README** ⭐
   - Update link from `01-installation.md` to `02-installation.md`
   - Update link from `02-architecture.md` to `04-architecture.md`
   - Verify all other documentation links work

## Track 1: Benchmark Validation (Mini's suggestions)

Tasks ordered from least to most cognitively complex:

**Additional completed work**: ✅ Created benchmark_logger.go to save benchmark results to JSON files

1. ✅ **Add benchmark labels** ⭐
   - Simply add "hot-cache (packed, persistent)" vs "cold-cache (randomized, flushed)" labels to benchmark output
   - Modify `b.ReportMetric()` calls to include cache state

2. ✅ **Create make bench-cold target** ⭐⭐
   - Add Makefile target that runs benchmarks with cache flushing
   - Use `sync && echo 3 > /proc/sys/vm/drop_caches` between runs (made safer)
   - Randomize memory allocation addresses

3. ✅ **Write 'How to reproduce (hot vs cold)' section** ⭐⭐
   - Document methodology for hot vs cold benchmarks
   - Explain cache effects on performance
   - Add to README or separate BENCHMARKS.md file (created benchmarking-guide.md)

4. **Add performance counter checks** ⭐⭐⭐
   - Integrate perf counters: L3 misses, memory bandwidth, FP32_FMA operations
   - Use `perf stat` or Go's x/sys/unix for programmatic access
   - Report alongside GFLOPS numbers

5. **Create roofline analysis** ⭐⭐⭐⭐
   - Calculate arithmetic intensity (FLOPS/byte)
   - Measure actual memory bandwidth
   - Plot operations on roofline model
   - Prove 2K GFLOPS sits in compute-bound region under hot cache

## Track 2: High-Priority Peer Review Items

Tasks ordered from least to most cognitively complex:

1. ✅ **Add comprehensive godoc comments** ⭐
   - Document all exported functions with proper godoc format
   - Include parameter descriptions and return values
   - Add usage examples where helpful

2. **Extract magic numbers to configuration** ⭐⭐
   - Create config.go with named constants
   - Replace hardcoded values like block sizes (256), cache sizes
   - Make tunable parameters accessible

3. **Create structured error types** ⭐⭐
   - Replace string errors with typed errors
   - Implement error interfaces for different error categories
   - Add context and wrapping support

4. **Implement size-class based memory pool** ⭐⭐⭐
   - Create buckets for common allocation sizes
   - Reduce fragmentation and allocation overhead
   - Track statistics per size class

5. **Add fuzz tests** ⭐⭐⭐
   - Create fuzz targets for kernel argument validation
   - Test edge cases in memory allocation
   - Verify numerical stability with random inputs

6. **Replace interface{} with generics** ⭐⭐⭐⭐
   - Convert KernelFunc to use generic type parameters
   - Make kernel arguments type-safe at compile time
   - Maintain backward compatibility where possible

## Track 3: Performance Enhancements

Tasks ordered from least to most cognitively complex:

1. ✅ **Implement memory prefetching** ⭐⭐
   - Add prefetch instructions for large matrix operations
   - Use `__builtin_prefetch` in C code or assembly
   - Focus on predictable access patterns (added PREFETCHT0 to AVX2 kernels)

2. **Add AVX-512 support** ⭐⭐⭐
   - Detect CPU capabilities at runtime
   - Create AVX-512 variants of hot kernels (16-element vectors)
   - Fall back to AVX2 on older CPUs

3. **Implement cache-aware adaptive blocking** ⭐⭐⭐⭐
   - Replace fixed 256-element blocks with dynamic sizing
   - Consider L1/L2/L3 cache sizes at runtime
   - Tune for different matrix dimensions

4. **Implement work stealing** ⭐⭐⭐⭐
   - Replace simple work distribution with work-stealing queues
   - Balance load across CPU cores dynamically
   - Handle irregular workloads better

5. **Add NUMA awareness** ⭐⭐⭐⭐⭐
   - Detect NUMA topology
   - Allocate memory on appropriate NUMA nodes
   - Schedule work with NUMA affinity
   - Minimize cross-socket traffic

## Track 4: Testing Strategy (from Uzu analysis)

Tasks ordered from least to most cognitively complex:

1. **Create reference implementations for all kernels** ⭐⭐
   - Simple, correct scalar implementations
   - Use as baseline for correctness verification
   - Example: `dotProductReference()` for `DotProductAVX2()`

2. ✅ **Add deterministic test data generation** ⭐
   - Use seeded random data for reproducibility
   - Test with known problem sizes (L1/L2/L3/RAM)
   - Include edge cases: empty, single element, unaligned
   - Created test_data.go with comprehensive test data generators

3. **Implement tolerance-based verification** ⭐⭐
   - Define acceptable error margins (e.g., 1e-5 for float32)
   - Support both absolute and relative error checking
   - Account for different precision in SIMD operations

4. **Add performance regression tests** ⭐⭐⭐
   - Benchmark against reference implementations
   - Ensure minimum speedup thresholds (e.g., 10x)
   - Track performance over time
   - Test with real menthar workloads

5. **Create CUDA compatibility test suite** ⭐⭐⭐⭐
   - Verify grid/block/thread semantics match CUDA
   - Test memory access patterns
   - Ensure reduction operations match CUDA behavior
   - Edge case handling compatibility

## Implementation Notes

- Start with Track 1 to validate performance claims
- Track 2 items improve code quality and safety
- Track 3 items push performance boundaries further
- Track 4 ensures correctness and prevents regressions
- Each ⭐ represents approximate cognitive complexity
- Consider pairing simpler tasks with complex ones for variety

## Success Criteria

- [x] Benchmarks clearly distinguish hot vs cold cache scenarios
- [ ] Performance counters prove we're not skipping work
- [ ] Roofline analysis validates high GFLOPS numbers
- [ ] All exported APIs have proper documentation
- [ ] Type safety improved through generics
- [ ] Memory allocation is more efficient
- [ ] Error handling is structured and informative
- [ ] Every SIMD kernel has a reference implementation
- [ ] All kernels pass tolerance-based correctness tests
- [ ] Performance regressions are automatically detected