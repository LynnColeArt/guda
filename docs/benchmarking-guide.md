# GUDA Benchmarking Guide: Hot vs Cold Cache

## Understanding Cache Effects in Benchmarks

When benchmarking high-performance computing libraries like GUDA, cache state significantly impacts results. This guide explains how to properly measure and interpret performance in both hot and cold cache scenarios.

## Hot Cache vs Cold Cache

### Hot Cache
- **Definition**: Data is already loaded in CPU caches (L1/L2/L3)
- **Characteristics**: 
  - Represents best-case performance
  - Common in repeated operations on same data
  - Shows theoretical peak performance
- **When it occurs**: 
  - Running same benchmark multiple times
  - Small working sets that fit in cache
  - Tight computational loops

### Cold Cache
- **Definition**: Data must be loaded from main memory
- **Characteristics**:
  - Represents real-world first-run performance
  - Memory bandwidth limited
  - More realistic for large datasets
- **When it occurs**:
  - First execution after system boot
  - Processing new data
  - Working sets larger than cache

## How to Reproduce Each Scenario

### Hot Cache Benchmarks

```bash
# Standard benchmark - data stays in cache between iterations
make benchmark

# Or directly with Go:
go test -bench=. -benchmem ./...
```

The Go benchmark harness runs each benchmark multiple times, naturally creating hot cache conditions after the first iteration.

### Cold Cache Benchmarks

⚠️ **CRITICAL SAFETY WARNING** ⚠️

The `echo 3 > /proc/sys/vm/drop_caches` command can cause **severe system instability**, including:
- Application crashes
- Service failures  
- Data loss
- System freezes
- Complete system failure requiring reinstallation

**SAFER ALTERNATIVES:**

```bash
# Option 1: Use the safer make target (simulates cold cache)
make bench-cold

# Option 2: Clear page cache only (less dangerous but still risky)
sudo sh -c 'sync && echo 1 > /proc/sys/vm/drop_caches'
go test -bench=. -benchmem -benchtime=3s ./...

# Option 3: Allocate large memory to flush caches (safest)
# Create a simple program that allocates and touches ~2x your RAM
# Then run benchmarks immediately after
```

**If you must use full cache clearing:**
1. Save all work first
2. Close all applications
3. Be prepared for system instability
4. Have a recovery plan
5. Consider running in a VM or container

**Note**: Cold cache benchmarks require root privileges to clear system caches.

## Interpreting Results

### Expected Performance Differences

| Operation | Hot Cache | Cold Cache | Ratio |
|-----------|-----------|------------|-------|
| GEMM (small) | ~100 GFLOPS | ~20 GFLOPS | 5x |
| GEMM (large) | ~80 GFLOPS | ~60 GFLOPS | 1.3x |
| DOT | ~50 GFLOPS | ~15 GFLOPS | 3.3x |
| AXPY | ~40 GFLOPS | ~10 GFLOPS | 4x |

### What the Numbers Mean

**Hot Cache Performance**:
- Shows computational efficiency
- Indicates SIMD utilization
- Measures ALU throughput

**Cold Cache Performance**:
- Shows memory bandwidth utilization
- Indicates real-world performance
- Measures system integration

## Best Practices

### 1. Always Report Both
When sharing benchmark results:
```
BenchmarkGEMM_1024x1024:
  Hot Cache:  95.2 GFLOPS
  Cold Cache: 42.1 GFLOPS
  Cache Effect: 2.26x
```

### 2. Consider Your Use Case
- **ML Training**: Often hot cache (repeated operations)
- **ML Inference**: Often cold cache (new data)
- **Scientific Computing**: Mixed (depends on algorithm)

### 3. System Configuration
Cache sizes affect results:
```bash
# Check your cache sizes
lscpu | grep cache
```

Typical modern CPU:
- L1: 32KB per core
- L2: 256KB per core  
- L3: 8-32MB shared

### 4. Benchmark Guidelines

**For Hot Cache**:
- Use standard benchmark commands
- Run multiple iterations
- Report best/average performance

**For Cold Cache**:
- Clear caches before each run
- Use longer benchmark times (-benchtime=10s)
- Consider single iteration measurements

## Advanced Topics

### Memory Prefetching
GUDA uses prefetching to hide memory latency:
- Software prefetch instructions
- Hardware prefetcher engagement
- Streaming patterns

### NUMA Effects
On multi-socket systems:
- Local vs remote memory access
- First-touch policy matters
- Consider NUMA binding

### Working Set Analysis
Determine if your data fits in cache:
```go
// L1: 32KB = 8K floats
// L2: 256KB = 64K floats  
// L3: 8MB = 2M floats
```

## Troubleshooting

### Inconsistent Results
- Ensure no other processes are running
- Disable CPU frequency scaling
- Use performance governor

### Cannot Clear Caches
- Requires sudo/root access
- Some systems may restrict this
- Alternative: allocate large arrays to flush cache

### Performance Lower Than Expected
- Check CPU frequency scaling
- Verify SIMD support (AVX2/AVX512)
- Monitor thermal throttling

## Example Benchmark Script

```bash
#!/bin/bash
# comprehensive_bench.sh

echo "=== GUDA Comprehensive Benchmark ==="
echo "System: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Cache: $(lscpu | grep 'L3 cache' | cut -d: -f2 | xargs)"
echo

echo "--- Hot Cache Benchmarks ---"
go test -bench=. -benchmem ./... | grep -E "Benchmark|GFLOPS"

echo
echo "--- Cold Cache Benchmarks ---"
sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
go test -bench=. -benchmem -benchtime=3s ./... | grep -E "Benchmark|GFLOPS"
```

## Conclusion

Understanding cache effects is crucial for:
- Accurate performance reporting
- Choosing appropriate algorithms
- Setting realistic expectations

Remember: both hot and cold cache numbers are valid - they represent different use cases and scenarios in real applications.