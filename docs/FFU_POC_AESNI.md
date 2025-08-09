# Proof of Concept: AES-NI Fixed-Function Unit Integration

## Goal

Demonstrate the FFU concept by implementing AES encryption that automatically uses AES-NI instructions when available, falling back to software implementation when not.

## Why AES-NI?

1. **Ubiquitous**: Available on most x86 CPUs since 2010
2. **Massive Speedup**: 3-10x over software AES
3. **Simple API**: Clear input/output semantics
4. **Real Use Case**: Many ML pipelines need encryption

## Design

### 1. FFU Interface

```go
type FFU interface {
    // Check if this FFU can handle the workload
    CanHandle(workload Workload) bool
    
    // Estimate performance for cost model
    EstimateCost(workload Workload) Cost
    
    // Execute the workload
    Execute(workload Workload) error
    
    // Get capabilities
    Capabilities() FFUCapability
}
```

### 2. AES Workload Definition

```go
type AESWorkload struct {
    Operation  string  // "encrypt" or "decrypt"
    Mode       string  // "ECB", "CBC", "CTR", etc.
    Key        []byte  // 128, 192, or 256 bit
    IV         []byte  // Initialization vector
    Input      []byte  // Plaintext or ciphertext
    Output     []byte  // Result buffer
}
```

### 3. Implementation Strategy

```
┌─────────────────┐
│ GUDA AES Kernel │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ FFU Dispatcher  │────▶│ Capability Check │
└────────┬────────┘     └──────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │ Has AES-NI?     │
         │              └───┬─────────┬───┘
         │                Yes│         │No
         │                  ▼         ▼
         │          ┌─────────────┐ ┌──────────────┐
         └─────────▶│ AES-NI FFU  │ │ Software AES │
                    └─────────────┘ └──────────────┘
```

### 4. Benchmark Plan

```go
// Compare three implementations:
// 1. Pure Go crypto/aes
// 2. GUDA with FFU disabled (software path)
// 3. GUDA with FFU enabled (AES-NI path)

sizes := []int{16, 1024, 65536, 1048576} // 16B to 1MB
for _, size := range sizes {
    // Measure throughput (MB/s)
    // Measure latency (µs)
    // Measure power (if available)
}
```

## Expected Results

| Data Size | Software AES | AES-NI FFU | Speedup |
|-----------|-------------|------------|---------|
| 16 B      | 100 MB/s    | 500 MB/s   | 5x      |
| 1 KB      | 500 MB/s    | 3 GB/s     | 6x      |
| 64 KB     | 600 MB/s    | 5 GB/s     | 8x      |
| 1 MB      | 650 MB/s    | 6 GB/s     | 9x      |

## Integration Points

### 1. Detection
```go
func detectAESNI() bool {
    // Check CPUID for AES-NI support
    return cpu.X86.HasAES
}
```

### 2. Dispatcher
```go
func (d *Dispatcher) Route(kernel Kernel) FFU {
    if kernel.Type == "crypto_aes" && hasAESNI {
        return aesniFFU
    }
    return cpuFFU
}
```

### 3. Metrics
```go
type FFUMetrics struct {
    CallCount     int64
    BytesProcessed int64
    TotalTime     time.Duration
    EnergyUsed    float64 // If available
}
```

## Code Structure

```
ffu/
├── ffu.go              # Core interfaces
├── detector.go         # FFU detection logic
├── dispatcher.go       # Workload routing
├── metrics.go          # Performance tracking
├── aesni/
│   ├── aesni.go       # AES-NI FFU implementation
│   ├── aesni_amd64.s  # Assembly implementation
│   └── aesni_test.go  # Tests and benchmarks
└── software/
    └── aes.go         # Software fallback
```

## Testing Strategy

1. **Correctness**: Verify bit-identical output
2. **Performance**: Benchmark across data sizes
3. **Fallback**: Test on CPUs without AES-NI
4. **Stress**: Concurrent operations
5. **Integration**: Use in larger GUDA kernel

## Success Criteria

- [ ] 5x+ speedup on AES operations
- [ ] Transparent fallback on older CPUs
- [ ] Less than 1% overhead when FFU not used
- [ ] Clean API that generalizes to other FFUs

## Lessons for Full Implementation

This POC will teach us:
1. How to abstract FFU interfaces effectively
2. Real overhead of the dispatch layer
3. Complexity of data movement
4. Testing strategies for hardware features

## Timeline

- Day 1-2: AES-NI detection and basic wrapper
- Day 3-4: Integration with dispatcher
- Day 5: Benchmarking and optimization
- Day 6-7: Documentation and cleanup

---

**Next**: If POC succeeds, apply pattern to AMX, video encode, etc.