# Chapter 7: Runtime API

> *"A well-designed API is like a perfect user interface—powerful enough for experts, simple enough for beginners."* — GUDA API Design Philosophy

The GUDA Runtime API is your gateway to high-performance computing. This comprehensive reference covers every function you need to master, from basic memory management to advanced kernel execution. Let's explore the complete toolkit!

## Core Memory Management

### Memory Allocation and Deallocation

The foundation of all GUDA operations starts with memory management:

```go
// Basic memory allocation
func Malloc(size int) DevicePtr

// Free allocated memory  
func Free(ptr DevicePtr) error

// Allocate with alignment for SIMD optimization
func MallocAligned(size, alignment int) DevicePtr

// Query available memory
func MemGetInfo() (free, total int64, err error)

// Set memory to a specific value
func Memset(ptr DevicePtr, value byte, size int) error

// Example: Comprehensive memory management
func demonstrateMemoryAPI() {
    // Query system memory
    free, total, err := guda.MemGetInfo()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Memory: %d MB free, %d MB total\n", free/1024/1024, total/1024/1024)
    
    // Allocate aligned memory for optimal SIMD performance
    size := 1024 * 1024 * 4 // 4MB
    ptr := guda.MallocAligned(size, 32) // 32-byte aligned for AVX2
    if ptr == 0 {
        log.Fatal("Failed to allocate memory")
    }
    defer func() {
        if err := guda.Free(ptr); err != nil {
            log.Printf("Warning: Failed to free memory: %v", err)
        }
    }()
    
    // Initialize memory to zero
    if err := guda.Memset(ptr, 0, size); err != nil {
        log.Fatal("Failed to initialize memory:", err)
    }
    
    fmt.Printf("Successfully allocated and initialized %d bytes\n", size)
}
```

### Memory Transfer Operations

Seamless data movement between host and device:

```go
// Host to Device transfer
func MemcpyHtoD(dst DevicePtr, src unsafe.Pointer, size int) error

// Device to Host transfer  
func MemcpyDtoH(dst unsafe.Pointer, src DevicePtr, size int) error

// Device to Device transfer
func MemcpyDtoD(dst, src DevicePtr, size int) error

// Asynchronous transfers (non-blocking)
func MemcpyHtoDAsync(dst DevicePtr, src unsafe.Pointer, size int, stream Stream) error
func MemcpyDtoHAsync(dst unsafe.Pointer, src DevicePtr, size int, stream Stream) error

// Convenience functions for common types
func MemcpyFloat32HtoD(dst DevicePtr, src []float32) error
func MemcpyFloat32DtoH(dst []float32, src DevicePtr) error

// Example: Efficient batch data transfers
func batchDataProcessing() {
    const batchSize = 10000
    const numBatches = 100
    
    // Allocate device buffer once
    deviceBuffer := guda.MallocAligned(batchSize*4, 32)
    defer guda.Free(deviceBuffer)
    
    for batch := 0; batch < numBatches; batch++ {
        // Prepare host data
        hostData := make([]float32, batchSize)
        for i := range hostData {
            hostData[i] = rand.Float32()
        }
        
        // Transfer to device
        if err := guda.MemcpyFloat32HtoD(deviceBuffer, hostData); err != nil {
            log.Fatal("Transfer failed:", err)
        }
        
        // Process on device (example: scale by 2.0)
        guda.Sscal(batchSize, 2.0, deviceBuffer, 1)
        
        // Transfer results back
        results := make([]float32, batchSize)
        if err := guda.MemcpyFloat32DtoH(results, deviceBuffer); err != nil {
            log.Fatal("Transfer back failed:", err)
        }
        
        fmt.Printf("Processed batch %d, first result: %.3f\n", batch, results[0])
    }
}
```

## Stream Management

Streams enable asynchronous execution and overlapping operations:

```go
// Stream creation and destruction
func StreamCreate() (Stream, error)
func StreamDestroy(stream Stream) error

// Stream synchronization
func StreamSynchronize(stream Stream) error
func StreamQuery(stream Stream) error // Returns error if not complete

// Default stream operations
func GetDefaultStream() Stream

// Example: Overlapping computation and data transfer
func demonstrateStreamOverlap() {
    stream1, _ := guda.StreamCreate()
    stream2, _ := guda.StreamCreate()
    defer guda.StreamDestroy(stream1)
    defer guda.StreamDestroy(stream2)
    
    const dataSize = 1024 * 1024
    buffer1 := guda.Malloc(dataSize * 4)
    buffer2 := guda.Malloc(dataSize * 4)
    defer guda.Free(buffer1)
    defer guda.Free(buffer2)
    
    hostData1 := make([]float32, dataSize)
    hostData2 := make([]float32, dataSize)
    
    // Initialize data
    for i := range hostData1 {
        hostData1[i] = rand.Float32()
        hostData2[i] = rand.Float32()
    }
    
    // Overlapping operations using different streams
    // Stream 1: Transfer + Compute
    guda.MemcpyHtoDAsync(buffer1, unsafe.Pointer(&hostData1[0]), 
                         dataSize*4, stream1)
    guda.SscalAsync(dataSize, 2.0, buffer1, 1, stream1)
    
    // Stream 2: Transfer + Compute (runs in parallel)
    guda.MemcpyHtoDAsync(buffer2, unsafe.Pointer(&hostData2[0]), 
                         dataSize*4, stream2)
    guda.SscalAsync(dataSize, 3.0, buffer2, 1, stream2)
    
    // Wait for both streams to complete
    guda.StreamSynchronize(stream1)
    guda.StreamSynchronize(stream2)
    
    fmt.Println("Parallel stream execution completed")
}
```

## Event Management

Events provide fine-grained timing and synchronization:

```go
// Event creation and destruction
func EventCreate() (Event, error)
func EventCreateWithFlags(flags EventFlag) (Event, error)
func EventDestroy(event Event) error

// Event operations
func EventRecord(event Event, stream Stream) error
func EventSynchronize(event Event) error
func EventQuery(event Event) error

// Timing measurements
func EventElapsedTime(start, end Event) (float32, error)

// Stream synchronization with events
func StreamWaitEvent(stream Stream, event Event) error

// Example: Precise timing measurements
func preciseTimingMeasurement() {
    // Create timing events
    start, _ := guda.EventCreate()
    end, _ := guda.EventCreate()
    defer guda.EventDestroy(start)
    defer guda.EventDestroy(end)
    
    const N = 1024 * 1024
    data := guda.Malloc(N * 4)
    defer guda.Free(data)
    
    stream, _ := guda.StreamCreate()
    defer guda.StreamDestroy(stream)
    
    // Record start event
    guda.EventRecord(start, stream)
    
    // Perform operation
    guda.SscalAsync(N, 2.0, data, 1, stream)
    
    // Record end event
    guda.EventRecord(end, stream)
    
    // Wait for completion
    guda.EventSynchronize(end)
    
    // Calculate elapsed time
    elapsed, _ := guda.EventElapsedTime(start, end)
    
    // Calculate performance
    operations := float32(N)
    gflops := operations / elapsed / 1000.0 // Convert to GFLOPS
    
    fmt.Printf("Operation took %.3f ms (%.2f GFLOPS)\n", elapsed, gflops)
}
```

## Device Management

Multi-device systems require careful device management:

```go
// Device queries and selection
func GetDeviceCount() (int, error)
func SetDevice(device int) error
func GetDevice() (int, error)

// Device properties
type DeviceProperties struct {
    Name           string
    TotalGlobalMem int64
    MaxThreadsPerBlock int
    MaxGridSize    [3]int
    WarpSize       int
    ClockRate      int
    MemoryBusWidth int
}

func GetDeviceProperties(device int) (DeviceProperties, error)

// Device synchronization
func DeviceSynchronize() error
func DeviceReset() error

// Example: Multi-device workload distribution
func multiDeviceProcessing() {
    deviceCount, err := guda.GetDeviceCount()
    if err != nil || deviceCount == 0 {
        log.Fatal("No GUDA devices available")
    }
    
    fmt.Printf("Found %d GUDA devices\n", deviceCount)
    
    // Query each device
    for i := 0; i < deviceCount; i++ {
        props, err := guda.GetDeviceProperties(i)
        if err != nil {
            continue
        }
        
        fmt.Printf("Device %d: %s\n", i, props.Name)
        fmt.Printf("  Memory: %.2f GB\n", float64(props.TotalGlobalMem)/1024/1024/1024)
        fmt.Printf("  Clock: %d MHz\n", props.ClockRate/1000)
    }
    
    // Distribute work across devices
    const totalWork = 1000000
    workPerDevice := totalWork / deviceCount
    
    var wg sync.WaitGroup
    
    for device := 0; device < deviceCount; device++ {
        wg.Add(1)
        
        go func(deviceID int) {
            defer wg.Done()
            
            // Set device context
            guda.SetDevice(deviceID)
            
            // Allocate memory on this device
            buffer := guda.Malloc(workPerDevice * 4)
            defer guda.Free(buffer)
            
            // Process work on this device
            guda.Sscal(workPerDevice, float32(deviceID+1), buffer, 1)
            
            fmt.Printf("Device %d completed its work\n", deviceID)
        }(device)
    }
    
    wg.Wait()
    fmt.Println("All devices completed processing")
}
```

## Error Handling

Robust error handling for production applications:

```go
// Error types
type GudaError int

const (
    Success GudaError = iota
    ErrorInvalidValue
    ErrorOutOfMemory
    ErrorNotInitialized
    ErrorDeinitialized
    ErrorProfilerDisabled
    ErrorInvalidConfiguration
    ErrorInvalidDevice
    ErrorInvalidKernelImage
    ErrorNoKernelImageForDevice
    ErrorInsufficientDriver
    ErrorUnsupportedLimit
)

// Error handling functions
func GetLastError() error
func PeekAtLastError() error
func GetErrorString(err GudaError) string

// Example: Comprehensive error handling
func robustGudaOperation() error {
    // Clear any existing errors
    guda.GetLastError()
    
    // Attempt memory allocation
    ptr := guda.Malloc(1024 * 1024)
    if ptr == 0 {
        lastErr := guda.GetLastError()
        return fmt.Errorf("memory allocation failed: %v", lastErr)
    }
    defer func() {
        if err := guda.Free(ptr); err != nil {
            log.Printf("Warning: Failed to free memory: %v", err)
        }
    }()
    
    // Attempt operation with error checking
    if err := guda.Memset(ptr, 0, 1024*1024); err != nil {
        return fmt.Errorf("memset operation failed: %v", err)
    }
    
    // Check for any accumulated errors
    if err := guda.PeekAtLastError(); err != nil {
        return fmt.Errorf("accumulated error detected: %v", err)
    }
    
    return nil
}

// Error recovery strategies
func recoverableOperation() {
    const maxRetries = 3
    var lastErr error
    
    for attempt := 0; attempt < maxRetries; attempt++ {
        ptr := guda.Malloc(1024 * 1024)
        if ptr != 0 {
            // Success
            defer guda.Free(ptr)
            fmt.Println("Operation succeeded on attempt", attempt+1)
            return
        }
        
        lastErr = guda.GetLastError()
        if lastErr != nil && lastErr.Error() != "out of memory" {
            // Non-recoverable error
            log.Fatal("Non-recoverable error:", lastErr)
        }
        
        // Wait before retry (exponential backoff)
        time.Sleep(time.Duration(attempt+1) * 100 * time.Millisecond)
        
        // Try to free some memory
        runtime.GC()
    }
    
    log.Fatal("Operation failed after", maxRetries, "attempts:", lastErr)
}
```

## Advanced Runtime Features

### Memory Pool Integration

```go
// Memory pool for high-performance allocation
type RuntimeMemoryPool struct {
    pools map[int][]DevicePtr
    mutex sync.Mutex
}

func NewRuntimeMemoryPool() *RuntimeMemoryPool {
    return &RuntimeMemoryPool{
        pools: make(map[int][]DevicePtr),
    }
}

func (rmp *RuntimeMemoryPool) Malloc(size int) DevicePtr {
    rmp.mutex.Lock()
    defer rmp.mutex.Unlock()
    
    // Round to next power of 2
    poolSize := 1
    for poolSize < size {
        poolSize <<= 1
    }
    
    // Try to reuse from pool
    if pool, exists := rmp.pools[poolSize]; exists && len(pool) > 0 {
        ptr := pool[len(pool)-1]
        rmp.pools[poolSize] = pool[:len(pool)-1]
        return ptr
    }
    
    // Allocate new
    return guda.Malloc(poolSize)
}

func (rmp *RuntimeMemoryPool) Free(ptr DevicePtr, size int) {
    rmp.mutex.Lock()
    defer rmp.mutex.Unlock()
    
    poolSize := 1
    for poolSize < size {
        poolSize <<= 1
    }
    
    // Return to pool
    rmp.pools[poolSize] = append(rmp.pools[poolSize], ptr)
}
```

### Profiling Integration

```go
// Profiling support for performance analysis
func ProfilerStart() error
func ProfilerStop() error

// Range profiling for specific code sections
func ProfilerRangeStart(name string) error
func ProfilerRangeEnd(name string) error

// Example: Comprehensive profiling
func profiledExecution() {
    if err := guda.ProfilerStart(); err != nil {
        log.Fatal("Failed to start profiler:", err)
    }
    defer guda.ProfilerStop()
    
    const N = 1024 * 1024
    data := guda.Malloc(N * 4)
    defer guda.Free(data)
    
    // Profile memory initialization
    guda.ProfilerRangeStart("memory_init")
    guda.Memset(data, 0, N*4)
    guda.ProfilerRangeEnd("memory_init")
    
    // Profile computation
    guda.ProfilerRangeStart("computation")
    guda.Sscal(N, 2.0, data, 1)
    guda.ProfilerRangeEnd("computation")
    
    // Profile memory transfer
    guda.ProfilerRangeStart("memory_transfer")
    result := make([]float32, N)
    guda.MemcpyFloat32DtoH(result, data)
    guda.ProfilerRangeEnd("memory_transfer")
    
    fmt.Println("Profiling completed - check profiler output")
}
```

## Runtime API Reference Tables

### Memory Management Functions

| Function | Purpose | Performance Notes |
|----------|---------|-------------------|
| `Malloc(size)` | Basic allocation | Use aligned version for SIMD |
| `MallocAligned(size, align)` | Aligned allocation | Optimal for vectorized ops |
| `Free(ptr)` | Deallocate memory | Always pair with allocations |
| `Memset(ptr, val, size)` | Initialize memory | Vectorized implementation |
| `MemcpyHtoD(dst, src, size)` | Host to device | Zero-copy in GUDA |
| `MemcpyDtoH(dst, src, size)` | Device to host | Zero-copy in GUDA |

### Stream and Event Functions

| Function | Purpose | Typical Use Case |
|----------|---------|------------------|
| `StreamCreate()` | Create stream | Async operations |
| `StreamSynchronize(stream)` | Wait for stream | Synchronization points |
| `EventRecord(event, stream)` | Mark timing point | Performance measurement |
| `EventElapsedTime(start, end)` | Calculate duration | Profiling |
| `StreamWaitEvent(stream, event)` | Cross-stream sync | Complex dependencies |

### Device Management Functions

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `GetDeviceCount()` | Query device count | Multi-device setup |
| `SetDevice(id)` | Select active device | Device switching |
| `GetDeviceProperties(id)` | Query capabilities | Resource planning |
| `DeviceSynchronize()` | Wait for all operations | Global sync point |

## Performance Tips and Best Practices

### ✅ **Optimal Runtime Usage**

```go
// 1. Batch allocations and minimize malloc/free calls
func efficientMemoryUsage() {
    // Good: Allocate once, use multiple times
    buffer := guda.MallocAligned(1024*1024*4, 32)
    defer guda.Free(buffer)
    
    for i := 0; i < 100; i++ {
        // Process data in the same buffer
        guda.Sscal(1024*1024, float32(i), buffer, 1)
    }
}

// 2. Use streams for overlapping operations
func efficientStreamUsage() {
    compute := guda.StreamCreate()
    transfer := guda.StreamCreate()
    defer guda.StreamDestroy(compute)
    defer guda.StreamDestroy(transfer)
    
    // Overlap computation with next data transfer
    guda.ComputeAsync(..., compute)
    guda.MemcpyAsync(..., transfer)
}

// 3. Check errors efficiently
func efficientErrorHandling() error {
    // Check critical operations
    ptr := guda.Malloc(size)
    if ptr == 0 {
        return guda.GetLastError()
    }
    defer guda.Free(ptr)
    
    // Batch non-critical error checking
    guda.Operation1(...)
    guda.Operation2(...)
    guda.Operation3(...)
    
    // Single error check for batch
    return guda.PeekAtLastError()
}
```

### ❌ **Runtime Anti-patterns**

```go
// DON'T: Allocate/free in tight loops
func inefficientMemoryUsage() {
    for i := 0; i < 1000; i++ {
        ptr := guda.Malloc(1024) // Expensive!
        guda.Sscal(256, 2.0, ptr, 1)
        guda.Free(ptr) // Expensive!
    }
}

// DON'T: Synchronize unnecessarily
func excessiveSynchronization() {
    for i := 0; i < 100; i++ {
        guda.Sscal(1000, 2.0, data, 1)
        guda.DeviceSynchronize() // Kills parallelism!
    }
}

// DON'T: Ignore error handling
func ignoreErrors() {
    ptr := guda.Malloc(size) // What if this fails?
    guda.Sscal(n, alpha, ptr, 1) // Could crash!
    // Missing Free() - memory leak!
}
```

## What's Next?

You now have complete mastery of GUDA's Runtime API! Ready to build on this foundation?

- [BLAS Operations](08-blas-api.md) - High-performance linear algebra
- [Neural Network Operations](09-nn-api.md) - ML-specific building blocks
- [Optimization Techniques](10-optimization.md) - Squeeze maximum performance

The Runtime API is your foundation—everything else builds upon these primitives. Use them wisely, and your GUDA applications will be robust, fast, and scalable!

---

*🔧 A solid foundation enables unlimited architectural possibilities. Master the runtime, master GUDA.*