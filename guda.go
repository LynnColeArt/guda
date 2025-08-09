// Package guda provides a CUDA-compatible API for CPU execution.
// It enables running CUDA applications on CPU-only infrastructure through
// aggressive SIMD optimization and native CPU implementations.
//
// Example usage:
//
//	ctx := guda.NewContext()
//	defer ctx.Destroy()
//	
//	// Allocate device memory
//	d_a, _ := ctx.Malloc(n * 4) // n float32s
//	d_b, _ := ctx.Malloc(n * 4)
//	
//	// Copy data to device
//	ctx.Memcpy(d_a, h_a, n*4, guda.MemcpyHostToDevice)
//	ctx.Memcpy(d_b, h_b, n*4, guda.MemcpyHostToDevice)
//	
//	// Launch kernel
//	grid := guda.Dim3{X: (n + 255) / 256}
//	block := guda.Dim3{X: 256}
//	ctx.LaunchKernel(myKernel, grid, block, args...)
package guda

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Device represents a compute device. In GUDA, this is the CPU with its
// cores and available memory. Each device has a unique ID and capabilities.
type Device struct {
	ID         int    // Unique device identifier
	Name       string // Human-readable device name
	TotalMem   uint64 // Total available memory in bytes
	NumCores   int    // Number of CPU cores
	MaxThreads int    // Maximum concurrent threads
}

// Context represents an execution context for GUDA operations.
// It manages device resources, memory allocation, and stream execution.
// A Context must be created before any GUDA operations and should be
// destroyed when no longer needed.
type Context struct {
	device     *Device
	streams    map[int]*Stream
	streamID   int32
	memory     *MemoryPool
	defaultStream *Stream
}

// Stream represents an ordered sequence of operations that execute
// asynchronously. Operations within a stream execute in order, but
// operations in different streams may execute concurrently.
type Stream struct {
	id       int
	tasks    chan func()
	done     chan struct{}
	wg       sync.WaitGroup
}

// Dim3 represents 3D dimensions for grid and block configurations.
// This matches CUDA's dim3 structure for kernel launch parameters.
type Dim3 struct {
	X, Y, Z int
}

// ThreadID identifies a thread's position within the execution hierarchy.
// It provides the same indexing semantics as CUDA's built-in variables:
// blockIdx, threadIdx, blockDim, and gridDim.
type ThreadID struct {
	BlockIdx  Dim3 // Block index within the grid
	ThreadIdx Dim3 // Thread index within the block
	BlockDim  Dim3 // Dimensions of the block
	GridDim   Dim3 // Dimensions of the grid
}

// Kernel represents a compute kernel that can be executed in parallel.
// Implementations should be thread-safe as Execute will be called
// concurrently from multiple threads.
type Kernel interface {
	Execute(tid ThreadID, args ...interface{})
}

// KernelFunc is a function that can be launched as a kernel.
// It receives thread identification and variadic arguments.
type KernelFunc func(tid ThreadID, args ...interface{})

// DevicePtr represents a pointer to device memory. It provides type-safe
// access to device memory and supports pointer arithmetic through the
// Offset method. Use the type conversion methods (Float32, Float64, etc.)
// to access the underlying data with proper type safety.
type DevicePtr struct {
	ptr    unsafe.Pointer
	size   int
	offset int
}

// Global runtime state
var (
	defaultDevice  *Device
	defaultContext *Context
	initOnce       sync.Once
)

// Initialize GUDA runtime
func init() {
	initOnce.Do(func() {
		defaultDevice = &Device{
			ID:         0,
			Name:       "CPU",
			TotalMem:   getSystemMemory(),
			NumCores:   runtime.NumCPU(),
			MaxThreads: runtime.NumCPU() * 2, // Hyperthreading
		}
		
		defaultContext = &Context{
			device:   defaultDevice,
			streams:  make(map[int]*Stream),
			memory:   NewMemoryPool(),
		}
		
		// Create default stream
		defaultContext.defaultStream = defaultContext.CreateStream()
	})
}

// Malloc allocates device memory of the specified size in bytes.
// In GUDA, this allocates CPU memory with proper alignment for SIMD operations.
// The returned DevicePtr can be used with all GUDA operations.
//
// Example:
//   d_data, err := guda.Malloc(1024 * 4) // Allocate 1024 float32s
//   if err != nil {
//       log.Fatal(err)
//   }
//   defer guda.Free(d_data)
func Malloc(size int) (DevicePtr, error) {
	return defaultContext.Malloc(size)
}

// Free releases device memory allocated by Malloc.
// It is safe to call Free with a zero-value DevicePtr.
//
// Example:
//   d_data, _ := guda.Malloc(1024 * 4)
//   defer guda.Free(d_data)
func Free(ptr DevicePtr) error {
	return defaultContext.Free(ptr)
}

// Memcpy copies memory between host and device.
// In GUDA's unified memory model, this may be a no-op or simple copy.
// Supports various Go slice types ([]float32, []float64, []int32, etc.).
//
// Parameters:
//   - dst: Destination (DevicePtr or Go slice)
//   - src: Source (DevicePtr or Go slice)  
//   - size: Number of bytes to copy
//   - kind: Transfer direction (MemcpyHostToDevice, MemcpyDeviceToHost, etc.)
//
// Example:
//   hostData := make([]float32, 1024)
//   d_data, _ := guda.Malloc(1024 * 4)
//   err := guda.Memcpy(d_data, hostData, 1024*4, guda.MemcpyHostToDevice)
func Memcpy(dst, src interface{}, size int, kind MemcpyKind) error {
	return defaultContext.Memcpy(dst, src, size, kind)
}

// Launch executes a kernel on the default stream.
// The kernel is executed across a grid of thread blocks.
//
// Parameters:
//   - kernel: The kernel to execute
//   - grid: Grid dimensions (number of blocks)
//   - block: Block dimensions (threads per block)
//   - args: Kernel arguments
//
// Example:
//   kernel := MyKernel{}
//   err := guda.Launch(kernel, guda.Dim3{X: 256, Y: 1, Z: 1}, guda.Dim3{X: 64, Y: 1, Z: 1})
func Launch(kernel Kernel, grid, block Dim3, args ...interface{}) error {
	return defaultContext.Launch(kernel, grid, block, args...)
}

// LaunchFunc executes a kernel function
func LaunchFunc(fn KernelFunc, grid, block Dim3, args ...interface{}) error {
	return defaultContext.LaunchFunc(fn, grid, block, args...)
}

// Synchronize waits for all operations on all streams to complete.
// This ensures all previously launched kernels and memory operations have finished.
//
// Example:
//   guda.Launch(kernel, grid, block)
//   err := guda.Synchronize() // Wait for kernel to complete
func Synchronize() error {
	return defaultContext.Synchronize()
}

// GetDevice returns the current device information.
// In GUDA, this always returns the CPU device.
//
// Example:
//   device := guda.GetDevice()
//   fmt.Printf("Running on: %s with %d cores\n", device.Name, device.NumCores)
func GetDevice() *Device {
	return defaultDevice
}

// SetDevice sets the active device (no-op for CPU)
func SetDevice(id int) error {
	if id != 0 {
		return ErrInvalidDevice
	}
	return nil
}

// GetDeviceCount returns the number of available devices.
// GUDA always returns 1 as it only supports CPU execution.
//
// Example:
//   count := guda.GetDeviceCount()
//   fmt.Printf("Available devices: %d\n", count)
func GetDeviceCount() int {
	return 1 // Only CPU
}

// GetDeviceProperties returns device properties
func GetDeviceProperties(id int) (*Device, error) {
	if id != 0 {
		return nil, NewInvalidArgError("GetDeviceProperties", fmt.Sprintf("invalid device ID: %d", id))
	}
	return defaultDevice, nil
}

// Context methods

// CreateStream creates a new execution stream
func (ctx *Context) CreateStream() *Stream {
	id := int(atomic.AddInt32(&ctx.streamID, 1))
	stream := &Stream{
		id:    id,
		tasks: make(chan func(), 1000),
		done:  make(chan struct{}),
	}
	
	// Start worker goroutine for stream
	go stream.worker()
	
	ctx.streams[id] = stream
	return stream
}

// Launch executes a kernel on the default stream
func (ctx *Context) Launch(kernel Kernel, grid, block Dim3, args ...interface{}) error {
	return ctx.LaunchStream(kernel, grid, block, ctx.defaultStream, args...)
}

// LaunchFunc executes a kernel function on the default stream
func (ctx *Context) LaunchFunc(fn KernelFunc, grid, block Dim3, args ...interface{}) error {
	return ctx.LaunchFuncStream(fn, grid, block, ctx.defaultStream, args...)
}

// LaunchStream executes a kernel on a specific stream
func (ctx *Context) LaunchStream(kernel Kernel, grid, block Dim3, stream *Stream, args ...interface{}) error {
	return ctx.launchInternal(kernel.Execute, grid, block, stream, args...)
}

// LaunchFuncStream executes a kernel function on a specific stream
func (ctx *Context) LaunchFuncStream(fn KernelFunc, grid, block Dim3, stream *Stream, args ...interface{}) error {
	return ctx.launchInternal(fn, grid, block, stream, args...)
}

// Synchronize waits for all streams to complete
func (ctx *Context) Synchronize() error {
	for _, stream := range ctx.streams {
		stream.Synchronize()
	}
	return nil
}

// Stream methods

// worker processes tasks for a stream
func (s *Stream) worker() {
	for task := range s.tasks {
		task()
		s.wg.Done()
	}
	close(s.done)
}

// Synchronize waits for all tasks in the stream to complete
func (s *Stream) Synchronize() {
	s.wg.Wait()
}

// Submit adds a task to the stream
func (s *Stream) Submit(task func()) {
	s.wg.Add(1)
	s.tasks <- task
}

// Helper functions

// Global returns the global thread index
func (tid ThreadID) Global() int {
	return tid.BlockIdx.X*tid.BlockDim.X + tid.ThreadIdx.X
}

// GlobalX returns the global X index
func (tid ThreadID) GlobalX() int {
	return tid.BlockIdx.X*tid.BlockDim.X + tid.ThreadIdx.X
}

// GlobalY returns the global Y index
func (tid ThreadID) GlobalY() int {
	return tid.BlockIdx.Y*tid.BlockDim.Y + tid.ThreadIdx.Y
}

// GlobalZ returns the global Z index
func (tid ThreadID) GlobalZ() int {
	return tid.BlockIdx.Z*tid.BlockDim.Z + tid.ThreadIdx.Z
}

// Size returns the total number of elements
func (d Dim3) Size() int {
	return d.X * d.Y * d.Z
}

// Implement KernelFunc as Kernel
func (fn KernelFunc) Execute(tid ThreadID, args ...interface{}) {
	fn(tid, args...)
}