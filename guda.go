// Package guda provides a CUDA-compatible API for CPU execution
package guda

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// Device represents a compute device (CPU in our case)
type Device struct {
	ID         int
	Name       string
	TotalMem   uint64
	NumCores   int
	MaxThreads int
}

// Context represents an execution context
type Context struct {
	device     *Device
	streams    map[int]*Stream
	streamID   int32
	memory     *MemoryPool
	defaultStream *Stream
}

// Stream represents an execution stream
type Stream struct {
	id       int
	tasks    chan func()
	done     chan struct{}
	wg       sync.WaitGroup
}

// Dim3 represents 3D dimensions for grid/block
type Dim3 struct {
	X, Y, Z int
}

// ThreadID identifies a thread within the execution hierarchy
type ThreadID struct {
	BlockIdx  Dim3
	ThreadIdx Dim3
	BlockDim  Dim3
	GridDim   Dim3
}

// Kernel represents a compute kernel
type Kernel interface {
	Execute(tid ThreadID, args ...interface{})
}

// KernelFunc is a function that can be launched as a kernel
type KernelFunc func(tid ThreadID, args ...interface{})

// DevicePtr represents a device memory pointer
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