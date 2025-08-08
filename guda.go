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

// Malloc allocates device memory
func Malloc(size int) (DevicePtr, error) {
	return defaultContext.Malloc(size)
}

// Free releases device memory
func Free(ptr DevicePtr) error {
	return defaultContext.Free(ptr)
}

// Memcpy copies memory between host and device
func Memcpy(dst, src interface{}, size int, kind MemcpyKind) error {
	return defaultContext.Memcpy(dst, src, size, kind)
}

// Launch executes a kernel
func Launch(kernel Kernel, grid, block Dim3, args ...interface{}) error {
	return defaultContext.Launch(kernel, grid, block, args...)
}

// LaunchFunc executes a kernel function
func LaunchFunc(fn KernelFunc, grid, block Dim3, args ...interface{}) error {
	return defaultContext.LaunchFunc(fn, grid, block, args...)
}

// Synchronize waits for all operations to complete
func Synchronize() error {
	return defaultContext.Synchronize()
}

// GetDevice returns the current device
func GetDevice() *Device {
	return defaultDevice
}

// SetDevice sets the active device (no-op for CPU)
func SetDevice(id int) error {
	if id != 0 {
		return fmt.Errorf("only device 0 (CPU) is available")
	}
	return nil
}

// GetDeviceCount returns the number of available devices
func GetDeviceCount() int {
	return 1 // Only CPU
}

// GetDeviceProperties returns device properties
func GetDeviceProperties(id int) (*Device, error) {
	if id != 0 {
		return nil, fmt.Errorf("invalid device ID: %d", id)
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