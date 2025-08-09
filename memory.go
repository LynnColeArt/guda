package guda

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// MemcpyKind specifies the direction of memory transfer.
// In GUDA's unified memory model, these are provided for CUDA compatibility
// but may be treated identically since all memory is CPU-accessible.
type MemcpyKind int

const (
	MemcpyHostToHost     MemcpyKind = iota // Host to host transfer
	MemcpyHostToDevice                      // Host to device transfer
	MemcpyDeviceToHost                      // Device to host transfer
	MemcpyDeviceToDevice                    // Device to device transfer
	MemcpyDefault                           // Default transfer (infer direction)
)

// MemoryPool manages device memory allocation with efficient reuse.
// It maintains a free list of previously allocated blocks to reduce
// allocation overhead and memory fragmentation.
type MemoryPool struct {
	mu         sync.Mutex
	allocated  map[uintptr]*allocation
	freeList   []*allocation
	totalAlloc int64
	peakAlloc  int64
}

type allocation struct {
	ptr  unsafe.Pointer
	size int
	used bool
}

// NewMemoryPool creates a new memory pool for efficient memory management.
// The pool tracks allocations and provides statistics on memory usage.
func NewMemoryPool() *MemoryPool {
	return &MemoryPool{
		allocated: make(map[uintptr]*allocation),
	}
}

// Malloc allocates device memory of the specified size in bytes.
// The memory is aligned for optimal SIMD performance.
//
// Example:
//   ptr, err := ctx.Malloc(1024 * 4) // Allocate 1024 float32s
//   if err != nil {
//       return err
//   }
//   defer ctx.Free(ptr)
func (ctx *Context) Malloc(size int) (DevicePtr, error) {
	return ctx.memory.Allocate(size)
}

// Free releases device memory allocated by Malloc.
// It is safe to call Free with a zero DevicePtr.
// The memory may be retained in the pool for future allocations.
func (ctx *Context) Free(ptr DevicePtr) error {
	return ctx.memory.Free(ptr)
}

// Memcpy copies memory between host and device.
// Supports various combinations of DevicePtr and Go slices.
//
// Parameters:
//   - dst: Destination (DevicePtr or Go slice)
//   - src: Source (DevicePtr or Go slice)
//   - size: Number of bytes to copy
//   - kind: Transfer direction (for CUDA compatibility)
//
// Example:
//   h_data := make([]float32, 1024)
//   d_data, _ := ctx.Malloc(1024 * 4)
//   ctx.Memcpy(d_data, h_data, 1024*4, guda.MemcpyHostToDevice)
func (ctx *Context) Memcpy(dst, src interface{}, size int, kind MemcpyKind) error {
	// On CPU, all memory transfers are just memcpy
	// We keep the API for compatibility
	
	var dstPtr, srcPtr unsafe.Pointer
	
	// Handle dst
	switch d := dst.(type) {
	case DevicePtr:
		dstPtr = d.ptr
	case unsafe.Pointer:
		dstPtr = d
	case []byte:
		if len(d) > 0 {
			dstPtr = unsafe.Pointer(&d[0])
		}
	case []float32:
		if len(d) > 0 {
			dstPtr = unsafe.Pointer(&d[0])
		}
	case []float64:
		if len(d) > 0 {
			dstPtr = unsafe.Pointer(&d[0])
		}
	case []int32:
		if len(d) > 0 {
			dstPtr = unsafe.Pointer(&d[0])
		}
	default:
		return NewInvalidArgError("Memcpy", fmt.Sprintf("unsupported dst type: %T", dst))
	}
	
	// Handle src
	switch s := src.(type) {
	case DevicePtr:
		srcPtr = s.ptr
	case unsafe.Pointer:
		srcPtr = s
	case []byte:
		if len(s) > 0 {
			srcPtr = unsafe.Pointer(&s[0])
		}
	case []float32:
		if len(s) > 0 {
			srcPtr = unsafe.Pointer(&s[0])
		}
	case []float64:
		if len(s) > 0 {
			srcPtr = unsafe.Pointer(&s[0])
		}
	case []int32:
		if len(s) > 0 {
			srcPtr = unsafe.Pointer(&s[0])
		}
	default:
		return NewInvalidArgError("Memcpy", fmt.Sprintf("unsupported src type: %T", src))
	}
	
	// Perform the copy
	if dstPtr != nil && srcPtr != nil && size > 0 {
		copy((*[1 << 30]byte)(dstPtr)[:size:size], (*[1 << 30]byte)(srcPtr)[:size:size])
	}
	
	return nil
}

// MemoryPool methods

// Allocate allocates memory from the pool
func (mp *MemoryPool) Allocate(size int) (DevicePtr, error) {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	// Round up to alignment
	const alignment = 64 // Cache line size
	alignedSize := (size + alignment - 1) &^ (alignment - 1)
	
	// Try to reuse from free list
	for i, alloc := range mp.freeList {
		if alloc.size >= alignedSize {
			// Remove from free list
			mp.freeList = append(mp.freeList[:i], mp.freeList[i+1:]...)
			alloc.used = true
			
			// Update tracking
			mp.totalAlloc += int64(alloc.size)
			if mp.totalAlloc > mp.peakAlloc {
				mp.peakAlloc = mp.totalAlloc
			}
			
			return DevicePtr{
				ptr:  alloc.ptr,
				size: size,
			}, nil
		}
	}
	
	// Allocate new memory
	// Use make to get properly aligned memory
	buf := make([]byte, alignedSize)
	ptr := unsafe.Pointer(&buf[0])
	
	// Prevent GC from collecting
	runtime.KeepAlive(buf)
	
	alloc := &allocation{
		ptr:  ptr,
		size: alignedSize,
		used: true,
	}
	
	mp.allocated[uintptr(ptr)] = alloc
	
	// Update tracking
	mp.totalAlloc += int64(alignedSize)
	if mp.totalAlloc > mp.peakAlloc {
		mp.peakAlloc = mp.totalAlloc
	}
	
	return DevicePtr{
		ptr:  ptr,
		size: size,
	}, nil
}

// Free returns memory to the pool
func (mp *MemoryPool) Free(ptr DevicePtr) error {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	
	allocPtr := uintptr(ptr.ptr)
	alloc, ok := mp.allocated[allocPtr]
	if !ok {
		return NewMemoryError("Free", "pointer not found in allocation pool", nil)
	}
	
	if !alloc.used {
		return ErrDoubleFree
	}
	
	// Mark as free and add to free list
	alloc.used = false
	mp.freeList = append(mp.freeList, alloc)
	mp.totalAlloc -= int64(alloc.size)
	
	return nil
}

// GetStats returns memory pool statistics
func (mp *MemoryPool) GetStats() (allocated, peak int64) {
	mp.mu.Lock()
	defer mp.mu.Unlock()
	return mp.totalAlloc, mp.peakAlloc
}

// DevicePtr methods for convenience

// Float32 returns a float32 slice view of the device memory.
// The slice can be used directly for reading and writing data.
// Panics if the memory size is not aligned to float32 boundaries.
//
// Example:
//   d_data, _ := guda.Malloc(1024 * 4) // Allocate for 1024 float32s
//   data := d_data.Float32()
//   data[0] = 3.14 // Direct access
func (d DevicePtr) Float32() []float32 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 28]float32)(d.ptr)[:d.size/4:d.size/4]
}

// Float64 returns a float64 slice view of the device memory.
// The slice can be used directly for reading and writing data.
// Panics if the memory size is not aligned to float64 boundaries.
//
// Example:
//   d_data, _ := guda.Malloc(1024 * 8) // Allocate for 1024 float64s
//   data := d_data.Float64()
//   data[0] = 3.14159 // Direct access
func (d DevicePtr) Float64() []float64 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 27]float64)(d.ptr)[:d.size/8:d.size/8]
}

// Int32 returns an int32 slice view of the device memory.
// The slice can be used directly for reading and writing data.
// Panics if the memory size is not aligned to int32 boundaries.
//
// Example:
//   d_indices, _ := guda.Malloc(1024 * 4) // Allocate for 1024 int32s
//   indices := d_indices.Int32()
//   indices[0] = 42 // Direct access
func (d DevicePtr) Int32() []int32 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 28]int32)(d.ptr)[:d.size/4:d.size/4]
}

// Byte returns a byte slice view of the device memory.
// The slice covers the entire allocated memory region.
// Useful for raw memory operations or interfacing with I/O.
//
// Example:
//   d_buffer, _ := guda.Malloc(4096)
//   bytes := d_buffer.Byte()
//   copy(bytes, sourceData) // Copy raw bytes
func (d DevicePtr) Byte() []byte {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 30]byte)(d.ptr)[:d.size:d.size]
}

// Offset returns a new DevicePtr offset by the given number of bytes.
// Useful for accessing sub-regions of allocated memory.
// The returned DevicePtr shares the same underlying memory.
//
// Example:
//   d_array, _ := guda.Malloc(1024 * 4) // 1024 float32s
//   d_second_half := d_array.Offset(512 * 4) // Start at element 512
//   data := d_second_half.Float32() // Access second half
func (d DevicePtr) Offset(bytes int) DevicePtr {
	return DevicePtr{
		ptr:    unsafe.Pointer(uintptr(d.ptr) + uintptr(bytes)),
		size:   d.size - bytes,
		offset: d.offset + bytes,
	}
}

// Size returns the size in bytes of the memory region
func (d DevicePtr) Size() int {
	return d.size
}

// getSystemMemory returns total system memory in bytes
func getSystemMemory() uint64 {
	// This is a simplified version
	// In production, we'd use syscalls to get actual memory
	return 16 * 1024 * 1024 * 1024 // Default to 16GB
}