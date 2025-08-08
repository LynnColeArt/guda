package guda

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// MemcpyKind specifies the direction of memory transfer
type MemcpyKind int

const (
	MemcpyHostToHost MemcpyKind = iota
	MemcpyHostToDevice
	MemcpyDeviceToHost
	MemcpyDeviceToDevice
	MemcpyDefault
)

// MemoryPool manages device memory allocation
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

// NewMemoryPool creates a new memory pool
func NewMemoryPool() *MemoryPool {
	return &MemoryPool{
		allocated: make(map[uintptr]*allocation),
	}
}

// Malloc allocates device memory
func (ctx *Context) Malloc(size int) (DevicePtr, error) {
	return ctx.memory.Allocate(size)
}

// Free releases device memory
func (ctx *Context) Free(ptr DevicePtr) error {
	return ctx.memory.Free(ptr)
}

// Memcpy copies memory between host and device
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
		return fmt.Errorf("unsupported dst type: %T", dst)
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
		return fmt.Errorf("unsupported src type: %T", src)
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
		return fmt.Errorf("pointer not found in allocation pool")
	}
	
	if !alloc.used {
		return fmt.Errorf("double free detected")
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

// Float32 returns a float32 slice view of the memory
func (d DevicePtr) Float32() []float32 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 28]float32)(d.ptr)[:d.size/4:d.size/4]
}

// Float64 returns a float64 slice view of the memory
func (d DevicePtr) Float64() []float64 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 27]float64)(d.ptr)[:d.size/8:d.size/8]
}

// Int32 returns an int32 slice view of the memory
func (d DevicePtr) Int32() []int32 {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 28]int32)(d.ptr)[:d.size/4:d.size/4]
}

// Byte returns a byte slice view of the memory
func (d DevicePtr) Byte() []byte {
	if d.ptr == nil {
		return nil
	}
	return (*[1 << 30]byte)(d.ptr)[:d.size:d.size]
}

// Offset returns a new DevicePtr offset by the given number of bytes
func (d DevicePtr) Offset(bytes int) DevicePtr {
	return DevicePtr{
		ptr:    unsafe.Pointer(uintptr(d.ptr) + uintptr(bytes)),
		size:   d.size - bytes,
		offset: d.offset + bytes,
	}
}

// getSystemMemory returns total system memory in bytes
func getSystemMemory() uint64 {
	// This is a simplified version
	// In production, we'd use syscalls to get actual memory
	return 16 * 1024 * 1024 * 1024 // Default to 16GB
}