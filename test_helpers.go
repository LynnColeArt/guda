package guda

import (
	"testing"
)

// MallocOrFail allocates device memory and fails the test if unsuccessful
func MallocOrFail(t testing.TB, size int) DevicePtr {
	t.Helper()
	ptr, err := Malloc(size)
	if err != nil {
		t.Fatalf("Failed to allocate %d bytes: %v", size, err)
	}
	return ptr
}

// MemcpyOrFail copies data and fails the test if unsuccessful
func MemcpyOrFail(t testing.TB, dst DevicePtr, src interface{}, size int, direction MemcpyKind) {
	t.Helper()
	err := Memcpy(dst, src, size, direction)
	if err != nil {
		t.Fatalf("Memcpy failed: %v", err)
	}
}

// LaunchOrFail launches a kernel and fails the test if unsuccessful
func LaunchOrFail(t testing.TB, kernel KernelFunc, grid, block Dim3, args ...interface{}) {
	t.Helper()
	err := Launch(kernel, grid, block, args...)
	if err != nil {
		t.Fatalf("Kernel launch failed: %v", err)
	}
}

// SynchronizeOrFail synchronizes and fails the test if unsuccessful
func SynchronizeOrFail(t testing.TB) {
	t.Helper()
	err := Synchronize()
	if err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
}