package guda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Test basic memory allocation and deallocation
func TestMemoryAllocation(t *testing.T) {
	sizes := []int{100, 1000, 10000, 1000000}
	
	for _, size := range sizes {
		ptr, err := Malloc(size * 4)
		if err != nil {
			t.Fatalf("Failed to allocate %d bytes: %v", size*4, err)
		}
		
		// Verify we can access the memory
		slice := ptr.Float32()
		if len(slice) != size {
			t.Errorf("Expected slice length %d, got %d", size, len(slice))
		}
		
		// Write and read test
		for i := 0; i < min(100, size); i++ {
			slice[i] = float32(i)
		}
		
		for i := 0; i < min(100, size); i++ {
			if slice[i] != float32(i) {
				t.Errorf("Memory corruption at index %d", i)
			}
		}
		
		err = Free(ptr)
		if err != nil {
			t.Fatalf("Failed to free memory: %v", err)
		}
	}
}

// Test memory copy operations
func TestMemcpy(t *testing.T) {
	const N = 1000
	
	// Create host data
	h_src := make([]float32, N)
	h_dst := make([]float32, N)
	for i := 0; i < N; i++ {
		h_src[i] = rand.Float32()
	}
	
	// Allocate device memory
	d_src, _ := Malloc(N * 4)
	d_dst, _ := Malloc(N * 4)
	defer Free(d_src)
	defer Free(d_dst)
	
	// Test H2D copy
	err := Memcpy(d_src, h_src, N*4, MemcpyHostToDevice)
	if err != nil {
		t.Fatalf("H2D copy failed: %v", err)
	}
	
	// Test D2D copy
	err = Memcpy(d_dst, d_src, N*4, MemcpyDeviceToDevice)
	if err != nil {
		t.Fatalf("D2D copy failed: %v", err)
	}
	
	// Test D2H copy
	err = Memcpy(h_dst, d_dst, N*4, MemcpyDeviceToHost)
	if err != nil {
		t.Fatalf("D2H copy failed: %v", err)
	}
	
	// Verify data
	for i := 0; i < N; i++ {
		if math.Abs(float64(h_src[i]-h_dst[i])) > 1e-6 {
			t.Errorf("Data mismatch at index %d: %f vs %f", i, h_src[i], h_dst[i])
		}
	}
}

// Test basic kernel launch
func TestKernelLaunch(t *testing.T) {
	const N = 10000
	
	// Allocate memory
	d_data, _ := Malloc(N * 4)
	defer Free(d_data)
	
	// Initialize to zero
	slice := d_data.Float32()
	for i := 0; i < N; i++ {
		slice[i] = 0
	}
	
	// Launch kernel to set values
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < N {
			slice[idx] = float32(idx)
		}
	})
	
	err := Launch(kernel, Dim3{X: (N+255)/256, Y: 1, Z: 1}, Dim3{X: 256, Y: 1, Z: 1})
	if err != nil {
		t.Fatalf("Kernel launch failed: %v", err)
	}
	
	err = Synchronize()
	if err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
	
	// Verify results
	for i := 0; i < N; i++ {
		if slice[i] != float32(i) {
			t.Errorf("Incorrect value at index %d: expected %f, got %f", i, float32(i), slice[i])
		}
	}
}

// Test vector operations
func TestVectorOperations(t *testing.T) {
	const N = 10000
	
	// Create test data
	h_A := make([]float32, N)
	h_B := make([]float32, N)
	for i := 0; i < N; i++ {
		h_A[i] = rand.Float32()
		h_B[i] = rand.Float32()
	}
	
	// Allocate device memory
	d_A, _ := Malloc(N * 4)
	d_B, _ := Malloc(N * 4)
	d_C, _ := Malloc(N * 4)
	defer Free(d_A)
	defer Free(d_B)
	defer Free(d_C)
	
	// Copy data
	Memcpy(d_A, h_A, N*4, MemcpyHostToDevice)
	Memcpy(d_B, h_B, N*4, MemcpyHostToDevice)
	
	// Test Add
	err := Add(d_A, d_B, d_C, N)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	Synchronize()
	
	// Verify
	result := d_C.Float32()
	for i := 0; i < N; i++ {
		expected := h_A[i] + h_B[i]
		if math.Abs(float64(result[i]-expected)) > 1e-5 {
			t.Errorf("Add mismatch at %d: expected %f, got %f", i, expected, result[i])
			break
		}
	}
	
	// Test AXPY
	alpha := float32(2.5)
	err = AXPY(alpha, d_A, d_B, N) // B = alpha*A + B
	if err != nil {
		t.Fatalf("AXPY failed: %v", err)
	}
	Synchronize()
	
	// Verify AXPY
	result2 := d_B.Float32()
	for i := 0; i < N; i++ {
		expected := alpha*h_A[i] + h_B[i]
		if math.Abs(float64(result2[i]-expected)) > 1e-5 {
			t.Errorf("AXPY mismatch at %d: expected %f, got %f", i, expected, result2[i])
			break
		}
	}
}

// Test fusion
func TestFusedOperations(t *testing.T) {
	const N = 1000
	
	// Create test data
	h_X := make([]float32, N)
	h_Bias := make([]float32, N)
	for i := 0; i < N; i++ {
		h_X[i] = rand.Float32()*2 - 1
		h_Bias[i] = rand.Float32() * 0.1
	}
	
	// Allocate device memory
	d_X, _ := Malloc(N * 4)
	d_Bias, _ := Malloc(N * 4)
	defer Free(d_X)
	defer Free(d_Bias)
	
	// Copy data
	Memcpy(d_X, h_X, N*4, MemcpyHostToDevice)
	Memcpy(d_Bias, h_Bias, N*4, MemcpyHostToDevice)
	
	// Test fused kernel: y = ReLU(2*x + bias)
	err := NewFusedKernel().
		MulScalar(2.0).
		Add(1.0).
		ReLU().
		Execute(d_X, []DevicePtr{d_Bias}, N)
	
	if err != nil {
		t.Fatalf("Fused kernel failed: %v", err)
	}
	Synchronize()
	
	// Verify
	result := d_X.Float32()
	for i := 0; i < N; i++ {
		expected := 2.0*h_X[i] + h_Bias[i]
		if expected < 0 {
			expected = 0
		}
		if math.Abs(float64(result[i]-expected)) > 1e-5 {
			t.Errorf("Fused mismatch at %d: expected %f, got %f", i, expected, result[i])
			break
		}
	}
}

// Benchmark vector addition
func BenchmarkVectorAdd(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	
	for _, N := range sizes {
		b.Run(fmt.Sprintf("Size_%d", N), func(b *testing.B) {
			// Allocate
			d_A, _ := Malloc(N * 4)
			d_B, _ := Malloc(N * 4)
			d_C, _ := Malloc(N * 4)
			defer Free(d_A)
			defer Free(d_B)
			defer Free(d_C)
			
			b.ResetTimer()
			b.SetBytes(int64(3 * N * 4)) // Read A, B, Write C
			
			for i := 0; i < b.N; i++ {
				Add(d_A, d_B, d_C, N)
				Synchronize()
			}
		})
	}
}

// Benchmark kernel fusion benefit
func BenchmarkFusion(b *testing.B) {
	const N = 1000000
	
	// Allocate
	d_X, _ := Malloc(N * 4)
	d_Bias, _ := Malloc(N * 4)
	d_Temp, _ := Malloc(N * 4)
	defer Free(d_X)
	defer Free(d_Bias)
	defer Free(d_Temp)
	
	b.Run("Separate", func(b *testing.B) {
		b.SetBytes(int64(6 * N * 4)) // 3 reads + 3 writes
		
		for i := 0; i < b.N; i++ {
			// Copy X to Temp
			Memcpy(d_Temp, d_X, N*4, MemcpyDeviceToDevice)
			// Scale
			Scale(2.0, d_Temp, N)
			// Add bias
			AXPY(1.0, d_Bias, d_Temp, N)
			// ReLU
			ReLU(d_Temp, N)
			Synchronize()
		}
	})
	
	b.Run("Fused", func(b *testing.B) {
		b.SetBytes(int64(3 * N * 4)) // 2 reads + 1 write
		
		for i := 0; i < b.N; i++ {
			// Copy X to Temp
			Memcpy(d_Temp, d_X, N*4, MemcpyDeviceToDevice)
			// Fused operation
			NewFusedKernel().
				MulScalar(2.0).
				Add(1.0).
				ReLU().
				Execute(d_Temp, []DevicePtr{d_Bias}, N)
			Synchronize()
		}
	})
}


// Test error conditions
func TestErrorHandling(t *testing.T) {
	// Test double free
	ptr, _ := Malloc(100)
	err := Free(ptr)
	if err != nil {
		t.Fatalf("First free failed: %v", err)
	}
	
	err = Free(ptr)
	if err == nil {
		t.Error("Double free should have failed")
	}
	
	// Test invalid device
	err = SetDevice(1)
	if err == nil {
		t.Error("SetDevice(1) should have failed")
	}
	
	// Test device count
	count := GetDeviceCount()
	if count != 1 {
		t.Errorf("Expected 1 device, got %d", count)
	}
}

// Test memory pool statistics
func TestMemoryPoolStats(t *testing.T) {
	// Get initial stats
	allocated1, _ := defaultContext.memory.GetStats()
	
	// Allocate some memory
	ptrs := make([]DevicePtr, 10)
	for i := range ptrs {
		ptrs[i], _ = Malloc(1024 * 1024) // 1MB each
	}
	
	// Check stats increased
	allocated2, peak2 := defaultContext.memory.GetStats()
	if allocated2 <= allocated1 {
		t.Error("Allocated memory should have increased")
	}
	if peak2 < allocated2 {
		t.Error("Peak should be at least current allocation")
	}
	
	// Free half
	for i := 0; i < 5; i++ {
		Free(ptrs[i])
	}
	
	// Check allocated decreased but peak unchanged
	allocated3, peak3 := defaultContext.memory.GetStats()
	if allocated3 >= allocated2 {
		t.Error("Allocated memory should have decreased")
	}
	if peak3 != peak2 {
		t.Error("Peak should not have changed")
	}
	
	// Clean up
	for i := 5; i < 10; i++ {
		Free(ptrs[i])
	}
}