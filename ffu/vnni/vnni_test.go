package vnni

import (
	"fmt"
	"testing"
	
	"github.com/LynnColeArt/guda/ffu"
)

func TestVNNI32x32(t *testing.T) {
	t.Skip("Skipping VNNI 32x32 test - EVEX encoding not yet implemented")
	
	if !HasVNNI() {
		t.Skip("VNNI not available")
	}
	
	// Create 32x32 test matrices
	M, N, K := 32, 32, 32
	A := make([]int8, M*K)
	B := make([]int8, K*N)
	C := make([]int32, M*N)
	Cref := make([]int32, M*N)
	
	// Initialize with simple pattern
	for i := range A {
		A[i] = int8(i % 7)
	}
	for i := range B {
		B[i] = int8(i % 5)
	}
	
	// Compute reference result
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := int32(0)
			for k := 0; k < K; k++ {
				sum += int32(A[i*K+k]) * int32(B[k*N+j])
			}
			Cref[i*N+j] = sum
		}
	}
	
	// Test VNNI implementation
	vnniInt8GEMM32x32(A, B, C)
	
	// Compare results
	maxDiff := int32(0)
	for i := range C {
		diff := C[i] - Cref[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	
	if maxDiff > 0 {
		t.Errorf("VNNI result differs from reference, max diff: %d", maxDiff)
		// Print first few differences
		count := 0
		for i := range C {
			if C[i] != Cref[i] && count < 10 {
				t.Errorf("  C[%d] = %d, expected %d", i, C[i], Cref[i])
				count++
			}
		}
	}
}

func TestVNNIFFU(t *testing.T) {
	vnniFFU := NewVNNIFFU()
	
	if vnniFFU.Name() != "AVX512-VNNI" {
		t.Errorf("Expected name AVX512-VNNI, got %s", vnniFFU.Name())
	}
	
	if vnniFFU.Type() != ffu.FFUTypeAVX512VNNI {
		t.Errorf("Expected type AVX512VNNI, got %v", vnniFFU.Type())
	}
	
	// Check if VNNI is available on this machine
	t.Logf("VNNI available: %v", vnniFFU.IsAvailable())
	t.Logf("BF16 available: %v", HasBF16())
}

func TestVNNIExecution(t *testing.T) {
	// Enable VNNI for testing
	SetVNNISupport(true, false)
	vnniFFU := NewVNNIFFU()
	
	// Test small matrix multiplication
	M, N, K := 64, 64, 64
	
	A := make([]int8, M*K)
	B := make([]int8, K*N)
	C := make([]int32, M*N)
	
	// Initialize with simple pattern
	for i := range A {
		A[i] = int8(i % 7)
	}
	for i := range B {
		B[i] = int8(i % 5)
	}
	
	workload := &ffu.VNNIWorkload{
		Operation: ffu.VNNIMatMul,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		Alpha:     1,
	}
	
	// Execute
	err := vnniFFU.Execute(workload)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}
	
	// Verify result is not all zeros
	allZero := true
	for _, v := range C {
		if v != 0 {
			allZero = false
			break
		}
	}
	
	if allZero {
		t.Error("Result matrix is all zeros")
	}
	
	// Check specific value for correctness
	// C[0,0] = sum(A[0,k] * B[k,0]) for k=0..63
	expected := int32(0)
	for k := 0; k < K; k++ {
		expected += int32(A[k]) * int32(B[k*N])
	}
	
	if C[0] != expected {
		t.Errorf("C[0,0] = %d, expected %d", C[0], expected)
	}
}

func TestVNNICanHandle(t *testing.T) {
	SetVNNISupport(true, false)
	vnniFFU := NewVNNIFFU()
	
	tests := []struct {
		name     string
		workload ffu.Workload
		expected bool
	}{
		{
			name: "Valid VNNI workload",
			workload: &ffu.VNNIWorkload{
				M: 64, N: 64, K: 64,
				A: make([]int8, 64*64),
				B: make([]int8, 64*64),
				C: make([]int32, 64*64),
			},
			expected: true,
		},
		{
			name: "K not divisible by 16",
			workload: &ffu.VNNIWorkload{
				M: 64, N: 64, K: 63,
				A: make([]int8, 64*63),
				B: make([]int8, 63*64),
				C: make([]int32, 64*64),
			},
			expected: false,
		},
		{
			name: "K too small",
			workload: &ffu.VNNIWorkload{
				M: 64, N: 64, K: 32,
				A: make([]int8, 64*32),
				B: make([]int8, 32*64),
				C: make([]int32, 64*64),
			},
			expected: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := vnniFFU.CanHandle(tt.workload)
			if result != tt.expected {
				t.Errorf("CanHandle() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func BenchmarkVNNI_INT8_GEMM(b *testing.B) {
	// Check if VNNI is actually available
	if !HasVNNI() {
		SetVNNISupport(true, false) // Enable for testing
	}
	
	vnniFFU := NewVNNIFFU()
	
	sizes := []int{64, 128, 256, 512}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%dx%d", size, size), func(b *testing.B) {
			M, N, K := size, size, size
			
			A := make([]int8, M*K)
			B := make([]int8, K*N)
			C := make([]int32, M*N)
			
			// Initialize
			for i := range A {
				A[i] = int8(i % 127)
			}
			for i := range B {
				B[i] = int8(i % 127)
			}
			
			workload := &ffu.VNNIWorkload{
				Operation: ffu.VNNIMatMul,
				M:         M,
				N:         N,
				K:         K,
				A:         A,
				B:         B,
				C:         C,
				Alpha:     1,
			}
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				err := vnniFFU.Execute(workload)
				if err != nil {
					b.Fatal(err)
				}
			}
			
			// Report metrics
			ops := int64(2 * M * N * K)
			b.SetBytes(int64(M*K + K*N + M*N*4))
			b.ReportMetric(float64(ops*int64(b.N))/b.Elapsed().Seconds()/1e9, "GOPS")
		})
	}
}

