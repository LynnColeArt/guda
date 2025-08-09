//go:build amd64
// +build amd64

package amx

import (
	"testing"
)

func TestTileConfig(t *testing.T) {
	cfg := ConfigureInt8GEMM()
	
	// Validate configuration
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("Invalid configuration: %v", err)
	}
	
	// Check specific values
	if cfg.Palette != 1 {
		t.Errorf("Expected palette 1, got %d", cfg.Palette)
	}
	
	// Check A tiles (16×64)
	if cfg.Rows[0] != 16 || cfg.ColsB[0] != 64 {
		t.Errorf("Tile 0 incorrect: %d×%d", cfg.Rows[0], cfg.ColsB[0])
	}
	
	// Check C tiles (16×16 INT32 = 16×64 bytes)
	if cfg.Rows[4] != 16 || cfg.ColsB[4] != 64 {
		t.Errorf("Tile 4 incorrect: %d×%d", cfg.Rows[4], cfg.ColsB[4])
	}
}

func TestAMXKernelSmall(t *testing.T) {
	// Skip if not in test mode
	if !HasAMX() {
		SetAMXSupport(true, true, true)
		if !HasAMX() {
			t.Skip("AMX not available")
		}
	}
	
	kernel := NewAMXKernel()
	defer kernel.Release()
	
	// Test 16×16×64 matrix multiply
	M, N, K := 16, 16, 64
	
	A := make([]byte, M*K)
	B := make([]byte, K*N)
	C := make([]int32, M*N)
	
	// Initialize with simple pattern
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 7)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte(i % 5)
	}
	
	// Execute kernel
	err := kernel.Int8GEMM(M, N, K, A, B, C, 1.0, 0.0)
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
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
}

func TestPackingFunctions(t *testing.T) {
	// Test PackA
	M, K := 32, 128
	A := make([]int8, M*K)
	packedA := make([]int8, M*K+1024) // Extra space for padding
	
	// Fill with pattern
	for i := 0; i < M*K; i++ {
		A[i] = int8(i % 127)
	}
	
	PackA(M, K, A, packedA)
	
	// Verify first tile
	if packedA[0] != A[0] {
		t.Error("First element of packed A incorrect")
	}
	
	// Test PackB
	N := 64
	B := make([]int8, K*N)
	packedB := make([]int8, K*N+1024)
	
	for i := 0; i < K*N; i++ {
		B[i] = int8(i % 127)
	}
	
	PackB(K, N, B, packedB)
	
	// Verify first element
	if packedB[0] != B[0] {
		t.Error("First element of packed B incorrect")
	}
}

func BenchmarkAMXKernel_64x64(b *testing.B) {
	if !HasAMX() {
		SetAMXSupport(true, true, true)
	}
	
	kernel := NewAMXKernel()
	defer kernel.Release()
	
	M, N, K := 64, 64, 64
	
	A := makeAlignedBuffer(M*K, 64)
	B := makeAlignedBuffer(K*N, 64)
	C32 := make([]int32, M*N)
	
	// Initialize
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 127)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte(i % 127)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := kernel.Int8GEMM(M, N, K, A, B, C32, 1.0, 0.0)
		if err != nil {
			b.Fatal(err)
		}
	}
	
	ops := int64(2 * M * N * K)
	b.SetBytes(int64(M*K + K*N + M*N*4))
	b.ReportMetric(float64(ops*int64(b.N))/b.Elapsed().Seconds()/1e9, "GOPS")
}