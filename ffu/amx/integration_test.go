package amx

import (
	"testing"
	"unsafe"

	"github.com/LynnColeArt/guda/ffu"
)

func TestAMXIntegration(t *testing.T) {
	// Enable AMX for testing
	SetAMXSupport(true, true, true)
	
	// Create registry and register AMX
	registry := ffu.NewRegistry()
	amxFFU := NewAMXFFU()
	
	if err := registry.Register(amxFFU); err != nil {
		t.Fatalf("Failed to register AMX FFU: %v", err)
	}
	
	// Create a 128x128x128 INT8 workload
	M, N, K := 128, 128, 128
	
	A := makeAlignedBuffer(M*K, 16)
	B := makeAlignedBuffer(K*N, 16)
	C := makeAlignedBuffer(M*N*4, 16) // INT32 output
	
	// Initialize with pattern
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 63)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte(i % 63)
	}
	
	workload := &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		ScaleA:    1.0,
		ScaleB:    1.0,
		ScaleC:    1.0,
	}
	
	// Find best FFU
	bestFFU, cost := registry.FindBest(workload)
	if bestFFU == nil {
		t.Fatal("No suitable FFU found")
	}
	
	// Verify cost estimate is reasonable
	if cost == nil {
		t.Fatal("No cost estimate returned")
	}
	if cost.Duration <= 0 {
		t.Errorf("Invalid duration estimate: %v", cost.Duration)
	}
	
	if bestFFU.Name() != "Intel AMX" {
		t.Errorf("Expected Intel AMX, got %s", bestFFU.Name())
	}
	
	// Execute
	if err := bestFFU.Execute(workload); err != nil {
		t.Fatalf("Execution failed: %v", err)
	}
	
	// Verify output is not zero
	cInt32 := (*[1 << 30]int32)(unsafe.Pointer(&C[0]))[:M*N]
	allZero := true
	for i := 0; i < 10; i++ {
		if cInt32[i] != 0 {
			allZero = false
			break
		}
	}
	
	if allZero {
		t.Error("Output matrix appears to be all zeros")
	}
	
	// Check metrics
	metrics := bestFFU.Metrics()
	if metrics.WorkloadCount != 1 {
		t.Errorf("Expected 1 workload, got %d", metrics.WorkloadCount)
	}
	
	expectedBytes := int64(M*K + K*N + M*N*4)
	if metrics.BytesProcessed != expectedBytes {
		t.Errorf("Expected %d bytes processed, got %d", expectedBytes, metrics.BytesProcessed)
	}
}

func BenchmarkAMXInt8_256x256(b *testing.B) {
	SetAMXSupport(true, true, true)
	amxFFU := NewAMXFFU()
	
	M, N, K := 256, 256, 256
	
	A := makeAlignedBuffer(M*K, 16)
	B := makeAlignedBuffer(K*N, 16)
	C := makeAlignedBuffer(M*N*4, 16)
	
	// Initialize
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 127)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte(i % 127)
	}
	
	workload := &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		ScaleA:    1.0,
		ScaleB:    1.0,
		ScaleC:    1.0,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := amxFFU.Execute(workload); err != nil {
			b.Fatal(err)
		}
	}
	
	ops := int64(2 * M * N * K)
	b.SetBytes(int64(M*K + K*N + M*N*4))
	b.ReportMetric(float64(ops*int64(b.N))/b.Elapsed().Seconds()/1e9, "GOPS")
}