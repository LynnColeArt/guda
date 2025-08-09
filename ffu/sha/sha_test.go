package sha

import (
	"crypto/sha256"
	"fmt"
	"testing"
	
	"github.com/LynnColeArt/guda/ffu"
)

func TestSHAFFU(t *testing.T) {
	shaFFU := NewSHAFFU()
	
	// Check properties
	if shaFFU.Name() != "SHA-NI" {
		t.Errorf("Expected name SHA-NI, got %s", shaFFU.Name())
	}
	
	if shaFFU.Type() != ffu.FFUTypeSHA {
		t.Errorf("Expected type SHA, got %v", shaFFU.Type())
	}
}

func TestSHAExecution(t *testing.T) {
	shaFFU := NewSHAFFU()
	
	// Test data
	data := []byte("The quick brown fox jumps over the lazy dog")
	output := make([]byte, 32) // SHA-256 output
	
	workload := &ffu.SHAWorkload{
		Algorithm: ffu.SHA256,
		Data:      data,
		Output:    output,
	}
	
	// Execute
	err := shaFFU.Execute(workload)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}
	
	// Verify result
	expected := sha256.Sum256(data)
	for i := 0; i < 32; i++ {
		if output[i] != expected[i] {
			t.Errorf("Output mismatch at byte %d: got %x, want %x", i, output[i], expected[i])
		}
	}
	
	// Check metrics
	metrics := shaFFU.Metrics()
	if metrics.WorkloadCount != 1 {
		t.Errorf("Expected 1 workload, got %d", metrics.WorkloadCount)
	}
	if metrics.BytesProcessed != int64(len(data)) {
		t.Errorf("Expected %d bytes processed, got %d", len(data), metrics.BytesProcessed)
	}
}

func BenchmarkSHA256_Software(b *testing.B) {
	sizes := []int{64, 1024, 8192, 65536, 1048576} // 64B to 1MB
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			data := make([]byte, size)
			for i := range data {
				data[i] = byte(i)
			}
			
			b.SetBytes(int64(size))
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				_ = sha256.Sum256(data)
			}
		})
	}
}

func BenchmarkSHA256_FFU(b *testing.B) {
	shaFFU := NewSHAFFU()
	sizes := []int{64, 1024, 8192, 65536, 1048576}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			data := make([]byte, size)
			output := make([]byte, 32)
			for i := range data {
				data[i] = byte(i)
			}
			
			workload := &ffu.SHAWorkload{
				Algorithm: ffu.SHA256,
				Data:      data,
				Output:    output,
			}
			
			b.SetBytes(int64(size))
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				err := shaFFU.Execute(workload)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

