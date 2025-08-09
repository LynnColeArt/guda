package ffu

import (
	"fmt"
)

// VNNIOperation represents the type of VNNI operation
type VNNIOperation int

const (
	VNNIDotProduct VNNIOperation = iota
	VNNIMatMul
)

// VNNIWorkload represents a VNNI (Vector Neural Network Instructions) workload
type VNNIWorkload struct {
	Operation VNNIOperation
	M, N, K   int    // Matrix dimensions
	A         []int8  // M×K matrix
	B         []int8  // K×N matrix  
	C         []int32 // M×N matrix (output)
	Alpha     int32   // Scaling factor
}

// Type returns the workload type
func (w *VNNIWorkload) Type() string {
	return "vnni_int8_matmul"
}

// Size returns the size of the workload in bytes
func (w *VNNIWorkload) Size() int64 {
	// Input: A + B, Output: C
	return int64(w.M*w.K + w.K*w.N + w.M*w.N*4)
}

// Validate checks if the workload is valid
func (w *VNNIWorkload) Validate() error {
	if w.M <= 0 || w.N <= 0 || w.K <= 0 {
		return fmt.Errorf("invalid dimensions: M=%d, N=%d, K=%d", w.M, w.N, w.K)
	}
	
	if len(w.A) < w.M*w.K {
		return fmt.Errorf("A buffer too small: %d < %d", len(w.A), w.M*w.K)
	}
	
	if len(w.B) < w.K*w.N {
		return fmt.Errorf("B buffer too small: %d < %d", len(w.B), w.K*w.N)
	}
	
	if len(w.C) < w.M*w.N {
		return fmt.Errorf("C buffer too small: %d < %d", len(w.C), w.M*w.N)
	}
	
	// VNNI works best with dimensions divisible by 16 (AVX512 width)
	if w.K%16 != 0 {
		return fmt.Errorf("K dimension should be divisible by 16 for optimal VNNI, got %d", w.K)
	}
	
	return nil
}

// Operations returns the number of operations
func (w *VNNIWorkload) Operations() int64 {
	return int64(2 * w.M * w.N * w.K) // multiply-add
}