package ffu

import (
	"fmt"
)

// AMXDataType represents the data type for AMX operations
type AMXDataType int

const (
	AMXInt8 AMXDataType = iota
	AMXBFloat16
	AMXFloat16 // Future
)

func (t AMXDataType) String() string {
	switch t {
	case AMXInt8:
		return "INT8"
	case AMXBFloat16:
		return "BF16"
	case AMXFloat16:
		return "FP16"
	default:
		return "Unknown"
	}
}

// AMXOperation represents the type of AMX operation
type AMXOperation int

const (
	AMXMatMul AMXOperation = iota
	AMXMatMulAccumulate
)

// AMXWorkload represents a matrix operation workload for AMX
type AMXWorkload struct {
	Operation AMXOperation
	DataType  AMXDataType
	
	// Matrix dimensions
	M int // Rows of A and C
	N int // Columns of B and C
	K int // Columns of A, Rows of B
	
	// Matrix data
	A []byte // M×K matrix (row-major)
	B []byte // K×N matrix (row-major)
	C []byte // M×N matrix (row-major)
	
	// Scaling factors for quantized operations
	ScaleA float32 // Scale factor for A
	ScaleB float32 // Scale factor for B
	ScaleC float32 // Scale factor for C
	
	// For INT8: C = ScaleC * (ScaleA * A) × (ScaleB * B)
	// Result is typically INT32 that needs to be scaled back
}

// Type returns the workload type
func (w *AMXWorkload) Type() string {
	return fmt.Sprintf("amx_%s_matmul", w.DataType)
}

// Size returns the size of the workload in bytes
func (w *AMXWorkload) Size() int64 {
	// Total bytes processed (read A, B and write C)
	var sizeA, sizeB, sizeC int64
	
	switch w.DataType {
	case AMXInt8:
		sizeA = int64(w.M * w.K)     // INT8 input
		sizeB = int64(w.K * w.N)     // INT8 input
		sizeC = int64(w.M * w.N * 4) // INT32 output
	case AMXBFloat16, AMXFloat16:
		sizeA = int64(w.M * w.K * 2) // BF16/FP16 input
		sizeB = int64(w.K * w.N * 2) // BF16/FP16 input
		sizeC = int64(w.M * w.N * 2) // BF16/FP16 output
	}
	
	return sizeA + sizeB + sizeC
}

// Validate checks if the workload is valid
func (w *AMXWorkload) Validate() error {
	// Check dimensions
	if w.M <= 0 || w.N <= 0 || w.K <= 0 {
		return fmt.Errorf("invalid dimensions: M=%d, N=%d, K=%d", w.M, w.N, w.K)
	}
	
	// Check data buffers
	expectedSizeA := w.M * w.K
	expectedSizeB := w.K * w.N
	expectedSizeC := w.M * w.N
	
	if w.DataType == AMXBFloat16 || w.DataType == AMXFloat16 {
		expectedSizeA *= 2
		expectedSizeB *= 2
		expectedSizeC *= 2
	} else if w.DataType == AMXInt8 {
		// For INT8, C matrix is INT32 (4 bytes)
		expectedSizeC *= 4
	}
	
	if len(w.A) < expectedSizeA {
		return fmt.Errorf("A buffer too small: %d < %d", len(w.A), expectedSizeA)
	}
	if len(w.B) < expectedSizeB {
		return fmt.Errorf("B buffer too small: %d < %d", len(w.B), expectedSizeB)
	}
	if len(w.C) < expectedSizeC {
		return fmt.Errorf("C buffer too small: %d < %d", len(w.C), expectedSizeC)
	}
	
	// AMX has specific alignment requirements (16-byte aligned)
	// In practice, we'd check pointer alignment here
	
	// AMX tile sizes have limits (max 16 rows, 64 columns for INT8)
	// These would be checked in the actual kernel
	
	return nil
}

// Operations returns the number of operations
func (w *AMXWorkload) Operations() int64 {
	// For matrix multiply: 2*M*N*K operations (multiply + add)
	return int64(2 * w.M * w.N * w.K)
}

// AMXCapability describes AMX-specific capabilities
type AMXCapability struct {
	// Supported data types
	SupportsInt8  bool
	SupportsBF16  bool
	SupportsFP16  bool
	
	// Tile configuration
	NumTiles      int
	MaxTileRows   int
	MaxTileCols   int
	MaxTileBytes  int
	
	// Performance characteristics
	PeakInt8TOPS  float64 // Peak INT8 operations per second
	PeakBF16TFLOPS float64 // Peak BF16 operations per second
	
	// Constraints
	RequiresAlignment int // Byte alignment requirement
}