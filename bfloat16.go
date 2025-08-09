package guda

import (
	"math"
)

// BFloat16 represents a 16-bit brain floating point number
// Format: 1 sign bit, 8 exponent bits, 7 mantissa bits
type BFloat16 uint16

// ToBFloat16 converts float32 to BFloat16 (truncate mantissa)
func ToBFloat16(f float32) BFloat16 {
	// BFloat16 is just the top 16 bits of float32
	bits := math.Float32bits(f)
	
	// Round to nearest even
	rounding := (bits >> 16) & 1
	if (bits & 0x7FFF) > 0x7FFF || ((bits & 0x7FFF) == 0x7FFF && rounding == 1) {
		bits += 0x10000
	}
	
	return BFloat16(bits >> 16)
}

// ToFloat32 converts BFloat16 to float32
func (b BFloat16) ToFloat32() float32 {
	// Just shift back to float32 position
	return math.Float32frombits(uint32(b) << 16)
}

// BFloat16Slice wraps a byte slice as BFloat16 values
type BFloat16Slice struct {
	data []byte
}

// NewBFloat16Slice creates a BFloat16 slice from a byte slice
func NewBFloat16Slice(data []byte) BFloat16Slice {
	return BFloat16Slice{data: data}
}

// Len returns the number of BFloat16 elements
func (s BFloat16Slice) Len() int {
	return len(s.data) / 2
}

// Get returns the BFloat16 at index i
func (s BFloat16Slice) Get(i int) BFloat16 {
	return BFloat16(uint16(s.data[i*2]) | (uint16(s.data[i*2+1]) << 8))
}

// Set sets the BFloat16 at index i
func (s BFloat16Slice) Set(i int, val BFloat16) {
	s.data[i*2] = byte(val)
	s.data[i*2+1] = byte(val >> 8)
}

// GetFloat32 returns the value at index i as float32
func (s BFloat16Slice) GetFloat32(i int) float32 {
	return s.Get(i).ToFloat32()
}

// SetFloat32 sets the value at index i from float32
func (s BFloat16Slice) SetFloat32(i int, val float32) {
	s.Set(i, ToBFloat16(val))
}

// DevicePtr methods for BFloat16

// BFloat16 returns a BFloat16 slice view of the memory
func (d DevicePtr) BFloat16() BFloat16Slice {
	if d.ptr == nil {
		return BFloat16Slice{}
	}
	return NewBFloat16Slice(d.Byte())
}

// BFloat16 operations - these are much simpler than Float16
// because BFloat16 preserves the same exponent range as Float32

// AddBFloat16 performs element-wise addition on BFloat16 arrays
func AddBFloat16(a, b, c DevicePtr, n int) error {
	grid := Dim3{X: (n + DefaultBlockSize - 1) / DefaultBlockSize, Y: 1, Z: 1}
	block := Dim3{X: DefaultBlockSize, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			aSlice := a.BFloat16()
			bSlice := b.BFloat16()
			cSlice := c.BFloat16()
			
			// BFloat16 arithmetic is simpler - just convert and compute
			aVal := aSlice.GetFloat32(idx)
			bVal := bSlice.GetFloat32(idx)
			cSlice.SetFloat32(idx, aVal + bVal)
		}
	})
	
	return Launch(kernel, grid, block)
}

// GEMMBFloat16 performs matrix multiplication with BFloat16
// This is what modern AI accelerators use
func GEMMBFloat16(transA, transB bool, m, n, k int, alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int) error {
	
	// For AI workloads, we often accumulate in FP32 even with BF16 inputs
	// This matches TPU/GPU tensor core behavior
	
	grid := Dim3{X: (n + 63) / 64, Y: (m + 63) / 64, Z: 1}
	block := Dim3{X: 8, Y: 8, Z: 1} // 64 threads per block
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		// Simple tiled matrix multiply
		tx := tid.ThreadIdx.X
		ty := tid.ThreadIdx.Y
		bx := tid.BlockIdx.X
		by := tid.BlockIdx.Y
		
		// Calculate global indices
		col := bx*64 + tx*8
		row := by*64 + ty*8
		
		if row >= m || col >= n {
			return
		}
		
		aSlice := a.BFloat16()
		bSlice := b.BFloat16()
		cSlice := c.BFloat16()
		
		// Accumulate in float32 for accuracy
		var sum [8][8]float32
		
		// Initialize with C if beta != 0
		if beta != 0 {
			for i := 0; i < 8 && row+i < m; i++ {
				for j := 0; j < 8 && col+j < n; j++ {
					sum[i][j] = beta * cSlice.GetFloat32((row+i)*ldc + col + j)
				}
			}
		}
		
		// Compute dot product
		for kk := 0; kk < k; kk++ {
			// In production, we'd tile this for cache efficiency
			for i := 0; i < 8 && row+i < m; i++ {
				for j := 0; j < 8 && col+j < n; j++ {
					aVal := aSlice.GetFloat32((row+i)*lda + kk)
					bVal := bSlice.GetFloat32(kk*ldb + col + j)
					sum[i][j] += alpha * aVal * bVal
				}
			}
		}
		
		// Write results back as BFloat16
		for i := 0; i < 8 && row+i < m; i++ {
			for j := 0; j < 8 && col+j < n; j++ {
				cSlice.SetFloat32((row+i)*ldc + col + j, sum[i][j])
			}
		}
	})
	
	return Launch(kernel, grid, block)
}

// Mixed precision operations common in AI

// MixedPrecisionLinear performs FP32 accumulation with BF16 storage
// This is the key to modern AI training efficiency
func MixedPrecisionLinear(x, weights, bias, output DevicePtr, 
	batchSize, inputDim, outputDim int) error {
	
	// x: [batchSize, inputDim] in BF16
	// weights: [inputDim, outputDim] in BF16  
	// bias: [outputDim] in BF16
	// output: [batchSize, outputDim] in BF16
	
	// Use BF16 GEMM with FP32 accumulation
	err := GEMMBFloat16(false, false, batchSize, outputDim, inputDim,
		1.0, x, inputDim, weights, outputDim,
		0.0, output, outputDim)
	if err != nil {
		return err
	}
	
	// Add bias
	grid := Dim3{X: (batchSize*outputDim + DefaultBlockSize - 1) / DefaultBlockSize, Y: 1, Z: 1}
	block := Dim3{X: DefaultBlockSize, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx >= batchSize*outputDim {
			return
		}
		
		// batch := idx / outputDim  // Not used in this simple version
		col := idx % outputDim
		
		outputSlice := output.BFloat16()
		biasSlice := bias.BFloat16()
		
		// Add bias
		val := outputSlice.GetFloat32(idx) + biasSlice.GetFloat32(col)
		outputSlice.SetFloat32(idx, val)
	})
	
	return Launch(kernel, grid, block)
}