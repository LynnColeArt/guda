package guda

import (
	"math"
)

// Float16 represents a 16-bit floating point number
type Float16 uint16

// Float16 conversion constants
const (
	float16SignMask     = 0x8000
	float16ExponentMask = 0x7C00
	float16MantissaMask = 0x03FF
	float16ExponentBias = 15
	float16MantissaBits = 10
	float16ExponentBits = 5
)

// ToFloat32 converts Float16 to float32
func (f Float16) ToFloat32() float32 {
	sign := uint32(f&float16SignMask) << 16
	exponent := (f & float16ExponentMask) >> float16MantissaBits
	mantissa := f & float16MantissaMask

	if exponent == 0 {
		// Subnormal or zero
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal - normalize it
		exp := uint32(1)
		for mantissa&0x200 == 0 {
			mantissa <<= 1
			exp++
		}
		mantissa &= 0x1FF
		exponentBits := 127 - 15 - uint16(exp) + 1
		return math.Float32frombits(sign | (uint32(exponentBits) << 23) | (uint32(mantissa) << 13))
	} else if exponent == 0x1F {
		// Infinity or NaN
		if mantissa == 0 {
			return math.Float32frombits(sign | 0x7F800000) // Infinity
		}
		return math.Float32frombits(sign | 0x7FC00000 | (uint32(mantissa) << 13)) // NaN
	}

	// Normal number
	return math.Float32frombits(sign | ((uint32(exponent)+127-15) << 23) | (uint32(mantissa) << 13))
}

// FromFloat32 converts float32 to Float16
func FromFloat32(f float32) Float16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & float16SignMask
	exponent := (bits >> 23) & 0xFF
	mantissa := bits & 0x7FFFFF

	// Handle special cases
	if exponent == 0xFF {
		// Infinity or NaN
		if mantissa == 0 {
			return Float16(sign | float16ExponentMask) // Infinity
		}
		return Float16(sign | float16ExponentMask | (mantissa >> 13)) // NaN
	}

	// Convert exponent
	exp := int(exponent) - 127 + float16ExponentBias
	
	if exp <= 0 {
		// Underflow to zero
		return Float16(sign)
	} else if exp >= 0x1F {
		// Overflow to infinity
		return Float16(sign | float16ExponentMask)
	}

	// Normal number
	return Float16(uint16(sign) | (uint16(exp) << float16MantissaBits) | uint16(mantissa>>13))
}

// Float16Slice wraps a byte slice as Float16 values
type Float16Slice struct {
	data []byte
}

// NewFloat16Slice creates a Float16 slice from a byte slice
func NewFloat16Slice(data []byte) Float16Slice {
	return Float16Slice{data: data}
}

// Len returns the number of Float16 elements
func (s Float16Slice) Len() int {
	return len(s.data) / 2
}

// Get returns the Float16 at index i
func (s Float16Slice) Get(i int) Float16 {
	return Float16(uint16(s.data[i*2]) | (uint16(s.data[i*2+1]) << 8))
}

// Set sets the Float16 at index i
func (s Float16Slice) Set(i int, val Float16) {
	s.data[i*2] = byte(val)
	s.data[i*2+1] = byte(val >> 8)
}

// GetFloat32 returns the value at index i as float32
func (s Float16Slice) GetFloat32(i int) float32 {
	return s.Get(i).ToFloat32()
}

// SetFloat32 sets the value at index i from float32
func (s Float16Slice) SetFloat32(i int, val float32) {
	s.Set(i, FromFloat32(val))
}

// DevicePtr methods for Float16

// Float16 returns a Float16 slice view of the memory
func (d DevicePtr) Float16() Float16Slice {
	if d.ptr == nil {
		return Float16Slice{}
	}
	return NewFloat16Slice(d.Byte())
}

// Float16 operations using SIMD where possible

// AddFloat16 performs element-wise addition on Float16 arrays
func AddFloat16(a, b, c DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			aSlice := a.Float16()
			bSlice := b.Float16()
			cSlice := c.Float16()
			
			// Convert to float32, compute, convert back
			// In production, we'd use SIMD F16C instructions
			aVal := aSlice.GetFloat32(idx)
			bVal := bSlice.GetFloat32(idx)
			cSlice.SetFloat32(idx, aVal + bVal)
		}
	})
	
	return Launch(kernel, grid, block)
}

// MultiplyFloat16 performs element-wise multiplication on Float16 arrays
func MultiplyFloat16(a, b, c DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			aSlice := a.Float16()
			bSlice := b.Float16()
			cSlice := c.Float16()
			
			aVal := aSlice.GetFloat32(idx)
			bVal := bSlice.GetFloat32(idx)
			cSlice.SetFloat32(idx, aVal * bVal)
		}
	})
	
	return Launch(kernel, grid, block)
}

// FusedFloat16 operations for neural networks

// LinearFloat16 performs y = alpha*x + beta with Float16
func LinearFloat16(x DevicePtr, alpha, beta float32, n int) error {
	alphaF16 := FromFloat32(alpha)
	betaF16 := FromFloat32(beta)
	
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			xSlice := x.Float16()
			
			// In production, this would use F16C SIMD instructions
			val := xSlice.GetFloat32(idx)
			result := alphaF16.ToFloat32()*val + betaF16.ToFloat32()
			xSlice.SetFloat32(idx, result)
		}
	})
	
	return Launch(kernel, grid, block)
}

// GEMMFloat16 performs matrix multiplication with Float16
func GEMMFloat16(transA, transB bool, m, n, k int, alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int) error {
	
	// For now, convert to float32 and use existing GEMM
	// In production, we'd implement native F16 GEMM with F16C instructions
	
	// Calculate sizes
	aSize := m * k
	bSize := k * n
	cSize := m * n
	
	// Allocate temp float32 buffers
	aF32, _ := Malloc(aSize * 4)
	bF32, _ := Malloc(bSize * 4)
	cF32, _ := Malloc(cSize * 4)
	defer Free(aF32)
	defer Free(bF32)
	defer Free(cF32)
	
	// Convert A and B to float32
	convertF16ToF32(a, aF32, aSize)
	convertF16ToF32(b, bF32, bSize)
	
	// If beta != 0, convert C as well
	if beta != 0 {
		convertF16ToF32(c, cF32, cSize)
	}
	
	// Perform GEMM in float32
	err := GEMM(transA, transB, m, n, k, alpha, aF32, lda, bF32, ldb, beta, cF32, ldc)
	if err != nil {
		return err
	}
	
	// Convert result back to float16
	convertF32ToF16(cF32, c, cSize)
	
	return nil
}

// Helper functions for bulk conversion

func convertF16ToF32(src, dst DevicePtr, n int) {
	srcF16 := src.Float16()
	dstF32 := dst.Float32()
	
	// In production, use SIMD F16C VCVTPH2PS instruction
	for i := 0; i < n; i++ {
		dstF32[i] = srcF16.GetFloat32(i)
	}
}

func convertF32ToF16(src, dst DevicePtr, n int) {
	srcF32 := src.Float32()
	dstF16 := dst.Float16()
	
	// In production, use SIMD F16C VCVTPS2PH instruction
	for i := 0; i < n; i++ {
		dstF16.SetFloat32(i, srcF32[i])
	}
}