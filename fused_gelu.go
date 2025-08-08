package guda

import (
	"github.com/LynnColeArt/guda/blas"
	"github.com/LynnColeArt/guda/blas/blas32"
	"github.com/LynnColeArt/guda/compute/asm/f32"
)

// AddBiasGELU performs x = GELU(x + bias) in one pass
// GELU(x) = x * Φ(x) where Φ is CDF of standard normal
// We use the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
func AddBiasGELU(x, bias DevicePtr, n int) error {
	biasLen := bias.size / 4 // number of float32s in bias
	
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx >= n {
			return
		}
		
		xSlice := x.Float32()
		biasSlice := bias.Float32()
		
		// Broadcast bias: bias_index = idx % output_dim
		biasIdx := idx % biasLen
		
		// Add bias
		val := xSlice[idx] + biasSlice[biasIdx]
		
		// Apply GELU activation
		xSlice[idx] = geluFloat32(val)
	})
	
	return Launch(kernel, grid, block)
}

// GELU activation uses centralized constants from math_constants.go

// geluFloat32 computes GELU activation for a single float32
// Uses the tanh approximation for consistency with AVX2 version
func geluFloat32(x float32) float32 {
	// Compute x^3
	x3 := x * x * x
	
	// Compute argument to tanh
	arg := GELUSqrt2OverPi * (x + GELUCoefficient*x3)
	
	// Use the accurate tanh implementation
	tanhVal := TanhFloat32(arg)
	
	// Final GELU computation
	return 0.5 * x * (1 + tanhVal)
}

// geluFloat32Accurate provides the most accurate GELU using erf
// For reference/testing purposes
func geluFloat32Accurate(x float32) float32 {
	return GeluFloat32Accurate(x)
}

// FusedGEMMBiasGELU performs C = GELU(alpha*A*B + bias) in a fused operation
func FusedGEMMBiasGELU(
	transA, transB bool,
	m, n, k int,
	alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int,
	bias DevicePtr,
) error {
	// For now, we don't support transposed matrices or beta != 0
	if transA || transB {
		return ErrNotSupported
	}
	if beta != 0 {
		return ErrNotSupported
	}
	
	// Get the actual slices from DevicePtr
	aSlice := a.Float32()
	bSlice := b.Float32()
	cSlice := c.Float32()
	biasSlice := bias.Float32()
	
	// Use our optimized fused implementation
	fusedGEMMBiasGELUOptimized(m, n, k, alpha, aSlice, lda, bSlice, ldb, biasSlice, cSlice, ldc)
	
	return nil
}

// fusedGEMMBiasGELUOptimized performs the actual fused computation
func fusedGEMMBiasGELUOptimized(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, bias []float32, c []float32, ldc int) {
	// First, do the GEMM with our optimized gonum BLAS
	blas32.Gemm(blas.NoTrans, blas.NoTrans, alpha, blas32.General{
		Rows: m, Cols: k, Data: a, Stride: lda,
	}, blas32.General{
		Rows: k, Cols: n, Data: b, Stride: ldb,
	}, 0, blas32.General{
		Rows: m, Cols: n, Data: c, Stride: ldc,
	})
	
	// Apply bias and GELU
	if hasAVX2 {
		// Process column by column for better cache locality
		for j := 0; j < n; j++ {
			biasVal := bias[j]
			
			// Create a temporary slice for this column
			col := make([]float32, m)
			for i := 0; i < m; i++ {
				col[i] = c[i*ldc+j] + biasVal
			}
			
			// Apply GELU using AVX2
			f32.GeluAVX2(col)
			
			// Copy back
			for i := 0; i < m; i++ {
				c[i*ldc+j] = col[i]
			}
		}
	} else {
		// Fallback to scalar implementation
		for j := 0; j < n; j++ {
			biasVal := bias[j]
			for i := 0; i < m; i++ {
				idx := i*ldc + j
				val := c[idx] + biasVal
				c[idx] = geluFloat32(val)
			}
		}
	}
}