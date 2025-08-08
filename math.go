package guda

import (
	"github.com/LynnColeArt/guda/compute"
	"github.com/LynnColeArt/guda/compute/blas"
	"github.com/LynnColeArt/guda/floats"
)

// MathOp represents a mathematical operation
type MathOp int

const (
	OpAdd MathOp = iota
	OpSub
	OpMul
	OpDiv
	OpMax
	OpMin
)

// BLAS operations using optimized gonum kernels

// AXPY performs y = alpha*x + y
func AXPY(alpha float32, x, y DevicePtr, n int) error {
	xf32 := x.Float32()[:n]
	yf32 := y.Float32()[:n]
	
	// Use our assimilated SIMD-optimized Float32 AXPY
	compute.Implementation{}.Saxpy(n, alpha, xf32, 1, yf32, 1)
	
	return nil
}

// DOT computes the dot product of two vectors
func DOT(x, y DevicePtr, n int) (float32, error) {
	xf32 := x.Float32()[:n]
	yf32 := y.Float32()[:n]
	
	// Use our assimilated SIMD-optimized Float32 DOT
	result := compute.Implementation{}.Sdot(n, xf32, 1, yf32, 1)
	return result, nil
}

// GEMM performs C = alpha*A*B + beta*C
func GEMM(transA, transB bool, m, n, k int, alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int) error {
	
	// Convert to BLAS types
	var tA, tB blas.Transpose
	if transA {
		tA = blas.Trans
	} else {
		tA = blas.NoTrans
	}
	if transB {
		tB = blas.Trans
	} else {
		tB = blas.NoTrans
	}
	
	// Get float32 slices
	af32 := a.Float32()
	bf32 := b.Float32()
	cf32 := c.Float32()
	
	// Use our assimilated SIMD-optimized Float32 GEMM
	compute.Implementation{}.Sgemm(tA, tB, m, n, k,
		alpha, af32, lda,
		bf32, ldb,
		beta, cf32, ldc)
	
	return nil
}

// Element-wise operations

// Add performs element-wise addition: c = a + b
func Add(a, b, c DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			aSlice := a.Float32()
			bSlice := b.Float32()
			cSlice := c.Float32()
			cSlice[idx] = aSlice[idx] + bSlice[idx]
		}
	})
	
	return Launch(kernel, grid, block)
}

// Multiply performs element-wise multiplication: c = a * b
func Multiply(a, b, c DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			aSlice := a.Float32()
			bSlice := b.Float32()
			cSlice := c.Float32()
			cSlice[idx] = aSlice[idx] * bSlice[idx]
		}
	})
	
	return Launch(kernel, grid, block)
}

// Scale multiplies all elements by a scalar: x = alpha * x
func Scale(alpha float32, x DevicePtr, n int) error {
	xf32 := x.Float32()[:n]
	
	// Use our assimilated SIMD-optimized Float32 Scale
	compute.Implementation{}.Sscal(n, alpha, xf32, 1)
	
	return nil
}

// Reduction operations

// Sum computes the sum of all elements
func Sum(x DevicePtr, n int) (float32, error) {
	// Use gonum's optimized Sum for large vectors
	if n > 1000 {
		xf32 := x.Float32()[:n]
		xf64 := make([]float64, n)
		for i := 0; i < n; i++ {
			xf64[i] = float64(xf32[i])
		}
		
		result := floats.Sum(xf64)
		return float32(result), nil
	}
	
	// Simple reduction for small vectors
	xSlice := x.Float32()[:n]
	var sum float32
	for i := 0; i < n; i++ {
		sum += xSlice[i]
	}
	return sum, nil
}

// Max finds the maximum element
func Max(x DevicePtr, n int) (float32, error) {
	xSlice := x.Float32()[:n]
	if n == 0 {
		return 0, nil
	}
	
	max := xSlice[0]
	for i := 1; i < n; i++ {
		if xSlice[i] > max {
			max = xSlice[i]
		}
	}
	return max, nil
}

// Min finds the minimum element
func Min(x DevicePtr, n int) (float32, error) {
	xSlice := x.Float32()[:n]
	if n == 0 {
		return 0, nil
	}
	
	min := xSlice[0]
	for i := 1; i < n; i++ {
		if xSlice[i] < min {
			min = xSlice[i]
		}
	}
	return min, nil
}

// Activation functions

// ReLU applies the ReLU activation function: x = max(0, x)
func ReLU(x DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			xSlice := x.Float32()
			if xSlice[idx] < 0 {
				xSlice[idx] = 0
			}
		}
	})
	
	return Launch(kernel, grid, block)
}

// Sigmoid applies the sigmoid activation function
func Sigmoid(x DevicePtr, n int) error {
	grid := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	
	kernel := KernelFunc(func(tid ThreadID, args ...interface{}) {
		idx := tid.Global()
		if idx < n {
			xSlice := x.Float32()
			val := xSlice[idx]
			// Use accurate sigmoid implementation
			xSlice[idx] = SigmoidFloat32(val)
		}
	})
	
	return Launch(kernel, grid, block)
}