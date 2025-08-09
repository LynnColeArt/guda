// Package guda reference implementations for verification
package guda

import (
	"math"
)

// Reference contains simple, correct implementations of all kernels
// These are used for testing and verification of optimized implementations
type Reference struct{}

// BLAS Level 1 Reference Implementations

// AXPYRef performs y = alpha*x + y (reference implementation)
func (r Reference) AXPY(alpha float32, x, y []float32) {
	for i := range x {
		y[i] = alpha*x[i] + y[i]
	}
}

// DOTRef computes dot product of x and y (reference implementation)
func (r Reference) DOT(x, y []float32) float32 {
	var sum float32
	for i := range x {
		sum += x[i] * y[i]
	}
	return sum
}

// ScaleRef performs x = alpha*x (reference implementation)
func (r Reference) Scale(alpha float32, x []float32) {
	for i := range x {
		x[i] *= alpha
	}
}

// NRM2Ref computes the 2-norm of vector x (reference implementation)
func (r Reference) NRM2(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	return float32(math.Sqrt(float64(sum)))
}

// ASUMRef computes the sum of absolute values (reference implementation)
func (r Reference) ASUM(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += float32(math.Abs(float64(v)))
	}
	return sum
}

// BLAS Level 2 Reference Implementations

// GEMVRef performs matrix-vector multiplication: y = alpha*A*x + beta*y
func (r Reference) GEMV(transA bool, m, n int, alpha float32, 
	a []float32, lda int, x []float32, incX int, 
	beta float32, y []float32, incY int) {
	
	if !transA {
		// y = alpha*A*x + beta*y (A is m x n)
		for i := 0; i < m; i++ {
			sum := float32(0)
			for j := 0; j < n; j++ {
				sum += a[i*lda+j] * x[j*incX]
			}
			y[i*incY] = alpha*sum + beta*y[i*incY]
		}
	} else {
		// y = alpha*A^T*x + beta*y (A^T is n x m)
		for j := 0; j < n; j++ {
			sum := float32(0)
			for i := 0; i < m; i++ {
				sum += a[i*lda+j] * x[i*incX]
			}
			y[j*incY] = alpha*sum + beta*y[j*incY]
		}
	}
}

// BLAS Level 3 Reference Implementations

// GEMMRef performs general matrix multiplication: C = alpha*A*B + beta*C
func (r Reference) GEMM(transA, transB bool, m, n, k int, alpha float32,
	a []float32, lda int, b []float32, ldb int,
	beta float32, c []float32, ldc int) {
	
	// Handle beta*C
	if beta != 1.0 {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] *= beta
			}
		}
	}
	
	// Compute alpha*A*B
	if !transA && !transB {
		// C = alpha*A*B + C
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a[i*lda+l] * b[l*ldb+j]
				}
				c[i*ldc+j] += alpha * sum
			}
		}
	} else if transA && !transB {
		// C = alpha*A^T*B + C
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a[l*lda+i] * b[l*ldb+j]
				}
				c[i*ldc+j] += alpha * sum
			}
		}
	} else if !transA && transB {
		// C = alpha*A*B^T + C
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a[i*lda+l] * b[j*ldb+l]
				}
				c[i*ldc+j] += alpha * sum
			}
		}
	} else {
		// C = alpha*A^T*B^T + C
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a[l*lda+i] * b[j*ldb+l]
				}
				c[i*ldc+j] += alpha * sum
			}
		}
	}
}

// Element-wise operations

// AddRef performs element-wise addition: c = a + b
func (r Reference) Add(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] + b[i]
	}
}

// SubRef performs element-wise subtraction: c = a - b
func (r Reference) Sub(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] - b[i]
	}
}

// MulRef performs element-wise multiplication: c = a * b
func (r Reference) Mul(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] * b[i]
	}
}

// DivRef performs element-wise division: c = a / b
func (r Reference) Div(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] / b[i]
	}
}

// Activation functions

// ReLURef applies ReLU activation: y = max(0, x)
func (r Reference) ReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// SigmoidRef applies sigmoid activation: y = 1 / (1 + exp(-x))
func (r Reference) Sigmoid(x []float32) {
	for i := range x {
		x[i] = 1.0 / (1.0 + float32(math.Exp(float64(-x[i]))))
	}
}

// TanhRef applies tanh activation
func (r Reference) Tanh(x []float32) {
	for i := range x {
		x[i] = float32(math.Tanh(float64(x[i])))
	}
}

// GELURef applies GELU activation using the accurate formula
func (r Reference) GELU(x []float32) {
	for i := range x {
		// GELU(x) = x * Φ(x) where Φ is CDF of standard normal
		// Using approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
		x3 := x[i] * x[i] * x[i]
		arg := math.Sqrt(2.0/math.Pi) * (float64(x[i]) + 0.044715*float64(x3))
		x[i] = 0.5 * x[i] * (1.0 + float32(math.Tanh(arg)))
	}
}

// Reduction operations

// SumRef computes the sum of all elements
func (r Reference) Sum(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v
	}
	return sum
}

// MaxRef finds the maximum element
func (r Reference) Max(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// MinRef finds the minimum element
func (r Reference) Min(x []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	min := x[0]
	for _, v := range x[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// ArgMaxRef finds the index of the maximum element
func (r Reference) ArgMax(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := x[0]
	for i, v := range x[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// ArgMinRef finds the index of the minimum element
func (r Reference) ArgMin(x []float32) int {
	if len(x) == 0 {
		return -1
	}
	minIdx := 0
	minVal := x[0]
	for i, v := range x[1:] {
		if v < minVal {
			minVal = v
			minIdx = i + 1
		}
	}
	return minIdx
}

// Neural network operations

// SoftmaxRef applies softmax: y[i] = exp(x[i]) / sum(exp(x))
func (r Reference) Softmax(x []float32) {
	// Find max for numerical stability
	max := r.Max(x)
	
	// Compute exp(x - max) and sum
	sum := float32(0)
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	
	// Normalize
	for i := range x {
		x[i] /= sum
	}
}

// LayerNormRef applies layer normalization
func (r Reference) LayerNorm(x []float32, gamma, beta []float32, eps float32) {
	n := len(x)
	
	// Compute mean
	mean := r.Sum(x) / float32(n)
	
	// Compute variance
	var variance float32
	for _, v := range x {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float32(n)
	
	// Normalize and apply gamma/beta
	invStd := 1.0 / float32(math.Sqrt(float64(variance + eps)))
	for i := range x {
		normalized := (x[i] - mean) * invStd
		if gamma != nil && beta != nil {
			x[i] = normalized*gamma[i] + beta[i]
		} else {
			x[i] = normalized
		}
	}
}

// Conv2DRef performs 2D convolution (NCHW format)
// This is a simple direct convolution, not optimized
func (r Reference) Conv2D(
	input []float32, batch, inC, inH, inW int,
	kernel []float32, outC, kernelH, kernelW int,
	output []float32,
	strideH, strideW, padH, padW int) {
	
	// Calculate output dimensions
	outH := (inH + 2*padH - kernelH) / strideH + 1
	outW := (inW + 2*padW - kernelW) / strideW + 1
	
	// Direct convolution
	for b := 0; b < batch; b++ {
		for oc := 0; oc < outC; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := float32(0)
					
					// Convolve
					for ic := 0; ic < inC; ic++ {
						for kh := 0; kh < kernelH; kh++ {
							for kw := 0; kw < kernelW; kw++ {
								ih := oh*strideH - padH + kh
								iw := ow*strideW - padW + kw
								
								// Check bounds
								if ih >= 0 && ih < inH && iw >= 0 && iw < inW {
									inIdx := ((b*inC + ic)*inH + ih)*inW + iw
									kernelIdx := ((oc*inC + ic)*kernelH + kh)*kernelW + kw
									sum += input[inIdx] * kernel[kernelIdx]
								}
							}
						}
					}
					
					outIdx := ((b*outC + oc)*outH + oh)*outW + ow
					output[outIdx] = sum
				}
			}
		}
	}
}

// Fused operations

// AddBiasReLURef performs x = ReLU(x + bias) where bias is broadcast
func (r Reference) AddBiasReLU(x, bias []float32, n, biasLen int) {
	for i := 0; i < n; i++ {
		biasIdx := i % biasLen
		x[i] = x[i] + bias[biasIdx]
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// LinearReLURef performs x = ReLU(alpha*x + beta)
func (r Reference) LinearReLU(x []float32, alpha, beta float32) {
	for i := range x {
		x[i] = alpha*x[i] + beta
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// GEMMBiasReLURef performs C = ReLU(alpha*A*B + bias)
func (r Reference) GEMMBiasReLU(
	m, n, k int, alpha float32,
	a []float32, lda int,
	b []float32, ldb int,
	bias []float32,
	c []float32, ldc int) {
	
	// First do GEMM with beta=0
	r.GEMM(false, false, m, n, k, alpha, a, lda, b, ldb, 0, c, ldc)
	
	// Then add bias and apply ReLU
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			idx := i*ldc + j
			c[idx] += bias[j]
			if c[idx] < 0 {
				c[idx] = 0
			}
		}
	}
}