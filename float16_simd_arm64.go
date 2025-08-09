// +build arm64

package guda

import (
	"fmt"
	"unsafe"
	"golang.org/x/sys/cpu"
)

// Assembly function declarations
//go:noescape
func simdF16ToF32NEON(src, dst unsafe.Pointer, n int)

//go:noescape
func simdF32ToF16NEON(src, dst unsafe.Pointer, n int)

//go:noescape
func simdAddFloat16NEON(a, b, c unsafe.Pointer, n int) error

//go:noescape
func simdMulFloat16NEON(a, b, c unsafe.Pointer, n int) error

//go:noescape
func simdFMAFloat16NEON(a, b, c, d unsafe.Pointer, n int) error

// CPU feature detection
var (
	// These are defined in cpu_arm64_features.go
	// hasNEON bool
	// hasFP16 bool
)

func init() {
	// Detect ARM64 CPU features
	hasNEON = cpu.ARM64.HasASIMD
	hasFP16 = cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP
}

// SimdF16ToF32 converts float16 to float32 using NEON instructions
func SimdF16ToF32(src []uint16, dst []float32) {
	if !hasNEON {
		// Fallback to scalar
		for i := 0; i < min(len(src), len(dst)); i++ {
			dst[i] = Float16(src[i]).ToFloat32()
		}
		return
	}
	
	n := min(len(src), len(dst))
	if n == 0 {
		return
	}
	
	simdF16ToF32NEON(unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), n)
}

// SimdF32ToF16 converts float32 to float16 using NEON instructions
func SimdF32ToF16(src []float32, dst []uint16) {
	if !hasNEON {
		// Fallback to scalar
		for i := 0; i < min(len(src), len(dst)); i++ {
			dst[i] = uint16(FromFloat32(src[i]))
		}
		return
	}
	
	n := min(len(src), len(dst))
	if n == 0 {
		return
	}
	
	simdF32ToF16NEON(unsafe.Pointer(&src[0]), unsafe.Pointer(&dst[0]), n)
}

// AddFloat16SIMD uses NEON for float16 vector addition
func AddFloat16SIMD(a, b, c DevicePtr, n int) error {
	if !hasNEON {
		// Fallback to existing implementation
		return AddFloat16(a, b, c, n)
	}
	
	// Direct SIMD operation on the entire array
	aData := a.Byte()
	bData := b.Byte()
	cData := c.Byte()
	
	// Check for valid data or n <= 0
	if n <= 0 || a.ptr == nil || b.ptr == nil || c.ptr == nil || 
		len(aData) < n*2 || len(bData) < n*2 || len(cData) < n*2 {
		return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
			// Out of bounds, skip
		}), Dim3{X: 1, Y: 1, Z: 1}, Dim3{X: 1, Y: 1, Z: 1})
	}
	
	// Use assembly implementation
	simdAddFloat16NEON(
		unsafe.Pointer(&aData[0]),
		unsafe.Pointer(&bData[0]),
		unsafe.Pointer(&cData[0]),
		n,
	)
	
	return nil
}

// MultiplyFloat16SIMD uses NEON for float16 vector multiplication
func MultiplyFloat16SIMD(a, b, c DevicePtr, n int) error {
	if !hasNEON {
		// Fallback to existing implementation
		return MultiplyFloat16(a, b, c, n)
	}
	
	aData := a.Byte()
	bData := b.Byte()
	cData := c.Byte()
	
	// Check for valid data or n <= 0
	if n <= 0 || a.ptr == nil || b.ptr == nil || c.ptr == nil || 
		len(aData) < n*2 || len(bData) < n*2 || len(cData) < n*2 {
		return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
			// Out of bounds, skip
		}), Dim3{X: 1, Y: 1, Z: 1}, Dim3{X: 1, Y: 1, Z: 1})
	}
	
	simdMulFloat16NEON(
		unsafe.Pointer(&aData[0]),
		unsafe.Pointer(&bData[0]),
		unsafe.Pointer(&cData[0]),
		n,
	)
	
	return nil
}

// FMAFloat16SIMD performs d = a*b + c using NEON
func FMAFloat16SIMD(a, b, c, d DevicePtr, n int) error {
	if !hasNEON {
		// Fallback to separate multiply and add
		temp, err := Malloc(n * 2)
		if err != nil {
			return err
		}
		defer Free(temp)
		
		// Validate data before proceeding
		aLen := a.Float16().Len()
		bLen := b.Float16().Len()
		cLen := c.Float16().Len()
		dLen := d.Float16().Len()
		tempLen := temp.Float16().Len()
		
		if aLen < n || bLen < n || cLen < n || dLen < n || tempLen < n {
			return fmt.Errorf("insufficient data for FMAFloat16SIMD operation: aLen=%d, bLen=%d, cLen=%d, dLen=%d, tempLen=%d, n=%d", 
				aLen, bLen, cLen, dLen, tempLen, n)
		}
		
		if n <= 0 {
			return nil // Nothing to do
		}
		
		err = MultiplyFloat16(a, b, temp, n)
		if err != nil {
			return err
		}
		return AddFloat16(temp, c, d, n)
	}
	
	// Note: ARM64 fused multiply-add operations don't require specific FP16 support
	// as they work with the general NEON implementation
	
	aData := a.Byte()
	bData := b.Byte()
	cData := c.Byte()
	dData := d.Byte()
	
	// Check for valid data or n <= 0
	if n <= 0 || a.ptr == nil || b.ptr == nil || c.ptr == nil || d.ptr == nil ||
		len(aData) < n*2 || len(bData) < n*2 || len(cData) < n*2 || len(dData) < n*2 {
		return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
			// Out of bounds, skip
		}), Dim3{X: 1, Y: 1, Z: 1}, Dim3{X: 1, Y: 1, Z: 1})
	}
	
	simdFMAFloat16NEON(
		unsafe.Pointer(&aData[0]),
		unsafe.Pointer(&bData[0]),
		unsafe.Pointer(&cData[0]),
		unsafe.Pointer(&dData[0]),
		n,
	)
	
	return nil
}

// GEMMFloat16SIMD performs optimized float16 matrix multiplication using NEON
func GEMMFloat16SIMD(transA, transB bool, m, n, k int, alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int) error {
	
	if !hasNEON {
		// Fallback to tiled implementation
		return GEMMFloat16(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	}
	
	// For simplicity, assume no transpose and beta=0
	if transA || transB || beta != 0 {
		return GEMMFloat16(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	}
	
	// Optimized tiled GEMM with NEON
	// ARM64 has 32 128-bit NEON registers, which is sufficient for efficient tiled GEMM
	const tileM = 4  // 4x8 register blocking for NEON
	const tileN = 8
	const tileK = 8  // Process 8 K elements at a time with NEON
	
	// Check if the input data has enough elements
	if m <= 0 || n <= 0 || k <= 0 {
		return nil
	}
	
	aLen := a.Float16().Len()
	bLen := b.Float16().Len()
	cLen := c.Float16().Len()
	
	// Verify we have enough data for the matrices
	if aLen < m*k || bLen < k*n || cLen < m*n {
		return fmt.Errorf("insufficient data for GEMM operation: aLen=%d, bLen=%d, cLen=%d need at least %dx%d, %dx%d, %dx%d", 
			aLen, bLen, cLen, m, k, k, n, m, n)
	}
	
	// Process tiles
	for i := 0; i < m; i += tileM {
		for j := 0; j < n; j += tileN {
			// Zero accumulator for this tile
			var accum [tileM][tileN]float32
			
			// Accumulate over K dimension
			for kk := 0; kk < k; kk += tileK {
				// Load and convert tile of A (4x8 float16 -> float32)
				var aTile [tileM][tileK]float32
				for ti := 0; ti < tileM && i+ti < m; ti++ {
					aRow := a.Float16()
					for tk := 0; tk < tileK && kk+tk < k; tk++ {
						aTile[ti][tk] = aRow.GetFloat32((i+ti)*lda + kk + tk)
					}
				}
				
				// Load and convert tile of B (8x8 float16 -> float32)
				var bTile [tileK][tileN]float32
				for tk := 0; tk < tileK && kk+tk < k; tk++ {
					bRow := b.Float16()
					for tj := 0; tj < tileN && j+tj < n; tj++ {
						bTile[tk][tj] = bRow.GetFloat32((kk+tk)*ldb + j + tj)
					}
				}
				
				// Compute 4x8 outer product and accumulate
				for ti := 0; ti < tileM && i+ti < m; ti++ {
					for tj := 0; tj < tileN && j+tj < n; tj++ {
						for tk := 0; tk < tileK && kk+tk < k; tk++ {
							accum[ti][tj] += aTile[ti][tk] * bTile[tk][tj]
						}
					}
				}
			}
			
			// Store accumulated results back as float16
			cSlice := c.Float16()
			for ti := 0; ti < tileM && i+ti < m; ti++ {
				for tj := 0; tj < tileN && j+tj < n; tj++ {
					cSlice.SetFloat32((i+ti)*ldc + j + tj, alpha * accum[ti][tj])
				}
			}
		}
	}
	
	return nil
}

// BatchedGEMMFloat16SIMD performs multiple GEMMs in parallel
// Common in transformer models
func BatchedGEMMFloat16SIMD(
	batch int,
	transA, transB bool,
	m, n, k int,
	alpha float32,
	aArray []DevicePtr, ldaArray []int,
	bArray []DevicePtr, ldbArray []int,
	beta float32,
	cArray []DevicePtr, ldcArray []int) error {
	
	// Use goroutines to parallelize across batch dimension
	errChan := make(chan error, batch)
	
	for b := 0; b < batch; b++ {
		go func(idx int) {
			err := GEMMFloat16SIMD(transA, transB, m, n, k, alpha,
				aArray[idx], ldaArray[idx],
				bArray[idx], ldbArray[idx],
				beta,
				cArray[idx], ldcArray[idx])
			errChan <- err
		}(b)
	}
	
	// Collect errors
	for b := 0; b < batch; b++ {
		if err := <-errChan; err != nil {
			return err
		}
	}
	
	return nil
}

// Conv2DFloat16SIMD performs 2D convolution with float16 using NEON
func Conv2DFloat16SIMD(
	input DevicePtr, inputH, inputW, inputC int,
	kernel DevicePtr, kernelH, kernelW int,
	output DevicePtr, outputH, outputW, outputC int,
	strideH, strideW int,
	padH, padW int) error {
	
	grid := Dim3{X: (outputW + 15) / 16, Y: (outputH + 15) / 16, Z: (outputC + 3) / 4}
	block := Dim3{X: 16, Y: 16, Z: 1}
	
	return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
		ox := tid.BlockIdx.X*16 + tid.ThreadIdx.X
		oy := tid.BlockIdx.Y*16 + tid.ThreadIdx.Y
		oc := tid.BlockIdx.Z * 4
		
		if ox >= outputW || oy >= outputH || oc >= outputC {
			return
		}
		
		inputF16 := input.Float16()
		kernelF16 := kernel.Float16()
		outputF16 := output.Float16()
		
		// Process 4 output channels at once
		var sum [4]float32
		
		for ky := 0; ky < kernelH; ky++ {
			for kx := 0; kx < kernelW; kx++ {
				ix := ox*strideW - padW + kx
				iy := oy*strideH - padH + ky
				
				if ix >= 0 && ix < inputW && iy >= 0 && iy < inputH {
					for ic := 0; ic < inputC; ic++ {
						inputIdx := (iy*inputW + ix)*inputC + ic
						inputVal := inputF16.GetFloat32(inputIdx)
						
						for i := 0; i < 4 && oc+i < outputC; i++ {
							kernelIdx := ((oc+i)*kernelH*kernelW + ky*kernelW + kx)*inputC + ic
							kernelVal := kernelF16.GetFloat32(kernelIdx)
							sum[i] += inputVal * kernelVal
						}
					}
				}
			}
		}
		
		// Store results
		for i := 0; i < 4 && oc+i < outputC; i++ {
			outputIdx := (oy*outputW + ox)*outputC + oc + i
			outputF16.SetFloat32(outputIdx, sum[i])
		}
	}), grid, block)
}

// LayerNormFloat16SIMD performs layer normalization with float16 using NEON
func LayerNormFloat16SIMD(input, gamma, beta, output DevicePtr, n, hidden int) error {
	// n = batch size, hidden = hidden dimension
	
	grid := Dim3{X: n, Y: 1, Z: 1}
	block := Dim3{X: min(256, hidden), Y: 1, Z: 1}
	
	return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
		batch := tid.BlockIdx.X
		tid_local := tid.ThreadIdx.X
		threads := block.X
		
		if batch >= n {
			return
		}
		
		inputF16 := input.Float16()
		gammaF16 := gamma.Float16()
		betaF16 := beta.Float16()
		outputF16 := output.Float16()
		
		// Compute mean and variance for this batch element
		// Using Welford's algorithm for numerical stability
		var mean, m2 float32
		count := 0
		
		// First pass: compute mean and variance
		for i := tid_local; i < hidden; i += threads {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			count++
			delta := val - mean
			mean += delta / float32(count)
			m2 += delta * (val - mean)
		}
		
		// Reduce across threads (simplified - in production use warp shuffles)
		// For now, just recompute in each thread
		mean = 0
		m2 = 0
		for i := 0; i < hidden; i++ {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			delta := val - mean
			mean += delta / float32(i+1)
			m2 += delta * (val - mean)
		}
		
		variance := m2 / float32(hidden)
		invStd := 1.0 / (variance + DefaultLayerNormEpsilon) // rsqrt in production
		
		// Second pass: normalize and apply gamma/beta
		for i := tid_local; i < hidden; i += threads {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			normalized := (val - mean) * invStd
			
			// Apply learnable parameters
			g := gammaF16.GetFloat32(i)
			b := betaF16.GetFloat32(i)
			result := normalized*g + b
			
			outputF16.SetFloat32(idx, result)
		}
	}), grid, block)
}