package guda

import (
	"runtime"
)

// CacheObliviousGEMM implements a cache-oblivious matrix multiplication algorithm
// using recursive subdivision. This approach automatically adapts to any cache
// hierarchy without needing to know cache sizes.
//
// The algorithm recursively divides matrices until they fit in cache, achieving
// near-optimal cache utilization across all cache levels simultaneously.
type CacheObliviousGEMM struct {
	// Threshold for switching to optimized kernel
	baseSize int
}

// NewCacheObliviousGEMM creates a new cache-oblivious GEMM implementation
func NewCacheObliviousGEMM() *CacheObliviousGEMM {
	// Base case size - when to switch to optimized kernel
	// This should be tuned but 64 is a reasonable starting point
	return &CacheObliviousGEMM{
		baseSize: 64,
	}
}

// Compute performs C = alpha * A * B + beta * C using recursive subdivision
func (cog *CacheObliviousGEMM) Compute(
	alpha float32,
	a []float32, lda int, aRows, aCols int,
	b []float32, ldb int, bRows, bCols int,
	beta float32,
	c []float32, ldc int,
) {
	// Handle beta scaling first
	if beta != 1.0 {
		if beta == 0.0 {
			// Zero C
			for i := 0; i < aRows; i++ {
				for j := 0; j < bCols; j++ {
					c[i*ldc+j] = 0
				}
			}
		} else {
			// Scale C by beta
			for i := 0; i < aRows; i++ {
				for j := 0; j < bCols; j++ {
					c[i*ldc+j] *= beta
				}
			}
		}
	}

	// Start recursive multiplication
	cog.recursiveMultiply(
		alpha,
		a, lda, 0, 0, aRows, aCols,
		b, ldb, 0, 0, bRows, bCols,
		c, ldc, 0, 0,
	)
}

// recursiveMultiply performs the recursive subdivision
func (cog *CacheObliviousGEMM) recursiveMultiply(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
) {
	// Base case - use optimized kernel
	if aRows <= cog.baseSize && aCols <= cog.baseSize && bCols <= cog.baseSize {
		cog.baseCase(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart, cColStart,
		)
		return
	}

	// Determine which dimension to split
	// Split the largest dimension to maintain balance
	if aRows >= max(aCols, bCols) {
		// Split A and C horizontally
		mid := aRows / 2
		
		// Top half: C_top = alpha * A_top * B
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, mid, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart, cColStart,
		)
		
		// Bottom half: C_bottom = alpha * A_bottom * B
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart+mid, aColStart, aRows-mid, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart+mid, cColStart,
		)
		
	} else if bCols >= max(aRows, aCols) {
		// Split B and C vertically
		mid := bCols / 2
		
		// Left half: C_left = alpha * A * B_left
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart, bRows, mid,
			c, ldc, cRowStart, cColStart,
		)
		
		// Right half: C_right = alpha * A * B_right
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart+mid, bRows, bCols-mid,
			c, ldc, cRowStart, cColStart+mid,
		)
		
	} else {
		// Split along K dimension (sum reduction)
		mid := aCols / 2
		
		// First half: C += alpha * A_left * B_top
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, mid,
			b, ldb, bRowStart, bColStart, mid, bCols,
			c, ldc, cRowStart, cColStart,
		)
		
		// Second half: C += alpha * A_right * B_bottom
		cog.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart+mid, aRows, aCols-mid,
			b, ldb, bRowStart+mid, bColStart, bRows-mid, bCols,
			c, ldc, cRowStart, cColStart,
		)
	}
}

// baseCase handles small matrices with our optimized kernels
func (cog *CacheObliviousGEMM) baseCase(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
) {
	// For very small matrices or when dimensions don't align well,
	// use simple implementation
	if aRows < 8 || bCols < 8 || aCols < 8 {
		for i := 0; i < aRows; i++ {
			for j := 0; j < bCols; j++ {
				sum := float32(0.0)
				for k := 0; k < aCols; k++ {
					aIdx := (aRowStart+i)*lda + (aColStart+k)
					bIdx := (bRowStart+k)*ldb + (bColStart+j)
					sum += a[aIdx] * b[bIdx]
				}
				cIdx := (cRowStart+i)*ldc + (cColStart+j)
				c[cIdx] += alpha * sum
			}
		}
		return
	}
	
	// For larger blocks, we need to be more careful about memory layout
	// Our GEMM kernels expect packed data in a specific format
	// For now, let's use a hybrid approach with tiling
	
	const tileM = 32
	const tileN = 32
	const tileK = 32
	
	// Process in tiles
	for ii := 0; ii < aRows; ii += tileM {
		iEnd := min(ii+tileM, aRows)
		for jj := 0; jj < bCols; jj += tileN {
			jEnd := min(jj+tileN, bCols)
			for kk := 0; kk < aCols; kk += tileK {
				kEnd := min(kk+tileK, aCols)
				
				// Compute tile
				for i := ii; i < iEnd; i++ {
					for j := jj; j < jEnd; j++ {
						sum := float32(0.0)
						for k := kk; k < kEnd; k++ {
							aIdx := (aRowStart+i)*lda + (aColStart+k)
							bIdx := (bRowStart+k)*ldb + (bColStart+j)
							sum += a[aIdx] * b[bIdx]
						}
						cIdx := (cRowStart+i)*ldc + (cColStart+j)
						if kk == 0 {
							c[cIdx] += alpha * sum
						} else {
							c[cIdx] += sum
						}
					}
				}
			}
		}
	}
}

// CacheObliviousGEMMParallel adds parallelism to the cache-oblivious algorithm
type CacheObliviousGEMMParallel struct {
	*CacheObliviousGEMM
	minParallelSize int
	maxWorkers      int
}

// NewCacheObliviousGEMMParallel creates a parallel cache-oblivious GEMM
func NewCacheObliviousGEMMParallel() *CacheObliviousGEMMParallel {
	return &CacheObliviousGEMMParallel{
		CacheObliviousGEMM: NewCacheObliviousGEMM(),
		minParallelSize:    128, // Don't parallelize below this size
		maxWorkers:         runtime.NumCPU(),
	}
}

// Compute performs parallel cache-oblivious GEMM
func (cogp *CacheObliviousGEMMParallel) Compute(
	alpha float32,
	a []float32, lda int, aRows, aCols int,
	b []float32, ldb int, bRows, bCols int,
	beta float32,
	c []float32, ldc int,
) {
	// Handle beta scaling
	if beta != 1.0 {
		if beta == 0.0 {
			// Parallel zero
			parallel(aRows, cogp.maxWorkers, func(start, end int) {
				for i := start; i < end; i++ {
					for j := 0; j < bCols; j++ {
						c[i*ldc+j] = 0
					}
				}
			})
		} else {
			// Parallel scale
			parallel(aRows, cogp.maxWorkers, func(start, end int) {
				for i := start; i < end; i++ {
					for j := 0; j < bCols; j++ {
						c[i*ldc+j] *= beta
					}
				}
			})
		}
	}

	// Start parallel recursive multiplication
	cogp.parallelRecursiveMultiply(
		alpha,
		a, lda, 0, 0, aRows, aCols,
		b, ldb, 0, 0, bRows, bCols,
		c, ldc, 0, 0,
		cogp.maxWorkers,
	)
}

// parallelRecursiveMultiply with work stealing
func (cogp *CacheObliviousGEMMParallel) parallelRecursiveMultiply(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
	workers int,
) {
	// Use serial version for small sizes or few workers
	if workers <= 1 || max(aRows, max(aCols, bCols)) < cogp.minParallelSize {
		cogp.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart, cColStart,
		)
		return
	}

	// Parallel subdivision based on largest dimension
	if aRows >= max(aCols, bCols) && aRows > cogp.minParallelSize {
		// Split A and C horizontally
		mid := aRows / 2
		done := make(chan struct{})
		
		// Top half in goroutine
		go func() {
			cogp.parallelRecursiveMultiply(
				alpha,
				a, lda, aRowStart, aColStart, mid, aCols,
				b, ldb, bRowStart, bColStart, bRows, bCols,
				c, ldc, cRowStart, cColStart,
				workers/2,
			)
			done <- struct{}{}
		}()
		
		// Bottom half in current goroutine
		cogp.parallelRecursiveMultiply(
			alpha,
			a, lda, aRowStart+mid, aColStart, aRows-mid, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart+mid, cColStart,
			workers-workers/2,
		)
		
		<-done
		
	} else if bCols >= max(aRows, aCols) && bCols > cogp.minParallelSize {
		// Split B and C vertically - can be done in parallel
		mid := bCols / 2
		done := make(chan struct{})
		
		go func() {
			cogp.parallelRecursiveMultiply(
				alpha,
				a, lda, aRowStart, aColStart, aRows, aCols,
				b, ldb, bRowStart, bColStart, bRows, mid,
				c, ldc, cRowStart, cColStart,
				workers/2,
			)
			done <- struct{}{}
		}()
		
		cogp.parallelRecursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart+mid, bRows, bCols-mid,
			c, ldc, cRowStart, cColStart+mid,
			workers-workers/2,
		)
		
		<-done
		
	} else {
		// K dimension split - must be sequential (accumulation)
		cogp.recursiveMultiply(
			alpha,
			a, lda, aRowStart, aColStart, aRows, aCols,
			b, ldb, bRowStart, bColStart, bRows, bCols,
			c, ldc, cRowStart, cColStart,
		)
	}
}

// Helper function for parallel execution
func parallel(n int, workers int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	
	if workers <= 1 || n < workers*2 {
		fn(0, n)
		return
	}
	
	blockSize := (n + workers - 1) / workers
	done := make(chan struct{}, workers)
	
	for i := 0; i < workers; i++ {
		start := i * blockSize
		end := min((i+1)*blockSize, n)
		if start >= end {
			break
		}
		
		go func(s, e int) {
			fn(s, e)
			done <- struct{}{}
		}(start, end)
	}
	
	// Wait for all workers
	for i := 0; i < workers && i*blockSize < n; i++ {
		<-done
	}
}

// Integration with GUDA's GEMM
func CacheObliviousGEMM_Float32(transA, transB bool, m, n, k int, alpha float32,
	a []float32, lda int, b []float32, ldb int,
	beta float32, c []float32, ldc int) {
	
	// For now, only handle non-transposed case
	if transA || transB {
		// Fall back to standard implementation
		ref := Reference{}
		ref.GEMM(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	
	// Use parallel cache-oblivious implementation
	gemm := NewCacheObliviousGEMMParallel()
	gemm.Compute(alpha, a, lda, m, k, b, ldb, k, n, beta, c, ldc)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Using min from fused_transformer_layer.go