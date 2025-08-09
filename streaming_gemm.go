package guda

import (
	"runtime"
	"sync"
)

// StreamingGEMM implements a memory-bandwidth optimized GEMM
// that minimizes memory traffic by processing data in streaming fashion
type StreamingGEMM struct {
	// L1 tile size (fits in L1 cache - 32KB)
	l1TileM int
	l1TileN int
	l1TileK int
	
	// L2 tile size (fits in L2 cache - 256KB)
	l2TileM int
	l2TileN int
	l2TileK int
	
	// L3 tile size (fits in L3 cache - 8MB)
	l3TileM int
	l3TileN int
	l3TileK int
	
	// Number of threads
	numThreads int
}

// NewStreamingGEMM creates a streaming GEMM optimized for memory bandwidth
func NewStreamingGEMM() *StreamingGEMM {
	return &StreamingGEMM{
		// L1 tiles: ~8KB per matrix (32KB total / 3 matrices)
		// For float32: 8KB = 2048 elements
		// Square root: ~45, but we want multiples of 8 for SIMD
		l1TileM: 16,
		l1TileN: 16,
		l1TileK: 8,
		
		// L2 tiles: ~85KB per matrix (256KB / 3)
		// For float32: 85KB = ~21K elements
		// Square root: ~145, round to 128
		l2TileM: 64,
		l2TileN: 64,
		l2TileK: 32,
		
		// L3 tiles: ~2.7MB per matrix (8MB / 3)
		// For float32: 2.7MB = ~675K elements
		// Square root: ~820, round to 512
		l3TileM: 256,
		l3TileN: 256,
		l3TileK: 128,
		
		numThreads: runtime.NumCPU(),
	}
}

// Compute performs C = alpha * A * B + beta * C
// using a streaming approach that maximizes cache reuse
func (sg *StreamingGEMM) Compute(
	alpha float32,
	a []float32, lda int, m, k int,
	b []float32, ldb int, k2, n int,
	beta float32,
	c []float32, ldc int,
) {
	// Handle beta scaling
	if beta != 1.0 {
		sg.scaleC(c, ldc, m, n, beta)
	}
	
	// Use parallel L3 blocking
	var wg sync.WaitGroup
	l3BlocksM := (m + sg.l3TileM - 1) / sg.l3TileM
	l3BlocksN := (n + sg.l3TileN - 1) / sg.l3TileN
	totalL3Blocks := l3BlocksM * l3BlocksN
	
	// Distribute L3 blocks among threads
	blocksPerThread := (totalL3Blocks + sg.numThreads - 1) / sg.numThreads
	
	for thread := 0; thread < sg.numThreads; thread++ {
		startBlock := thread * blocksPerThread
		endBlock := min((thread+1)*blocksPerThread, totalL3Blocks)
		
		if startBlock >= endBlock {
			break
		}
		
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			
			// Process assigned L3 blocks
			for blockIdx := start; blockIdx < end; blockIdx++ {
				l3i := (blockIdx / l3BlocksN) * sg.l3TileM
				l3j := (blockIdx % l3BlocksN) * sg.l3TileN
				
				l3iEnd := min(l3i+sg.l3TileM, m)
				l3jEnd := min(l3j+sg.l3TileN, n)
				
				// Process this L3 block with L2 tiling
				sg.processL3Block(
					alpha,
					a, lda, l3i, 0, l3iEnd-l3i, k,
					b, ldb, 0, l3j, k, l3jEnd-l3j,
					c, ldc, l3i, l3j,
				)
			}
		}(startBlock, endBlock)
	}
	
	wg.Wait()
}

// processL3Block processes one L3-sized block with L2 tiling
func (sg *StreamingGEMM) processL3Block(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
) {
	// L2 blocking within the L3 block
	for ii := 0; ii < aRows; ii += sg.l2TileM {
		iEnd := min(ii+sg.l2TileM, aRows)
		
		for jj := 0; jj < bCols; jj += sg.l2TileN {
			jEnd := min(jj+sg.l2TileN, bCols)
			
			// Process K dimension in L2-sized chunks
			for kk := 0; kk < aCols; kk += sg.l2TileK {
				kEnd := min(kk+sg.l2TileK, aCols)
				
				// L1 tiling within L2 block
				sg.processL2Block(
					alpha,
					a, lda, aRowStart+ii, aColStart+kk, iEnd-ii, kEnd-kk,
					b, ldb, bRowStart+kk, bColStart+jj, kEnd-kk, jEnd-jj,
					c, ldc, cRowStart+ii, cColStart+jj,
					aColStart+kk == 0, // first K block globally?
				)
			}
		}
	}
}

// processL2Block processes one L2-sized block with L1 tiling
func (sg *StreamingGEMM) processL2Block(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
	firstK bool,
) {
	// L1 blocking within the L2 block
	for ii := 0; ii < aRows; ii += sg.l1TileM {
		iEnd := min(ii+sg.l1TileM, aRows)
		
		for jj := 0; jj < bCols; jj += sg.l1TileN {
			jEnd := min(jj+sg.l1TileN, bCols)
			
			// Process the entire K dimension for this L1 tile
			// This keeps the C tile in L1 cache
			for kk := 0; kk < aCols; kk += sg.l1TileK {
				kEnd := min(kk+sg.l1TileK, aCols)
				
				// Call optimized kernel for L1 tile
				sg.computeL1Tile(
					alpha,
					a, lda, aRowStart+ii, aColStart+kk, iEnd-ii, kEnd-kk,
					b, ldb, bRowStart+kk, bColStart+jj, kEnd-kk, jEnd-jj,
					c, ldc, cRowStart+ii, cColStart+jj,
					firstK && kk == 0,
				)
			}
		}
	}
}

// computeL1Tile computes a small tile that fits in L1 cache
func (sg *StreamingGEMM) computeL1Tile(
	alpha float32,
	a []float32, lda int, aRowStart, aColStart, aRows, aCols int,
	b []float32, ldb int, bRowStart, bColStart, bRows, bCols int,
	c []float32, ldc int, cRowStart, cColStart int,
	firstK bool,
) {
	// For small tiles, use optimized micro-kernels
	// This is where we'd call our AVX2/AVX512 kernels
	
	// For now, use a simple implementation with good memory access pattern
	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			sum := float32(0.0)
			
			// Unroll by 4 for better performance
			k := 0
			for ; k <= aCols-4; k += 4 {
				aIdx0 := (aRowStart+i)*lda + (aColStart+k)
				bIdx0 := (bRowStart+k)*ldb + (bColStart+j)
				
				sum += a[aIdx0] * b[bIdx0]
				sum += a[aIdx0+1] * b[bIdx0+ldb]
				sum += a[aIdx0+2] * b[bIdx0+2*ldb]
				sum += a[aIdx0+3] * b[bIdx0+3*ldb]
			}
			
			// Handle remainder
			for ; k < aCols; k++ {
				aIdx := (aRowStart+i)*lda + (aColStart+k)
				bIdx := (bRowStart+k)*ldb + (bColStart+j)
				sum += a[aIdx] * b[bIdx]
			}
			
			cIdx := (cRowStart+i)*ldc + (cColStart+j)
			if firstK {
				c[cIdx] += alpha * sum
			} else {
				c[cIdx] += sum
			}
		}
	}
}

// scaleC scales matrix C by beta
func (sg *StreamingGEMM) scaleC(c []float32, ldc int, m, n int, beta float32) {
	if beta == 0.0 {
		// Simple serial zero for now
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] = 0
			}
		}
	} else {
		// Simple serial scale for now
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				c[i*ldc+j] *= beta
			}
		}
	}
}

// StreamingGEMM_Float32 provides a convenient interface
func StreamingGEMM_Float32(transA, transB bool, m, n, k int, alpha float32,
	a []float32, lda int, b []float32, ldb int,
	beta float32, c []float32, ldc int) {
	
	if transA || transB {
		// Fall back to reference for transposed cases
		ref := Reference{}
		ref.GEMM(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	
	sg := NewStreamingGEMM()
	sg.Compute(alpha, a, lda, m, k, b, ldb, k, n, beta, c, ldc)
}