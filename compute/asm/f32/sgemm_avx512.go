//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import (
	"unsafe"
)

// Assembly functions defined in sgemm_16x4_avx512_amd64.s
//go:noescape
func sgemmKernel16x4AVX512(a, b, c unsafe.Pointer, kc int64, ldc int64)

// Blocking parameters for AVX-512 GEMM
const (
	// Microkernel dimensions
	MR_AVX512 = 16  // Rows per microkernel (ZMM register width)
	NR_AVX512 = 4   // Columns per microkernel
	
	// Cache blocking parameters (tunable)
	KC_AVX512 = 256 // K dimension blocking (L2 cache)
	MC_AVX512 = 128 // M dimension blocking (L2 cache)
	NC_AVX512 = 256 // N dimension blocking (L3 cache)
)

// PackAMatrixAVX512 packs A into MR x KC blocks for AVX-512 microkernel
// Layout: for each k, store 16 contiguous rows
func PackAMatrixAVX512(dst []float32, src []float32, lda, m, k int) {
	mr := MR_AVX512
	dstIdx := 0
	
	// Pack m x k matrix into blocks of MR x k
	for i := 0; i < m; i += mr {
		mBlock := min(mr, m-i)
		
		// For each column in the K dimension
		for kk := 0; kk < k; kk++ {
			// Pack MR rows at current k
			for ii := 0; ii < mBlock; ii++ {
				dst[dstIdx] = src[(i+ii)*lda+kk]
				dstIdx++
			}
			// Pad with zeros if mBlock < MR
			for ii := mBlock; ii < mr; ii++ {
				dst[dstIdx] = 0
				dstIdx++
			}
		}
	}
}

// PackBMatrixAVX512 packs B into KC x NR panels for AVX-512 microkernel
// Layout: for each of NR columns, store KC contiguous elements
func PackBMatrixAVX512(dst []float32, src []float32, ldb, k, n int) {
	nr := NR_AVX512
	dstIdx := 0
	
	// Pack k x n matrix into panels of k x NR
	for j := 0; j < n; j += nr {
		nBlock := min(nr, n-j)
		
		// For each column in the panel
		for jj := 0; jj < nBlock; jj++ {
			// Pack K elements of this column
			for kk := 0; kk < k; kk++ {
				dst[dstIdx+jj*k+kk] = src[kk*ldb+j+jj]
			}
		}
		// Pad with zeros if nBlock < NR
		for jj := nBlock; jj < nr; jj++ {
			for kk := 0; kk < k; kk++ {
				dst[dstIdx+jj*k+kk] = 0
			}
		}
		dstIdx += nr * k
	}
}

// GemmAVX512 performs C = alpha * A * B + beta * C using AVX-512 instructions
func GemmAVX512(transA, transB bool, m, n, k int, alpha float32,
	a []float32, lda int, b []float32, ldb int,
	beta float32, c []float32, ldc int) {
	
	// Only handle non-transposed case for now
	if transA || transB {
		panic("AVX-512 GEMM: transposed matrices not yet implemented")
	}
	
	// Handle edge cases
	if m == 0 || n == 0 || k == 0 {
		return
	}
	
	// Apply beta to C if needed
	if beta != 1.0 {
		if beta == 0.0 {
			// Zero C
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					c[i*ldc+j] = 0
				}
			}
		} else {
			// Scale C by beta
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					c[i*ldc+j] *= beta
				}
			}
		}
	}
	
	// Allocate packing buffers
	// TODO: Use memory pool to avoid allocation
	aPackSize := MC_AVX512 * KC_AVX512
	bPackSize := KC_AVX512 * NC_AVX512
	aPack := make([]float32, aPackSize)
	bPack := make([]float32, bPackSize)
	
	// Main blocking loops
	for jc := 0; jc < n; jc += NC_AVX512 {
		nc := min(NC_AVX512, n-jc)
		
		for pc := 0; pc < k; pc += KC_AVX512 {
			kc := min(KC_AVX512, k-pc)
			
			// Pack B panel
			PackBMatrixAVX512(bPack, b[pc*ldb+jc:], ldb, kc, nc)
			
			for ic := 0; ic < m; ic += MC_AVX512 {
				mc := min(MC_AVX512, m-ic)
				
				// Pack A block
				PackAMatrixAVX512(aPack, a[ic*lda+pc:], lda, mc, kc)
				
				// Multiply packed blocks
				for jr := 0; jr < nc; jr += NR_AVX512 {
					nr := min(NR_AVX512, nc-jr)
					
					for ir := 0; ir < mc; ir += MR_AVX512 {
						mr := min(MR_AVX512, mc-ir)
						
						// Call microkernel
						if mr == MR_AVX512 && nr == NR_AVX512 {
							// Full tile - use optimized kernel
							aPtr := unsafe.Pointer(&aPack[ir*kc])
							bPtr := unsafe.Pointer(&bPack[jr*kc])
							cPtr := unsafe.Pointer(&c[(ic+ir)*ldc+jc+jr])
							
							// Apply alpha scaling to result
							// TODO: Integrate alpha into packing or kernel
							if alpha != 1.0 {
								sgemmKernel16x4AVX512(aPtr, bPtr, cPtr, int64(kc), int64(ldc*4))
								// Scale result by alpha
								for i := 0; i < mr; i++ {
									for j := 0; j < nr; j++ {
										c[(ic+ir+i)*ldc+jc+jr+j] *= alpha
									}
								}
							} else {
								sgemmKernel16x4AVX512(aPtr, bPtr, cPtr, int64(kc), int64(ldc*4))
							}
						} else {
							// Tail case - use reference implementation for now
							// TODO: Implement tail kernels
							for i := 0; i < mr; i++ {
								for j := 0; j < nr; j++ {
									sum := float32(0.0)
									for p := 0; p < kc; p++ {
										// A is packed as MR x KC (row panels)
										// B is packed as KC x NR (column panels)
										// B layout: for each column j, KC elements are stored contiguously
										aVal := aPack[ir*kc+p*MR_AVX512+i]
										bVal := bPack[jr*kc+j*kc+p]
										sum += aVal * bVal
									}
									c[(ic+ir+i)*ldc+jc+jr+j] += alpha * sum
								}
							}
						}
					}
				}
			}
		}
	}
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}