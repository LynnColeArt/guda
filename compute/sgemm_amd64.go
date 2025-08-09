//go:build amd64 && !noasm && !gccgo && !safe
// +build amd64,!noasm,!gccgo,!safe

package compute

import (
	"github.com/LynnColeArt/guda/compute/asm/f32"
)

// tryOptimizedGemm attempts to use optimized GEMM kernels if available
// This version overrides the generic implementation for AMD64
func tryOptimizedGemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) bool {
	// Check if AVX-512 is available and matrix is large enough
	if f32.HasAVX512Support {
		// Use AVX-512 kernel
		// Note: GemmAVX512 expects beta=1.0 (accumulate into C)
		// Since we're called from sgemmSerialNotNot which handles beta separately,
		// we use beta=1.0 here
		f32.GemmAVX512(false, false, m, n, k, alpha, a, lda, b, ldb, 1.0, c, ldc)
		return true
	}
	
	// Could add AVX2 dispatch here in the future
	// For now, return false to use the default AXPY-based approach
	return false
}