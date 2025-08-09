//go:build (!amd64 || noasm || gccgo || safe)
// +build !amd64 noasm gccgo safe

package compute

// tryOptimizedGemm attempts to use optimized GEMM kernels if available
// Generic version always returns false
func tryOptimizedGemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, c []float32, ldc int) bool {
	return false
}