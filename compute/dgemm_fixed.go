package compute

import "github.com/LynnColeArt/guda/compute/asm/f64"

// dgemmSerialNotNotKernelFixed is a fixed version that handles edge cases properly
func dgemmSerialNotNotKernelFixed(m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int) {
	const mr8 = 8
	const nr8 = 8
	const mr4 = 4
	const nr4 = 4
	
	// Use 8x8 kernels for the bulk
	mi8 := m / mr8 * mr8
	ni8 := n / nr8 * nr8
	
	for i := 0; i < mi8; i += mr8 {
		for j := 0; j < ni8; j += nr8 {
			f64.GemmKernel8x8(&a[i*lda], &b[j], &c[i*ldc+j], k, lda, ldb, ldc)
		}
		
		// Handle remaining columns with 4x4 kernels
		ni4 := n / nr4 * nr4
		for j := ni8; j < ni4; j += nr4 {
			f64.GemmKernel4x4(&a[i*lda], &b[j], &c[i*ldc+j], k, lda, ldb, ldc)
			f64.GemmKernel4x4(&a[(i+4)*lda], &b[j], &c[(i+4)*ldc+j], k, lda, ldb, ldc)
		}
		
		// Handle truly remaining columns (less than 4)
		for j := ni4; j < n; j++ {
			for ii := i; ii < i+mr8 && ii < m; ii++ {
				sum := 0.0
				for l := 0; l < k; l++ {
					sum += a[ii*lda+l] * b[l*ldb+j]
				}
				c[ii*ldc+j] += sum
			}
		}
	}
	
	// Handle remaining rows with 4x4 kernels
	mi4 := m / mr4 * mr4
	for i := mi8; i < mi4; i += mr4 {
		ni4 := n / nr4 * nr4
		for j := 0; j < ni4; j += nr4 {
			f64.GemmKernel4x4(&a[i*lda], &b[j], &c[i*ldc+j], k, lda, ldb, ldc)
		}
		// Handle remaining columns
		for j := ni4; j < n; j++ {
			for ii := i; ii < i+mr4 && ii < m; ii++ {
				sum := 0.0
				for l := 0; l < k; l++ {
					sum += a[ii*lda+l] * b[l*ldb+j]
				}
				c[ii*ldc+j] += sum
			}
		}
	}
	
	// Handle any truly remaining rows (less than 4)
	for i := mi4; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for l := 0; l < k; l++ {
				sum += a[i*lda+l] * b[l*ldb+j]
			}
			c[i*ldc+j] += sum
		}
	}
}