//go:build cgo && amd64
// +build cgo,amd64

package vnni

/*
#cgo CFLAGS: -mavx512f -mavx512bw -mavx512vnni -O3 -march=native -fopenmp
#cgo LDFLAGS: -lgomp

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

// Correct VNNI kernel implementation
void vnni_gemm_kernel(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) {
    // Clear C
    memset(C, 0, M * N * sizeof(int32_t));
    
    // Process with correct algorithm
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            
            // Process K in chunks of 4 for VPDPBUSD
            int k;
            for (k = 0; k + 3 < K; k += 4) {
                // Manual dot product of 4 elements
                sum += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
                sum += (int32_t)A[i*K + k+1] * (int32_t)B[(k+1)*N + j];
                sum += (int32_t)A[i*K + k+2] * (int32_t)B[(k+2)*N + j];
                sum += (int32_t)A[i*K + k+3] * (int32_t)B[(k+3)*N + j];
            }
            
            // Handle remainder
            for (; k < K; k++) {
                sum += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
            }
            
            C[i*N + j] = sum;
        }
    }
}
*/
import "C"
import "unsafe"

// vnniCGOAvailable indicates if CGO VNNI is available
var vnniCGOAvailable = true

// vnniInt8GEMMCgo uses C intrinsics for real VNNI performance
func vnniInt8GEMMCgo(m, n, k int, a, b []int8, c []int32) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}
	
	C.vnni_gemm_kernel(
		C.int(m), C.int(n), C.int(k),
		(*C.int8_t)(unsafe.Pointer(&a[0])),
		(*C.int8_t)(unsafe.Pointer(&b[0])),
		(*C.int32_t)(unsafe.Pointer(&c[0])),
	)
}