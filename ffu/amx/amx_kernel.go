//go:build amd64
// +build amd64

package amx

import (
	"fmt"
)

// Assembly functions
//go:noescape
func amxConfigureTiles(cfg *TileConfigData)

//go:noescape
func amxReleaseTiles()

//go:noescape
func amxInt8GEMM_16x16(a, b, c []byte, lda, ldb, ldc int)

//go:noescape
func amxInt8GEMM_32x32(a, b, c []byte, lda, ldb, ldc int, k int)

//go:noescape
func amxCheckSupport() bool

//go:noescape
func amxInt8GEMMReference(m, n, k int, a, b []byte, c []int32)

// AMXKernel provides high-level AMX operations
type AMXKernel struct {
	configured bool
	config     *TileConfigData
}

// NewAMXKernel creates a new AMX kernel
func NewAMXKernel() *AMXKernel {
	return &AMXKernel{}
}

// Configure sets up AMX tiles for INT8 GEMM
func (k *AMXKernel) Configure() error {
	if k.configured {
		return nil
	}
	
	k.config = ConfigureInt8GEMM()
	if err := ValidateConfig(k.config); err != nil {
		return fmt.Errorf("invalid tile configuration: %w", err)
	}
	
	amxConfigureTiles(k.config)
	k.configured = true
	return nil
}

// Release releases AMX tile configuration
func (k *AMXKernel) Release() {
	if k.configured {
		amxReleaseTiles()
		k.configured = false
	}
}

// Int8GEMM performs INT8 matrix multiplication using AMX
// C += A * B where A is M×K, B is K×N, C is M×N
func (k *AMXKernel) Int8GEMM(M, N, K int, A, B []byte, C []int32, alpha, beta float32) error {
	// Validate dimensions
	if M <= 0 || N <= 0 || K <= 0 {
		return fmt.Errorf("invalid dimensions: M=%d, N=%d, K=%d", M, N, K)
	}
	
	// Check buffer sizes
	if len(A) < M*K {
		return fmt.Errorf("A buffer too small: %d < %d", len(A), M*K)
	}
	if len(B) < K*N {
		return fmt.Errorf("B buffer too small: %d < %d", len(B), K*N)
	}
	if len(C) < M*N {
		return fmt.Errorf("C buffer too small: %d < %d", len(C), M*N)
	}
	
	// Configure tiles if needed
	if err := k.Configure(); err != nil {
		return err
	}
	
	// For now, use reference implementation until AMX instructions are ready
	// This gives us correctness while we work on the real AMX encoding
	amxInt8GEMMReference(M, N, K, A, B, C)
	
	// Apply scaling if needed
	if alpha != 1.0 || beta != 0.0 {
		for i := range C {
			C[i] = int32(float32(C[i])*alpha + beta)
		}
	}
	
	return nil
}

// PackA packs matrix A for optimal AMX tile access
// Converts from row-major to tile-friendly layout
func PackA(M, K int, A []int8, packedA []int8) {
	// Pack into 16×64 tiles
	tileM := (M + 15) / 16
	tileK := (K + 63) / 64
	
	idx := 0
	for tm := 0; tm < tileM; tm++ {
		for tk := 0; tk < tileK; tk++ {
			// Pack one 16×64 tile
			for i := 0; i < 16 && tm*16+i < M; i++ {
				for j := 0; j < 64 && tk*64+j < K; j++ {
					row := tm*16 + i
					col := tk*64 + j
					packedA[idx] = A[row*K+col]
					idx++
				}
			}
		}
	}
}

// PackB packs matrix B for optimal AMX tile access
// Converts from row-major to tile-friendly layout
func PackB(K, N int, B []int8, packedB []int8) {
	// Pack into 64×16 tiles (transposed for TDPBSSD)
	tileK := (K + 63) / 64
	tileN := (N + 15) / 16
	
	idx := 0
	for tk := 0; tk < tileK; tk++ {
		for tn := 0; tn < tileN; tn++ {
			// Pack one 64×16 tile
			for i := 0; i < 64 && tk*64+i < K; i++ {
				for j := 0; j < 16 && tn*16+j < N; j++ {
					row := tk*64 + i
					col := tn*16 + j
					packedB[idx] = B[row*N+col]
					idx++
				}
			}
		}
	}
}