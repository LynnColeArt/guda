//go:build amd64
// +build amd64

#include "textflag.h"

// func amxCheckSupport() bool
TEXT ·amxCheckSupport(SB), NOSPLIT, $0-1
	// Check CPUID for AMX support
	MOVL $7, AX      // CPUID function 7
	MOVL $0, CX      // Subleaf 0
	CPUID
	
	// Check EDX for AMX bits:
	// bit 22: AMX-INT8
	// bit 24: AMX-BF16
	// bit 25: AMX-TILE
	MOVL DX, AX
	SHRL $22, AX     // Shift AMX-INT8 bit to position 0
	ANDL $1, AX      // Isolate the bit
	
	MOVL DX, BX
	SHRL $25, BX     // Shift AMX-TILE bit to position 0
	ANDL $1, BX      // Isolate the bit
	
	// Both bits must be set
	ANDL BX, AX
	MOVB AX, ret+0(FP)
	RET

// For now, stub implementations for AMX instructions
// Real AMX instructions require special encoding

// func amxConfigureTiles(cfg *TileConfigData)
TEXT ·amxConfigureTiles(SB), NOSPLIT, $0-8
	// Stub: would use LDTILECFG instruction
	RET

// func amxReleaseTiles()
TEXT ·amxReleaseTiles(SB), NOSPLIT, $0-0
	// Stub: would use TILERELEASE instruction
	RET

// func amxInt8GEMM_16x16(a, b, c []byte, lda, ldb, ldc int)
TEXT ·amxInt8GEMM_16x16(SB), NOSPLIT, $0-72
	// Stub: would use TILELOADD and TDPBSSD instructions
	RET

// func amxInt8GEMM_32x32(a, b, c []byte, lda, ldb, ldc int, k int)
TEXT ·amxInt8GEMM_32x32(SB), NOSPLIT, $0-80
	// Stub: would use TILELOADD and TDPBSSD instructions
	RET
