package guda

import (
	"runtime"
)

// ArchToleranceConfig provides architecture-specific tolerance configurations
type ArchToleranceConfig struct {
	// Base tolerance for all architectures
	Base ToleranceConfig
	
	// Architecture-specific overrides
	AMD64   *ToleranceConfig
	ARM64   *ToleranceConfig
	Generic *ToleranceConfig
}

// GetArchTolerance returns the appropriate tolerance for the current architecture
func GetArchTolerance(config ArchToleranceConfig) ToleranceConfig {
	base := config.Base
	
	switch runtime.GOARCH {
	case "amd64":
		if config.AMD64 != nil {
			return mergeTolerances(base, *config.AMD64)
		}
	case "arm64", "arm64be":
		if config.ARM64 != nil {
			return mergeTolerances(base, *config.ARM64)
		}
	default:
		if config.Generic != nil {
			return mergeTolerances(base, *config.Generic)
		}
	}
	
	return base
}

// mergeTolerances applies overrides to base tolerance
func mergeTolerances(base, override ToleranceConfig) ToleranceConfig {
	result := base
	
	// Only override non-zero values
	if override.AbsTol > 0 {
		result.AbsTol = override.AbsTol
	}
	if override.RelTol > 0 {
		result.RelTol = override.RelTol
	}
	if override.ULPTol > 0 {
		result.ULPTol = override.ULPTol
	}
	
	return result
}

// GEMMArchTolerance provides architecture-aware tolerances for GEMM operations
var GEMMArchTolerance = ArchToleranceConfig{
	Base: ToleranceConfig{
		AbsTol:   1e-6,
		RelTol:   1e-5,
		ULPTol:   4,
		CheckNaN: true,
		CheckInf: true,
	},
	ARM64: &ToleranceConfig{
		// ARM64 NEON can have slightly different rounding behavior
		// Relax ULP tolerance to account for FMA differences
		AbsTol: 1e-5,
		RelTol: 1e-4,
		ULPTol: 16,
	},
	Generic: &ToleranceConfig{
		// Other architectures might need even more relaxed tolerances
		AbsTol: 1e-4,
		RelTol: 1e-3,
		ULPTol: 32,
	},
}

// ConvArchTolerance provides architecture-aware tolerances for convolution operations
var ConvArchTolerance = ArchToleranceConfig{
	Base: ToleranceConfig{
		AbsTol:   1e-6,
		RelTol:   1e-5,
		ULPTol:   8,
		CheckNaN: true,
		CheckInf: true,
	},
	ARM64: &ToleranceConfig{
		AbsTol: 1e-5,
		RelTol: 1e-4,
		ULPTol: 32,
	},
}

// ReduceArchTolerance provides architecture-aware tolerances for reduction operations
var ReduceArchTolerance = ArchToleranceConfig{
	Base: ToleranceConfig{
		AbsTol:   1e-5,
		RelTol:   1e-4,
		ULPTol:   16,
		CheckNaN: true,
		CheckInf: true,
	},
	ARM64: &ToleranceConfig{
		// Reductions can accumulate more error on ARM64
		AbsTol: 1e-4,
		RelTol: 1e-3,
		ULPTol: 64,
	},
}

// GetOperationTolerance returns architecture-specific tolerance for an operation
func GetOperationTolerance(operation string) ToleranceConfig {
	switch operation {
	case "gemm":
		return GetArchTolerance(GEMMArchTolerance)
	case "conv2d":
		return GetArchTolerance(ConvArchTolerance)
	case "reduce_sum", "softmax":
		return GetArchTolerance(ReduceArchTolerance)
	default:
		return DefaultTolerance()
	}
}

// IsARM64 returns true if running on ARM64 architecture
func IsARM64() bool {
	return runtime.GOARCH == "arm64" || runtime.GOARCH == "arm64be"
}

// Note for ARM64 implementers:
// ARM64 NEON instructions may have different rounding modes and FMA behavior
// compared to x86 AVX/SSE. Key differences:
// 1. NEON uses "round to nearest, ties to even" by default
// 2. FMA operations may produce different results due to no intermediate rounding
// 3. Denormal handling may differ between architectures
// 4. Vector reduction operations may use different accumulation orders
//
// If you're seeing tolerance errors on ARM64, check:
// - FMA vs separate multiply-add sequences
// - Reduction operation order (tree reduction vs sequential)
// - Denormal number handling
// - Compiler optimization flags affecting FP behavior