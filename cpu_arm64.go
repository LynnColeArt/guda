// +build arm64

package guda

import (
	"golang.org/x/sys/cpu"
)

// CPUFeatures represents the SIMD capabilities detected on the system
type CPUFeatures struct {
	HasNEON bool
	HasFP16 bool
}

// DetectARM64Features detects ARM64 CPU features using golang.org/x/sys/cpu
// HasASIMD corresponds to NEON support
// HasFPHP and HasASIMDHP indicate support for half-precision floating point operations
func DetectARM64Features() (hasNEON, hasFP16 bool) {
	// Check for NEON support (ASIMD - Advanced SIMD)
	hasNEON = cpu.ARM64.HasASIMD
	
	// Check for FP16 support (FPHP - Floating Point Half Precision, ASIMDHP - Advanced SIMD Half Precision)
	hasFP16 = cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP
	
	return hasNEON, hasFP16
}

// getCPUFeatures returns the CPU features for ARM64 systems
func getCPUFeatures() *CPUFeatures {
	hasNEON, hasFP16 := DetectARM64Features()
	return &CPUFeatures{
		HasNEON: hasNEON,
		HasFP16: hasFP16,
	}
}