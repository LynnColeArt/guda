//go:build arm64

package guda

import "golang.org/x/sys/cpu"

// CPU feature detection for ARM64
var (
	hasAVX2 = false  // ARM64 doesn't have AVX2
	hasFMA  = false  // ARM64 has FMA but it's different from x86 FMA
	
	// ARM64-specific features
	hasNEON = cpu.ARM64.HasASIMD
	hasFP16 = cpu.ARM64.HasFPHP && cpu.ARM64.HasASIMDHP
)