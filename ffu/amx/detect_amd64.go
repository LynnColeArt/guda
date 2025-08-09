//go:build amd64
// +build amd64

package amx

import (
	"golang.org/x/sys/cpu"
)

// AMX feature flags
var (
	hasAMXTile  bool
	hasAMXInt8  bool
	hasAMXBF16  bool
)

func init() {
	detectAMX()
}

// detectAMX checks for AMX support via CPUID
func detectAMX() {
	// Check if we're on Intel
	if cpu.X86.HasAVX512F {
		// AMX detection requires checking specific CPUID leaves
		// For now, we'll use a simplified check
		// Real implementation would use CPUID directly
		
		// These would be set by actual CPUID checks:
		// CPUID.07H:EDX[bit 24] = AMX-BF16
		// CPUID.07H:EDX[bit 25] = AMX-TILE  
		// CPUID.07H:EDX[bit 25] = AMX-INT8
		
		// For development, we'll assume false unless on Sapphire Rapids
		hasAMXTile = false
		hasAMXInt8 = false
		hasAMXBF16 = false
	}
}

// HasAMX returns true if any AMX features are available
func HasAMX() bool {
	return hasAMXTile
}

// HasAMXInt8 returns true if AMX INT8 operations are available
func HasAMXInt8() bool {
	return hasAMXInt8
}

// HasAMXBF16 returns true if AMX BF16 operations are available
func HasAMXBF16() bool {
	return hasAMXBF16
}

// SetAMXSupport allows manual override for testing
func SetAMXSupport(tile, int8, bf16 bool) {
	hasAMXTile = tile
	hasAMXInt8 = int8
	hasAMXBF16 = bf16
}