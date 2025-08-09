//go:build amd64
// +build amd64

package amx

import (
	_ "golang.org/x/sys/cpu"
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
	// Use our assembly function to check CPUID
	if amxCheckSupport() {
		hasAMXTile = true
		hasAMXInt8 = true
		hasAMXBF16 = true
	} else {
		// Check if we're in test mode
		// This allows testing on non-AMX hardware
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