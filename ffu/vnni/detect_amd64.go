//go:build amd64
// +build amd64

package vnni

import (
	"golang.org/x/sys/cpu"
)

var (
	hasVNNI = false
	hasBF16 = false
)

func init() {
	detectVNNI()
}

// detectVNNI checks for AVX512-VNNI support
func detectVNNI() {
	// Check for AVX512-VNNI
	hasVNNI = cpu.X86.HasAVX512VNNI
	
	// Check for AVX512-BF16 while we're at it
	hasBF16 = cpu.X86.HasAVX512BF16
}

// HasVNNI returns true if AVX512-VNNI is available
func HasVNNI() bool {
	return hasVNNI
}

// HasBF16 returns true if AVX512-BF16 is available
func HasBF16() bool {
	return hasBF16
}

// For testing
func SetVNNISupport(vnni, bf16 bool) {
	hasVNNI = vnni
	hasBF16 = bf16
}