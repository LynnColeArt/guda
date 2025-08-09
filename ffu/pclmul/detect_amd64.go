//go:build amd64
// +build amd64

package pclmul

import (
	"golang.org/x/sys/cpu"
)

var (
	hasPCLMUL  = false
	hasVPCLMUL = false
)

func init() {
	detectPCLMUL()
}

// detectPCLMUL checks for PCLMULQDQ support
func detectPCLMUL() {
	// PCLMULQDQ - carryless multiplication
	hasPCLMUL = cpu.X86.HasPCLMULQDQ
	
	// VPCLMULQDQ - AVX512 version (4x wider)
	hasVPCLMUL = cpu.X86.HasAVX512VPCLMULQDQ
}

// HasPCLMUL returns true if PCLMULQDQ is available
func HasPCLMUL() bool {
	return hasPCLMUL
}

// HasVPCLMUL returns true if VPCLMULQDQ (AVX512) is available
func HasVPCLMUL() bool {
	return hasVPCLMUL
}