//go:build amd64
// +build amd64

package aesni

import (
	"golang.org/x/sys/cpu"
)

// hasAESNI detects if the CPU supports AES-NI instructions
func hasAESNI() bool {
	return cpu.X86.HasAES
}