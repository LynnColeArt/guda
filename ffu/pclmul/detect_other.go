//go:build !amd64
// +build !amd64

package pclmul

// HasPCLMUL returns false on non-AMD64 architectures
func HasPCLMUL() bool {
	return false
}

// HasVPCLMUL returns false on non-AMD64 architectures
func HasVPCLMUL() bool {
	return false
}