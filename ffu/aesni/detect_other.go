//go:build !amd64
// +build !amd64

package aesni

// hasAESNI returns false on non-AMD64 platforms
func hasAESNI() bool {
	return false
}