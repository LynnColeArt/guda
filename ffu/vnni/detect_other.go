//go:build !amd64
// +build !amd64

package vnni

// VNNI is x86-64 only
func detectVNNI() {}

func HasVNNI() bool { return false }
func HasBF16() bool { return false }
func SetVNNISupport(vnni, bf16 bool) {}