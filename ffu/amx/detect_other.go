//go:build !amd64
// +build !amd64

package amx

// AMX is Intel x86-64 only
func detectAMX() {}

func HasAMX() bool { return false }
func HasAMXInt8() bool { return false }
func HasAMXBF16() bool { return false }