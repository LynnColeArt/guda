//go:build !amd64
// +build !amd64

package sha

// SHA-NI is x86-64 only
func detectSHA() {}

func HasSHA() bool { return false }

func SetSHASupport(enabled bool) {}