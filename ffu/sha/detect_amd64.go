//go:build amd64
// +build amd64

package sha


var (
	hasSHA = false
)

// Assembly function
//go:noescape
func hasSHANI() bool

func init() {
	detectSHA()
}

// detectSHA checks for SHA-NI support
func detectSHA() {
	// SHA-NI is indicated by the "sha_ni" flag in /proc/cpuinfo
	// For now, we'll use our assembly detection
	hasSHA = hasSHANI()
}

// HasSHA returns true if SHA-NI is available
func HasSHA() bool {
	return hasSHA
}

// For testing
func SetSHASupport(enabled bool) {
	hasSHA = enabled
}