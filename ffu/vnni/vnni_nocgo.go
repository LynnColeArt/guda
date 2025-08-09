//go:build !cgo || !amd64
// +build !cgo !amd64

package vnni

// vnniCGOAvailable indicates if CGO VNNI is available
var vnniCGOAvailable = false

// vnniInt8GEMMCgo fallback when CGO is not available
func vnniInt8GEMMCgo(m, n, k int, a, b []int8, c []int32) {
	// Fall back to reference implementation
	vnniInt8GEMMRef(m, n, k, a, b, c)
}