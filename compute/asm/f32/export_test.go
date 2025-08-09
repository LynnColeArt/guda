//go:build amd64 && !noasm
// +build amd64,!noasm

package f32

import "unsafe"

// Export for testing
func CallKernel16x4(a, b, c unsafe.Pointer, kc int64, ldc int64) {
	sgemmKernel16x4AVX512(a, b, c, kc, ldc)
}