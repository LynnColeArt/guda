//go:build amd64 && !cgo
// +build amd64,!cgo

package vnni

// Assembly function declarations

//go:noescape
func vnniInt8DotProduct(a, b []int8) int32

//go:noescape
func vnniInt8GEMMRef(m, n, k int, a, b []int8, c []int32)

//go:noescape
func vnniInt8GEMMKernel(m, n, k int, a, b []int8, c []int32)

//go:noescape
func vnniInt8GEMM32x32(a, b []int8, c []int32)

//go:noescape
func vnniInt8GEMM16x16(a, b []int8, c []int32)

//go:noescape
func vnniBreakthroughKernel(a, b []int8, c []int32)

//go:noescape
func vnniMemoryBreakthrough(m, n, k int, a, b []int8, c []int32)

//go:noescape
func vnniFinalKernel(m, n, k int, a, b []int8, c []int32)

//go:noescape
func vnniFinal256x256x256(a, b []int8, c []int32)