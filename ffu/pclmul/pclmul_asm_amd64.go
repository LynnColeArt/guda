//go:build amd64 && !cgo
// +build amd64,!cgo

package pclmul

// Assembly function declarations

//go:noescape
func crc32PCLMUL(data []byte, polynomial uint64) uint64

//go:noescape
func crc32VPCLMUL(data []byte, polynomial uint64) uint64

//go:noescape
func rsEncodePCLMUL(data []byte, parity []byte, matrix []byte)

//go:noescape
func galoisMulPCLMUL(a, b []byte, result []byte)