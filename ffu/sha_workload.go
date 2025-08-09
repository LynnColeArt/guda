package ffu

import (
	"fmt"
)

// SHAAlgorithm represents the SHA variant
type SHAAlgorithm int

const (
	SHA1 SHAAlgorithm = iota
	SHA256
	SHA512 // Not hardware accelerated, but for completeness
)

func (a SHAAlgorithm) String() string {
	switch a {
	case SHA1:
		return "SHA-1"
	case SHA256:
		return "SHA-256"
	case SHA512:
		return "SHA-512"
	default:
		return "Unknown"
	}
}

// SHAWorkload represents a SHA hashing workload
type SHAWorkload struct {
	Algorithm SHAAlgorithm
	Data      []byte
	Output    []byte // Pre-allocated output buffer
}

// Type returns the workload type
func (w *SHAWorkload) Type() string {
	return fmt.Sprintf("sha_%s", w.Algorithm)
}

// Size returns the size of the workload in bytes
func (w *SHAWorkload) Size() int64 {
	return int64(len(w.Data))
}

// Validate checks if the workload is valid
func (w *SHAWorkload) Validate() error {
	if len(w.Data) == 0 {
		return fmt.Errorf("empty data")
	}
	
	// Check output buffer size
	requiredSize := 0
	switch w.Algorithm {
	case SHA1:
		requiredSize = 20 // 160 bits
	case SHA256:
		requiredSize = 32 // 256 bits
	case SHA512:
		requiredSize = 64 // 512 bits
	default:
		return fmt.Errorf("unsupported algorithm: %v", w.Algorithm)
	}
	
	if len(w.Output) < requiredSize {
		return fmt.Errorf("output buffer too small: %d < %d", len(w.Output), requiredSize)
	}
	
	return nil
}