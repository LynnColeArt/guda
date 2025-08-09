package ffu

import (
	"fmt"
)

// AESMode represents AES encryption mode
type AESMode int

const (
	AESModeECB AESMode = iota
	AESModeCBC
	AESModeCTR
	AESModeGCM
)

func (m AESMode) String() string {
	switch m {
	case AESModeECB:
		return "ECB"
	case AESModeCBC:
		return "CBC"
	case AESModeCTR:
		return "CTR"
	case AESModeGCM:
		return "GCM"
	default:
		return "Unknown"
	}
}

// AESOperation represents the AES operation type
type AESOperation int

const (
	AESEncrypt AESOperation = iota
	AESDecrypt
)

// AESWorkload represents an AES encryption/decryption workload
type AESWorkload struct {
	Operation AESOperation
	Mode      AESMode
	Key       []byte // 128, 192, or 256 bit
	IV        []byte // Initialization vector (for modes that need it)
	Input     []byte // Plaintext or ciphertext
	Output    []byte // Result buffer (must be pre-allocated)
	
	// Optional: Additional authenticated data for GCM mode
	AAD []byte
}

// Type returns the workload type
func (w *AESWorkload) Type() string {
	return "crypto_aes"
}

// Size returns the size of the workload in bytes
func (w *AESWorkload) Size() int64 {
	return int64(len(w.Input))
}

// Validate checks if the workload is valid
func (w *AESWorkload) Validate() error {
	// Validate key size
	keyLen := len(w.Key)
	if keyLen != 16 && keyLen != 24 && keyLen != 32 {
		return fmt.Errorf("invalid AES key size: %d bytes (must be 16, 24, or 32)", keyLen)
	}
	
	// Validate IV for modes that need it
	switch w.Mode {
	case AESModeCBC, AESModeCTR, AESModeGCM:
		if len(w.IV) != 16 {
			return fmt.Errorf("invalid IV size for %s mode: %d bytes (must be 16)", w.Mode, len(w.IV))
		}
	}
	
	// Validate input/output buffers
	if len(w.Input) == 0 {
		return fmt.Errorf("empty input buffer")
	}
	
	if len(w.Output) < len(w.Input) {
		return fmt.Errorf("output buffer too small: %d bytes (need at least %d)", len(w.Output), len(w.Input))
	}
	
	// For ECB and CBC modes, input must be multiple of block size
	if w.Mode == AESModeECB || w.Mode == AESModeCBC {
		if len(w.Input)%16 != 0 {
			return fmt.Errorf("input size must be multiple of 16 for %s mode", w.Mode)
		}
	}
	
	return nil
}

// AESCapability describes AES-specific capabilities
type AESCapability struct {
	// Supported key sizes in bits
	KeySizes []int
	
	// Supported modes
	Modes []AESMode
	
	// Maximum throughput in MB/s
	ThroughputMBps map[AESMode]float64
	
	// Hardware acceleration available
	HardwareAccelerated bool
	
	// Supports parallel operations
	SupportsParallel bool
	
	// Maximum parallel operations
	MaxParallel int
}