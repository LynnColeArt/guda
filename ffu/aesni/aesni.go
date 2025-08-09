package aesni

import (
	"crypto/aes"
	"crypto/cipher"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
)

// AESNIFFU implements hardware-accelerated AES using AES-NI instructions
type AESNIFFU struct {
	available bool
	metrics   atomic.Value // *ffu.Metrics
	mu        sync.Mutex
}

// NewAESNIFFU creates a new AES-NI FFU
func NewAESNIFFU() *AESNIFFU {
	a := &AESNIFFU{
		available: hasAESNI(),
	}
	
	// Initialize metrics
	a.metrics.Store(&ffu.Metrics{
		LastUsed: time.Now(),
	})
	
	return a
}

// Name returns the FFU name
func (a *AESNIFFU) Name() string {
	return "AES-NI"
}

// Type returns the FFU type
func (a *AESNIFFU) Type() ffu.FFUType {
	return ffu.FFUTypeAESNI
}

// CanHandle checks if this FFU can handle the workload
func (a *AESNIFFU) CanHandle(workload ffu.Workload) bool {
	if !a.available {
		return false
	}
	
	aesWork, ok := workload.(*ffu.AESWorkload)
	if !ok || workload.Type() != "crypto_aes" {
		return false
	}
	
	// We can handle all standard AES operations
	// crypto/aes automatically uses AES-NI when available
	return aesWork.Validate() == nil
}

// EstimateCost estimates the cost of executing the workload
func (a *AESNIFFU) EstimateCost(workload ffu.Workload) ffu.Cost {
	size := workload.Size()
	
	// AES-NI throughput estimates (conservative)
	// Based on real-world measurements
	throughputMBps := 5000.0 // 5 GB/s for AES-NI
	
	duration := time.Duration(float64(size) / (throughputMBps * 1e6) * 1e9)
	
	// Energy estimate: ~0.5 nJ per byte for AES-NI
	energy := float64(size) * 0.5e-9
	
	return ffu.Cost{
		Duration:        duration,
		Energy:          energy,
		MemoryBandwidth: size * 2, // Read input + write output
		Confidence:      0.9,      // High confidence for AES-NI
	}
}

// Execute runs the workload on this FFU
func (a *AESNIFFU) Execute(workload ffu.Workload) error {
	aesWork, ok := workload.(*ffu.AESWorkload)
	if !ok {
		return fmt.Errorf("invalid workload type: expected AESWorkload")
	}
	
	if err := aesWork.Validate(); err != nil {
		return fmt.Errorf("invalid workload: %w", err)
	}
	
	start := time.Now()
	defer a.updateMetrics(start, int64(len(aesWork.Input)), nil)
	
	// Create cipher
	block, err := aes.NewCipher(aesWork.Key)
	if err != nil {
		a.updateMetrics(start, 0, err)
		return fmt.Errorf("failed to create cipher: %w", err)
	}
	
	// Execute based on mode
	switch aesWork.Mode {
	case ffu.AESModeECB:
		return a.executeECB(block, aesWork)
	case ffu.AESModeCBC:
		return a.executeCBC(block, aesWork)
	case ffu.AESModeCTR:
		return a.executeCTR(block, aesWork)
	case ffu.AESModeGCM:
		return a.executeGCM(block, aesWork)
	default:
		return fmt.Errorf("unsupported AES mode: %v", aesWork.Mode)
	}
}

// executeECB handles ECB mode (not recommended for real use!)
func (a *AESNIFFU) executeECB(block cipher.Block, work *ffu.AESWorkload) error {
	// ECB mode - process each block independently
	blockSize := block.BlockSize()
	
	for i := 0; i < len(work.Input); i += blockSize {
		if work.Operation == ffu.AESEncrypt {
			block.Encrypt(work.Output[i:i+blockSize], work.Input[i:i+blockSize])
		} else {
			block.Decrypt(work.Output[i:i+blockSize], work.Input[i:i+blockSize])
		}
	}
	
	return nil
}

// executeCBC handles CBC mode
func (a *AESNIFFU) executeCBC(block cipher.Block, work *ffu.AESWorkload) error {
	if work.Operation == ffu.AESEncrypt {
		mode := cipher.NewCBCEncrypter(block, work.IV)
		mode.CryptBlocks(work.Output, work.Input)
	} else {
		mode := cipher.NewCBCDecrypter(block, work.IV)
		mode.CryptBlocks(work.Output, work.Input)
	}
	
	return nil
}

// executeCTR handles CTR mode
func (a *AESNIFFU) executeCTR(block cipher.Block, work *ffu.AESWorkload) error {
	// CTR mode works the same for encryption and decryption
	stream := cipher.NewCTR(block, work.IV)
	stream.XORKeyStream(work.Output, work.Input)
	
	return nil
}

// executeGCM handles GCM mode
func (a *AESNIFFU) executeGCM(block cipher.Block, work *ffu.AESWorkload) error {
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return fmt.Errorf("failed to create GCM: %w", err)
	}
	
	if work.Operation == ffu.AESEncrypt {
		// For GCM encryption, output needs extra space for tag
		if len(work.Output) < len(work.Input)+gcm.Overhead() {
			return fmt.Errorf("output buffer too small for GCM tag")
		}
		
		// Use provided IV/nonce
		nonce := work.IV[:gcm.NonceSize()]
		
		// Encrypt and append tag
		result := gcm.Seal(work.Output[:0], nonce, work.Input, work.AAD)
		if len(result) != len(work.Input)+gcm.Overhead() {
			return fmt.Errorf("unexpected GCM output size")
		}
	} else {
		// For GCM decryption, input includes tag
		if len(work.Input) < gcm.Overhead() {
			return fmt.Errorf("input too small for GCM tag")
		}
		
		nonce := work.IV[:gcm.NonceSize()]
		
		// Decrypt and verify tag
		result, err := gcm.Open(work.Output[:0], nonce, work.Input, work.AAD)
		if err != nil {
			return fmt.Errorf("GCM decryption failed: %w", err)
		}
		
		if len(result) != len(work.Input)-gcm.Overhead() {
			return fmt.Errorf("unexpected GCM output size")
		}
	}
	
	return nil
}

// IsAvailable checks if the FFU is currently available
func (a *AESNIFFU) IsAvailable() bool {
	return a.available
}

// Metrics returns performance metrics for this FFU
func (a *AESNIFFU) Metrics() ffu.Metrics {
	m := a.metrics.Load().(*ffu.Metrics)
	return *m
}

// updateMetrics updates the performance metrics
func (a *AESNIFFU) updateMetrics(start time.Time, bytes int64, err error) {
	duration := time.Since(start)
	
	a.mu.Lock()
	defer a.mu.Unlock()
	
	m := a.metrics.Load().(*ffu.Metrics)
	newMetrics := *m
	
	newMetrics.WorkloadCount++
	newMetrics.BytesProcessed += bytes
	newMetrics.TotalDuration += duration
	newMetrics.LastUsed = time.Now()
	
	if err != nil {
		newMetrics.ErrorCount++
		newMetrics.LastError = err
	}
	
	a.metrics.Store(&newMetrics)
}

// GetCapabilities returns AES-specific capabilities
func (a *AESNIFFU) GetCapabilities() ffu.AESCapability {
	return ffu.AESCapability{
		KeySizes: []int{128, 192, 256},
		Modes: []ffu.AESMode{
			ffu.AESModeECB,
			ffu.AESModeCBC,
			ffu.AESModeCTR,
			ffu.AESModeGCM,
		},
		ThroughputMBps: map[ffu.AESMode]float64{
			ffu.AESModeECB: 6000.0, // 6 GB/s
			ffu.AESModeCBC: 5000.0, // 5 GB/s
			ffu.AESModeCTR: 5500.0, // 5.5 GB/s
			ffu.AESModeGCM: 3000.0, // 3 GB/s (includes auth)
		},
		HardwareAccelerated: a.available,
		SupportsParallel:    true,
		MaxParallel:         8, // Reasonable default
	}
}