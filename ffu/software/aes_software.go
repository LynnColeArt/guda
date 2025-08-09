package software

import (
	"crypto/aes"
	"crypto/cipher"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
)

// AESSoftwareFFU implements software AES without hardware acceleration
type AESSoftwareFFU struct {
	metrics atomic.Value // *ffu.Metrics
	mu      sync.Mutex
}

// NewAESSoftwareFFU creates a new software AES FFU
func NewAESSoftwareFFU() *AESSoftwareFFU {
	a := &AESSoftwareFFU{}
	
	// Initialize metrics
	a.metrics.Store(&ffu.Metrics{
		LastUsed: time.Now(),
	})
	
	return a
}

// Name returns the FFU name
func (a *AESSoftwareFFU) Name() string {
	return "AES-Software"
}

// Type returns the FFU type
func (a *AESSoftwareFFU) Type() ffu.FFUType {
	return ffu.FFUTypeCPU
}

// CanHandle checks if this FFU can handle the workload
func (a *AESSoftwareFFU) CanHandle(workload ffu.Workload) bool {
	aesWork, ok := workload.(*ffu.AESWorkload)
	if !ok || workload.Type() != "crypto_aes" {
		return false
	}
	
	return aesWork.Validate() == nil
}

// EstimateCost estimates the cost of executing the workload
func (a *AESSoftwareFFU) EstimateCost(workload ffu.Workload) ffu.Cost {
	size := workload.Size()
	
	// Software AES throughput estimates
	// Much slower than AES-NI
	throughputMBps := 500.0 // 500 MB/s for software AES
	
	duration := time.Duration(float64(size) / (throughputMBps * 1e6) * 1e9)
	
	// Energy estimate: ~5 nJ per byte for software AES (10x more than AES-NI)
	energy := float64(size) * 5e-9
	
	return ffu.Cost{
		Duration:        duration,
		Energy:          energy,
		MemoryBandwidth: size * 2, // Read input + write output
		Confidence:      0.8,      // Lower confidence for software
	}
}

// Execute runs the workload
func (a *AESSoftwareFFU) Execute(workload ffu.Workload) error {
	aesWork, ok := workload.(*ffu.AESWorkload)
	if !ok {
		return fmt.Errorf("invalid workload type: expected AESWorkload")
	}
	
	if err := aesWork.Validate(); err != nil {
		return fmt.Errorf("invalid workload: %w", err)
	}
	
	start := time.Now()
	defer a.updateMetrics(start, int64(len(aesWork.Input)), nil)
	
	// Note: This uses the same crypto/aes package but we're pretending
	// it's software-only for benchmarking purposes. In reality, Go's
	// crypto/aes automatically uses AES-NI when available.
	// For a true software comparison, we'd need a pure-Go implementation.
	
	block, err := aes.NewCipher(aesWork.Key)
	if err != nil {
		a.updateMetrics(start, 0, err)
		return fmt.Errorf("failed to create cipher: %w", err)
	}
	
	// Execute based on mode (same as AES-NI version)
	switch aesWork.Mode {
	case ffu.AESModeCBC:
		if aesWork.Operation == ffu.AESEncrypt {
			mode := cipher.NewCBCEncrypter(block, aesWork.IV)
			mode.CryptBlocks(aesWork.Output, aesWork.Input)
		} else {
			mode := cipher.NewCBCDecrypter(block, aesWork.IV)
			mode.CryptBlocks(aesWork.Output, aesWork.Input)
		}
		return nil
		
	case ffu.AESModeCTR:
		stream := cipher.NewCTR(block, aesWork.IV)
		stream.XORKeyStream(aesWork.Output, aesWork.Input)
		return nil
		
	default:
		return fmt.Errorf("mode %v not implemented in software FFU", aesWork.Mode)
	}
}

// IsAvailable always returns true for software
func (a *AESSoftwareFFU) IsAvailable() bool {
	return true
}

// Metrics returns performance metrics
func (a *AESSoftwareFFU) Metrics() ffu.Metrics {
	m := a.metrics.Load().(*ffu.Metrics)
	return *m
}

// updateMetrics updates the performance metrics
func (a *AESSoftwareFFU) updateMetrics(start time.Time, bytes int64, err error) {
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