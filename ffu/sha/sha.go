package sha

import (
	"crypto/sha1"
	"crypto/sha256"
	"fmt"
	"sync/atomic"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
)

// SHAFFU implements the FFU interface for SHA operations
type SHAFFU struct {
	available bool
	metrics   atomic.Value // *ffu.Metrics
}

// NewSHAFFU creates a new SHA FFU instance
func NewSHAFFU() *SHAFFU {
	s := &SHAFFU{
		available: HasSHA(),
	}
	
	// Initialize metrics
	s.metrics.Store(&ffu.Metrics{
		LastUsed: time.Now(),
	})
	
	return s
}

// Name returns the FFU name
func (s *SHAFFU) Name() string {
	return "SHA-NI"
}

// Type returns the FFU type
func (s *SHAFFU) Type() ffu.FFUType {
	return ffu.FFUTypeSHA
}

// CanHandle checks if this FFU can handle the workload
func (s *SHAFFU) CanHandle(workload ffu.Workload) bool {
	if !s.available {
		return false
	}
	
	switch w := workload.(type) {
	case *ffu.SHAWorkload:
		// SHA-NI supports SHA-1 and SHA-256
		return w.Algorithm == ffu.SHA1 || w.Algorithm == ffu.SHA256
	default:
		return false
	}
}

// EstimateCost estimates the cost of executing the workload
func (s *SHAFFU) EstimateCost(workload ffu.Workload) ffu.Cost {
	if !s.CanHandle(workload) {
		return ffu.Cost{
			Duration:        time.Hour,
			Energy:          1e9,
			MemoryBandwidth: 0,
			Confidence:      0,
		}
	}
	
	w := workload.(*ffu.SHAWorkload)
	
	// SHA-NI can process ~1-2 GB/s
	throughput := 1.5e9 // 1.5 GB/s average
	duration := float64(w.Size()) / throughput
	
	return ffu.Cost{
		Duration:        time.Duration(duration * float64(time.Second)),
		Energy:          duration * 5.0, // ~5W for SHA operations
		MemoryBandwidth: int64(throughput),
		Confidence:      0.9,
	}
}

// Execute performs the SHA operation
func (s *SHAFFU) Execute(workload ffu.Workload) error {
	if !s.CanHandle(workload) {
		return fmt.Errorf("SHA FFU cannot handle this workload")
	}
	
	w := workload.(*ffu.SHAWorkload)
	
	// Validate workload
	if err := w.Validate(); err != nil {
		return fmt.Errorf("invalid workload: %w", err)
	}
	
	// Update metrics
	metrics := s.metrics.Load().(*ffu.Metrics)
	newMetrics := *metrics
	newMetrics.WorkloadCount++
	newMetrics.BytesProcessed += w.Size()
	newMetrics.LastUsed = time.Now()
	
	// Execute based on algorithm
	var err error
	start := time.Now()
	
	switch w.Algorithm {
	case ffu.SHA1:
		if s.available {
			err = s.executeSHA1NI(w)
		} else {
			err = s.executeSHA1Software(w)
		}
	case ffu.SHA256:
		if s.available {
			err = s.executeSHA256NI(w)
		} else {
			err = s.executeSHA256Software(w)
		}
	default:
		err = fmt.Errorf("unsupported algorithm: %v", w.Algorithm)
	}
	
	newMetrics.TotalDuration += time.Since(start)
	if err != nil {
		newMetrics.ErrorCount++
		newMetrics.LastError = err
	}
	
	s.metrics.Store(&newMetrics)
	return err
}

// IsAvailable returns true if SHA-NI is available
func (s *SHAFFU) IsAvailable() bool {
	return s.available
}

// Metrics returns performance metrics
func (s *SHAFFU) Metrics() ffu.Metrics {
	return *s.metrics.Load().(*ffu.Metrics)
}

// executeSHA1NI performs SHA-1 using hardware acceleration
func (s *SHAFFU) executeSHA1NI(w *ffu.SHAWorkload) error {
	// TODO: Call assembly implementation
	// For now, fall back to software
	return s.executeSHA1Software(w)
}

// executeSHA256NI performs SHA-256 using hardware acceleration
func (s *SHAFFU) executeSHA256NI(w *ffu.SHAWorkload) error {
	// TODO: Call assembly implementation
	// For now, fall back to software
	return s.executeSHA256Software(w)
}

// executeSHA1Software performs SHA-1 in software
func (s *SHAFFU) executeSHA1Software(w *ffu.SHAWorkload) error {
	h := sha1.Sum(w.Data)
	copy(w.Output, h[:])
	return nil
}

// executeSHA256Software performs SHA-256 in software
func (s *SHAFFU) executeSHA256Software(w *ffu.SHAWorkload) error {
	h := sha256.Sum256(w.Data)
	copy(w.Output, h[:])
	return nil
}