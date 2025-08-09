package vnni

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
)


// VNNIFFU implements the FFU interface for AVX512-VNNI operations
type VNNIFFU struct {
	available bool
	hasVNNI   bool
	hasBF16   bool
	metrics   atomic.Value // *ffu.Metrics
}

// NewVNNIFFU creates a new VNNI FFU instance
func NewVNNIFFU() *VNNIFFU {
	v := &VNNIFFU{
		available: HasVNNI(),
		hasVNNI:   HasVNNI(),
		hasBF16:   HasBF16(),
	}
	
	// Initialize metrics
	v.metrics.Store(&ffu.Metrics{
		LastUsed: time.Now(),
	})
	
	return v
}

// Name returns the FFU name
func (v *VNNIFFU) Name() string {
	return "AVX512-VNNI"
}

// Type returns the FFU type
func (v *VNNIFFU) Type() ffu.FFUType {
	return ffu.FFUTypeAVX512VNNI
}

// CanHandle checks if this FFU can handle the workload
func (v *VNNIFFU) CanHandle(workload ffu.Workload) bool {
	if !v.available {
		return false
	}
	
	switch w := workload.(type) {
	case *ffu.VNNIWorkload:
		// VNNI works best with K divisible by 64 (ZMM width)
		return w.K >= 64 && w.K%16 == 0
	default:
		return false
	}
}

// EstimateCost estimates the cost of executing the workload
func (v *VNNIFFU) EstimateCost(workload ffu.Workload) ffu.Cost {
	if !v.CanHandle(workload) {
		return ffu.Cost{
			Duration:        time.Hour,
			Energy:          1e9,
			MemoryBandwidth: 0,
			Confidence:      0,
		}
	}
	
	w := workload.(*ffu.VNNIWorkload)
	
	// VNNI can achieve ~200-400 GOPS for INT8
	// Estimate based on matrix size
	ops := w.Operations()
	gopsRate := 300e9 // 300 GOPS average
	
	duration := float64(ops) / gopsRate
	
	return ffu.Cost{
		Duration:        time.Duration(duration * float64(time.Second)),
		Energy:          duration * 20.0, // ~20W for VNNI operations
		MemoryBandwidth: int64(float64(w.Size()) / duration),
		Confidence:      0.85,
	}
}

// Execute performs the VNNI operation
func (v *VNNIFFU) Execute(workload ffu.Workload) error {
	if !v.CanHandle(workload) {
		return fmt.Errorf("VNNI FFU cannot handle this workload")
	}
	
	w := workload.(*ffu.VNNIWorkload)
	
	// Validate workload
	if err := w.Validate(); err != nil {
		return fmt.Errorf("invalid workload: %w", err)
	}
	
	// Update metrics
	metrics := v.metrics.Load().(*ffu.Metrics)
	newMetrics := *metrics
	newMetrics.WorkloadCount++
	newMetrics.BytesProcessed += w.Size()
	newMetrics.LastUsed = time.Now()
	
	// Execute
	start := time.Now()
	var err error
	
	switch w.Operation {
	case ffu.VNNIDotProduct:
		if w.M == 1 && w.N == 1 {
			// Special case: dot product
			result := vnniInt8DotProduct(w.A, w.B)
			w.C[0] = result * w.Alpha
		} else {
			err = fmt.Errorf("dot product requires M=1, N=1")
		}
		
	case ffu.VNNIMatMul:
		// Use optimized VNNI kernel when available
		if runtime.GOARCH == "amd64" && v.hasVNNI {
			if vnniCGOAvailable {
				// Use real VNNI with intrinsics!
				vnniInt8GEMMCgo(w.M, w.N, w.K, w.A, w.B, w.C)
			} else if w.M == 32 && w.N == 32 && w.K == 32 {
				// Special case for 32x32 matrices
				vnniInt8GEMM32x32(w.A, w.B, w.C)
			} else {
				// Use reference assembly
				vnniInt8GEMMRef(w.M, w.N, w.K, w.A, w.B, w.C)
			}
		} else {
			err = v.executeReferenceGEMM(w)
		}
		// Apply scaling
		if w.Alpha != 1 {
			for i := range w.C {
				w.C[i] *= w.Alpha
			}
		}
		
	default:
		err = fmt.Errorf("unsupported operation: %v", w.Operation)
	}
	
	newMetrics.TotalDuration += time.Since(start)
	if err != nil {
		newMetrics.ErrorCount++
		newMetrics.LastError = err
	}
	
	v.metrics.Store(&newMetrics)
	return err
}

// IsAvailable returns true if VNNI is available
func (v *VNNIFFU) IsAvailable() bool {
	return v.available
}

// Metrics returns performance metrics
func (v *VNNIFFU) Metrics() ffu.Metrics {
	return *v.metrics.Load().(*ffu.Metrics)
}

// executeReferenceGEMM performs INT8 GEMM in software
func (v *VNNIFFU) executeReferenceGEMM(w *ffu.VNNIWorkload) error {
	// Clear C
	for i := range w.C {
		w.C[i] = 0
	}
	
	// Triple nested loop
	for i := 0; i < w.M; i++ {
		for j := 0; j < w.N; j++ {
			sum := int32(0)
			for k := 0; k < w.K; k++ {
				sum += int32(w.A[i*w.K+k]) * int32(w.B[k*w.N+j])
			}
			w.C[i*w.N+j] = sum * w.Alpha
		}
	}
	
	return nil
}