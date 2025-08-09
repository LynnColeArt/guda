package amx

import (
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/LynnColeArt/guda/ffu"
)

// AMXFFU implements the FFU interface for Intel AMX operations
type AMXFFU struct {
	available  bool
	hasInt8    bool
	hasBF16    bool
	tileConfig TileConfig
	metrics    atomic.Value // *ffu.Metrics
}

// TileConfig represents AMX tile configuration
type TileConfig struct {
	NumTiles     int
	MaxRows      int
	MaxCols      int
	MaxTileBytes int
}

// NewAMXFFU creates a new AMX FFU instance
func NewAMXFFU() *AMXFFU {
	amx := &AMXFFU{
		available: HasAMX(),
		hasInt8:   HasAMXInt8(),
		hasBF16:   HasAMXBF16(),
	}
	
	if amx.available {
		// AMX tile configuration for Sapphire Rapids
		amx.tileConfig = TileConfig{
			NumTiles:     8,
			MaxRows:      16,
			MaxCols:      64, // For INT8
			MaxTileBytes: 1024,
		}
	}
	
	// Initialize metrics
	amx.metrics.Store(&ffu.Metrics{
		LastUsed: time.Now(),
	})
	
	return amx
}

// Name returns the FFU name
func (a *AMXFFU) Name() string {
	return "Intel AMX"
}

// Type returns the FFU type
func (a *AMXFFU) Type() ffu.FFUType {
	return ffu.FFUTypeAMX
}

// CanHandle checks if this FFU can handle the workload
func (a *AMXFFU) CanHandle(workload ffu.Workload) bool {
	if !a.available {
		return false
	}
	
	switch w := workload.(type) {
	case *ffu.AMXWorkload:
		// Check data type support
		switch w.DataType {
		case ffu.AMXInt8:
			if !a.hasInt8 {
				return false
			}
		case ffu.AMXBFloat16:
			if !a.hasBF16 {
				return false
			}
		default:
			return false
		}
		
		// Check size constraints
		// AMX works best with tile-sized operations
		if w.M < 16 || w.N < 16 || w.K < 64 {
			return false // Too small for AMX
		}
		
		// Check alignment requirements
		// AMX requires 16-byte aligned data
		if !isAligned(unsafe.Pointer(&w.A[0]), 16) ||
		   !isAligned(unsafe.Pointer(&w.B[0]), 16) ||
		   !isAligned(unsafe.Pointer(&w.C[0]), 16) {
			return false
		}
		
		return true
	default:
		return false
	}
}

// EstimateCost estimates the cost of executing the workload
func (a *AMXFFU) EstimateCost(workload ffu.Workload) ffu.Cost {
	if !a.CanHandle(workload) {
		return ffu.Cost{
			Duration:        time.Hour * 24 * 365, // Effectively infinite
			Energy:          1e9,
			MemoryBandwidth: 0,
			Confidence:      0,
		}
	}
	
	w := workload.(*ffu.AMXWorkload)
	
	// Estimate based on peak performance
	var peakOpsPerSec float64
	switch w.DataType {
	case ffu.AMXInt8:
		peakOpsPerSec = 2e12 // 2 TOPS
	case ffu.AMXBFloat16:
		peakOpsPerSec = 1e12 // 1 TFLOPS
	}
	
	// Calculate operations
	ops := w.Operations()
	estimatedTime := float64(ops) / peakOpsPerSec
	
	// AMX is very power efficient for matrix ops
	// Estimate ~50W for full AMX utilization
	estimatedEnergy := estimatedTime * 50.0
	
	// Calculate throughput
	throughput := float64(w.Size()) / estimatedTime
	
	return ffu.Cost{
		Duration:        time.Duration(estimatedTime * float64(time.Second)),
		Energy:          estimatedEnergy,
		MemoryBandwidth: int64(throughput),
		Confidence:      0.85, // Good confidence for AMX estimates
	}
}

// Execute performs the AMX operation
func (a *AMXFFU) Execute(workload ffu.Workload) error {
	if !a.CanHandle(workload) {
		return fmt.Errorf("AMX FFU cannot handle this workload")
	}
	
	w := workload.(*ffu.AMXWorkload)
	
	// Validate workload
	if err := w.Validate(); err != nil {
		return fmt.Errorf("invalid workload: %w", err)
	}
	
	// Update metrics
	metrics := a.metrics.Load().(*ffu.Metrics)
	newMetrics := *metrics
	newMetrics.WorkloadCount++
	newMetrics.BytesProcessed += w.Size()
	newMetrics.LastUsed = time.Now()
	a.metrics.Store(&newMetrics)
	
	// Execute based on data type
	switch w.DataType {
	case ffu.AMXInt8:
		return a.executeInt8(w)
	case ffu.AMXBFloat16:
		return a.executeBF16(w)
	default:
		return fmt.Errorf("unsupported data type: %v", w.DataType)
	}
}

// executeInt8 performs INT8 matrix multiplication using AMX
func (a *AMXFFU) executeInt8(w *ffu.AMXWorkload) error {
	// Check if we have real AMX support
	if amxCheckSupport() && a.hasInt8 {
		// Use AMX kernel
		kernel := NewAMXKernel()
		defer kernel.Release()
		
		// Convert byte slices to int8 for the kernel
		aInt8 := (*[1 << 30]int8)(unsafe.Pointer(&w.A[0]))[:w.M*w.K]
		bInt8 := (*[1 << 30]int8)(unsafe.Pointer(&w.B[0]))[:w.K*w.N]
		cInt32 := (*[1 << 30]int32)(unsafe.Pointer(&w.C[0]))[:w.M*w.N]
		
		// Clear C matrix
		for i := range cInt32 {
			cInt32[i] = 0
		}
		
		// Convert []int8 to []byte for assembly
		aBytes := (*[1 << 30]byte)(unsafe.Pointer(&aInt8[0]))[:len(aInt8)]
		bBytes := (*[1 << 30]byte)(unsafe.Pointer(&bInt8[0]))[:len(bInt8)]
		
		// Execute AMX kernel
		return kernel.Int8GEMM(w.M, w.N, w.K, aBytes, bBytes, cInt32, 
			w.ScaleA*w.ScaleB*w.ScaleC, 0.0)
	}
	
	// Fallback to reference implementation
	return a.executeInt8Reference(w)
}

// executeInt8Reference is the reference implementation
func (a *AMXFFU) executeInt8Reference(w *ffu.AMXWorkload) error {
	// INT8 matrix multiply: C = alpha * A * B
	// A is M×K, B is K×N, C is M×N
	// Result is INT32 that needs scaling
	
	// Clear C matrix (INT32)
	cInt32 := (*[1 << 30]int32)(unsafe.Pointer(&w.C[0]))[:w.M*w.N]
	for i := range cInt32 {
		cInt32[i] = 0
	}
	
	// Reference implementation
	for i := 0; i < w.M; i++ {
		for j := 0; j < w.N; j++ {
			sum := int32(0)
			for k := 0; k < w.K; k++ {
				aVal := int32(int8(w.A[i*w.K+k]))
				bVal := int32(int8(w.B[k*w.N+j]))
				sum += aVal * bVal
			}
			// Apply scaling: C = ScaleC * (ScaleA * A) × (ScaleB * B)
			scaledSum := float32(sum) * w.ScaleA * w.ScaleB * w.ScaleC
			cInt32[i*w.N+j] = int32(scaledSum)
		}
	}
	
	return nil
}

// executeBF16 performs BF16 matrix multiplication using AMX
func (a *AMXFFU) executeBF16(w *ffu.AMXWorkload) error {
	// For now, use a reference implementation
	// Real implementation would use AMX assembly instructions
	
	// BF16 matrix multiply
	// This is a placeholder - real implementation would handle BF16 format
	return fmt.Errorf("BF16 not yet implemented")
}

// IsAvailable returns true if the AMX FFU is available
func (a *AMXFFU) IsAvailable() bool {
	return a.available
}

// Metrics returns performance metrics for this FFU
func (a *AMXFFU) Metrics() ffu.Metrics {
	return *a.metrics.Load().(*ffu.Metrics)
}

// isAligned checks if a pointer is aligned to the given boundary
func isAligned(p unsafe.Pointer, align uintptr) bool {
	return uintptr(p)&(align-1) == 0
}

// Capability returns AMX-specific capabilities
func (a *AMXFFU) Capability() *ffu.AMXCapability {
	if !a.available {
		return nil
	}
	
	return &ffu.AMXCapability{
		SupportsInt8: a.hasInt8,
		SupportsBF16: a.hasBF16,
		SupportsFP16: false, // Not yet on current hardware
		
		NumTiles:     a.tileConfig.NumTiles,
		MaxTileRows:  a.tileConfig.MaxRows,
		MaxTileCols:  a.tileConfig.MaxCols,
		MaxTileBytes: a.tileConfig.MaxTileBytes,
		
		PeakInt8TOPS:   2.0,  // 2 TOPS for INT8
		PeakBF16TFLOPS: 1.0,  // 1 TFLOPS for BF16
		
		RequiresAlignment: 16, // 16-byte alignment
	}
}