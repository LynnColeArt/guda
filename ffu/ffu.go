package ffu

import (
	"fmt"
	"time"
)

// Workload represents a unit of work that can be executed on an FFU
type Workload interface {
	// Type returns the workload type (e.g., "crypto_aes", "matmul_int8")
	Type() string
	
	// Size returns the size of the workload in bytes
	Size() int64
	
	// Validate checks if the workload is valid
	Validate() error
}

// Cost represents the estimated cost of executing a workload
type Cost struct {
	// Estimated execution time
	Duration time.Duration
	
	// Estimated energy usage in joules
	Energy float64
	
	// Estimated memory bandwidth usage in bytes
	MemoryBandwidth int64
	
	// Confidence level (0-1) in the estimate
	Confidence float64
}

// FFU represents a Fixed-Function Unit
type FFU interface {
	// Name returns the FFU name
	Name() string
	
	// Type returns the FFU type
	Type() FFUType
	
	// CanHandle checks if this FFU can handle the workload
	CanHandle(workload Workload) bool
	
	// EstimateCost estimates the cost of executing the workload
	EstimateCost(workload Workload) Cost
	
	// Execute runs the workload on this FFU
	Execute(workload Workload) error
	
	// IsAvailable checks if the FFU is currently available
	IsAvailable() bool
	
	// Metrics returns performance metrics for this FFU
	Metrics() Metrics
}

// FFUType represents the type of fixed-function unit
type FFUType int

const (
	FFUTypeUnknown FFUType = iota
	FFUTypeCPU              // General purpose CPU
	FFUTypeAESNI            // AES New Instructions
	FFUTypeSHA              // SHA extensions
	FFUTypeAVX512VNNI       // AVX-512 Vector Neural Network Instructions
	FFUTypeAMX              // Intel Advanced Matrix Extensions
	FFUTypeGPU              // General GPU compute
	FFUTypeVideoEncode      // Hardware video encoder
	FFUTypeVideoDecode      // Hardware video decoder
	FFUTypeNPU              // Neural Processing Unit
)

func (t FFUType) String() string {
	switch t {
	case FFUTypeCPU:
		return "CPU"
	case FFUTypeAESNI:
		return "AES-NI"
	case FFUTypeSHA:
		return "SHA"
	case FFUTypeAVX512VNNI:
		return "AVX512-VNNI"
	case FFUTypeAMX:
		return "AMX"
	case FFUTypeGPU:
		return "GPU"
	case FFUTypeVideoEncode:
		return "VideoEncode"
	case FFUTypeVideoDecode:
		return "VideoDecode"
	case FFUTypeNPU:
		return "NPU"
	default:
		return "Unknown"
	}
}

// Metrics tracks FFU performance metrics
type Metrics struct {
	// Total number of workloads executed
	WorkloadCount int64
	
	// Total bytes processed
	BytesProcessed int64
	
	// Total execution time
	TotalDuration time.Duration
	
	// Total energy consumed (if available)
	TotalEnergy float64
	
	// Number of errors
	ErrorCount int64
	
	// Last error (if any)
	LastError error
	
	// Timestamp of last use
	LastUsed time.Time
}

// Registry manages available FFUs
type Registry struct {
	ffus map[string]FFU
}

// NewRegistry creates a new FFU registry
func NewRegistry() *Registry {
	return &Registry{
		ffus: make(map[string]FFU),
	}
}

// Register adds an FFU to the registry
func (r *Registry) Register(ffu FFU) error {
	if ffu == nil {
		return fmt.Errorf("cannot register nil FFU")
	}
	
	name := ffu.Name()
	if _, exists := r.ffus[name]; exists {
		return fmt.Errorf("FFU %s already registered", name)
	}
	
	r.ffus[name] = ffu
	return nil
}

// Get returns an FFU by name
func (r *Registry) Get(name string) (FFU, bool) {
	ffu, exists := r.ffus[name]
	return ffu, exists
}

// List returns all registered FFUs
func (r *Registry) List() []FFU {
	result := make([]FFU, 0, len(r.ffus))
	for _, ffu := range r.ffus {
		result = append(result, ffu)
	}
	return result
}

// FindBest returns the best FFU for a workload based on cost estimation
func (r *Registry) FindBest(workload Workload) (FFU, *Cost) {
	var bestFFU FFU
	var bestCost *Cost
	
	for _, ffu := range r.ffus {
		if !ffu.IsAvailable() || !ffu.CanHandle(workload) {
			continue
		}
		
		cost := ffu.EstimateCost(workload)
		
		// Simple selection: lowest duration
		// TODO: Add more sophisticated cost model
		if bestCost == nil || cost.Duration < bestCost.Duration {
			bestFFU = ffu
			bestCost = &cost
		}
	}
	
	return bestFFU, bestCost
}