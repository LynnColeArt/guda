package guda

import (
	"golang.org/x/sys/cpu"
)

// CPUFeatures tracks available CPU instruction set extensions
type CPUFeatures struct {
	HasAVX     bool
	HasAVX2    bool
	HasAVX512F bool // Foundation
	HasAVX512DQ bool // Double/Quad precision
	HasAVX512BW bool // Byte/Word
	HasAVX512VL bool // Vector Length
	HasFMA     bool
	HasSSE4    bool
}

// Global CPU feature detection
var cpuFeatures CPUFeatures

func init() {
	detectCPUFeatures()
}

// detectCPUFeatures populates the global cpuFeatures struct
func detectCPUFeatures() {
	cpuFeatures = CPUFeatures{
		HasSSE4:     cpu.X86.HasSSE41 || cpu.X86.HasSSE42,
		HasAVX:      cpu.X86.HasAVX,
		HasAVX2:     cpu.X86.HasAVX2,
		HasAVX512F:  cpu.X86.HasAVX512F,
		HasAVX512DQ: cpu.X86.HasAVX512DQ,
		HasAVX512BW: cpu.X86.HasAVX512BW,
		HasAVX512VL: cpu.X86.HasAVX512VL,
		HasFMA:      cpu.X86.HasFMA,
	}
}

// HasAVX512 returns true if the CPU supports AVX-512 operations needed for GEMM
func HasAVX512() bool {
	// For GEMM, we need at least AVX512F (foundation)
	// AVX512DQ is nice to have for better FP operations
	return cpuFeatures.HasAVX512F
}

// HasAVX2 returns true if the CPU supports AVX2 operations
func HasAVX2() bool {
	return cpuFeatures.HasAVX2 && cpuFeatures.HasFMA
}

// GetBestGemmImplementation returns the optimal GEMM implementation for the CPU
func GetBestGemmImplementation() string {
	if HasAVX512() {
		return "AVX512"
	}
	if HasAVX2() {
		return "AVX2"
	}
	if cpuFeatures.HasSSE4 {
		return "SSE4"
	}
	return "scalar"
}

// GetCPUInfo returns a string describing available CPU features
func GetCPUInfo() string {
	features := []string{}
	
	if cpuFeatures.HasSSE4 {
		features = append(features, "SSE4")
	}
	if cpuFeatures.HasAVX {
		features = append(features, "AVX")
	}
	if cpuFeatures.HasAVX2 {
		features = append(features, "AVX2")
	}
	if cpuFeatures.HasFMA {
		features = append(features, "FMA")
	}
	if cpuFeatures.HasAVX512F {
		features = append(features, "AVX512F")
	}
	if cpuFeatures.HasAVX512DQ {
		features = append(features, "AVX512DQ")
	}
	if cpuFeatures.HasAVX512BW {
		features = append(features, "AVX512BW")
	}
	if cpuFeatures.HasAVX512VL {
		features = append(features, "AVX512VL")
	}
	
	if len(features) == 0 {
		return "No SIMD extensions detected"
	}
	
	result := "CPU features: "
	for i, f := range features {
		if i > 0 {
			result += ", "
		}
		result += f
	}
	return result
}