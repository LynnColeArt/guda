package guda

import (
	"fmt"
	"runtime"
)

// FFUType represents the type of fixed-function unit
type FFUType int

const (
	FFUTypeUnknown FFUType = iota
	FFUTypeCPU              // General purpose CPU
	FFUTypeAVX512VNNI       // AVX-512 Vector Neural Network Instructions
	FFUTypeAMX              // Intel Advanced Matrix Extensions
	FFUTypeAESNI            // AES New Instructions
	FFUTypeSHA              // SHA extensions
	FFUTypeGPU              // General GPU compute
	FFUTypeVideoEncode      // Hardware video encoder (NVENC, VCN, QuickSync)
	FFUTypeVideoDecode      // Hardware video decoder
	FFUTypeTexture          // GPU texture units
	FFUTypeDSP              // Digital Signal Processor
	FFUTypeFPGA             // Field Programmable Gate Array
	FFUTypeNPU              // Neural Processing Unit
)

// FFUCapability describes a fixed-function unit's capabilities
type FFUCapability struct {
	Type        FFUType
	Name        string
	DeviceID    int
	Available   bool
	
	// Performance characteristics
	Throughput  float64 // Operations per second
	Latency     float64 // Microseconds
	PowerDraw   float64 // Watts (estimated)
	
	// Supported operations
	SupportsInt8     bool
	SupportsInt16    bool
	SupportsFloat16  bool
	SupportsFloat32  bool
	SupportsFloat64  bool
	
	// Specific capabilities
	MaxMatrixSize    int    // For matrix operations
	MaxVectorLength  int    // For vector operations
	SharedMemorySize int64  // Bytes
	
	// Backend information
	Backend     string // "cpu", "rocm", "directml", "metal", etc.
	VendorID    uint32
	DeviceInfo  string
}

// FFUDetector manages detection and enumeration of fixed-function units
type FFUDetector struct {
	capabilities []FFUCapability
	cpuFeatures  *CPUFeatures
}

// NewFFUDetector creates a new FFU detector
func NewFFUDetector() *FFUDetector {
	return &FFUDetector{
		capabilities: make([]FFUCapability, 0),
		cpuFeatures:  GetCPUFeatures(),
	}
}

// Detect discovers all available fixed-function units
func (d *FFUDetector) Detect() error {
	// Always detect CPU capabilities first
	d.detectCPUFFUs()
	
	// Platform-specific detection
	switch runtime.GOOS {
	case "linux":
		d.detectLinuxFFUs()
	case "windows":
		d.detectWindowsFFUs()
	case "darwin":
		d.detectMacOSFFUs()
	}
	
	// Detect GPUs and their FFUs
	d.detectGPUFFUs()
	
	return nil
}

// detectCPUFFUs detects CPU-based fixed-function units
func (d *FFUDetector) detectCPUFFUs() {
	// Basic CPU capability
	cpu := FFUCapability{
		Type:            FFUTypeCPU,
		Name:            "CPU Compute",
		DeviceID:        0,
		Available:       true,
		Backend:         "cpu",
		SupportsFloat32: true,
		SupportsFloat64: true,
	}
	
	// Check for AVX-512 VNNI
	if d.cpuFeatures.HasAVX512VNNI {
		vnni := FFUCapability{
			Type:           FFUTypeAVX512VNNI,
			Name:           "AVX-512 VNNI",
			DeviceID:       0,
			Available:      true,
			Backend:        "cpu",
			SupportsInt8:   true,
			SupportsInt16:  true,
			Throughput:     1e12, // 1 TOPS estimate
			Latency:        0.1,  // 100ns
			MaxVectorLength: 64,  // 512 bits / 8 bits
			DeviceInfo:     "AVX-512 Vector Neural Network Instructions",
		}
		d.capabilities = append(d.capabilities, vnni)
	}
	
	// Check for Intel AMX
	if d.cpuFeatures.HasAMX {
		amx := FFUCapability{
			Type:           FFUTypeAMX,
			Name:           "Intel AMX",
			DeviceID:       0,
			Available:      true,
			Backend:        "cpu",
			SupportsInt8:   true,
			SupportsFloat16: true,
			Throughput:     2e12,  // 2 TOPS estimate
			Latency:        1.0,   // 1us for tile ops
			MaxMatrixSize:  16,    // 16x16 tiles
			DeviceInfo:     "Intel Advanced Matrix Extensions",
		}
		d.capabilities = append(d.capabilities, amx)
	}
	
	// Check for AES-NI
	if d.cpuFeatures.HasAES {
		aes := FFUCapability{
			Type:       FFUTypeAESNI,
			Name:       "AES-NI",
			DeviceID:   0,
			Available:  true,
			Backend:    "cpu",
			Throughput: 100e9, // 100 GB/s for AES
			Latency:    0.05,  // 50ns
			DeviceInfo: "AES New Instructions",
		}
		d.capabilities = append(d.capabilities, aes)
	}
	
	// Check for SHA extensions
	if d.cpuFeatures.HasSHA {
		sha := FFUCapability{
			Type:       FFUTypeSHA,
			Name:       "SHA Extensions",
			DeviceID:   0,
			Available:  true,
			Backend:    "cpu",
			Throughput: 50e9, // 50 GB/s for SHA
			Latency:    0.1,  // 100ns
			DeviceInfo: "SHA-1 and SHA-256 acceleration",
		}
		d.capabilities = append(d.capabilities, sha)
	}
	
	d.capabilities = append(d.capabilities, cpu)
}

// detectGPUFFUs detects GPU and integrated graphics FFUs
func (d *FFUDetector) detectGPUFFUs() {
	// TODO: Implement GPU detection via Vulkan/ROCm/DirectML
	// This is a placeholder for the concept
	
	// Example: Intel integrated graphics with QuickSync
	if d.hasIntelGPU() {
		quicksync := FFUCapability{
			Type:       FFUTypeVideoEncode,
			Name:       "Intel QuickSync",
			DeviceID:   1,
			Available:  true,
			Backend:    "quicksync",
			Throughput: 120 * 1920 * 1080, // 120 FPS @ 1080p
			Latency:    8.3,                // ms per frame
			DeviceInfo: "Intel QuickSync Video",
		}
		d.capabilities = append(d.capabilities, quicksync)
	}
}

// hasIntelGPU checks if Intel integrated graphics is present
func (d *FFUDetector) hasIntelGPU() bool {
	// TODO: Implement actual detection
	// For now, check if we're on an Intel CPU
	return d.cpuFeatures.Vendor == "GenuineIntel"
}

// detectLinuxFFUs detects Linux-specific FFUs
func (d *FFUDetector) detectLinuxFFUs() {
	// TODO: Parse /proc/cpuinfo, check for DSPs, etc.
	// Check for ROCm devices
	// Check for Video4Linux devices
}

// detectWindowsFFUs detects Windows-specific FFUs
func (d *FFUDetector) detectWindowsFFUs() {
	// TODO: Use DirectML enumeration
	// Check for DirectX Video Acceleration
}

// detectMacOSFFUs detects macOS-specific FFUs
func (d *FFUDetector) detectMacOSFFUs() {
	// TODO: Use Metal Performance Shaders enumeration
	// Check for Apple Neural Engine
}

// GetCapabilities returns all detected FFU capabilities
func (d *FFUDetector) GetCapabilities() []FFUCapability {
	return d.capabilities
}

// GetCapabilitiesByType returns FFUs of a specific type
func (d *FFUDetector) GetCapabilitiesByType(ffuType FFUType) []FFUCapability {
	var result []FFUCapability
	for _, cap := range d.capabilities {
		if cap.Type == ffuType {
			result = append(result, cap)
		}
	}
	return result
}

// FindBestFFUForWorkload returns the best FFU for a given workload type
func (d *FFUDetector) FindBestFFUForWorkload(workloadType string) *FFUCapability {
	// Simple heuristic-based selection
	switch workloadType {
	case "crypto_aes":
		ffus := d.GetCapabilitiesByType(FFUTypeAESNI)
		if len(ffus) > 0 {
			return &ffus[0]
		}
	case "matmul_int8":
		// Prefer AMX over VNNI
		ffus := d.GetCapabilitiesByType(FFUTypeAMX)
		if len(ffus) > 0 {
			return &ffus[0]
		}
		ffus = d.GetCapabilitiesByType(FFUTypeAVX512VNNI)
		if len(ffus) > 0 {
			return &ffus[0]
		}
	case "video_encode":
		ffus := d.GetCapabilitiesByType(FFUTypeVideoEncode)
		if len(ffus) > 0 {
			return &ffus[0]
		}
	}
	
	// Default to CPU
	ffus := d.GetCapabilitiesByType(FFUTypeCPU)
	if len(ffus) > 0 {
		return &ffus[0]
	}
	
	return nil
}

// String returns a string representation of an FFU capability
func (c FFUCapability) String() string {
	return fmt.Sprintf("%s (Type: %d, Backend: %s, Throughput: %.2e ops/s, Latency: %.2f µs)",
		c.Name, c.Type, c.Backend, c.Throughput, c.Latency)
}

// Global FFU detector instance
var globalFFUDetector *FFUDetector

// InitFFUDetection initializes the global FFU detector
func InitFFUDetection() error {
	globalFFUDetector = NewFFUDetector()
	return globalFFUDetector.Detect()
}

// GetGlobalFFUDetector returns the global FFU detector
func GetGlobalFFUDetector() *FFUDetector {
	if globalFFUDetector == nil {
		InitFFUDetection()
	}
	return globalFFUDetector
}