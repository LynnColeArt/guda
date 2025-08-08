// Package guda configuration constants
package guda

// Cache sizes for different levels (in bytes)
const (
	// L1 cache size per core (typical for modern CPUs)
	L1CacheSize = 32 * 1024 // 32KB
	
	// L2 cache size per core (typical for modern CPUs)
	L2CacheSize = 256 * 1024 // 256KB
	
	// L3 cache size (shared, typical for modern CPUs)
	L3CacheSize = 8 * 1024 * 1024 // 8MB
)

// SIMD vector sizes
const (
	// AVX2 vector width in float32 elements
	AVX2VectorSize = 8
	
	// AVX512 vector width in float32 elements
	AVX512VectorSize = 16
	
	// Default SIMD alignment in bytes
	SIMDAlignment = 64
)

// Thread and block dimensions
const (
	// Default block size for kernels
	DefaultBlockSize = 256
	
	// Maximum threads per block (CUDA compatibility)
	MaxThreadsPerBlock = 1024
	
	// Default grid size multiplier
	DefaultGridMultiplier = 4
)

// Memory pool parameters
const (
	// Minimum allocation size to prevent fragmentation
	MinAllocationSize = 64
	
	// Memory alignment for allocations
	MemoryAlignment = 64
	
	// Free list size threshold for reuse
	FreeListThreshold = 100
)

// Performance tuning parameters
const (
	// Tile size for matrix operations (optimized for L1 cache)
	MatrixTileSize = 64
	
	// Prefetch distance in cache lines
	PrefetchDistance = 8
	
	// Unroll factor for loops
	LoopUnrollFactor = 4
)

// Convolution parameters
const (
	// im2col workspace multiplier
	Im2colWorkspaceMultiplier = 2
	
	// Minimum size for using im2col (below this, use direct convolution)
	Im2colThreshold = 16
)

// Numerical constants
const (
	// Machine epsilon for float32
	Float32Epsilon = 1.192092896e-07
	
	// Maximum ULP difference for float32 comparisons
	MaxULPDiff = 4
)