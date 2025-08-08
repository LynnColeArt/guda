# GUDA Gonum Assimilation Plan

## The Vision
Fully integrate gonum's compute kernels into GUDA as our native math engine.
No external dependencies, total control, maximum performance.

## Step 1: Core Extraction
Pull only what we need:
- `blas/gonum` - The BLAS implementation
- `internal/asm/f32` - Float32 assembly
- `internal/asm/f64` - Float64 assembly  
- `floats` - Helper functions

Skip what we don't:
- Complex number support
- Plotting
- Statistics (for now)
- Sparse matrices

## Step 2: Restructure
```
guda/
├── compute/           # Renamed from blas/gonum
│   ├── level1.go     # AXPY, DOT, etc
│   ├── level2.go     # GEMV, GER, etc
│   ├── level3.go     # GEMM, etc
│   └── asm/          # Merged internal/asm
│       ├── f16/      # NEW: Add Float16
│       ├── f32/      # Enhanced with AVX2
│       └── f64/      # Keep the good ones
└── guda.go           # Single entry point
```

## Step 3: Enhance
With full control, we can:
1. Add Float16 versions of everything
2. Implement missing Float32 AVX2 
3. Add ML-specific operations directly:
   - AddBiasGELU
   - LayerNorm  
   - Attention primitives
4. Remove abstraction penalties
5. Optimize for our memory patterns

## Step 4: Integrate
Replace all gonum imports:
```go
// Before
import "gonum.org/v1/gonum/blas/gonum"

// After  
import "github.com/guda/guda/compute"
```

## Benefits
- Single binary deployment
- No version conflicts
- Can break/change anything
- Inline across boundaries
- Our own numerical standards
- GPU-like semantics when needed

## The Code is Already Ours
We've already fixed critical bugs. We understand it. Let's own it.