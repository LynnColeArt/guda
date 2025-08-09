# Chapter 2: Installation

> *"The journey of a thousand optimizations begins with a single `go get`."* — Ancient Developer Wisdom

Getting GUDA running on your system is designed to be as smooth as matrix multiplication on a well-tuned CPU. Let's get you set up!

## Prerequisites

### Go Installation
GUDA requires **Go 1.19 or later** (though Go 1.24+ is recommended for the latest optimizations):

```bash
# Check your Go version
go version

# Should show: go version go1.24.x linux/amd64 (or similar)
```

If you need to install or upgrade Go:
- **Linux/macOS**: Download from [golang.org](https://golang.org/dl/)
- **Windows**: Use the official installer or Chocolatey: `choco install golang`
- **macOS**: Use Homebrew: `brew install go`

### System Requirements

**CPU Architecture:**
- **x86-64**: Full AVX2/FMA support for maximum performance
- **ARM64**: Native support with NEON optimizations
- **Other architectures**: Pure Go fallbacks available

**Memory:**
- Minimum: 1GB RAM for basic operations
- Recommended: 4GB+ for large matrix operations
- Optimal: 8GB+ for neural network workloads

**Operating System:**
- Linux (Ubuntu 20.04+, RHEL 8+, etc.)
- macOS (10.15+)
- Windows (10/11 with WSL2 recommended)

## Installation Methods

### Method 1: Go Modules (Recommended)

Add GUDA to your project:

```bash
# Initialize your Go module (if not already done)
go mod init your-awesome-project

# Add GUDA
go get github.com/LynnColeArt/guda

# Import in your Go code
```

```go
package main

import (
    "fmt"
    "github.com/LynnColeArt/guda"
)

func main() {
    fmt.Println("🧀 GUDA is ready!")
    
    // Your high-performance computing journey starts here
}
```

### Method 2: Direct Clone

For development or contributing:

```bash
# Clone the repository
git clone https://github.com/LynnColeArt/guda.git
cd guda

# Build and test
go build ./...
go test ./...

# Verify installation
go run examples/matrix_multiply/main.go
```

### Method 3: Docker Container

For isolated environments:

```dockerfile
FROM golang:1.24-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o guda-app ./cmd/example

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/guda-app .
CMD ["./guda-app"]
```

## Verification

Let's make sure everything is working perfectly:

### Quick Smoke Test

Create `test-guda.go`:

```go
package main

import (
    "fmt"
    "time"
    
    "github.com/LynnColeArt/guda"
)

func main() {
    fmt.Println("🧀 GUDA Installation Test")
    
    // Test basic functionality
    const N = 100
    
    // Create test matrices
    A := make([]float32, N*N)
    B := make([]float32, N*N)
    C := make([]float32, N*N)
    
    // Fill with test data
    for i := range A {
        A[i] = float32(i % 10)
        B[i] = float32((i + 1) % 10)
    }
    
    // Time a matrix multiplication
    start := time.Now()
    
    // This is where GUDA magic happens!
    guda.Sgemm(false, false, N, N, N,
               1.0, A, N,
               B, N,
               0.0, C, N)
    
    duration := time.Since(start)
    
    // Calculate performance
    ops := 2.0 * float64(N*N*N)
    gflops := ops / duration.Seconds() / 1e9
    
    fmt.Printf("✅ Matrix multiplication successful!\n")
    fmt.Printf("⚡ Performance: %.2f GFLOPS\n", gflops)
    fmt.Printf("🎯 Result checksum: %.6f\n", C[0]+C[N-1]+C[N*N-1])
    
    if gflops > 0.1 {
        fmt.Println("🎉 GUDA is working perfectly!")
    } else {
        fmt.Println("⚠️  Performance seems low, check your setup")
    }
}
```

Run it:
```bash
go run test-guda.go
```

Expected output:
```
🧀 GUDA Installation Test
✅ Matrix multiplication successful!
⚡ Performance: 12.34 GFLOPS
🎯 Result checksum: 495.000000
🎉 GUDA is working perfectly!
```

### Performance Benchmark

Run GUDA's built-in benchmarks:

```bash
# Quick benchmark
go test -bench=BenchmarkSgemm -benchtime=2s

# Full benchmark suite
go test -bench=. -benchtime=1s
```

### Feature Detection

Check what optimizations are available on your system:

```go
package main

import (
    "fmt"
    "github.com/LynnColeArt/guda"
)

func main() {
    info := guda.GetSystemInfo()
    fmt.Printf("🖥️  Architecture: %s\n", info.Architecture)
    fmt.Printf("🧮 CPU Features: %v\n", info.Features)
    fmt.Printf("⚡ SIMD Support: %s\n", info.SIMDLevel)
    fmt.Printf("🧵 CPU Cores: %d\n", info.NumCores)
}
```

## Platform-Specific Notes

### Linux Optimization

For maximum performance on Linux:

```bash
# Install build essentials (if needed)
sudo apt update
sudo apt install build-essential

# For AVX-512 support (Intel Skylake-X and newer)
export CGO_CFLAGS="-march=native -O3"
go build -a ./...
```

### macOS with Apple Silicon

GUDA automatically detects and uses Apple's optimized BLAS:

```bash
# Verify ARM64 optimizations are enabled
go env GOARCH  # Should show: arm64

# Build with optimizations
go build -ldflags="-s -w" ./...
```

### Windows Setup

Using WSL2 (recommended):

```powershell
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2
sudo apt update && sudo apt install golang-go
go version
```

Or native Windows:
```powershell
# Install using Chocolatey
choco install golang git

# Or download from golang.org
```

## Troubleshooting Common Issues

### "Cannot find package"
```bash
# Clear module cache
go clean -modcache

# Refresh dependencies
go mod tidy
go mod download
```

### Low Performance
1. Check Go version: `go version` (use 1.24+)
2. Verify CPU features: Run feature detection above
3. Check system load: `htop` or `top`
4. Disable debug mode: `go build -ldflags="-s -w"`

### Build Errors
```bash
# Update Go to latest version
# Clean everything
go clean -cache -modcache -testcache

# Rebuild from scratch
go mod tidy
go build -a ./...
```

### Memory Issues
```bash
# Increase Go's memory limit
export GOMEMLIMIT=4GiB

# Monitor memory usage
go test -bench=. -memprofile=mem.prof
go tool pprof mem.prof
```

## What's Next?

Now that GUDA is installed and verified, you're ready to dive into the fun stuff! 

Head over to [Quick Start Guide](03-quickstart.md) to write your first high-performance programs, or jump to [Architecture Overview](04-architecture.md) to understand how GUDA works its magic.

For detailed information about ARM64 implementation and optimizations, see [ARM64 Support Documentation](../README_ARM64.md).

---

*🎉 Welcome to the GUDA family! You're now ready to make CPUs do amazing things.*