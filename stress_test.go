package guda

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// StressTestMatrix represents a challenging matrix configuration
type StressTestMatrix struct {
	Name        string
	Generator   func(m, n int) []float32
	Description string
}

// Constants for stress test configurations
const (
	// Numerical thresholds
	PerturbationSize     = 1e-6
	CancellationBase    = 1e6
	DenormalBase        = 1e-40
	LargeValueThreshold = 1e19
	SmallestNormalF32   = 1.175494e-38
	
	// Probability thresholds for NaN/Inf infection
	NaNProbability      = 0.001
	InfProbability      = 0.002
	NegInfProbability   = 0.003
	
	// Tanh approximation constants
	TanhRationalConst27 = 27.0
	TanhRationalConst9  = 9.0
	TanhClampThreshold  = 3.0
)

// Collection of numerically challenging matrices
var stressMatrices = []StressTestMatrix{
	{
		Name: "IllConditioned",
		Description: "Matrix with huge condition number (near-singular)",
		Generator: func(m, n int) []float32 {
			// Create a matrix with exponentially decaying singular values
			data := make([]float32, m*n)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					if i == j {
						// Diagonal: exponentially decaying values
						data[i*n+j] = float32(math.Pow(10, float64(-i)/2))
					} else {
						// Off-diagonal: small perturbations
						data[i*n+j] = float32(rand.NormFloat64()) * PerturbationSize
					}
				}
			}
			return data
		},
	},
	{
		Name: "CatastrophicCancellation",
		Description: "Values that cause severe cancellation when added/subtracted",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			base := float32(CancellationBase)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					// Alternate between large positive and nearly-canceling negative
					if (i+j)%2 == 0 {
						data[i*n+j] = base + float32(i)*1e-3
					} else {
						data[i*n+j] = -base + float32(j)*1e-3
					}
				}
			}
			return data
		},
	},
	{
		Name: "DenormalHeavy",
		Description: "Matrix filled with denormal numbers",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			// Denormals are smaller than SmallestNormalF32
			denormalVal := float32(DenormalBase)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					data[i*n+j] = denormalVal * float32(1+i+j)
				}
			}
			return data
		},
	},
	{
		Name: "OverflowRisk",
		Description: "Values near float32 max that risk overflow in dot products",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			// Max float32 ≈ 3.4e38
			largeVal := float32(LargeValueThreshold) // Square of this approaches max
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					data[i*n+j] = largeVal * (1 + float32(rand.NormFloat64())*0.1)
				}
			}
			return data
		},
	},
	{
		Name: "CacheKiller",
		Description: "Access pattern designed to thrash CPU cache",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			// Use prime numbers for stride to avoid cache line reuse
			primes := []int{17, 31, 61, 127, 251, 509}
			for i := 0; i < m*n; i++ {
				// Scatter values across memory with prime strides
				idx := (i * primes[i%len(primes)]) % (m * n)
				data[idx] = float32(i) * 0.01
			}
			return data
		},
	},
	{
		Name: "NaNInfected",
		Description: "Matrix with strategic NaN/Inf placement",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					r := rand.Float32()
					if r < NaNProbability {
						data[i*n+j] = float32(math.NaN())
					} else if r < InfProbability {
						data[i*n+j] = float32(math.Inf(1))
					} else if r < NegInfProbability {
						data[i*n+j] = float32(math.Inf(-1))
					} else {
						data[i*n+j] = rand.Float32()
					}
				}
			}
			return data
		},
	},
	{
		Name: "HighFrequency",
		Description: "Rapid oscillation pattern (stress SIMD)",
		Generator: func(m, n int) []float32 {
			data := make([]float32, m*n)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					// High frequency sine wave
					x := float64(i*n+j) * 0.1
					data[i*n+j] = float32(math.Sin(x) * math.Cos(x*17) * math.Sin(x*31))
				}
			}
			return data
		},
	},
}

// TestStressMatrices runs GEMM operations on numerically challenging matrices
func TestStressMatrices(t *testing.T) {
	sizes := []struct{ m, n, k int }{
		{32, 32, 32},
		{128, 128, 128},
		{512, 512, 512},
	}
	
	for _, size := range sizes {
		for _, matrixType := range stressMatrices {
			testName := fmt.Sprintf("%s_%dx%dx%d", matrixType.Name, size.m, size.n, size.k)
			t.Run(testName, func(t *testing.T) {
				// Skip NaN test for correctness checking
				if matrixType.Name == "NaNInfected" {
					t.Skip("NaN propagation test - correctness check not applicable")
				}
				
				// Generate challenging matrices
				aData := matrixType.Generator(size.m, size.k)
				bData := matrixType.Generator(size.k, size.n)
				
				// Allocate device memory
				d_a, _ := Malloc(size.m * size.k * 4)
				d_b, _ := Malloc(size.k * size.n * 4)
				d_c, _ := Malloc(size.m * size.n * 4)
				defer Free(d_a)
				defer Free(d_b)
				defer Free(d_c)
				
				// Copy to device
				Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
				Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
				
				// Run GEMM
				start := time.Now()
				err := GEMM(false, false, size.m, size.n, size.k, 1.0, 
					d_a, size.k, d_b, size.n, 0.0, d_c, size.n)
				if err != nil {
					t.Errorf("GEMM failed: %v", err)
					return
				}
				Synchronize()
				elapsed := time.Since(start)
				
				// Get result
				result := d_c.Float32()[:size.m*size.n]
				
				// Check for NaN/Inf in result
				nanCount, infCount := 0, 0
				for _, v := range result {
					if math.IsNaN(float64(v)) {
						nanCount++
					}
					if math.IsInf(float64(v), 0) {
						infCount++
					}
				}
				
				t.Logf("Completed in %v - NaN: %d, Inf: %d", elapsed, nanCount, infCount)
				
				// For denormal test, check if denormals were preserved
				if matrixType.Name == "DenormalHeavy" {
					denormalCount := 0
					for _, v := range result {
						if v != 0 && math.Abs(float64(v)) < SmallestNormalF32 {
							denormalCount++
						}
					}
					t.Logf("Denormal values in result: %d/%d", denormalCount, len(result))
				}
			})
		}
	}
}

// BenchmarkStressMatrices benchmarks performance on challenging matrices
func BenchmarkStressMatrices(b *testing.B) {
	m, n, k := 512, 512, 512
	
	for _, matrixType := range stressMatrices {
		b.Run(matrixType.Name, func(b *testing.B) {
			// Generate matrices once
			aData := matrixType.Generator(m, k)
			bData := matrixType.Generator(k, n)
			
			// Allocate device memory
			d_a, _ := Malloc(m * k * 4)
			d_b, _ := Malloc(k * n * 4)
			d_c, _ := Malloc(m * n * 4)
			defer Free(d_a)
			defer Free(d_b)
			defer Free(d_c)
			
			// Copy to device
			Memcpy(d_a, aData, len(aData)*4, MemcpyHostToDevice)
			Memcpy(d_b, bData, len(bData)*4, MemcpyHostToDevice)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GEMM(false, false, m, n, k, 1.0, d_a, k, d_b, n, 0.0, d_c, n)
				Synchronize()
			}
		})
	}
}

// TestNumericalStability checks numerical stability of operations
func TestNumericalStability(t *testing.T) {
	// Test 1: Check if A*B*C == A*(B*C) within tolerance
	t.Run("Associativity", func(t *testing.T) {
		size := 64
		
		// Create three random matrices
		A := make([]float32, size*size)
		B := make([]float32, size*size)
		C := make([]float32, size*size)
		
		rand.Seed(time.Now().UnixNano())
		for i := range A {
			A[i] = rand.Float32()*2 - 1
			B[i] = rand.Float32()*2 - 1
			C[i] = rand.Float32()*2 - 1
		}
		
		// Allocate device memory
		d_A, _ := Malloc(size * size * 4)
		d_B, _ := Malloc(size * size * 4)
		d_C, _ := Malloc(size * size * 4)
		d_temp1, _ := Malloc(size * size * 4)
		d_temp2, _ := Malloc(size * size * 4)
		d_result1, _ := Malloc(size * size * 4)
		d_result2, _ := Malloc(size * size * 4)
		defer Free(d_A)
		defer Free(d_B)
		defer Free(d_C)
		defer Free(d_temp1)
		defer Free(d_temp2)
		defer Free(d_result1)
		defer Free(d_result2)
		
		// Copy to device
		Memcpy(d_A, A, len(A)*4, MemcpyHostToDevice)
		Memcpy(d_B, B, len(B)*4, MemcpyHostToDevice)
		Memcpy(d_C, C, len(C)*4, MemcpyHostToDevice)
		
		// Compute (A*B)*C
		GEMM(false, false, size, size, size, 1.0, d_A, size, d_B, size, 0.0, d_temp1, size)
		GEMM(false, false, size, size, size, 1.0, d_temp1, size, d_C, size, 0.0, d_result1, size)
		
		// Compute A*(B*C)
		GEMM(false, false, size, size, size, 1.0, d_B, size, d_C, size, 0.0, d_temp2, size)
		GEMM(false, false, size, size, size, 1.0, d_A, size, d_temp2, size, 0.0, d_result2, size)
		
		Synchronize()
		
		// Compare results
		result1 := d_result1.Float32()[:size*size]
		result2 := d_result2.Float32()[:size*size]
		
		maxDiff := float32(0)
		maxRelDiff := float32(0)
		for i := range result1 {
			diff := abs32(result1[i] - result2[i])
			if diff > maxDiff {
				maxDiff = diff
			}
			if result1[i] != 0 {
				relDiff := diff / abs32(result1[i])
				if relDiff > maxRelDiff {
					maxRelDiff = relDiff
				}
			}
		}
		
		t.Logf("Max absolute difference: %e", maxDiff)
		t.Logf("Max relative difference: %e", maxRelDiff)
		
		// With float32, we expect some loss of associativity
		// 3 matrix multiplications can accumulate significant error
		if maxRelDiff > 1e-2 {
			t.Errorf("Associativity error too large: %e", maxRelDiff)
		}
	})
}