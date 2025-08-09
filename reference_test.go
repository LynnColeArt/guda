package guda

import (
	"math"
	"math/rand"
	"testing"
)

// TestReferenceImplementations verifies that optimized implementations
// match the reference implementations within tolerance
func TestReferenceImplementations(t *testing.T) {
	const (
		n       = 1024
		tol     = 1e-5
		seed    = 42
	)
	
	rng := rand.New(rand.NewSource(seed))
	ref := Reference{}
	
	// Helper to generate random data
	randomSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	
	// Helper to check float equality with tolerance
	nearEqual := func(a, b, tol float32) bool {
		diff := float32(math.Abs(float64(a - b)))
		return diff <= tol || diff <= tol*float32(math.Max(math.Abs(float64(a)), math.Abs(float64(b))))
	}
	
	// Helper to check slice equality
	slicesNearEqual := func(a, b []float32, tol float32) bool {
		if len(a) != len(b) {
			return false
		}
		for i := range a {
			if !nearEqual(a[i], b[i], tol) {
				return false
			}
		}
		return true
	}
	
	// Helper to ensure synchronization after kernel launch
	sync := func(t *testing.T) {
		if err := Synchronize(); err != nil {
			t.Fatal("Synchronize failed:", err)
		}
	}
	
	t.Run("AXPY", func(t *testing.T) {
		alpha := float32(2.5)
		
		// Create test data
		x := randomSlice(n)
		yRef := randomSlice(n)
		yOpt := make([]float32, n)
		copy(yOpt, yRef)
		
		// Reference implementation
		ref.AXPY(alpha, x, yRef)
		
		// Optimized implementation
		dx, _ := Malloc(n * 4)
		dy, _ := Malloc(n * 4)
		defer Free(dx)
		defer Free(dy)
		
		copy(dx.Float32(), x)
		copy(dy.Float32(), yOpt)
		
		err := AXPY(alpha, dx, dy, n)
		if err != nil {
			t.Fatal(err)
		}
		
		sync(t)
		
		copy(yOpt, dy.Float32())
		
		// Compare
		if !slicesNearEqual(yRef, yOpt, tol) {
			t.Errorf("AXPY mismatch: reference and optimized differ")
		}
	})
	
	t.Run("DOT", func(t *testing.T) {
		x := randomSlice(n)
		y := randomSlice(n)
		
		// Reference implementation
		dotRef := ref.DOT(x, y)
		
		// Optimized implementation
		dx, _ := Malloc(n * 4)
		dy, _ := Malloc(n * 4)
		defer Free(dx)
		defer Free(dy)
		
		copy(dx.Float32(), x)
		copy(dy.Float32(), y)
		
		dotOpt, err := DOT(dx, dy, n)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		// Compare
		if !nearEqual(dotRef, dotOpt, tol) {
			t.Errorf("DOT mismatch: ref=%v, opt=%v", dotRef, dotOpt)
		}
	})
	
	t.Run("GEMM_Small", func(t *testing.T) {
		const m, n, k = 32, 32, 32
		alpha := float32(1.5)
		beta := float32(0.5)
		
		// Create test data
		a := randomSlice(m * k)
		b := randomSlice(k * n)
		cRef := randomSlice(m * n)
		cOpt := make([]float32, m*n)
		copy(cOpt, cRef)
		
		// Reference implementation
		ref.GEMM(false, false, m, n, k, alpha, a, k, b, n, beta, cRef, n)
		
		// Optimized implementation
		da, _ := Malloc(m * k * 4)
		db, _ := Malloc(k * n * 4)
		dc, _ := Malloc(m * n * 4)
		defer Free(da)
		defer Free(db)
		defer Free(dc)
		
		copy(da.Float32(), a)
		copy(db.Float32(), b)
		copy(dc.Float32(), cOpt)
		
		err := GEMM(false, false, m, n, k, alpha, da, k, db, n, beta, dc, n)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		copy(cOpt, dc.Float32())
		
		// Compare
		if !slicesNearEqual(cRef, cOpt, tol) {
			// Find max difference for debugging
			maxDiff := float32(0)
			for i := range cRef {
				diff := float32(math.Abs(float64(cRef[i] - cOpt[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Errorf("GEMM mismatch: max diff=%v", maxDiff)
		}
	})
	
	t.Run("ReLU", func(t *testing.T) {
		// Create test data with mix of positive and negative
		xRef := randomSlice(n)
		xOpt := make([]float32, n)
		copy(xOpt, xRef)
		
		// Reference implementation
		ref.ReLU(xRef)
		
		// Optimized implementation
		dx, _ := Malloc(n * 4)
		defer Free(dx)
		
		copy(dx.Float32(), xOpt)
		
		err := ReLU(dx, n)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		copy(xOpt, dx.Float32())
		
		// Compare
		if !slicesNearEqual(xRef, xOpt, tol) {
			t.Errorf("ReLU mismatch")
		}
	})
	
	t.Run("Softmax", func(t *testing.T) {
		// Create test data
		xRef := randomSlice(n)
		xOpt := make([]float32, n)
		copy(xOpt, xRef)
		
		// Reference implementation
		ref.Softmax(xRef)
		
		// Optimized implementation
		dx, _ := Malloc(n * 4)
		defer Free(dx)
		
		copy(dx.Float32(), xOpt)
		
		err := Softmax(dx, n)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		copy(xOpt, dx.Float32())
		
		// Compare - softmax needs slightly higher tolerance due to exp
		if !slicesNearEqual(xRef, xOpt, tol*10) {
			t.Errorf("Softmax mismatch")
		}
		
		// Verify sum is 1
		sum := float32(0)
		for _, v := range xOpt {
			sum += v
		}
		if !nearEqual(sum, 1.0, tol) {
			t.Errorf("Softmax sum = %v, expected 1.0", sum)
		}
	})
	
	t.Run("ElementwiseAdd", func(t *testing.T) {
		a := randomSlice(n)
		b := randomSlice(n)
		cRef := make([]float32, n)
		cOpt := make([]float32, n)
		
		// Reference implementation
		ref.Add(a, b, cRef)
		
		// Optimized implementation
		da, _ := Malloc(n * 4)
		db, _ := Malloc(n * 4)
		dc, _ := Malloc(n * 4)
		defer Free(da)
		defer Free(db)
		defer Free(dc)
		
		copy(da.Float32(), a)
		copy(db.Float32(), b)
		
		// Use Add function directly
		err := Add(da, db, dc, n)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		copy(cOpt, dc.Float32())
		
		// Compare
		if !slicesNearEqual(cRef, cOpt, tol) {
			t.Errorf("ElementwiseAdd mismatch")
		}
	})
	
	t.Run("ReduceSum", func(t *testing.T) {
		x := randomSlice(n)
		
		// Reference implementation
		sumRef := ref.Sum(x)
		
		// Optimized implementation
		dx, _ := Malloc(n * 4)
		defer Free(dx)
		
		copy(dx.Float32(), x)
		
		// Use the Reduce function with sum operation
		sumOpt, err := Reduce(dx, n, func(a, b float32) float32 { return a + b })
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		// Compare - reduction can accumulate more error
		if !nearEqual(sumRef, sumOpt, tol*float32(n)) {
			t.Errorf("ReduceSum mismatch: ref=%v, opt=%v", sumRef, sumOpt)
		}
	})
	
	t.Run("LayerNorm", func(t *testing.T) {
		const hiddenDim = 256
		eps := float32(1e-5)
		
		// Create test data
		x := randomSlice(hiddenDim)
		gamma := make([]float32, hiddenDim)
		beta := make([]float32, hiddenDim)
		for i := range gamma {
			gamma[i] = 1.0 // Standard initialization
			beta[i] = 0.0
		}
		
		xRef := make([]float32, hiddenDim)
		xOpt := make([]float32, hiddenDim)
		copy(xRef, x)
		copy(xOpt, x)
		
		// Reference implementation
		ref.LayerNorm(xRef, gamma, beta, eps)
		
		// Optimized implementation
		dx, _ := Malloc(hiddenDim * 4)
		dgamma, _ := Malloc(hiddenDim * 4)
		dbeta, _ := Malloc(hiddenDim * 4)
		defer Free(dx)
		defer Free(dgamma)
		defer Free(dbeta)
		
		copy(dx.Float32(), xOpt)
		copy(dgamma.Float32(), gamma)
		copy(dbeta.Float32(), beta)
		
		err := LayerNorm(dx, dgamma, dbeta, hiddenDim, eps)
		if err != nil {
			t.Fatal(err)
		}
		sync(t)
		
		copy(xOpt, dx.Float32())
		
		// Compare
		if !slicesNearEqual(xRef, xOpt, tol*10) {
			t.Errorf("LayerNorm mismatch")
		}
	})
}

// BenchmarkReferenceVsOptimized compares performance
func BenchmarkReferenceVsOptimized(b *testing.B) {
	const n = 1024
	ref := Reference{}
	
	// Prepare data
	x := make([]float32, n)
	y := make([]float32, n)
	for i := range x {
		x[i] = float32(i)
		y[i] = float32(n - i)
	}
	
	b.Run("AXPY_Reference", func(b *testing.B) {
		yCopy := make([]float32, n)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(yCopy, y)
			ref.AXPY(2.5, x, yCopy)
		}
	})
	
	b.Run("AXPY_Optimized", func(b *testing.B) {
		dx, _ := Malloc(n * 4)
		dy, _ := Malloc(n * 4)
		defer Free(dx)
		defer Free(dy)
		
		copy(dx.Float32(), x)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(dy.Float32(), y)
			AXPY(2.5, dx, dy, n)
		}
	})
}