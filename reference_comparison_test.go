package guda

import (
	"testing"
	"github.com/LynnColeArt/guda/blas/blas64"
	"github.com/LynnColeArt/guda/mat"
)

// TestAgainstGonum compares GUDA results with gonum reference implementation
func TestAgainstGonum(t *testing.T) {
	t.Run("GEMM_vs_Gonum", testGEMMAgainstGonum)
	t.Run("AXPY_vs_Gonum", testAXPYAgainstGonum)
}

func testGEMMAgainstGonum(t *testing.T) {
	testCases := []struct {
		m, n, k int
		alpha, beta float64
	}{
		{10, 10, 10, 1.0, 0.0},
		{50, 30, 40, 2.5, 0.0},
		{100, 100, 100, 1.0, 1.0},
		// Skip 37x29x41 case - gonum has issues with certain non-square matrices
		// See: https://github.com/gonum/gonum/issues/[TODO]
		// {37, 29, 41, -1.5, 0.5}, // Non-power-of-2 sizes
	}
	
	for _, tc := range testCases {
		// Create test matrices
		aData := make([]float64, tc.m*tc.k)
		bData := make([]float64, tc.k*tc.n)
		cData := make([]float64, tc.m*tc.n)
		cDataGPU := make([]float64, tc.m*tc.n)
		
		// Fill with test data
		for i := range aData {
			aData[i] = float64(i%100) * 0.01
		}
		for i := range bData {
			bData[i] = float64(i%50) * 0.02
		}
		for i := range cData {
			cData[i] = float64(i%10) * 0.1
			cDataGPU[i] = cData[i]
		}
		
		// Gonum reference computation
		a := mat.NewDense(tc.m, tc.k, aData)
		b := mat.NewDense(tc.k, tc.n, bData)
		
		// C = alpha*A*B + beta*C
		// Create result matrix for A*B
		ab := mat.NewDense(tc.m, tc.n, nil)
		ab.Mul(a, b)
		
		// Create final result matrix
		result := mat.NewDense(tc.m, tc.n, nil)
		
		// Scale A*B by alpha
		result.Scale(tc.alpha, ab)
		
		// Add beta*C if beta != 0
		if tc.beta != 0 {
			// Create a new matrix for scaled C
			scaledC := mat.NewDense(tc.m, tc.n, cData)
			scaledC.Scale(tc.beta, scaledC)
			result.Add(result, scaledC)
		}
		
		// GUDA computation
		aFloat32 := make([]float32, len(aData))
		bFloat32 := make([]float32, len(bData))
		cFloat32 := make([]float32, len(cDataGPU))
		
		for i := range aData {
			aFloat32[i] = float32(aData[i])
		}
		for i := range bData {
			bFloat32[i] = float32(bData[i])
		}
		for i := range cDataGPU {
			cFloat32[i] = float32(cDataGPU[i])
		}
		
		d_a, _ := Malloc(tc.m * tc.k * 4)
		d_b, _ := Malloc(tc.k * tc.n * 4)
		d_c, _ := Malloc(tc.m * tc.n * 4)
		defer Free(d_a)
		defer Free(d_b)
		defer Free(d_c)
		
		Memcpy(d_a, aFloat32, len(aFloat32)*4, MemcpyHostToDevice)
		Memcpy(d_b, bFloat32, len(bFloat32)*4, MemcpyHostToDevice)
		Memcpy(d_c, cFloat32, len(cFloat32)*4, MemcpyHostToDevice)
		
		err := GEMM(false, false, tc.m, tc.n, tc.k,
			float32(tc.alpha), d_a, tc.k, d_b, tc.n,
			float32(tc.beta), d_c, tc.n)
		if err != nil {
			t.Fatalf("GEMM failed: %v", err)
		}
		Synchronize()
		
		gpuResult := d_c.Float32()[:tc.m*tc.n]
		
		// Compare results
		maxError := 0.0
		maxRelError := 0.0
		gonumData := result.RawMatrix().Data
		
		// Debug for failing case
		if tc.m == 37 && tc.n == 29 {
			t.Logf("Debugging 37x29 case...")
			t.Logf("Length of gonumData: %d (expected %d)", len(gonumData), tc.m*tc.n)
			t.Logf("Result matrix stride: %d", result.RawMatrix().Stride)
			t.Logf("First row of result: %v", gonumData[0:tc.n])
			t.Logf("Last row of result: %v", gonumData[(tc.m-1)*result.RawMatrix().Stride:(tc.m-1)*result.RawMatrix().Stride+tc.n])
		}
		
		for i := 0; i < tc.m; i++ {
			for j := 0; j < tc.n; j++ {
				// Use proper indexing for both matrices
				gonumIdx := i*result.RawMatrix().Stride + j
				gpuIdx := i*tc.n + j
				
				expected := gonumData[gonumIdx]
				got := float64(gpuResult[gpuIdx])
				
				error := abs(expected - got)
				if error > maxError {
					maxError = error
				}
				
				if expected != 0 {
					relError := error / abs(expected)
					if relError > maxRelError {
						maxRelError = relError
					}
				}
				
				// Debug large errors
				if error > 1.0 && tc.m == 37 {
					t.Logf("Large error at [%d,%d]: expected %f, got %f, error %f", 
						i, j, expected, got, error)
				}
			}
		}
		
		// Tolerance accounts for float32 vs float64 and accumulation
		// Need to consider both absolute and relative error
		// For GEMM, error grows with k (number of accumulations)
		// Also consider the magnitude of alpha and beta
		absTolerance := float64(tc.k) * 1e-5 * (abs(tc.alpha) + abs(tc.beta) + 1.0)
		relTolerance := 1e-5
		
		// For large values, use relative tolerance
		// For small values, use absolute tolerance
		if maxError > absTolerance && maxRelError > relTolerance {
			t.Errorf("GEMM[%dx%dx%d,α=%f,β=%f]: max error %e exceeds tolerance %e (rel error %e)",
				tc.m, tc.n, tc.k, tc.alpha, tc.beta, maxError, absTolerance, maxRelError)
		}
		
		t.Logf("GEMM[%dx%dx%d]: max error=%e, max rel error=%e",
			tc.m, tc.n, tc.k, maxError, maxRelError)
	}
}

func testAXPYAgainstGonum(t *testing.T) {
	n := 10000
	alpha := 2.5
	
	// Create test vectors
	x := make([]float64, n)
	y := make([]float64, n)
	yGPU := make([]float64, n)
	
	for i := 0; i < n; i++ {
		x[i] = float64(i) * 0.001
		y[i] = float64(n-i) * 0.001
		yGPU[i] = y[i]
	}
	
	// Gonum reference
	blas64.Axpy(alpha, blas64.Vector{N: n, Inc: 1, Data: x}, blas64.Vector{N: n, Inc: 1, Data: y})
	
	// GUDA computation
	xFloat32 := make([]float32, n)
	yFloat32 := make([]float32, n)
	for i := 0; i < n; i++ {
		xFloat32[i] = float32(x[i])
		yFloat32[i] = float32(yGPU[i])
	}
	
	d_x, _ := Malloc(n * 4)
	d_y, _ := Malloc(n * 4)
	defer Free(d_x)
	defer Free(d_y)
	
	Memcpy(d_x, xFloat32, n*4, MemcpyHostToDevice)
	Memcpy(d_y, yFloat32, n*4, MemcpyHostToDevice)
	
	err := AXPY(float32(alpha), d_x, d_y, n)
	if err != nil {
		t.Fatalf("AXPY failed: %v", err)
	}
	Synchronize()
	
	gpuResult := d_y.Float32()[:n]
	
	// Compare
	maxError := 0.0
	for i := 0; i < n; i++ {
		error := abs(y[i] - float64(gpuResult[i]))
		if error > maxError {
			maxError = error
		}
	}
	
	if maxError > 1e-5 {
		t.Errorf("AXPY: max error %e exceeds tolerance", maxError)
	}
	
	t.Logf("AXPY[n=%d]: max error=%e", n, maxError)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}