// +build !amd64

package guda

// SimdF16ToF32 converts float16 to float32 (generic fallback)
func SimdF16ToF32(src []uint16, dst []float32) {
	for i := 0; i < min(len(src), len(dst)); i++ {
		dst[i] = Float16(src[i]).ToFloat32()
	}
}

// SimdF32ToF16 converts float32 to float16 (generic fallback)
func SimdF32ToF16(src []float32, dst []uint16) {
	for i := 0; i < min(len(src), len(dst)); i++ {
		dst[i] = uint16(FromFloat32(src[i]))
	}
}

// AddFloat16SIMD uses standard implementation for float16 vector addition
func AddFloat16SIMD(a, b, c DevicePtr, n int) error {
	return AddFloat16(a, b, c, n)
}

// MultiplyFloat16SIMD uses standard implementation for float16 vector multiplication
func MultiplyFloat16SIMD(a, b, c DevicePtr, n int) error {
	return MultiplyFloat16(a, b, c, n)
}

// FMAFloat16SIMD performs d = a*b + c (generic fallback)
func FMAFloat16SIMD(a, b, c, d DevicePtr, n int) error {
	temp, _ := Malloc(n * 2)
	defer Free(temp)
	
	err := MultiplyFloat16(a, b, temp, n)
	if err != nil {
		return err
	}
	return AddFloat16(temp, c, d, n)
}

// GEMMFloat16SIMD performs matrix multiplication with float16 (generic)
func GEMMFloat16SIMD(transA, transB bool, m, n, k int, alpha float32,
	a DevicePtr, lda int,
	b DevicePtr, ldb int,
	beta float32,
	c DevicePtr, ldc int) error {
	
	return GEMMFloat16(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// BatchedGEMMFloat16SIMD performs multiple GEMMs in parallel (generic)
func BatchedGEMMFloat16SIMD(
	batch int,
	transA, transB bool,
	m, n, k int,
	alpha float32,
	aArray []DevicePtr, ldaArray []int,
	bArray []DevicePtr, ldbArray []int,
	beta float32,
	cArray []DevicePtr, ldcArray []int) error {
	
	errChan := make(chan error, batch)
	
	for b := 0; b < batch; b++ {
		go func(idx int) {
			err := GEMMFloat16(transA, transB, m, n, k, alpha,
				aArray[idx], ldaArray[idx],
				bArray[idx], ldbArray[idx],
				beta,
				cArray[idx], ldcArray[idx])
			errChan <- err
		}(b)
	}
	
	for b := 0; b < batch; b++ {
		if err := <-errChan; err != nil {
			return err
		}
	}
	
	return nil
}

// Conv2DFloat16SIMD performs 2D convolution with float16 (generic)
func Conv2DFloat16SIMD(
	input DevicePtr, inputH, inputW, inputC int,
	kernel DevicePtr, kernelH, kernelW int,
	output DevicePtr, outputH, outputW, outputC int,
	strideH, strideW int,
	padH, padW int) error {
	
	grid := Dim3{X: (outputW + 15) / 16, Y: (outputH + 15) / 16, Z: (outputC + 3) / 4}
	block := Dim3{X: 16, Y: 16, Z: 1}
	
	return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
		ox := tid.BlockIdx.X*16 + tid.ThreadIdx.X
		oy := tid.BlockIdx.Y*16 + tid.ThreadIdx.Y
		oc := tid.BlockIdx.Z * 4
		
		if ox >= outputW || oy >= outputH || oc >= outputC {
			return
		}
		
		inputF16 := input.Float16()
		kernelF16 := kernel.Float16()
		outputF16 := output.Float16()
		
		var sum [4]float32
		
		for ky := 0; ky < kernelH; ky++ {
			for kx := 0; kx < kernelW; kx++ {
				ix := ox*strideW - padW + kx
				iy := oy*strideH - padH + ky
				
				if ix >= 0 && ix < inputW && iy >= 0 && iy < inputH {
					for ic := 0; ic < inputC; ic++ {
						inputIdx := (iy*inputW + ix)*inputC + ic
						inputVal := inputF16.GetFloat32(inputIdx)
						
						for i := 0; i < 4 && oc+i < outputC; i++ {
							kernelIdx := ((oc+i)*kernelH*kernelW + ky*kernelW + kx)*inputC + ic
							kernelVal := kernelF16.GetFloat32(kernelIdx)
							sum[i] += inputVal * kernelVal
						}
					}
				}
			}
		}
		
		for i := 0; i < 4 && oc+i < outputC; i++ {
			outputIdx := (oy*outputW + ox)*outputC + oc + i
			outputF16.SetFloat32(outputIdx, sum[i])
		}
	}), grid, block)
}

// LayerNormFloat16SIMD performs layer normalization with float16 (generic)
func LayerNormFloat16SIMD(input, gamma, beta, output DevicePtr, n, hidden int) error {
	grid := Dim3{X: n, Y: 1, Z: 1}
	block := Dim3{X: min(256, hidden), Y: 1, Z: 1}
	
	return Launch(KernelFunc(func(tid ThreadID, args ...interface{}) {
		batch := tid.BlockIdx.X
		tid_local := tid.ThreadIdx.X
		threads := block.X
		
		if batch >= n {
			return
		}
		
		inputF16 := input.Float16()
		gammaF16 := gamma.Float16()
		betaF16 := beta.Float16()
		outputF16 := output.Float16()
		
		var mean, m2 float32
		count := 0
		
		for i := tid_local; i < hidden; i += threads {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			count++
			delta := val - mean
			mean += delta / float32(count)
			m2 += delta * (val - mean)
		}
		
		mean = 0
		m2 = 0
		for i := 0; i < hidden; i++ {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			delta := val - mean
			mean += delta / float32(i+1)
			m2 += delta * (val - mean)
		}
		
		variance := m2 / float32(hidden)
		invStd := 1.0 / (variance + DefaultLayerNormEpsilon)
		
		for i := tid_local; i < hidden; i += threads {
			idx := batch*hidden + i
			val := inputF16.GetFloat32(idx)
			normalized := (val - mean) * invStd
			
			g := gammaF16.GetFloat32(i)
			b := betaF16.GetFloat32(i)
			result := normalized*g + b
			
			outputF16.SetFloat32(idx, result)
		}
	}), grid, block)
}