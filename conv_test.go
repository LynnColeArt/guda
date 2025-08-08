package guda

import (
	"math"
	"testing"
)

func TestConv2DBasic(t *testing.T) {
	// Test basic 3x3 convolution
	params := &ConvParams{
		BatchSize:    1,
		InChannels:   1,
		InHeight:     4,
		InWidth:      4,
		OutChannels:  1,
		KernelHeight: 3,
		KernelWidth:  3,
		StrideH:      1,
		StrideW:      1,
		PadH:         1,
		PadW:         1,
		DilationH:    1,
		DilationW:    1,
		UseBias:      false,
	}
	
	// Input: 4x4 image
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	
	// Kernel: 3x3 edge detector
	kernel := []float32{
		-1, -1, -1,
		-1, 8, -1,
		-1, -1, -1,
	}
	
	// Allocate memory
	inputPtr := MallocOrFail(t, len(input)*4)
	kernelPtr := MallocOrFail(t, len(kernel)*4)
	outputPtr := MallocOrFail(t, 16*4) // 4x4 output
	defer Free(inputPtr)
	defer Free(kernelPtr)
	defer Free(outputPtr)
	
	// Copy data
	copy(inputPtr.Float32(), input)
	copy(kernelPtr.Float32(), kernel)
	
	// Run convolution
	err := Conv2D(inputPtr, kernelPtr, DevicePtr{}, outputPtr, params)
	if err != nil {
		t.Fatalf("Conv2D failed: %v", err)
	}
	
	// Check output
	output := outputPtr.Float32()[:16]
	
	// Expected output (verified with direct convolution)
	expected := []float32{
		-5, -6, -3, 14,
		12, 0, 0, 27,
		24, 0, 0, 39,
		71, 54, 57, 90,
	}
	
	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
			t.Errorf("Output[%d]: expected %f, got %f", i, expected[i], output[i])
		}
	}
}

func TestConv2DWithBias(t *testing.T) {
	params := &ConvParams{
		BatchSize:    1,
		InChannels:   1,
		InHeight:     2,
		InWidth:      2,
		OutChannels:  2,
		KernelHeight: 2,
		KernelWidth:  2,
		StrideH:      1,
		StrideW:      1,
		PadH:         0,
		PadW:         0,
		DilationH:    1,
		DilationW:    1,
		UseBias:      true,
	}
	
	// Input: 2x2
	input := []float32{1, 2, 3, 4}
	
	// Two kernels: 2x2 each
	kernel := []float32{
		1, 0, 0, 1, // First kernel (identity-like)
		1, 1, 1, 1,  // Second kernel (sum all)
	}
	
	// Bias for each output channel
	bias := []float32{10, 20}
	
	// Allocate
	inputPtr := MallocOrFail(t, len(input)*4)
	kernelPtr := MallocOrFail(t, len(kernel)*4)
	biasPtr := MallocOrFail(t, len(bias)*4)
	outputPtr := MallocOrFail(t, 2*4) // 2 output channels, 1x1 each
	defer Free(inputPtr)
	defer Free(kernelPtr)
	defer Free(biasPtr)
	defer Free(outputPtr)
	
	copy(inputPtr.Float32(), input)
	copy(kernelPtr.Float32(), kernel)
	copy(biasPtr.Float32(), bias)
	
	err := Conv2D(inputPtr, kernelPtr, biasPtr, outputPtr, params)
	if err != nil {
		t.Fatalf("Conv2D with bias failed: %v", err)
	}
	
	output := outputPtr.Float32()[:2]
	
	// Expected: 
	// Channel 0: 1*1 + 2*0 + 3*0 + 4*1 + bias[0] = 1 + 4 + 10 = 15
	// Channel 1: 1*1 + 2*1 + 3*1 + 4*1 + bias[1] = 10 + 20 = 30
	expected := []float32{15, 30}
	
	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
			t.Errorf("Output[%d]: expected %f, got %f", i, expected[i], output[i])
		}
	}
}

func TestConv2DStride(t *testing.T) {
	params := &ConvParams{
		BatchSize:    1,
		InChannels:   1,
		InHeight:     4,
		InWidth:      4,
		OutChannels:  1,
		KernelHeight: 2,
		KernelWidth:  2,
		StrideH:      2,
		StrideW:      2,
		PadH:         0,
		PadW:         0,
		DilationH:    1,
		DilationW:    1,
		UseBias:      false,
	}
	
	// Input: 4x4
	input := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	
	// Kernel: 2x2 average
	kernel := []float32{0.25, 0.25, 0.25, 0.25}
	
	inputPtr := MallocOrFail(t, len(input)*4)
	kernelPtr := MallocOrFail(t, len(kernel)*4)
	outputPtr := MallocOrFail(t, 4*4) // 2x2 output
	defer Free(inputPtr)
	defer Free(kernelPtr)
	defer Free(outputPtr)
	
	copy(inputPtr.Float32(), input)
	copy(kernelPtr.Float32(), kernel)
	
	err := Conv2D(inputPtr, kernelPtr, DevicePtr{}, outputPtr, params)
	if err != nil {
		t.Fatalf("Conv2D with stride failed: %v", err)
	}
	
	output := outputPtr.Float32()[:4]
	
	// Expected: average of each 2x2 block
	// Top-left: (1+2+5+6)/4 = 3.5
	// Top-right: (3+4+7+8)/4 = 5.5
	// Bottom-left: (9+10+13+14)/4 = 11.5
	// Bottom-right: (11+12+15+16)/4 = 13.5
	expected := []float32{3.5, 5.5, 11.5, 13.5}
	
	for i := range expected {
		if math.Abs(float64(output[i]-expected[i])) > 1e-5 {
			t.Errorf("Output[%d]: expected %f, got %f", i, expected[i], output[i])
		}
	}
}

func TestConv2DMultiChannel(t *testing.T) {
	params := &ConvParams{
		BatchSize:    1,
		InChannels:   2,
		InHeight:     2,
		InWidth:      2,
		OutChannels:  1,
		KernelHeight: 2,
		KernelWidth:  2,
		StrideH:      1,
		StrideW:      1,
		PadH:         0,
		PadW:         0,
		DilationH:    1,
		DilationW:    1,
		UseBias:      false,
	}
	
	// Input: 2 channels, 2x2 each
	input := []float32{
		1, 2, 3, 4,     // Channel 0
		5, 6, 7, 8,     // Channel 1
	}
	
	// Kernel: 1 output channel, 2 input channels, 2x2 each
	kernel := []float32{
		1, 0, 0, 1,     // Kernel for channel 0
		1, 0, 0, 1,     // Kernel for channel 1
	}
	
	inputPtr := MallocOrFail(t, len(input)*4)
	kernelPtr := MallocOrFail(t, len(kernel)*4)
	outputPtr := MallocOrFail(t, 1*4) // 1x1 output
	defer Free(inputPtr)
	defer Free(kernelPtr)
	defer Free(outputPtr)
	
	copy(inputPtr.Float32(), input)
	copy(kernelPtr.Float32(), kernel)
	
	err := Conv2D(inputPtr, kernelPtr, DevicePtr{}, outputPtr, params)
	if err != nil {
		t.Fatalf("Conv2D multi-channel failed: %v", err)
	}
	
	output := outputPtr.Float32()[:1]
	
	// Expected: 
	// Channel 0 contribution: 1*1 + 2*0 + 3*0 + 4*1 = 5
	// Channel 1 contribution: 5*1 + 6*0 + 7*0 + 8*1 = 13
	// Total: 5 + 13 = 18
	expected := float32(18)
	
	if math.Abs(float64(output[0]-expected)) > 1e-5 {
		t.Errorf("Output: expected %f, got %f", expected, output[0])
	}
}

// Benchmark convolution operations
func BenchmarkConv2D(b *testing.B) {
	configs := []struct {
		name   string
		params ConvParams
	}{
		{
			name: "3x3_stride1_small",
			params: ConvParams{
				BatchSize: 1, InChannels: 32, InHeight: 32, InWidth: 32,
				OutChannels: 64, KernelHeight: 3, KernelWidth: 3,
				StrideH: 1, StrideW: 1, PadH: 1, PadW: 1,
				DilationH: 1, DilationW: 1,
			},
		},
		{
			name: "3x3_stride1_medium",
			params: ConvParams{
				BatchSize: 1, InChannels: 64, InHeight: 56, InWidth: 56,
				OutChannels: 128, KernelHeight: 3, KernelWidth: 3,
				StrideH: 1, StrideW: 1, PadH: 1, PadW: 1,
				DilationH: 1, DilationW: 1,
			},
		},
		{
			name: "1x1_pointwise",
			params: ConvParams{
				BatchSize: 1, InChannels: 256, InHeight: 28, InWidth: 28,
				OutChannels: 256, KernelHeight: 1, KernelWidth: 1,
				StrideH: 1, StrideW: 1, PadH: 0, PadW: 0,
				DilationH: 1, DilationW: 1,
			},
		},
		{
			name: "5x5_stride2",
			params: ConvParams{
				BatchSize: 1, InChannels: 3, InHeight: 224, InWidth: 224,
				OutChannels: 32, KernelHeight: 5, KernelWidth: 5,
				StrideH: 2, StrideW: 2, PadH: 2, PadW: 2,
				DilationH: 1, DilationW: 1,
			},
		},
	}
	
	for _, cfg := range configs {
		b.Run("Im2col_"+cfg.name, func(b *testing.B) {
			benchmarkConv2D(b, &cfg.params, false)
		})
		
		b.Run("Direct_"+cfg.name, func(b *testing.B) {
			benchmarkConv2D(b, &cfg.params, true)
		})
	}
}

func benchmarkConv2D(b *testing.B, params *ConvParams, useDirect bool) {
	// Calculate sizes
	inputSize := params.BatchSize * params.InChannels * params.InHeight * params.InWidth
	kernelSize := params.OutChannels * params.InChannels * params.KernelHeight * params.KernelWidth
	outputSize := params.BatchSize * params.OutChannels * params.OutputHeight() * params.OutputWidth()
	
	// Allocate
	input := MallocOrFail(b, inputSize*4)
	kernel := MallocOrFail(b, kernelSize*4)
	output := MallocOrFail(b, outputSize*4)
	defer Free(input)
	defer Free(kernel)
	defer Free(output)
	
	// Initialize with some data
	inputData := input.Float32()
	kernelData := kernel.Float32()
	for i := range inputData {
		inputData[i] = float32(i % 10)
	}
	for i := range kernelData {
		kernelData[i] = float32(i % 10) * 0.1
	}
	
	// Calculate FLOPs
	outH := params.OutputHeight()
	outW := params.OutputWidth()
	flops := int64(params.BatchSize) * int64(params.OutChannels) * int64(outH) * int64(outW) *
		int64(params.InChannels) * int64(params.KernelHeight) * int64(params.KernelWidth) * 2
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		var err error
		if useDirect {
			err = Conv2DDirect(input, kernel, DevicePtr{}, output, params)
		} else {
			err = Conv2D(input, kernel, DevicePtr{}, output, params)
		}
		if err != nil {
			b.Fatalf("Convolution failed: %v", err)
		}
	}
	
	// Report metrics
	gflops := float64(flops*int64(b.N)) / b.Elapsed().Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}