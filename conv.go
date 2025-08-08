package guda

import (
	"fmt"
	
	"github.com/LynnColeArt/guda/compute"
	"github.com/LynnColeArt/guda/compute/blas"
)

// ConvParams defines parameters for convolution operations
type ConvParams struct {
	// Input dimensions: [batch, channels, height, width]
	BatchSize    int
	InChannels   int
	InHeight     int
	InWidth      int
	
	// Kernel dimensions: [out_channels, in_channels, kernel_height, kernel_width]
	OutChannels  int
	KernelHeight int
	KernelWidth  int
	
	// Convolution parameters
	StrideH      int
	StrideW      int
	PadH         int
	PadW         int
	DilationH    int
	DilationW    int
	
	// Optional bias: [out_channels]
	UseBias      bool
}

// Validate checks if convolution parameters are valid
func (p *ConvParams) Validate() error {
	if p.BatchSize <= 0 || p.InChannels <= 0 || p.InHeight <= 0 || p.InWidth <= 0 {
		return fmt.Errorf("invalid input dimensions")
	}
	if p.OutChannels <= 0 || p.KernelHeight <= 0 || p.KernelWidth <= 0 {
		return fmt.Errorf("invalid kernel dimensions")
	}
	if p.StrideH <= 0 || p.StrideW <= 0 {
		return fmt.Errorf("invalid stride")
	}
	if p.DilationH <= 0 || p.DilationW <= 0 {
		return fmt.Errorf("invalid dilation")
	}
	if p.PadH < 0 || p.PadW < 0 {
		return fmt.Errorf("invalid padding")
	}
	return nil
}

// OutputHeight computes the output height after convolution
func (p *ConvParams) OutputHeight() int {
	effectiveKH := (p.KernelHeight - 1) * p.DilationH + 1
	return (p.InHeight + 2*p.PadH - effectiveKH) / p.StrideH + 1
}

// OutputWidth computes the output width after convolution
func (p *ConvParams) OutputWidth() int {
	effectiveKW := (p.KernelWidth - 1) * p.DilationW + 1
	return (p.InWidth + 2*p.PadW - effectiveKW) / p.StrideW + 1
}

// Conv2D performs 2D convolution: output = conv(input, kernel) + bias
// Input shape: [batch, in_channels, height, width]
// Kernel shape: [out_channels, in_channels, kernel_height, kernel_width]
// Output shape: [batch, out_channels, out_height, out_width]
func Conv2D(input, kernel, bias, output DevicePtr, params *ConvParams) error {
	if err := params.Validate(); err != nil {
		return err
	}
	
	outH := params.OutputHeight()
	outW := params.OutputWidth()
	
	// For now, use im2col + GEMM approach
	// This is memory-hungry but allows us to reuse optimized GEMM
	return conv2DIm2col(input, kernel, bias, output, params, outH, outW)
}

// conv2DIm2col implements convolution using im2col transformation
func conv2DIm2col(input, kernel, bias, output DevicePtr, params *ConvParams, outH, outW int) error {
	// Calculate sizes
	colHeight := params.InChannels * params.KernelHeight * params.KernelWidth
	colWidth := outH * outW
	
	// Allocate workspace for im2col transformation
	workspaceSize := params.BatchSize * colHeight * colWidth * 4 // float32
	workspace, err := Malloc(workspaceSize)
	if err != nil {
		return fmt.Errorf("failed to allocate workspace: %w", err)
	}
	defer Free(workspace)
	
	// Process each batch
	inputData := input.Float32()
	kernelData := kernel.Float32()
	outputData := output.Float32()
	workspaceData := workspace.Float32()
	
	inputStride := params.InChannels * params.InHeight * params.InWidth
	outputStride := params.OutChannels * outH * outW
	
	for b := 0; b < params.BatchSize; b++ {
		// Extract image patches into column matrix
		err := im2col(
			inputData[b*inputStride:],
			params.InChannels, params.InHeight, params.InWidth,
			params.KernelHeight, params.KernelWidth,
			params.PadH, params.PadW,
			params.StrideH, params.StrideW,
			params.DilationH, params.DilationW,
			workspaceData[b*colHeight*colWidth:],
		)
		if err != nil {
			return err
		}
	}
	
	// Reshape kernel for GEMM: [out_channels, in_channels * kh * kw]
	// kernel is already in the right format
	
	// Perform GEMM: output = kernel * im2col_data
	// kernel shape: [out_channels, in_channels * kernel_h * kernel_w]
	// im2col shape: [in_channels * kernel_h * kernel_w, batch_size * out_h * out_w]
	// output shape: [out_channels, batch_size * out_h * out_w]
	impl := compute.Implementation{}
	
	M := params.OutChannels
	N := params.BatchSize * outH * outW
	K := colHeight // in_channels * kernel_h * kernel_w
	
	// Compute: output = kernel * workspace
	// kernel is M x K, workspace is K x N, output is M x N
	impl.Sgemm(
		blas.NoTrans, blas.NoTrans,
		M, N, K,
		1.0,
		kernelData, K,      // lda = K (kernel is row-major)
		workspaceData, N,   // ldb = N (workspace is row-major)
		0.0,
		outputData, N,      // ldc = N (output is row-major)
	)
	
	// The GEMM output is in shape [out_channels, batch_size * out_h * out_w]
	// We need to transpose it to [batch_size, out_channels, out_h, out_w]
	// Create a temporary buffer for the transposed output
	tempOutput := make([]float32, len(outputData[:params.BatchSize*params.OutChannels*outH*outW]))
	copy(tempOutput, outputData[:len(tempOutput)])
	
	// Transpose the output
	for b := 0; b < params.BatchSize; b++ {
		for oc := 0; oc < params.OutChannels; oc++ {
			for h := 0; h < outH; h++ {
				for w := 0; w < outW; w++ {
					srcIdx := oc*(params.BatchSize*outH*outW) + b*outH*outW + h*outW + w
					dstIdx := b*outputStride + oc*outH*outW + h*outW + w
					outputData[dstIdx] = tempOutput[srcIdx]
				}
			}
		}
	}
	
	// Add bias if needed
	if params.UseBias && bias.ptr != nil {
		biasData := bias.Float32()
		for b := 0; b < params.BatchSize; b++ {
			for oc := 0; oc < params.OutChannels; oc++ {
				biasVal := biasData[oc]
				offset := b*outputStride + oc*outH*outW
				for i := 0; i < outH*outW; i++ {
					outputData[offset+i] += biasVal
				}
			}
		}
	}
	
	return nil
}

// im2col extracts image patches and arranges them as columns
// Output layout: [in_channels * kernel_h * kernel_w, out_h * out_w]
func im2col(
	input []float32,
	channels, height, width int,
	kernelH, kernelW int,
	padH, padW int,
	strideH, strideW int,
	dilationH, dilationW int,
	output []float32,
) error {
	outH := (height + 2*padH - (kernelH-1)*dilationH - 1) / strideH + 1
	outW := (width + 2*padW - (kernelW-1)*dilationW - 1) / strideW + 1
	
	// The output is organized as:
	// Each column represents one output position (h, w)
	// Each row represents one input position in the kernel (c, kh, kw)
	
	for c := 0; c < channels; c++ {
		for kh := 0; kh < kernelH; kh++ {
			for kw := 0; kw < kernelW; kw++ {
				rowIdx := (c*kernelH + kh)*kernelW + kw
				
				for h := 0; h < outH; h++ {
					for w := 0; w < outW; w++ {
						// Calculate input position
						inH := h*strideH - padH + kh*dilationH
						inW := w*strideW - padW + kw*dilationW
						
						colIdx := h*outW + w
						outIdx := rowIdx*outH*outW + colIdx
						
						// Check bounds and extract value
						if inH >= 0 && inH < height && inW >= 0 && inW < width {
							output[outIdx] = input[c*height*width + inH*width + inW]
						} else {
							output[outIdx] = 0 // Padding
						}
					}
				}
			}
		}
	}
	
	return nil
}

// Conv2DDirect performs direct convolution without im2col transformation
// This is more memory efficient for small kernels
func Conv2DDirect(input, kernel, bias, output DevicePtr, params *ConvParams) error {
	if err := params.Validate(); err != nil {
		return err
	}
	
	outH := params.OutputHeight()
	outW := params.OutputWidth()
	
	inputData := input.Float32()
	kernelData := kernel.Float32()
	outputData := output.Float32()
	
	// Zero output
	for i := range outputData[:params.BatchSize*params.OutChannels*outH*outW] {
		outputData[i] = 0
	}
	
	// Direct convolution loops
	for b := 0; b < params.BatchSize; b++ {
		for oc := 0; oc < params.OutChannels; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := float32(0)
					
					// Convolve
					for ic := 0; ic < params.InChannels; ic++ {
						for kh := 0; kh < params.KernelHeight; kh++ {
							for kw := 0; kw < params.KernelWidth; kw++ {
								ih := oh*params.StrideH - params.PadH + kh*params.DilationH
								iw := ow*params.StrideW - params.PadW + kw*params.DilationW
								
								if ih >= 0 && ih < params.InHeight && iw >= 0 && iw < params.InWidth {
									inputIdx := b*params.InChannels*params.InHeight*params.InWidth +
										ic*params.InHeight*params.InWidth +
										ih*params.InWidth + iw
									
									kernelIdx := oc*params.InChannels*params.KernelHeight*params.KernelWidth +
										ic*params.KernelHeight*params.KernelWidth +
										kh*params.KernelWidth + kw
									
									sum += inputData[inputIdx] * kernelData[kernelIdx]
								}
							}
						}
					}
					
					outputIdx := b*params.OutChannels*outH*outW +
						oc*outH*outW +
						oh*outW + ow
					
					outputData[outputIdx] = sum
				}
			}
		}
	}
	
	// Add bias if needed
	if params.UseBias && bias.ptr != nil {
		biasData := bias.Float32()
		for b := 0; b < params.BatchSize; b++ {
			for oc := 0; oc < params.OutChannels; oc++ {
				biasVal := biasData[oc]
				offset := b*params.OutChannels*outH*outW + oc*outH*outW
				for i := 0; i < outH*outW; i++ {
					outputData[offset+i] += biasVal
				}
			}
		}
	}
	
	return nil
}