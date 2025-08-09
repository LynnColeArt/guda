package vnni

import (
	"testing"
	"golang.org/x/sys/cpu"
)

func TestCheckCPUFeatures(t *testing.T) {
	t.Logf("CPU Features:")
	t.Logf("  AVX:          %v", cpu.X86.HasAVX)
	t.Logf("  AVX2:         %v", cpu.X86.HasAVX2)
	t.Logf("  AVX512F:      %v", cpu.X86.HasAVX512F)
	t.Logf("  AVX512DQ:     %v", cpu.X86.HasAVX512DQ)
	t.Logf("  AVX512BW:     %v", cpu.X86.HasAVX512BW)
	t.Logf("  AVX512VL:     %v", cpu.X86.HasAVX512VL)
	t.Logf("  AVX512VNNI:   %v", cpu.X86.HasAVX512VNNI)
	t.Logf("  AVX512BF16:   %v", cpu.X86.HasAVX512BF16)
	t.Logf("  AVX512VBMI:   %v", cpu.X86.HasAVX512VBMI)
	t.Logf("  AVX512VBMI2:  %v", cpu.X86.HasAVX512VBMI2)
	
	// Check what golang.org/x/sys/cpu thinks
	t.Logf("\nDetection results:")
	t.Logf("  HasVNNI():    %v", HasVNNI())
	t.Logf("  HasBF16():    %v", HasBF16())
}