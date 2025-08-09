//go:build amd64
// +build amd64

package compute

import (
	"github.com/LynnColeArt/guda/compute/asm/f32"
	"golang.org/x/sys/cpu"
)

// InitCPUFeatures initializes CPU feature detection for optimized kernels
func InitCPUFeatures() {
	// Detect CPU features and inform the f32 package
	hasAVX512 := cpu.X86.HasAVX512F
	hasAVX2 := cpu.X86.HasAVX2 && cpu.X86.HasFMA
	
	f32.SetCPUFeatures(hasAVX512, hasAVX2)
}

// init ensures CPU features are detected at package load time
func init() {
	InitCPUFeatures()
}