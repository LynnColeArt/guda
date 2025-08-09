//go:build amd64
// +build amd64

package compute

import (
	"fmt"
	"testing"
	"github.com/LynnColeArt/guda/compute/asm/f32"
)

func TestCPUFeatureDetection(t *testing.T) {
	// Force initialization
	InitCPUFeatures()
	
	fmt.Printf("f32.HasAVX512Support: %v\n", f32.HasAVX512Support)
	fmt.Printf("f32.HasAVX2Support: %v\n", f32.HasAVX2Support)
	
	// On the test system which has AVX-512, this should be true
	if !f32.HasAVX512Support {
		t.Log("AVX-512 not detected - this might be expected on some systems")
	}
}