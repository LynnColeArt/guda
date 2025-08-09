//go:build !linux
// +build !linux

// Package guda provides performance counter stubs for non-Linux platforms
package guda

import (
	"testing"
)

// LinuxPerfMonitor stub for non-Linux platforms
type LinuxPerfMonitor struct{}

// NewLinuxPerfMonitor returns a stub monitor on non-Linux platforms
func NewLinuxPerfMonitor() *LinuxPerfMonitor {
	return &LinuxPerfMonitor{}
}

// Start is a no-op on non-Linux platforms
func (pm *LinuxPerfMonitor) Start() error {
	return nil
}

// Stop returns empty counters on non-Linux platforms
func (pm *LinuxPerfMonitor) Stop() *PerfCounters {
	return &PerfCounters{}
}

// MeasureWithHardwareCounters falls back to basic timing on non-Linux platforms
func MeasureWithHardwareCounters(name string, fn func() error) (*PerfCounters, error) {
	return MeasureKernel(name, fn)
}

// IntegratePerfCounters is simplified on non-Linux platforms
func IntegratePerfCounters(b interface{}, name string, fn func()) {
	// Just run the function on non-Linux platforms
	fn()
}