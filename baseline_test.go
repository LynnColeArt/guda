// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package guda

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"
)

// BaselineResult captures the state before we start optimizing
type BaselineResult struct {
	TestName      string    `json:"test_name"`
	Timestamp     time.Time `json:"timestamp"`
	
	// Performance metrics
	Duration      time.Duration `json:"duration_ns"`
	BytesPerOp    int64         `json:"bytes_per_op"`
	
	// Numerical results (for accuracy verification)
	Checksum      float64   `json:"checksum"`
	SampleValues  []float32 `json:"sample_values"`
}

// TestCaptureBaseline captures baseline performance for our core operations
func TestCaptureBaseline(t *testing.T) {
	results := []BaselineResult{}
	
	// Test 1: AXPY operation (using our existing implementation)
	t.Run("AXPY_1000", func(t *testing.T) {
		n := 1000
		x := make([]float32, n)
		y := make([]float32, n)
		alpha := float32(2.5)
		
		// Initialize test data
		for i := range x {
			x[i] = float32(i) * 0.001
			y[i] = float32(i) * 0.002
		}
		
		// Warm up
		AxpyUnitary(alpha, x[:n], y[:n])
		
		// Measure
		start := time.Now()
		AxpyUnitary(alpha, x[:n], y[:n])
		duration := time.Since(start)
		
		// Calculate checksum
		checksum := 0.0
		for _, v := range y {
			checksum += float64(v)
		}
		
		result := BaselineResult{
			TestName:     "AXPY_1000",
			Timestamp:    time.Now(),
			Duration:     duration,
			BytesPerOp:   int64(n * 4 * 3), // 3 arrays of float32
			Checksum:     checksum,
			SampleValues: y[:10], // First 10 values
		}
		
		results = append(results, result)
		t.Logf("AXPY_1000: %v, checksum=%f", duration, checksum)
	})
	
	// Test 2: DOT product
	t.Run("DOT_1000", func(t *testing.T) {
		n := 1000
		x := make([]float32, n)
		y := make([]float32, n)
		
		// Initialize test data
		for i := range x {
			x[i] = float32(i) * 0.001
			y[i] = float32(i) * 0.002
		}
		
		// Warm up
		_ = DotUnitary(x[:n], y[:n])
		
		// Measure
		start := time.Now()
		result := DotUnitary(x[:n], y[:n])
		duration := time.Since(start)
		
		baselineResult := BaselineResult{
			TestName:     "DOT_1000",
			Timestamp:    time.Now(),
			Duration:     duration,
			BytesPerOp:   int64(n * 4 * 2), // 2 arrays of float32
			Checksum:     float64(result),
			SampleValues: []float32{result},
		}
		
		results = append(results, baselineResult)
		t.Logf("DOT_1000: %v, result=%f", duration, result)
	})
	
	// Test 3: Simple GEMM (if we have it implemented)
	t.Run("GEMM_128x128", func(t *testing.T) {
		m, n, k := 128, 128, 128
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		c := make([]float32, m*n)
		
		// Initialize matrices
		for i := range a {
			a[i] = float32(i%100) * 0.01
		}
		for i := range b {
			b[i] = float32(i%100) * 0.01
		}
		
		// Simple GEMM implementation for baseline
		// C = A * B
		start := time.Now()
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a[i*k+l] * b[l*n+j]
				}
				c[i*n+j] = sum
			}
		}
		duration := time.Since(start)
		
		// Calculate checksum
		checksum := 0.0
		for _, v := range c {
			checksum += float64(v)
		}
		
		result := BaselineResult{
			TestName:     "GEMM_128x128",
			Timestamp:    time.Now(),
			Duration:     duration,
			BytesPerOp:   int64(m*k*4 + k*n*4 + m*n*4),
			Checksum:     checksum,
			SampleValues: c[:10],
		}
		
		results = append(results, result)
		t.Logf("GEMM_128x128: %v, checksum=%f", duration, checksum)
	})
	
	// Save results if requested
	if os.Getenv("SAVE_BASELINE") == "1" {
		data, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			t.Fatalf("Failed to marshal results: %v", err)
		}
		
		filename := fmt.Sprintf("baseline_%s.json", time.Now().Format("20060102_150405"))
		if err := os.WriteFile(filename, data, 0644); err != nil {
			t.Fatalf("Failed to save baseline: %v", err)
		}
		
		t.Logf("Baseline saved to %s", filename)
	}
}

// Helper functions for the tests (these would normally be in compute/asm/f32)
func AxpyUnitary(alpha float32, x, y []float32) {
	n := min(len(x), len(y))
	for i := 0; i < n; i++ {
		y[i] += alpha * x[i]
	}
}

func DotUnitary(x, y []float32) float32 {
	n := min(len(x), len(y))
	sum := float32(0)
	for i := 0; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}

