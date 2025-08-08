// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command compare compares current results against baseline
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/LynnColeArt/guda"
)

type ComparisonResult struct {
	TestName string
	Status   string // "PASS", "FAIL", "SLOWER", "FASTER"
	
	// Performance comparison
	BaselineDuration time.Duration
	CurrentDuration  time.Duration
	SpeedupFactor    float64
	
	// Numerical comparison
	ChecksumDiff float64
	MaxAbsDiff   float32
	MaxRelDiff   float32
	
	// Detailed differences
	DifferingIndices []int
	Message         string
}

func main() {
	var (
		baselineFile = flag.String("baseline", "baseline.json", "Baseline results file")
		currentFile  = flag.String("current", "current.json", "Current results file")
		tolerance    = flag.Float64("tol", 1e-6, "Numerical tolerance")
		perfRegress  = flag.Float64("perf-regress", 1.1, "Performance regression threshold (1.1 = 10% slower)")
	)
	flag.Parse()
	
	// Load baseline
	baseline, err := loadResults(*baselineFile)
	if err != nil {
		log.Fatalf("Failed to load baseline: %v", err)
	}
	
	// Load current results
	current, err := loadResults(*currentFile)
	if err != nil {
		log.Fatalf("Failed to load current results: %v", err)
	}
	
	// Compare results
	comparisons := compareResults(baseline, current, *tolerance, *perfRegress)
	
	// Print summary
	printSummary(comparisons)
	
	// Exit with error if any failures
	for _, comp := range comparisons {
		if comp.Status == "FAIL" {
			os.Exit(1)
		}
	}
}

func loadResults(filename string) ([]guda.BaselineResult, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	
	var results []guda.BaselineResult
	if err := json.Unmarshal(data, &results); err != nil {
		return nil, err
	}
	
	return results, nil
}

func compareResults(baseline, current []guda.BaselineResult, tolerance, perfRegress float64) []ComparisonResult {
	// Create map for easy lookup
	currentMap := make(map[string]guda.BaselineResult)
	for _, result := range current {
		currentMap[result.TestName] = result
	}
	
	comparisons := make([]ComparisonResult, 0, len(baseline))
	
	for _, base := range baseline {
		comp := ComparisonResult{
			TestName:         base.TestName,
			BaselineDuration: base.Duration,
		}
		
		curr, exists := currentMap[base.TestName]
		if !exists {
			comp.Status = "FAIL"
			comp.Message = "Test missing in current results"
			comparisons = append(comparisons, comp)
			continue
		}
		
		comp.CurrentDuration = curr.Duration
		comp.SpeedupFactor = float64(base.Duration) / float64(curr.Duration)
		
		// Check performance regression
		if comp.SpeedupFactor < 1.0/perfRegress {
			comp.Status = "SLOWER"
			comp.Message = fmt.Sprintf("Performance regression: %.2fx slower", 1.0/comp.SpeedupFactor)
		} else if comp.SpeedupFactor > 1.2 {
			comp.Status = "FASTER"
			comp.Message = fmt.Sprintf("Performance improvement: %.2fx faster", comp.SpeedupFactor)
		}
		
		// Check numerical accuracy
		comp.ChecksumDiff = math.Abs(base.Checksum - curr.Checksum)
		
		// Compare sample values
		maxAbsDiff, maxRelDiff := compareArrays(base.First10, curr.First10)
		comp.MaxAbsDiff = maxAbsDiff
		comp.MaxRelDiff = maxRelDiff
		
		// Check if within tolerance
		if comp.ChecksumDiff > tolerance || float64(maxAbsDiff) > tolerance {
			comp.Status = "FAIL"
			comp.Message = fmt.Sprintf("Numerical difference: checksum_diff=%e, max_abs_diff=%e", 
				comp.ChecksumDiff, maxAbsDiff)
			
			// Find differing indices
			comp.DifferingIndices = findDifferences(base.First10, curr.First10, float32(tolerance))
		}
		
		// If no issues found
		if comp.Status == "" {
			comp.Status = "PASS"
		}
		
		comparisons = append(comparisons, comp)
	}
	
	return comparisons
}

func compareArrays(a, b []float32) (maxAbsDiff, maxRelDiff float32) {
	if len(a) != len(b) {
		return math.MaxFloat32, math.MaxFloat32
	}
	
	for i := range a {
		absDiff := float32(math.Abs(float64(a[i] - b[i])))
		if absDiff > maxAbsDiff {
			maxAbsDiff = absDiff
		}
		
		if a[i] != 0 {
			relDiff := absDiff / float32(math.Abs(float64(a[i])))
			if relDiff > maxRelDiff {
				maxRelDiff = relDiff
			}
		}
	}
	
	return
}

func findDifferences(a, b []float32, tolerance float32) []int {
	var indices []int
	for i := range a {
		if i < len(b) {
			diff := float32(math.Abs(float64(a[i] - b[i])))
			if diff > tolerance {
				indices = append(indices, i)
			}
		}
	}
	return indices
}

func printSummary(comparisons []ComparisonResult) {
	fmt.Println("=== GUDA Baseline Comparison ===")
	fmt.Println()
	
	// Count by status
	statusCount := make(map[string]int)
	for _, comp := range comparisons {
		statusCount[comp.Status]++
	}
	
	// Print summary
	fmt.Printf("Total tests: %d\n", len(comparisons))
	fmt.Printf("  PASS:   %d\n", statusCount["PASS"])
	fmt.Printf("  FAIL:   %d\n", statusCount["FAIL"])
	fmt.Printf("  SLOWER: %d\n", statusCount["SLOWER"])
	fmt.Printf("  FASTER: %d\n", statusCount["FASTER"])
	fmt.Println()
	
	// Print failures first
	if statusCount["FAIL"] > 0 {
		fmt.Println("FAILURES:")
		for _, comp := range comparisons {
			if comp.Status == "FAIL" {
				fmt.Printf("  %s: %s\n", comp.TestName, comp.Message)
				if len(comp.DifferingIndices) > 0 {
					fmt.Printf("    Differing at indices: %v\n", comp.DifferingIndices)
				}
			}
		}
		fmt.Println()
	}
	
	// Print performance changes
	if statusCount["SLOWER"] > 0 || statusCount["FASTER"] > 0 {
		fmt.Println("PERFORMANCE CHANGES:")
		for _, comp := range comparisons {
			if comp.Status == "SLOWER" || comp.Status == "FASTER" {
				fmt.Printf("  %s: %s (%.1fms -> %.1fms)\n", 
					comp.TestName, comp.Message,
					float64(comp.BaselineDuration)/1e6,
					float64(comp.CurrentDuration)/1e6)
			}
		}
		fmt.Println()
	}
	
	// Print detailed table for all tests
	fmt.Println("DETAILED RESULTS:")
	fmt.Printf("%-40s %-6s %10s %10s %8s %12s\n", 
		"Test", "Status", "Baseline", "Current", "Speedup", "Checksum Δ")
	fmt.Println(strings.Repeat("-", 90))
	
	for _, comp := range comparisons {
		fmt.Printf("%-40s %-6s %10.1f %10.1f %8.2f %12.2e\n",
			comp.TestName,
			comp.Status,
			float64(comp.BaselineDuration)/1e6, // Convert to ms
			float64(comp.CurrentDuration)/1e6,
			comp.SpeedupFactor,
			comp.ChecksumDiff)
	}
}