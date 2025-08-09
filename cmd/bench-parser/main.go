package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

type TestEvent struct {
	Time    time.Time `json:"Time"`
	Action  string    `json:"Action"`
	Package string    `json:"Package"`
	Test    string    `json:"Test,omitempty"`
	Output  string    `json:"Output,omitempty"`
	Elapsed float64   `json:"Elapsed,omitempty"`
}

type BenchmarkResult struct {
	Name     string
	NsPerOp  float64
	MBPerSec float64
	Status   string
	Error    string
}

func main() {
	var jsonFile string
	flag.StringVar(&jsonFile, "file", "", "JSON benchmark file to parse")
	flag.Parse()

	if jsonFile == "" {
		fmt.Println("Usage: bench-parser -file <benchmark.json>")
		os.Exit(1)
	}

	file, err := os.Open(jsonFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var results []BenchmarkResult
	scanner := bufio.NewScanner(file)
	
	currentTest := ""
	for scanner.Scan() {
		var event TestEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			continue
		}

		// Track current test
		if event.Test != "" {
			currentTest = event.Test
		}

		// Parse benchmark output
		if event.Action == "output" && strings.Contains(event.Output, "ns/op") {
			result := parseBenchmarkLine(currentTest, event.Output)
			if result != nil {
				results = append(results, *result)
			}
		}

		// Check for failures
		if event.Action == "fail" && currentTest != "" {
			results = append(results, BenchmarkResult{
				Name:   currentTest,
				Status: "FAIL",
			})
		}
	}

	// Print summary
	fmt.Println("\nBenchmark Results Summary")
	fmt.Println("========================")
	fmt.Printf("%-50s %15s %15s %10s\n", "Benchmark", "ns/op", "MB/s", "Status")
	fmt.Println(strings.Repeat("-", 92))

	for _, r := range results {
		status := "PASS"
		if r.Status != "" {
			status = r.Status
		}

		if r.MBPerSec > 0 {
			fmt.Printf("%-50s %15.2f %15.2f %10s\n", r.Name, r.NsPerOp, r.MBPerSec, status)
		} else if r.NsPerOp > 0 {
			fmt.Printf("%-50s %15.2f %15s %10s\n", r.Name, r.NsPerOp, "-", status)
		} else {
			fmt.Printf("%-50s %15s %15s %10s\n", r.Name, "-", "-", status)
		}
	}
}

func parseBenchmarkLine(testName, line string) *BenchmarkResult {
	// Example: BenchmarkPrefetchComparison/AXPY_Size_1024-16         	32973074	        33.91 ns/op	241591.11 MB/s	       0 B/op	       0 allocs/op
	fields := strings.Fields(line)
	if len(fields) < 4 {
		return nil
	}

	result := &BenchmarkResult{
		Name: testName,
	}

	// Find ns/op value
	for i, field := range fields {
		if field == "ns/op" && i > 0 {
			fmt.Sscanf(fields[i-1], "%f", &result.NsPerOp)
		}
		if field == "MB/s" && i > 0 {
			fmt.Sscanf(fields[i-1], "%f", &result.MBPerSec)
		}
	}

	return result
}