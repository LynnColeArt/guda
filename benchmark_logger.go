package guda

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// BenchmarkResult captures the result of a single benchmark run
type BenchmarkResult struct {
	Name           string        `json:"name"`
	Status         string        `json:"status"` // "pass", "fail", "timeout"
	Operations     int64         `json:"operations,omitempty"`
	NsPerOp        float64       `json:"ns_per_op,omitempty"`
	MBPerSec       float64       `json:"mb_per_sec,omitempty"`
	AllocsPerOp    int64         `json:"allocs_per_op,omitempty"`
	BytesPerOp     int64         `json:"bytes_per_op,omitempty"`
	Duration       time.Duration `json:"duration,omitempty"`
	Error          string        `json:"error,omitempty"`
	Timestamp      time.Time     `json:"timestamp"`
	CacheCondition string        `json:"cache_condition,omitempty"` // "hot" or "cold"
}

// BenchmarkLogger manages logging of benchmark results to file
type BenchmarkLogger struct {
	mu          sync.Mutex
	results     []BenchmarkResult
	logDir      string
	sessionFile string
}

var globalLogger = &BenchmarkLogger{
	logDir: "benchmark_logs",
}

// InitBenchmarkLogger initializes the logger for a new benchmark session
func InitBenchmarkLogger(sessionName string) error {
	globalLogger.mu.Lock()
	defer globalLogger.mu.Unlock()

	// Create log directory if it doesn't exist
	if err := os.MkdirAll(globalLogger.logDir, 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %w", err)
	}

	// Create session file name with timestamp
	timestamp := time.Now().Format("20060102_150405")
	globalLogger.sessionFile = filepath.Join(globalLogger.logDir, 
		fmt.Sprintf("%s_%s.json", sessionName, timestamp))

	// Reset results for new session
	globalLogger.results = nil

	// Write initial file
	return globalLogger.flush()
}

// LogBenchmarkResult logs a single benchmark result
func LogBenchmarkResult(result BenchmarkResult) {
	globalLogger.mu.Lock()
	defer globalLogger.mu.Unlock()

	result.Timestamp = time.Now()
	globalLogger.results = append(globalLogger.results, result)

	// Flush to disk immediately to avoid losing data on crash
	globalLogger.flush()
}

// LogBenchmarkPass logs a successful benchmark
func LogBenchmarkPass(name string, nsPerOp float64, mbPerSec float64, ops int64) {
	LogBenchmarkResult(BenchmarkResult{
		Name:        name,
		Status:      "pass",
		Operations:  ops,
		NsPerOp:     nsPerOp,
		MBPerSec:    mbPerSec,
		Timestamp:   time.Now(),
	})
}

// LogBenchmarkFail logs a failed benchmark
func LogBenchmarkFail(name string, err error) {
	LogBenchmarkResult(BenchmarkResult{
		Name:      name,
		Status:    "fail",
		Error:     err.Error(),
		Timestamp: time.Now(),
	})
}

// LogBenchmarkTimeout logs a timed out benchmark
func LogBenchmarkTimeout(name string, duration time.Duration) {
	LogBenchmarkResult(BenchmarkResult{
		Name:      name,
		Status:    "timeout",
		Duration:  duration,
		Timestamp: time.Now(),
	})
}

// flush writes results to disk
func (bl *BenchmarkLogger) flush() error {
	if bl.sessionFile == "" {
		return nil // Not initialized
	}

	data, err := json.MarshalIndent(bl.results, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal results: %w", err)
	}

	return os.WriteFile(bl.sessionFile, data, 0644)
}

// GetLatestLogFile returns the path to the most recent log file
func GetLatestLogFile() (string, error) {
	files, err := filepath.Glob(filepath.Join(globalLogger.logDir, "*.json"))
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no log files found")
	}

	// Sort by modification time to get latest
	var latest string
	var latestTime time.Time
	for _, file := range files {
		info, err := os.Stat(file)
		if err != nil {
			continue
		}
		if info.ModTime().After(latestTime) {
			latest = file
			latestTime = info.ModTime()
		}
	}

	return latest, nil
}

// PrintBenchmarkSummary prints a summary of the latest benchmark session
func PrintBenchmarkSummary() error {
	logFile, err := GetLatestLogFile()
	if err != nil {
		return err
	}

	data, err := os.ReadFile(logFile)
	if err != nil {
		return err
	}

	var results []BenchmarkResult
	if err := json.Unmarshal(data, &results); err != nil {
		return err
	}

	fmt.Printf("\nBenchmark Summary from %s:\n", filepath.Base(logFile))
	fmt.Println(strings.Repeat("=", 62))

	passed, failed, timeout := 0, 0, 0
	for _, r := range results {
		switch r.Status {
		case "pass":
			passed++
			fmt.Printf("✓ %-40s %10.2f ns/op", r.Name, r.NsPerOp)
			if r.MBPerSec > 0 {
				fmt.Printf(" %10.2f MB/s", r.MBPerSec)
			}
			fmt.Println()
		case "fail":
			failed++
			fmt.Printf("✗ %-40s FAILED: %s\n", r.Name, r.Error)
		case "timeout":
			timeout++
			fmt.Printf("⏱ %-40s TIMEOUT after %v\n", r.Name, r.Duration)
		}
	}

	fmt.Println(strings.Repeat("=", 62))
	fmt.Printf("Total: %d | Passed: %d | Failed: %d | Timeout: %d\n",
		len(results), passed, failed, timeout)

	return nil
}