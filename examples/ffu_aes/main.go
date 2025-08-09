package main

import (
	"crypto/rand"
	"fmt"
	"log"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
	"github.com/LynnColeArt/guda/ffu/aesni"
	"github.com/LynnColeArt/guda/ffu/software"
)

func main() {
	fmt.Println("GUDA FFU AES Demonstration")
	fmt.Println("==========================")
	
	// Create FFU registry
	registry := ffu.NewRegistry()
	
	// Register available FFUs
	aesniFFU := aesni.NewAESNIFFU()
	softwareFFU := software.NewAESSoftwareFFU()
	
	if err := registry.Register(aesniFFU); err != nil {
		log.Fatal(err)
	}
	if err := registry.Register(softwareFFU); err != nil {
		log.Fatal(err)
	}
	
	// List available FFUs
	fmt.Println("\nAvailable FFUs:")
	for _, f := range registry.List() {
		fmt.Printf("- %s (Type: %s, Available: %v)\n", 
			f.Name(), f.Type(), f.IsAvailable())
	}
	
	// Prepare test data
	sizes := []int{1024, 64 * 1024, 1024 * 1024} // 1KB, 64KB, 1MB
	key := make([]byte, 32) // AES-256
	iv := make([]byte, 16)
	
	if _, err := rand.Read(key); err != nil {
		log.Fatal(err)
	}
	if _, err := rand.Read(iv); err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("\nPerformance Comparison:")
	fmt.Println("Size\t\tDirect AES-NI\tFFU Dispatch\tSpeedup")
	fmt.Println("----\t\t-------------\t------------\t-------")
	
	for _, size := range sizes {
		plaintext := make([]byte, size)
		ciphertext := make([]byte, size)
		
		if _, err := rand.Read(plaintext); err != nil {
			log.Fatal(err)
		}
		
		work := &ffu.AESWorkload{
			Operation: ffu.AESEncrypt,
			Mode:      ffu.AESModeCTR,
			Key:       key,
			IV:        iv,
			Input:     plaintext,
			Output:    ciphertext,
		}
		
		// Direct AES-NI call
		var directTime time.Duration
		if aesniFFU.IsAvailable() {
			start := time.Now()
			for i := 0; i < 100; i++ {
				if err := aesniFFU.Execute(work); err != nil {
					log.Fatal(err)
				}
			}
			directTime = time.Since(start) / 100
		}
		
		// FFU dispatch
		start := time.Now()
		for i := 0; i < 100; i++ {
			bestFFU, _ := registry.FindBest(work)
			if bestFFU == nil {
				log.Fatal("No suitable FFU found")
			}
			
			if err := bestFFU.Execute(work); err != nil {
				log.Fatal(err)
			}
		}
		dispatchTime := time.Since(start) / 100
		
		// Calculate throughput
		throughputDirect := float64(size) / directTime.Seconds() / 1e6  // MB/s
		throughputDispatch := float64(size) / dispatchTime.Seconds() / 1e6
		
		fmt.Printf("%s\t\t%.0f MB/s\t%.0f MB/s\t%.2fx\n",
			formatSize(size),
			throughputDirect,
			throughputDispatch,
			throughputDirect/throughputDispatch)
	}
	
	// Show metrics
	fmt.Println("\nFFU Metrics:")
	for _, f := range registry.List() {
		metrics := f.Metrics()
		if metrics.WorkloadCount > 0 {
			avgTime := metrics.TotalDuration / time.Duration(metrics.WorkloadCount)
			throughput := float64(metrics.BytesProcessed) / metrics.TotalDuration.Seconds() / 1e6
			
			fmt.Printf("\n%s:\n", f.Name())
			fmt.Printf("  Workloads: %d\n", metrics.WorkloadCount)
			fmt.Printf("  Total bytes: %s\n", formatBytes(metrics.BytesProcessed))
			fmt.Printf("  Avg time: %v\n", avgTime)
			fmt.Printf("  Throughput: %.0f MB/s\n", throughput)
			fmt.Printf("  Errors: %d\n", metrics.ErrorCount)
		}
	}
	
	// Demonstrate cost estimation
	fmt.Println("\nCost Estimation for 1GB workload:")
	largeWork := &ffu.AESWorkload{
		Operation: ffu.AESEncrypt,
		Mode:      ffu.AESModeCTR,
		Key:       key,
		IV:        iv,
		Input:     make([]byte, 1024*1024*1024), // 1GB
		Output:    make([]byte, 1024*1024*1024),
	}
	
	for _, f := range registry.List() {
		if f.CanHandle(largeWork) {
			cost := f.EstimateCost(largeWork)
			fmt.Printf("\n%s:\n", f.Name())
			fmt.Printf("  Estimated time: %v\n", cost.Duration)
			fmt.Printf("  Estimated energy: %.3f J\n", cost.Energy)
			fmt.Printf("  Confidence: %.0f%%\n", cost.Confidence*100)
		}
	}
}

func formatSize(size int) string {
	if size < 1024 {
		return fmt.Sprintf("%d B", size)
	} else if size < 1024*1024 {
		return fmt.Sprintf("%d KB", size/1024)
	} else {
		return fmt.Sprintf("%d MB", size/(1024*1024))
	}
}

func formatBytes(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	} else if bytes < 1024*1024 {
		return fmt.Sprintf("%.1f KB", float64(bytes)/1024)
	} else if bytes < 1024*1024*1024 {
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	} else {
		return fmt.Sprintf("%.1f GB", float64(bytes)/(1024*1024*1024))
	}
}