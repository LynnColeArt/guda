package aesni

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"runtime"
	"testing"
	"time"
	
	"github.com/LynnColeArt/guda/ffu"
	"github.com/LynnColeArt/guda/ffu/software"
)

func TestAESNIDetection(t *testing.T) {
	aesni := NewAESNIFFU()
	
	t.Logf("Platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	t.Logf("AES-NI available: %v", aesni.IsAvailable())
	
	if runtime.GOARCH == "amd64" && !aesni.IsAvailable() {
		t.Log("Warning: Running on AMD64 but AES-NI not detected")
	}
}

func TestAESNICorrectness(t *testing.T) {
	if !NewAESNIFFU().IsAvailable() {
		t.Skip("AES-NI not available")
	}
	
	testCases := []struct {
		name     string
		mode     ffu.AESMode
		keySize  int
		dataSize int
	}{
		{"AES-128-CBC-Small", ffu.AESModeCBC, 16, 64},
		{"AES-256-CBC-Large", ffu.AESModeCBC, 32, 4096},
		{"AES-128-CTR-Small", ffu.AESModeCTR, 16, 63}, // CTR doesn't need padding
		{"AES-256-CTR-Large", ffu.AESModeCTR, 32, 4097},
	}
	
	aesni := NewAESNIFFU()
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate random key and IV
			key := make([]byte, tc.keySize)
			iv := make([]byte, 16)
			if _, err := rand.Read(key); err != nil {
				t.Fatal(err)
			}
			if _, err := rand.Read(iv); err != nil {
				t.Fatal(err)
			}
			
			// Generate random plaintext
			plaintext := make([]byte, tc.dataSize)
			if _, err := rand.Read(plaintext); err != nil {
				t.Fatal(err)
			}
			
			// Pad for CBC mode
			if tc.mode == ffu.AESModeCBC {
				padding := 16 - (len(plaintext) % 16)
				if padding > 0 {
					plaintext = append(plaintext, make([]byte, padding)...)
				}
			}
			
			// Encrypt
			ciphertext := make([]byte, len(plaintext))
			encWork := &ffu.AESWorkload{
				Operation: ffu.AESEncrypt,
				Mode:      tc.mode,
				Key:       key,
				IV:        iv,
				Input:     plaintext,
				Output:    ciphertext,
			}
			
			if err := aesni.Execute(encWork); err != nil {
				t.Fatalf("Encryption failed: %v", err)
			}
			
			// Decrypt
			decrypted := make([]byte, len(ciphertext))
			decWork := &ffu.AESWorkload{
				Operation: ffu.AESDecrypt,
				Mode:      tc.mode,
				Key:       key,
				IV:        iv,
				Input:     ciphertext,
				Output:    decrypted,
			}
			
			if err := aesni.Execute(decWork); err != nil {
				t.Fatalf("Decryption failed: %v", err)
			}
			
			// Verify
			if !bytes.Equal(plaintext, decrypted) {
				t.Errorf("Decryption mismatch: plaintext and decrypted don't match")
			}
		})
	}
}

func BenchmarkAESNI(b *testing.B) {
	aesni := NewAESNIFFU()
	software := software.NewAESSoftwareFFU()
	
	if !aesni.IsAvailable() {
		b.Skip("AES-NI not available")
	}
	
	sizes := []int{
		16,        // 16 B (single block)
		1024,      // 1 KB
		64 * 1024, // 64 KB
		1024 * 1024, // 1 MB
		16 * 1024 * 1024, // 16 MB
	}
	
	modes := []struct {
		name string
		mode ffu.AESMode
	}{
		{"CBC", ffu.AESModeCBC},
		{"CTR", ffu.AESModeCTR},
	}
	
	for _, mode := range modes {
		for _, size := range sizes {
			// Generate test data
			key := make([]byte, 32) // AES-256
			iv := make([]byte, 16)
			rand.Read(key)
			rand.Read(iv)
			
			plaintext := make([]byte, size)
			rand.Read(plaintext)
			
			// Pad for CBC
			if mode.mode == ffu.AESModeCBC && size%16 != 0 {
				padding := 16 - (size % 16)
				plaintext = append(plaintext, make([]byte, padding)...)
			}
			
			ciphertext := make([]byte, len(plaintext))
			
			// Benchmark AES-NI
			b.Run(fmt.Sprintf("%s/%s/AESNI", mode.name, formatSize(size)), func(b *testing.B) {
				work := &ffu.AESWorkload{
					Operation: ffu.AESEncrypt,
					Mode:      mode.mode,
					Key:       key,
					IV:        iv,
					Input:     plaintext,
					Output:    ciphertext,
				}
				
				b.SetBytes(int64(size))
				b.ResetTimer()
				
				for i := 0; i < b.N; i++ {
					if err := aesni.Execute(work); err != nil {
						b.Fatal(err)
					}
				}
				
				reportMetrics(b, aesni, size)
			})
			
			// Benchmark Software (for comparison)
			// Note: This isn't a true software implementation since Go's
			// crypto/aes uses AES-NI when available. For accurate comparison,
			// we'd need a pure-Go AES implementation.
			b.Run(fmt.Sprintf("%s/%s/Software", mode.name, formatSize(size)), func(b *testing.B) {
				work := &ffu.AESWorkload{
					Operation: ffu.AESEncrypt,
					Mode:      mode.mode,
					Key:       key,
					IV:        iv,
					Input:     plaintext,
					Output:    ciphertext,
				}
				
				b.SetBytes(int64(size))
				b.ResetTimer()
				
				for i := 0; i < b.N; i++ {
					if err := software.Execute(work); err != nil {
						b.Fatal(err)
					}
				}
				
				reportMetrics(b, software, size)
			})
		}
	}
}

func BenchmarkFFUDispatch(b *testing.B) {
	// Test the overhead of FFU dispatch
	registry := ffu.NewRegistry()
	
	aesni := NewAESNIFFU()
	software := software.NewAESSoftwareFFU()
	
	registry.Register(aesni)
	registry.Register(software)
	
	// Test data
	size := 64 * 1024 // 64KB
	key := make([]byte, 32)
	iv := make([]byte, 16)
	plaintext := make([]byte, size)
	ciphertext := make([]byte, size)
	
	rand.Read(key)
	rand.Read(iv)
	rand.Read(plaintext)
	
	work := &ffu.AESWorkload{
		Operation: ffu.AESEncrypt,
		Mode:      ffu.AESModeCTR,
		Key:       key,
		IV:        iv,
		Input:     plaintext,
		Output:    ciphertext,
	}
	
	b.Run("DirectCall", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			if err := aesni.Execute(work); err != nil {
				b.Fatal(err)
			}
		}
	})
	
	b.Run("RegistryDispatch", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		
		for i := 0; i < b.N; i++ {
			ffu, _ := registry.FindBest(work)
			if ffu == nil {
				b.Fatal("No suitable FFU found")
			}
			
			if err := ffu.Execute(work); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func reportMetrics(b *testing.B, ffu ffu.FFU, size int) {
	metrics := ffu.Metrics()
	
	if metrics.WorkloadCount > 0 {
		avgDuration := metrics.TotalDuration / time.Duration(metrics.WorkloadCount)
		throughput := float64(size) / avgDuration.Seconds() / 1e6 // MB/s
		
		b.ReportMetric(throughput, "MB/s")
		
		// Calculate speedup if we have both implementations
		// This is approximate since we're running them separately
		b.ReportMetric(float64(avgDuration.Nanoseconds()), "ns/op")
	}
}

func formatSize(size int) string {
	if size < 1024 {
		return fmt.Sprintf("%dB", size)
	} else if size < 1024*1024 {
		return fmt.Sprintf("%dKB", size/1024)
	} else {
		return fmt.Sprintf("%dMB", size/(1024*1024))
	}
}