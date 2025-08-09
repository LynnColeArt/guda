package amx

import (
	"testing"
	"time"
	"unsafe"

	"github.com/LynnColeArt/guda/ffu"
)

func TestAMXFFU(t *testing.T) {
	amx := NewAMXFFU()
	
	// Test basic properties
	if amx.Name() != "Intel AMX" {
		t.Errorf("Expected name 'Intel AMX', got %s", amx.Name())
	}
	
	if amx.Type() != ffu.FFUTypeAMX {
		t.Errorf("Expected type AMX, got %v", amx.Type())
	}
}

func TestAMXCanHandle(t *testing.T) {
	amx := NewAMXFFU()
	
	// For testing, enable AMX support
	SetAMXSupport(true, true, true)
	amx.available = true
	amx.hasInt8 = true
	amx.hasBF16 = true
	
	tests := []struct {
		name     string
		workload ffu.Workload
		expected bool
	}{
		{
			name: "Valid INT8 workload",
			workload: &ffu.AMXWorkload{
				Operation: ffu.AMXMatMul,
				DataType:  ffu.AMXInt8,
				M:         64,
				N:         64,
				K:         64,
				A:         makeAlignedBuffer(64*64, 16),
				B:         makeAlignedBuffer(64*64, 16),
				C:         makeAlignedBuffer(64*64*4, 16), // INT32 output
			},
			expected: true,
		},
		{
			name: "Too small for AMX",
			workload: &ffu.AMXWorkload{
				Operation: ffu.AMXMatMul,
				DataType:  ffu.AMXInt8,
				M:         8,
				N:         8,
				K:         8,
				A:         makeAlignedBuffer(8*8, 16),
				B:         makeAlignedBuffer(8*8, 16),
				C:         makeAlignedBuffer(8*8*4, 16),
			},
			expected: false,
		},
		{
			name: "Unaligned data",
			workload: &ffu.AMXWorkload{
				Operation: ffu.AMXMatMul,
				DataType:  ffu.AMXInt8,
				M:         64,
				N:         64,
				K:         64,
				A:         make([]byte, 64*64+1)[1:], // Misaligned
				B:         makeAlignedBuffer(64*64, 16),
				C:         makeAlignedBuffer(64*64*4, 16),
			},
			expected: false,
		},
		{
			name:     "Non-AMX workload",
			workload: &mockWorkload{},
			expected: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := amx.CanHandle(tt.workload)
			if result != tt.expected {
				t.Errorf("CanHandle() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestAMXInt8Execution(t *testing.T) {
	amx := NewAMXFFU()
	
	// For testing, enable AMX support
	SetAMXSupport(true, true, true)
	amx.available = true
	amx.hasInt8 = true
	
	// Create a simple 2x2 matrix multiplication test
	// A = [[1, 2], [3, 4]]
	// B = [[5, 6], [7, 8]]
	// C = A * B = [[19, 22], [43, 50]]
	
	M, N, K := 2, 2, 2
	
	A := makeAlignedBuffer(M*K, 16)
	B := makeAlignedBuffer(K*N, 16)
	C := makeAlignedBuffer(M*N*4, 16) // INT32 output
	
	// Fill A matrix
	A[0] = 1
	A[1] = 2
	A[2] = 3
	A[3] = 4
	
	// Fill B matrix
	B[0] = 5
	B[1] = 6
	B[2] = 7
	B[3] = 8
	
	workload := &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		ScaleA:    1.0,
		ScaleB:    1.0,
		ScaleC:    1.0,
	}
	
	// AMX requires larger matrices
	if amx.CanHandle(workload) {
		t.Fatal("Small matrix should not be handled by AMX")
	}
	
	// Test with larger matrices that AMX would handle
	M, N, K = 64, 64, 64
	A = makeAlignedBuffer(M*K, 16)
	B = makeAlignedBuffer(K*N, 16)
	C = makeAlignedBuffer(M*N*4, 16)
	
	// Fill with simple pattern
	for i := 0; i < M*K; i++ {
		A[i] = byte(i % 127)
	}
	for i := 0; i < K*N; i++ {
		B[i] = byte(i % 127)
	}
	
	workload = &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         M,
		N:         N,
		K:         K,
		A:         A,
		B:         B,
		C:         C,
		ScaleA:    1.0,
		ScaleB:    1.0,
		ScaleC:    1.0,
	}
	
	if !amx.CanHandle(workload) {
		t.Fatal("Large matrix should be handled by AMX")
	}
	
	err := amx.Execute(workload)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	
	// Verify some results (spot check)
	cInt32 := (*[1 << 30]int32)(unsafe.Pointer(&C[0]))[:M*N]
	if cInt32[0] == 0 && cInt32[1] == 0 {
		t.Error("Result matrix appears to be all zeros")
	}
}

func TestAMXCostEstimate(t *testing.T) {
	amx := NewAMXFFU()
	
	// For testing, enable AMX support
	SetAMXSupport(true, true, true)
	amx.available = true
	amx.hasInt8 = true
	
	workload := &ffu.AMXWorkload{
		Operation: ffu.AMXMatMul,
		DataType:  ffu.AMXInt8,
		M:         1024,
		N:         1024,
		K:         1024,
		A:         makeAlignedBuffer(1024*1024, 16),
		B:         makeAlignedBuffer(1024*1024, 16),
		C:         makeAlignedBuffer(1024*1024*4, 16),
	}
	
	cost := amx.EstimateCost(workload)
	
	// Check that cost is reasonable
	if cost.Duration <= 0 {
		t.Errorf("Expected positive duration estimate, got %v", cost.Duration)
	}
	
	if cost.MemoryBandwidth <= 0 {
		t.Errorf("Expected positive memory bandwidth, got %v", cost.MemoryBandwidth)
	}
	
	// For 1024x1024x1024 INT8 GEMM at 2 TOPS
	// Operations = 2*1024*1024*1024 = 2,147,483,648
	// Time = ops / 2e12 ≈ 0.001 seconds
	expectedSeconds := 2.147483648e9 / 2e12
	expectedDuration := time.Duration(expectedSeconds * float64(time.Second))
	if cost.Duration > expectedDuration*2 || cost.Duration < expectedDuration/2 {
		t.Errorf("Duration estimate %v is not close to expected %v", cost.Duration, expectedDuration)
	}
}

// Helper function to create aligned buffer
func makeAlignedBuffer(size int, align int) []byte {
	// Allocate extra space for alignment
	buf := make([]byte, size+align)
	
	// Find aligned offset
	ptr := uintptr(unsafe.Pointer(&buf[0]))
	offset := (align - int(ptr%uintptr(align))) % align
	
	return buf[offset : offset+size]
}

// Mock workload for testing
type mockWorkload struct{}

func (m *mockWorkload) Type() string     { return "mock" }
func (m *mockWorkload) Size() int64      { return 0 }
func (m *mockWorkload) Validate() error  { return nil }