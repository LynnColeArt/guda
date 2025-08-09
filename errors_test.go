package guda

import (
	"errors"
	"testing"
)

func TestStructuredErrors(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantType ErrorType
		wantOp   string
		wantMsg  string
		checkFn  func(error) bool
	}{
		{
			name:     "Memory Error",
			err:      ErrOutOfMemory,
			wantType: ErrTypeMemory,
			wantOp:   "Malloc",
			wantMsg:  "out of memory",
			checkFn:  IsMemoryError,
		},
		{
			name:     "Invalid Arg Error",
			err:      ErrInvalidSize,
			wantType: ErrTypeInvalidArg,
			wantOp:   "Malloc",
			wantMsg:  "size must be positive",
			checkFn:  IsInvalidArgError,
		},
		{
			name:     "Invalid Device Error",
			err:      ErrInvalidDevice,
			wantType: ErrTypeInvalidArg,
			wantOp:   "SetDevice",
			wantMsg:  "invalid device ID",
			checkFn:  IsInvalidArgError,
		},
		{
			name:     "No Device Error",
			err:      ErrNoDevice,
			wantType: ErrTypeDevice,
			wantOp:   "Device",
			wantMsg:  "no compute device available",
			checkFn:  IsDeviceError,
		},
		{
			name:     "Execution Error",
			err:      ErrKernelFailed,
			wantType: ErrTypeExecution,
			wantOp:   "Kernel",
			wantMsg:  "kernel execution failed",
			checkFn:  IsExecutionError,
		},
		{
			name:     "Numerical Error",
			err:      ErrNaN,
			wantType: ErrTypeNumerical,
			wantOp:   "Compute",
			wantMsg:  "NaN detected in computation",
			checkFn:  IsNumericalError,
		},
		{
			name:     "Not Implemented Error",
			err:      ErrNotSupported,
			wantType: ErrTypeNotImplemented,
			wantOp:   "FusedOp",
			wantMsg:  "operation not supported",
			checkFn:  IsNotImplementedError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Check if it's a GUDAError
			gudaErr, ok := tt.err.(*GUDAError)
			if !ok {
				t.Fatalf("Expected GUDAError, got %T", tt.err)
			}

			// Check type
			if gudaErr.Type != tt.wantType {
				t.Errorf("Type = %v, want %v", gudaErr.Type, tt.wantType)
			}

			// Check operation
			if gudaErr.Op != tt.wantOp {
				t.Errorf("Op = %v, want %v", gudaErr.Op, tt.wantOp)
			}

			// Check message
			if gudaErr.Message != tt.wantMsg {
				t.Errorf("Message = %v, want %v", gudaErr.Message, tt.wantMsg)
			}

			// Check type-specific function
			if !tt.checkFn(tt.err) {
				t.Errorf("Type check function returned false")
			}

			// Check error string contains expected parts
			errStr := tt.err.Error()
			if errStr == "" {
				t.Error("Error string is empty")
			}
		})
	}
}

func TestErrorUnwrap(t *testing.T) {
	baseErr := errors.New("base error")
	wrappedErr := NewMemoryError("Test", "wrapped error", baseErr)

	// Test Unwrap
	gudaErr, ok := wrappedErr.(*GUDAError)
	if !ok {
		t.Fatal("Expected GUDAError")
	}

	unwrapped := gudaErr.Unwrap()
	if unwrapped != baseErr {
		t.Errorf("Unwrap() = %v, want %v", unwrapped, baseErr)
	}

	// Test errors.Is
	if !errors.Is(wrappedErr, baseErr) {
		t.Error("errors.Is() should return true for wrapped error")
	}
}

func TestErrorTypeString(t *testing.T) {
	tests := []struct {
		errType ErrorType
		want    string
	}{
		{ErrTypeMemory, "Memory"},
		{ErrTypeInvalidArg, "InvalidArgument"},
		{ErrTypeExecution, "Execution"},
		{ErrTypeNumerical, "Numerical"},
		{ErrTypeDevice, "Device"},
		{ErrTypeNotImplemented, "NotImplemented"},
		{ErrorType(999), "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := tt.errType.String()
			if got != tt.want {
				t.Errorf("String() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvolutionErrors(t *testing.T) {
	// Test invalid parameters
	params := &ConvParams{
		BatchSize: -1, // Invalid
	}

	err := params.Validate()
	if !IsInvalidArgError(err) {
		t.Errorf("Expected invalid argument error, got %v", err)
	}

	// Check error message
	if err != nil {
		gudaErr, ok := err.(*GUDAError)
		if !ok {
			t.Fatal("Expected GUDAError")
		}
		if gudaErr.Op != "Conv2D" {
			t.Errorf("Expected Op = Conv2D, got %v", gudaErr.Op)
		}
	}
}