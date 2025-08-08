// Package guda structured error types for better error handling
package guda

import (
	"fmt"
)

// ErrorType represents categories of errors
type ErrorType int

const (
	// Memory errors
	ErrTypeMemory ErrorType = iota
	// Invalid argument errors
	ErrTypeInvalidArg
	// Execution errors
	ErrTypeExecution
	// Numerical errors
	ErrTypeNumerical
	// Device errors
	ErrTypeDevice
	// Not implemented errors
	ErrTypeNotImplemented
)

// GUDAError represents a structured error with context
type GUDAError struct {
	Type    ErrorType
	Op      string      // Operation that failed
	Message string      // Human-readable message
	Err     error       // Underlying error if any
	Context interface{} // Additional context
}

// Error implements the error interface
func (e *GUDAError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("GUDA %s error in %s: %s (caused by: %v)", 
			e.Type.String(), e.Op, e.Message, e.Err)
	}
	return fmt.Sprintf("GUDA %s error in %s: %s", 
		e.Type.String(), e.Op, e.Message)
}

// Unwrap allows error chain inspection
func (e *GUDAError) Unwrap() error {
	return e.Err
}

// String returns the error type as a string
func (t ErrorType) String() string {
	switch t {
	case ErrTypeMemory:
		return "Memory"
	case ErrTypeInvalidArg:
		return "InvalidArgument"
	case ErrTypeExecution:
		return "Execution"
	case ErrTypeNumerical:
		return "Numerical"
	case ErrTypeDevice:
		return "Device"
	case ErrTypeNotImplemented:
		return "NotImplemented"
	default:
		return "Unknown"
	}
}

// Common error constructors

// NewMemoryError creates a memory-related error
func NewMemoryError(op string, message string, err error) error {
	return &GUDAError{
		Type:    ErrTypeMemory,
		Op:      op,
		Message: message,
		Err:     err,
	}
}

// NewInvalidArgError creates an invalid argument error
func NewInvalidArgError(op string, message string) error {
	return &GUDAError{
		Type:    ErrTypeInvalidArg,
		Op:      op,
		Message: message,
	}
}

// NewExecutionError creates an execution error
func NewExecutionError(op string, message string, err error) error {
	return &GUDAError{
		Type:    ErrTypeExecution,
		Op:      op,
		Message: message,
		Err:     err,
	}
}

// NewNumericalError creates a numerical error
func NewNumericalError(op string, message string, context interface{}) error {
	return &GUDAError{
		Type:    ErrTypeNumerical,
		Op:      op,
		Message: message,
		Context: context,
	}
}

// Common pre-defined errors

var (
	// ErrOutOfMemory indicates memory allocation failure
	ErrOutOfMemory = NewMemoryError("Malloc", "out of memory", nil)
	
	// ErrInvalidSize indicates invalid size parameter
	ErrInvalidSize = NewInvalidArgError("Malloc", "size must be positive")
	
	// ErrNullPointer indicates null pointer access
	ErrNullPointer = NewInvalidArgError("Memory", "null pointer")
	
	// ErrDoubleFree indicates double free attempt
	ErrDoubleFree = NewMemoryError("Free", "double free detected", nil)
	
	// ErrInvalidDevice indicates invalid device ID
	ErrInvalidDevice = NewInvalidArgError("SetDevice", "invalid device ID")
)

// IsMemoryError checks if an error is a memory error
func IsMemoryError(err error) bool {
	if e, ok := err.(*GUDAError); ok {
		return e.Type == ErrTypeMemory
	}
	return false
}

// IsInvalidArgError checks if an error is an invalid argument error
func IsInvalidArgError(err error) bool {
	if e, ok := err.(*GUDAError); ok {
		return e.Type == ErrTypeInvalidArg
	}
	return false
}