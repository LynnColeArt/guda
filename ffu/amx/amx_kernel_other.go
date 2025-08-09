//go:build !amd64
// +build !amd64

package amx

import "errors"

// Stub implementations for non-AMD64 architectures

func amxCheckSupport() bool {
	return false
}

type AMXKernel struct{}

func NewAMXKernel() *AMXKernel {
	return &AMXKernel{}
}

func (k *AMXKernel) Configure() error {
	return errors.New("AMX not supported on this architecture")
}

func (k *AMXKernel) Release() {}

func (k *AMXKernel) Int8GEMM(M, N, K int, A, B []byte, C []int32, alpha, beta float32) error {
	return errors.New("AMX not supported on this architecture")
}