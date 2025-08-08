// Copyright ©2019 The Gonum Authors. All rights reserved.
// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package guda provides a CUDA-compatible API for CPU execution with high performance.
// 
// GUDA has fully assimilated the Gonum numerical computing libraries as its native
// compute engine, providing optimized BLAS operations, linear algebra, and scientific
// computing capabilities specifically tuned for machine learning workloads.
//
// The integrated compute engine includes:
//   - Optimized BLAS implementations with AVX2/FMA support
//   - Float32-first design for ML/AI applications  
//   - Fused operations for common neural network patterns
//   - SIMD-accelerated Float16/BFloat16 operations
//
// This assimilation preserves the full optimization history from Gonum while
// adapting it for modern machine learning requirements.
package guda