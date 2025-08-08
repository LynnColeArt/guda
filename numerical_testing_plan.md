# GUDA Numerical Testing Plan

## Goal
Achieve numerical parity with CUDA to ensure models produce identical results across platforms.

## Key Principles

1. **Bit-exact is impossible** - Float32 operations have different rounding across hardware
2. **Target: ULP (Units in Last Place) differences** - Aim for ≤ 1-2 ULP difference
3. **Accumulation patterns matter** - Order of operations affects results
4. **Test at scale** - Small errors compound in deep networks

## Testing Strategy

### 1. Operation-Level Testing

#### Basic Operations
- [ ] Element-wise ops (add, mul, etc.) - Should be bit-exact
- [ ] Reductions (sum, max, etc.) - Order-dependent, test error bounds
- [ ] Transcendentals (exp, tanh, etc.) - May use different approximations

#### BLAS Operations  
- [ ] GEMM - Most critical, different blocking = different rounding
- [ ] GEMV, GER - Less sensitive but still important
- [ ] Batched operations - Ensure consistency across batch

#### Neural Network Operations
- [ ] Convolution - im2col vs direct affects numerics
- [ ] Pooling - Edge handling differences
- [ ] Normalization - Numerical stability critical
- [ ] Activations - Approximation differences (e.g., GELU)

### 2. Test Methodology

```python
# Pseudo-code for numerical comparison
def compare_with_cuda(op_name, inputs, cuda_result, guda_result):
    # Absolute error for small values
    abs_diff = abs(cuda_result - guda_result)
    
    # Relative error for large values  
    rel_diff = abs_diff / max(abs(cuda_result), 1e-10)
    
    # ULP difference for detailed analysis
    ulp_diff = compute_ulp_difference(cuda_result, guda_result)
    
    # Different tolerances for different operations
    if op_name in ['add', 'mul']:
        assert ulp_diff <= 1  # Should be exact
    elif op_name in ['gemm', 'conv']:
        assert rel_diff < 1e-6  # ~1-2 ULP for float32
    elif op_name in ['exp', 'gelu']:
        assert rel_diff < 1e-4  # Approximations differ
```

### 3. Test Data Generation

1. **Edge cases**
   - Zeros, ones, negative values
   - Denormal numbers
   - Values near overflow/underflow
   - Powers of 2 (exact representation)

2. **Realistic data**
   - Normal distributions (like neural network activations)
   - Uniform distributions
   - Actual model weights/activations

3. **Stress patterns**
   - Large matrices (accumulation errors)
   - Repeated operations (error propagation)
   - Mixed scales (numerical stability)

### 4. Specific Concerns for GUDA

#### GEMM Numerical Behavior
- CUDA uses different algorithms based on size
- Our blocking pattern may accumulate differently
- Solution: Test various sizes, document max deviation

#### Reduction Operations
- Our AVX2 horizontal adds have different order than CUDA
- Solution: Document expected error bounds

#### Fast Math Approximations
- Our GELU uses tanh approximation
- Our exp uses polynomial approximation
- Solution: Provide accurate variants for testing

### 5. Test Suite Structure

```go
// numerical_parity_test.go
type NumericalTest struct {
    Name        string
    CUDAResult  []float32
    GUDAResult  []float32
    AbsTol      float32
    RelTol      float32
    ULPTol      int
}

func TestNumericalParity(t *testing.T) {
    tests := []NumericalTest{
        // Load from golden files or compute on-demand
    }
    
    for _, test := range tests {
        CheckNumericalParity(t, test)
    }
}
```

### 6. Continuous Validation

1. **Golden file tests** - Store CUDA outputs for regression testing
2. **Property tests** - Mathematical properties that must hold
3. **Differential testing** - Compare multiple implementations
4. **End-to-end model tests** - Run small models, compare outputs

### 7. Known Differences to Document

1. **Reduction order** - Our reductions are left-to-right
2. **GEMM blocking** - Different than cuBLAS
3. **Fast math flags** - Document which approximations we use
4. **Denormal handling** - May differ from GPU

## Implementation Priority

1. **First**: GEMM numerical tests (most critical)
2. **Second**: Reduction operations (affect batch norm, etc.)
3. **Third**: Activation functions (especially GELU, softmax)
4. **Fourth**: Convolution (various algorithms)

## Success Criteria

- [ ] Core operations within 2 ULP of CUDA
- [ ] Neural network forward pass < 1e-5 relative error
- [ ] Documentation of all known differences
- [ ] Ability to run PyTorch models with < 1e-4 output difference