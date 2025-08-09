# Issue: `sliceView32` creates unsafe contiguous views of non-contiguous submatrices

## Summary

The `sliceView32` function in `compute/sgemm.go` assumes that submatrices are stored contiguously in memory. This is incorrect when the leading dimension is greater than the number of columns being viewed, causing the function to create slices that span row boundaries and include padding/garbage data.

## Current Impact

- **Not currently causing failures** because problematic matrix sizes (e.g., 37x29) use the serial path (`parBlocks < minParBlock`)
- **Will cause correctness issues** when larger matrices trigger the parallel path with partial blocks

## When This Bug Triggers

The bug occurs when `sliceView32(data, ld, i, j, r, c)` creates a contiguous slice for multiple rows of a submatrix where rows aren't actually contiguous in memory. Specifically:

### 1. Right/Bottom Edge Tiles (Tail Blocks)
- When `i` or `j` create a partial block at matrix edges
- When `r > 1` (multiple rows) AND `c < ld` (partial width)
- The slice spans past row ends into padding/next row data

### 2. Matrices with Padding (`ld > n`)
- Common in blocked kernels where rows are padded for alignment
- Any submatrix with `rows > 1` and `cols < ld` is non-contiguous
- `sliceView32` treats it as contiguous, causing misindexing

### 3. Submatrix with Non-Zero Offset
- When tiling (`i > 0` or `j > 0`), the range includes inter-row slack
- Linear iteration through the slice walks into garbage data

### 4. Transpose Paths
- With `aTrans`/`bTrans`, logical tile dimensions flip but `ld` still refers to physical row stride
- Incorrect handling causes similar contiguity violations

## Reproduction

To reproduce, use any matrix where:
- `n = 29`, `ld = 64` (padded for alignment)
- Tile at `(i, j) = (any > 0, any > 0)`
- `r >= 2`, `c <= 29` (multi-row, partial width)

The corruption occurs when transitioning from column `c-1` of row `t` to column `0` of row `t+1`.

## Why Column 28 Errors Appeared

For `n=29` matrices, the rightmost partial tile has `c < ld`. Column 28 is exactly where linearized traversal crosses row boundaries and hits padding. This explains why all errors in our tests clustered in the last column.

## Proposed Solutions

### Option 1: Offset + Stride (Recommended)
- Don't create contiguous submatrix slices
- Pass `base[offset:]` plus true `ld` to kernels
- Kernels index with `(row*ld + col)`
- API: `kernel(..., a []float32, lda int, b []float32, ldb int, c []float32, ldc int, ...)`

### Option 2: Tile Packing (BLAS-style)
- Copy each tile to contiguous scratch buffer with width=`c`, no padding
- Compute on packed data, write back
- Extra copy cost but simplifies inner loops and improves cache/vector efficiency

### Option 3: Row Iterator Wrapper
- Expose `rowView(t int) []float32` returning each row separately
- Iterate rows outside microkernel
- Never construct flat `r*c` view across rows

## Immediate Mitigations

1. **Add assertion**: Require `r == 1 || c == ld` for contiguous views
2. **Bounds check**: Verify `offset + (r-1)*ld + (c-1) < len(data)`
3. **Unit tests**: Property-based testing with random `(m,n,ld,i,j,r,c)`
4. **Parallel/serial parity**: Force both paths and compare outputs

## Code Location

The problematic function is at:
```go
// compute/sgemm.go:336
func sliceView32(a []float32, lda, i, j, r, c int) []float32 {
	return a[i*lda+j : (i+r-1)*lda+j+c]
}
```

Used in `sgemmParallel` around line 231-260.

## Priority

- **Current**: Low (serial path handles current test sizes correctly)
- **Future**: High (required for correctness when parallel path activates)
- **Recommendation**: Fix before enabling larger matrix sizes or reducing `minParBlock`

## References

This issue was discovered during debugging of test failures in PR #2. While the test failures turned out to have a different cause (Gonum reference test issue), the analysis revealed this latent correctness bug.