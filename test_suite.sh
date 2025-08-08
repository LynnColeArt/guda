#!/bin/bash

echo "GUDA Comprehensive Test Suite"
echo "============================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a test
run_test() {
    echo -n "Running $1... "
    if $2 > /tmp/guda_test.log 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Error output:"
        cat /tmp/guda_test.log
    fi
}

# 1. Unit Tests
echo "1. Unit Tests"
echo "-------------"
run_test "Memory Tests" "go test -v -run TestMemory"
run_test "Kernel Tests" "go test -v -run TestKernel"
run_test "Vector Tests" "go test -v -run TestVector"
run_test "Fusion Tests" "go test -v -run TestFused"
run_test "All Tests" "go test -v"
echo

# 2. Benchmarks
echo "2. Performance Benchmarks"
echo "------------------------"
echo "Running benchmarks (this may take a while)..."
go test -bench=. -benchmem -benchtime=10s | grep -E "Benchmark|ns/op|MB/s|GFLOPS"
echo

# 3. Examples
echo "3. Example Programs"
echo "------------------"
for example in vector_add matrix_multiply fusion simd_test; do
    if [ -d "examples/$example" ]; then
        echo "Running $example example:"
        (cd examples/$example && go run main.go | head -20)
        echo
    fi
done

# 4. Memory Leak Test
echo "4. Memory Leak Test"
echo "------------------"
echo "Testing for memory leaks..."
go test -run TestMemoryPoolStats -count=100 > /tmp/guda_mem.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}No memory leaks detected${NC}"
else
    echo -e "${RED}Potential memory leak${NC}"
fi
echo

# 5. Race Condition Test
echo "5. Race Condition Test"
echo "---------------------"
echo "Testing for race conditions..."
go test -race -run TestKernelLaunch -count=10 > /tmp/guda_race.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}No race conditions detected${NC}"
else
    echo -e "${RED}Race condition detected${NC}"
    cat /tmp/guda_race.log
fi
echo

# 6. Stress Test
echo "6. Stress Test"
echo "--------------"
echo "Running stress test with large data..."
go test -run TestVectorOperations -timeout=60s > /tmp/guda_stress.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Stress test passed${NC}"
else
    echo -e "${RED}Stress test failed${NC}"
fi

echo
echo "Test suite complete!"