#!/bin/bash
# Cold cache benchmark script for GUDA
# This script runs benchmarks with cache flushing between tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}This script requires sudo privileges for cache flushing${NC}"
    echo "Please run: sudo $0"
    exit 1
fi

# Configuration
BENCHTIME="${BENCHTIME:-5s}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}GUDA Cold Cache Benchmark Suite${NC}"
echo "================================="
echo "Timestamp: $(date)"
echo "Benchmark time: $BENCHTIME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to flush caches
flush_caches() {
    echo -e "${YELLOW}Flushing caches...${NC}"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    sleep 1
}

# Function to run benchmark with cold cache
run_cold_benchmark() {
    local bench_name=$1
    local output_file=$2
    
    echo -e "${GREEN}Running $bench_name (cold cache)...${NC}"
    
    # Flush caches
    flush_caches
    
    # Run benchmark
    go test -bench="$bench_name" -benchtime="$BENCHTIME" -run=^$ 2>&1 | tee "$output_file"
    
    echo ""
}

# Run all benchmarks
echo -e "${GREEN}Starting cold cache benchmarks...${NC}"
echo ""

# AXPY benchmark
run_cold_benchmark "BenchmarkAXPY" "$OUTPUT_DIR/axpy_cold_${TIMESTAMP}.txt"

# DOT benchmark  
run_cold_benchmark "BenchmarkDOT" "$OUTPUT_DIR/dot_cold_${TIMESTAMP}.txt"

# GEMM benchmark
run_cold_benchmark "BenchmarkGEMM" "$OUTPUT_DIR/gemm_cold_${TIMESTAMP}.txt"

# Memory bandwidth benchmark
run_cold_benchmark "BenchmarkMemoryBandwidth" "$OUTPUT_DIR/memory_cold_${TIMESTAMP}.txt"

# Fusion benchmark
run_cold_benchmark "BenchmarkFusionSpeedup" "$OUTPUT_DIR/fusion_cold_${TIMESTAMP}.txt"

# Combine all results
echo -e "${GREEN}Combining results...${NC}"
cat "$OUTPUT_DIR"/*_cold_${TIMESTAMP}.txt > "$OUTPUT_DIR/all_cold_${TIMESTAMP}.txt"

# Summary
echo -e "${GREEN}Cold cache benchmark complete!${NC}"
echo "Results saved to:"
echo "  Individual: $OUTPUT_DIR/*_cold_${TIMESTAMP}.txt"
echo "  Combined: $OUTPUT_DIR/all_cold_${TIMESTAMP}.txt"
echo ""

# Compare with hot cache if available
if [ -f "axpy_hot_results.txt" ] || [ -f "gemm_hot_results.txt" ]; then
    echo -e "${YELLOW}Hot cache results found. To compare:${NC}"
    echo "  Hot cache AXPY: axpy_hot_results.txt"
    echo "  Hot cache GEMM: gemm_hot_results.txt"
    echo "  Cold cache: $OUTPUT_DIR/all_cold_${TIMESTAMP}.txt"
fi