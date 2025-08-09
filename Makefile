# GUDA Makefile

.PHONY: all test benchmark baseline compare clean

# Default target
all: test

# Run all tests
test:
	go test -v ./...

# Run tests with race detector
test-race:
	go test -race -v ./...

# Run benchmarks
benchmark:
	@mkdir -p benchmark_logs
	go test -bench=. -benchmem -json ./... | tee benchmark_logs/benchmark_$(shell date +%Y%m%d_%H%M%S).json || true
	@echo "\nBenchmark results saved to benchmark_logs/"

# Run benchmarks with cold cache (requires sudo)
bench-cold:
	@if [ "$$(id -u)" -ne 0 ]; then \
		echo "Cold cache benchmarks require sudo privileges."; \
		echo "Run: sudo make bench-cold"; \
		echo ""; \
		echo "Or use the script directly:"; \
		echo "  sudo ./scripts/bench_cold.sh"; \
		exit 1; \
	fi
	@./scripts/bench_cold.sh

# Compare hot vs cold cache results
bench-compare:
	@if [ ! -f "$(HOT)" ] || [ ! -f "$(COLD)" ]; then \
		echo "Usage: make bench-compare HOT=hot_results.txt COLD=cold_results.txt"; \
		echo ""; \
		echo "Example:"; \
		echo "  make bench-compare HOT=gemm_hot_results.txt COLD=benchmark_results/gemm_cold_*.txt"; \
		exit 1; \
	fi
	@python3 scripts/compare_benchmarks.py "$(HOT)" "$(COLD)"

# Capture baseline before optimization
baseline:
	@echo "Capturing baseline performance and numerical results..."
	@mkdir -p testdata
	go run cmd/baseline/main.go -output testdata/baseline_$(shell date +%Y%m%d_%H%M%S).json
	@echo "Baseline saved. Remember to commit this file!"

# Compare current performance against baseline
compare: 
	@echo "Comparing current implementation against baseline..."
	@if [ -z "$(BASELINE)" ]; then \
		echo "Usage: make compare BASELINE=testdata/baseline_YYYYMMDD_HHMMSS.json"; \
		exit 1; \
	fi
	go run cmd/compare/main.go -baseline $(BASELINE) -current testdata/current.json

# Run AVX2 benchmarks
benchmark-avx2:
	go test -bench=BenchmarkAxpyComparison -benchmem ./compute/asm/f32
	go test -bench=BenchmarkDotComparison -benchmem ./compute/asm/f32

# Profile CPU usage
profile:
	go test -cpuprofile=cpu.prof -bench=. ./...
	go tool pprof -http=:8080 cpu.prof

# Memory profile
memprofile:
	go test -memprofile=mem.prof -bench=. ./...
	go tool pprof -http=:8080 mem.prof

# Clean build artifacts
clean:
	go clean -cache
	rm -f *.prof
	rm -f testdata/current.json

# Build all binaries
build:
	go build ./...

# Install git hooks for pre-commit testing
install-hooks:
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "make test" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git pre-commit hook installed"

# Check for numerical stability issues
stability-check:
	go test -tags=debug -run=TestNumericalStability ./...

# Format code
fmt:
	go fmt ./...
	goimports -w .

# Lint code
lint:
	golangci-lint run

# Generate assembly listings for inspection
asm-dump:
	go build -gcflags="-S" ./compute/asm/f32 2> f32_asm.s
	@echo "Assembly dumped to f32_asm.s"

# Quick sanity check before commits
pre-commit: fmt test benchmark-avx2
	@echo "✅ All pre-commit checks passed!"

# View benchmark results
bench-summary:
	@if [ -d benchmark_logs ] && [ "$$(ls -A benchmark_logs)" ]; then \
		go run ./cmd/bench-parser/main.go -file $$(ls -t benchmark_logs/*.json | head -1); \
	else \
		echo "No benchmark logs found. Run 'make benchmark' first."; \
	fi

# Help target
help:
	@echo "GUDA Makefile targets:"
	@echo "  make test          - Run all tests"
	@echo "  make benchmark     - Run all benchmarks (with logging)"
	@echo "  make bench-cold    - Run benchmarks with cold cache (requires sudo)"
	@echo "  make bench-compare - Compare hot vs cold cache results"
	@echo "  make bench-summary - View latest benchmark results"
	@echo "  make baseline      - Capture performance baseline"
	@echo "  make compare       - Compare against baseline"
	@echo "  make profile       - CPU profiling"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make pre-commit    - Run checks before committing"