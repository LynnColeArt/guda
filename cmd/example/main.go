package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("GUDA Examples")
		fmt.Println("=============")
		fmt.Println()
		fmt.Println("Usage: go run cmd/example/main.go <example>")
		fmt.Println()
		fmt.Println("Available examples:")
		fmt.Println("  vector    - Vector addition")
		fmt.Println("  matrix    - Matrix multiplication")
		fmt.Println("  fusion    - Kernel fusion demonstration")
		fmt.Println("  benchmark - Run all benchmarks")
		return
	}
	
	switch os.Args[1] {
	case "vector":
		fmt.Println("Run: go run examples/vector_add.go")
	case "matrix":
		fmt.Println("Run: go run examples/matrix_multiply.go")
	case "fusion":
		fmt.Println("Run: go run examples/fusion.go")
	case "benchmark":
		fmt.Println("Run: go test -bench=.")
	default:
		fmt.Printf("Unknown example: %s\n", os.Args[1])
	}
}