#!/usr/bin/env python3
"""
Compare hot vs cold cache benchmark results for GUDA
"""

import re
import sys
import argparse
from collections import defaultdict

def parse_benchmark_file(filename):
    """Parse benchmark output file and extract metrics"""
    results = defaultdict(dict)
    
    with open(filename, 'r') as f:
        for line in f:
            # Match benchmark lines
            # Example: BenchmarkAXPY/N_1024-16         	85623910	        63.29 ns/op	194159.74 MB/s	         0.1667 FLOPS/byte	        32.36 GFLOPS(hot-cache)
            match = re.search(r'Benchmark(\w+)/(\S+)-\d+\s+\d+\s+([\d.]+)\s+ns/op\s+([\d.]+)\s+MB/s.*?([\d.]+)\s+GFLOPS', line)
            if match:
                bench_type = match.group(1)
                bench_size = match.group(2)
                ns_per_op = float(match.group(3))
                bandwidth = float(match.group(4))
                gflops = float(match.group(5))
                
                key = f"{bench_type}/{bench_size}"
                results[key] = {
                    'ns_per_op': ns_per_op,
                    'bandwidth_gb_s': bandwidth / 1000.0,  # Convert MB/s to GB/s
                    'gflops': gflops
                }
    
    return results

def compare_results(hot_results, cold_results):
    """Compare hot and cold cache results"""
    print(f"{'Benchmark':<30} {'Metric':<15} {'Hot Cache':<15} {'Cold Cache':<15} {'Difference':<15} {'Ratio':<10}")
    print("-" * 110)
    
    all_keys = sorted(set(hot_results.keys()) | set(cold_results.keys()))
    
    for key in all_keys:
        if key in hot_results and key in cold_results:
            hot = hot_results[key]
            cold = cold_results[key]
            
            # GFLOPS comparison
            hot_gflops = hot['gflops']
            cold_gflops = cold['gflops']
            diff_gflops = hot_gflops - cold_gflops
            ratio_gflops = hot_gflops / cold_gflops if cold_gflops > 0 else 0
            
            print(f"{key:<30} {'GFLOPS':<15} {hot_gflops:<15.2f} {cold_gflops:<15.2f} {diff_gflops:<15.2f} {ratio_gflops:<10.2f}x")
            
            # Bandwidth comparison
            hot_bw = hot['bandwidth_gb_s']
            cold_bw = cold['bandwidth_gb_s']
            diff_bw = hot_bw - cold_bw
            ratio_bw = hot_bw / cold_bw if cold_bw > 0 else 0
            
            print(f"{'':<30} {'Bandwidth GB/s':<15} {hot_bw:<15.1f} {cold_bw:<15.1f} {diff_bw:<15.1f} {ratio_bw:<10.2f}x")
            
            # Time comparison (lower is better)
            hot_time = hot['ns_per_op']
            cold_time = cold['ns_per_op']
            diff_time = cold_time - hot_time  # Reversed because lower is better
            ratio_time = cold_time / hot_time if hot_time > 0 else 0
            
            print(f"{'':<30} {'Time (ns)':<15} {hot_time:<15.1f} {cold_time:<15.1f} {diff_time:<15.1f} {ratio_time:<10.2f}x")
            print()

def analyze_cache_impact(hot_results, cold_results):
    """Analyze cache impact by operation type"""
    print("\nCache Impact Analysis:")
    print("=" * 80)
    
    # Group by operation type
    operation_impact = defaultdict(list)
    
    for key in hot_results:
        if key in cold_results:
            parts = key.split('/')
            op_type = parts[0].replace('Benchmark', '')
            
            hot_gflops = hot_results[key]['gflops']
            cold_gflops = cold_results[key]['gflops']
            
            if hot_gflops > 0 and cold_gflops > 0:
                impact = (hot_gflops - cold_gflops) / hot_gflops * 100
                operation_impact[op_type].append(impact)
    
    # Calculate average impact by operation
    for op_type, impacts in operation_impact.items():
        avg_impact = sum(impacts) / len(impacts)
        max_impact = max(impacts)
        min_impact = min(impacts)
        
        print(f"\n{op_type}:")
        print(f"  Average performance drop: {avg_impact:.1f}%")
        print(f"  Range: {min_impact:.1f}% to {max_impact:.1f}%")
        
        # Categorize impact
        if avg_impact < 10:
            print("  → Memory bandwidth limited (minimal cache impact)")
        elif avg_impact < 30:
            print("  → Moderate cache sensitivity")
        else:
            print("  → Highly cache dependent")

def main():
    parser = argparse.ArgumentParser(description='Compare GUDA hot vs cold cache benchmarks')
    parser.add_argument('hot_file', help='Hot cache benchmark results file')
    parser.add_argument('cold_file', help='Cold cache benchmark results file')
    
    args = parser.parse_args()
    
    print("GUDA Benchmark Comparison: Hot vs Cold Cache")
    print("=" * 110)
    print(f"Hot cache file:  {args.hot_file}")
    print(f"Cold cache file: {args.cold_file}")
    print()
    
    # Parse results
    hot_results = parse_benchmark_file(args.hot_file)
    cold_results = parse_benchmark_file(args.cold_file)
    
    if not hot_results:
        print(f"Error: No results found in {args.hot_file}")
        sys.exit(1)
    
    if not cold_results:
        print(f"Error: No results found in {args.cold_file}")
        sys.exit(1)
    
    # Compare results
    compare_results(hot_results, cold_results)
    
    # Analyze cache impact
    analyze_cache_impact(hot_results, cold_results)

if __name__ == "__main__":
    main()