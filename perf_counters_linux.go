//go:build linux
// +build linux

// Package guda provides Linux-specific performance counter implementation
package guda

import (
	"fmt"
	"syscall"
	"unsafe"
)

// perf_event_attr structure for perf_event_open syscall
type perfEventAttr struct {
	Type               uint32
	Size               uint32
	Config             uint64
	SamplePeriod       uint64
	SampleType         uint64
	ReadFormat         uint64
	Flags              uint64
	WakeupEvents       uint32
	BpType             uint32
	ConfigOne          uint64
	ConfigTwo          uint64
	BranchSampleType   uint64
	SampleRegsUser     uint64
	SampleStackUser    uint32
	ClockID            int32
	SampleRegsIntr     uint64
	AuxWatermark       uint32
	SampleMaxStack     uint16
	_                  uint16
}

// Performance counter types
const (
	PERF_TYPE_HARDWARE   = 0
	PERF_TYPE_SOFTWARE   = 1
	PERF_TYPE_TRACEPOINT = 2
	PERF_TYPE_HW_CACHE   = 3
	PERF_TYPE_RAW        = 4
)

// Hardware performance counter events
const (
	PERF_COUNT_HW_CPU_CYCLES              = 0
	PERF_COUNT_HW_INSTRUCTIONS            = 1
	PERF_COUNT_HW_CACHE_REFERENCES        = 2
	PERF_COUNT_HW_CACHE_MISSES            = 3
	PERF_COUNT_HW_BRANCH_INSTRUCTIONS     = 4
	PERF_COUNT_HW_BRANCH_MISSES           = 5
	PERF_COUNT_HW_BUS_CYCLES              = 6
	PERF_COUNT_HW_STALLED_CYCLES_FRONTEND = 7
	PERF_COUNT_HW_STALLED_CYCLES_BACKEND  = 8
	PERF_COUNT_HW_REF_CPU_CYCLES          = 9
)

// Cache levels and operations for cache events
const (
	PERF_COUNT_HW_CACHE_L1D  = 0
	PERF_COUNT_HW_CACHE_L1I  = 1
	PERF_COUNT_HW_CACHE_LL   = 2 // Last Level Cache (L3)
	PERF_COUNT_HW_CACHE_DTLB = 3
	PERF_COUNT_HW_CACHE_ITLB = 4
	PERF_COUNT_HW_CACHE_BPU  = 5
	PERF_COUNT_HW_CACHE_NODE = 6
)

const (
	PERF_COUNT_HW_CACHE_OP_READ     = 0
	PERF_COUNT_HW_CACHE_OP_WRITE    = 1
	PERF_COUNT_HW_CACHE_OP_PREFETCH = 2
)

const (
	PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
	PERF_COUNT_HW_CACHE_RESULT_MISS   = 1
)

// Flags for perf_event_open
const (
	PERF_FLAG_FD_NO_GROUP = 1 << 0
	PERF_FLAG_FD_OUTPUT   = 1 << 1
	PERF_FLAG_PID_CGROUP  = 1 << 2
)

// perfEventOpen wraps the perf_event_open syscall
func perfEventOpen(attr *perfEventAttr, pid int, cpu int, groupFd int, flags uint64) (int, error) {
	fd, _, errno := syscall.Syscall6(
		syscall.SYS_PERF_EVENT_OPEN,
		uintptr(unsafe.Pointer(attr)),
		uintptr(pid),
		uintptr(cpu),
		uintptr(groupFd),
		uintptr(flags),
		0,
	)
	
	if errno != 0 {
		return -1, errno
	}
	
	return int(fd), nil
}

// LinuxPerfMonitor provides direct access to hardware performance counters
type LinuxPerfMonitor struct {
	fds      []int
	counters []perfEventConfig
}

type perfEventConfig struct {
	name   string
	typ    uint32
	config uint64
}

// NewLinuxPerfMonitor creates a performance monitor using perf_event_open
func NewLinuxPerfMonitor() *LinuxPerfMonitor {
	return &LinuxPerfMonitor{
		counters: []perfEventConfig{
			{"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
			{"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
			{"branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
			{"cache-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
			{"L1-dcache-misses", PERF_TYPE_HW_CACHE, cacheConfig(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
			{"LLC-misses", PERF_TYPE_HW_CACHE, cacheConfig(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS)},
		},
	}
}

// cacheConfig creates a cache event configuration
func cacheConfig(cache, op, result int) uint64 {
	return uint64(cache) | (uint64(op) << 8) | (uint64(result) << 16)
}

// Start begins performance counter collection
func (pm *LinuxPerfMonitor) Start() error {
	// Close any existing file descriptors
	pm.Stop()
	
	pm.fds = make([]int, 0, len(pm.counters))
	
	for _, counter := range pm.counters {
		attr := &perfEventAttr{
			Type:   counter.typ,
			Size:   uint32(unsafe.Sizeof(perfEventAttr{})),
			Config: counter.config,
			Flags:  0,
		}
		
		// Monitor current process on any CPU
		fd, err := perfEventOpen(attr, 0, -1, -1, 0)
		if err != nil {
			// Cleanup on error
			pm.Stop()
			return fmt.Errorf("failed to open perf event %s: %w", counter.name, err)
		}
		
		pm.fds = append(pm.fds, fd)
		
		// Reset counter
		var zero uint64
		syscall.Write(fd, (*[8]byte)(unsafe.Pointer(&zero))[:])
		
		// Enable counter
		syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), 0x2400, 0) // PERF_EVENT_IOC_ENABLE
	}
	
	return nil
}

// Stop ends collection and returns counters
func (pm *LinuxPerfMonitor) Stop() *PerfCounters {
	if len(pm.fds) == 0 {
		return &PerfCounters{}
	}
	
	counters := &PerfCounters{}
	
	// Read counter values
	for i, fd := range pm.fds {
		var value uint64
		n, err := syscall.Read(fd, (*[8]byte)(unsafe.Pointer(&value))[:])
		if err == nil && n == 8 {
			switch pm.counters[i].name {
			case "cycles":
				counters.Cycles = value
			case "instructions":
				counters.Instructions = value
			case "branch-misses":
				counters.BranchMisses = value
			case "cache-misses":
				counters.CacheMisses = value
			case "L1-dcache-misses":
				counters.L1DCacheMisses = value
			case "LLC-misses":
				counters.L3CacheMisses = value
			}
		}
		
		// Disable and close
		syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), 0x2401, 0) // PERF_EVENT_IOC_DISABLE
		syscall.Close(fd)
	}
	
	pm.fds = nil
	
	// Calculate derived metrics
	if counters.Cycles > 0 {
		counters.IPC = float64(counters.Instructions) / float64(counters.Cycles)
	}
	
	if counters.CacheMisses > 0 && counters.L3CacheMisses > 0 {
		counters.CacheMissRate = float64(counters.L3CacheMisses) / float64(counters.CacheMisses)
	}
	
	return counters
}

// MeasureWithHardwareCounters runs a function and collects hardware counters
func MeasureWithHardwareCounters(name string, fn func() error) (*PerfCounters, error) {
	monitor := NewLinuxPerfMonitor()
	
	err := monitor.Start()
	if err != nil {
		// Fall back to basic timing if perf counters unavailable
		return MeasureKernel(name, fn)
	}
	
	// Run the function
	err = fn()
	if err != nil {
		monitor.Stop()
		return nil, err
	}
	
	// Collect counters
	counters := monitor.Stop()
	
	return counters, nil
}

// IntegratePerfCounters enhances benchmarks with hardware counter collection
func IntegratePerfCounters(b *testing.B, name string, fn func()) {
	// Try to use hardware counters
	monitor := NewLinuxPerfMonitor()
	
	// Warm up
	fn()
	
	b.ResetTimer()
	
	// Start monitoring
	err := monitor.Start()
	useHWCounters := err == nil
	
	for i := 0; i < b.N; i++ {
		fn()
	}
	
	if useHWCounters {
		counters := monitor.Stop()
		
		// Report hardware metrics
		if counters.IPC > 0 {
			b.ReportMetric(counters.IPC, "IPC")
		}
		if counters.L3CacheMisses > 0 {
			b.ReportMetric(float64(counters.L3CacheMisses)/float64(b.N), "L3misses/op")
		}
		if counters.BranchMisses > 0 {
			b.ReportMetric(float64(counters.BranchMisses)/float64(b.N), "branch-misses/op")
		}
		if counters.Instructions > 0 {
			b.ReportMetric(float64(counters.Instructions)/float64(b.N), "instructions/op")
		}
	}
}