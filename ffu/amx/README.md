# AMX (Advanced Matrix Extensions) Support - EXPERIMENTAL

⚠️ **WARNING: This is experimental code that requires Intel Sapphire Rapids or newer CPUs. It has NOT been tested on real hardware.**

## Status

- ✅ FFU Framework Integration
- ✅ Reference Implementation (4.6 GOPS)
- ✅ Tile Configuration Management
- ⚠️ AMX Assembly Instructions (UNTESTED)
- ❌ Hardware Validation
- ❌ Performance Verification

## What Works

1. **Reference Implementation** (`amx_instructions_amd64.s`)
   - Fully tested assembly code
   - Achieves 4.6 GOPS on any x86-64 CPU
   - Validates algorithmic correctness

2. **FFU Integration**
   - Seamless integration with heterogeneous compute framework
   - Runtime detection and fallback
   - Clean abstraction layer

## What's Experimental

1. **AMX Instruction Encodings** (`amx_real_amd64.s`)
   - Manual VEX byte encodings
   - Based on Intel documentation
   - **COMPLETELY UNTESTED**
   - May not work at all

2. **Performance Claims**
   - 2 TOPS is theoretical
   - No hardware validation
   - Could be wildly wrong

## Usage

### Safe Usage (Recommended)
```go
// This will use the tested reference implementation
amxFFU := amx.NewAMXFFU()
```

### Experimental Usage (At Your Own Risk)
```bash
# This MAY work on Sapphire Rapids
go build -tags=amx

# But we have no idea if it actually will
```

## Why This Exists

We implemented AMX support to:
1. Explore the FFU abstraction design
2. Understand AMX programming model
3. Prepare for future hardware

However, we acknowledge this violates "test what you ship" principles.

## TODO When Hardware Available

- [ ] Verify CPUID detection actually works
- [ ] Test instruction encodings are correct
- [ ] Validate tile configuration
- [ ] Measure real performance
- [ ] Fix inevitable bugs

## Bottom Line

Use the reference implementation. Treat AMX instructions as untested research code that might inform future development when hardware becomes available.