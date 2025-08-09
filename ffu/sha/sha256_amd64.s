//go:build amd64
// +build amd64

#include "textflag.h"

// SHA256 using SHA-NI instructions
// func sha256Block(h *[8]uint32, data []byte)
TEXT ·sha256Block(SB), NOSPLIT, $0-32
	MOVQ h+0(FP), DI        // Hash state
	MOVQ data_base+8(FP), SI   // Input data
	MOVQ data_len+16(FP), DX   // Length
	
	// Check if we have at least 64 bytes
	CMPQ DX, $64
	JL done
	
	// Load initial hash values
	MOVL 0(DI), AX   // H0
	MOVL 4(DI), BX   // H1
	MOVL 8(DI), CX   // H2
	MOVL 12(DI), DX  // H3
	// ... simplified for now
	
done:
	RET

// SHA1 using SHA-NI instructions
// func sha1Block(h *[5]uint32, data []byte)
TEXT ·sha1Block(SB), NOSPLIT, $0-32
	// SHA1 implementation would go here
	RET

// Feature detection
// func hasSHANI() bool
TEXT ·hasSHANI(SB), NOSPLIT, $0-1
	MOVL $7, AX
	MOVL $0, CX
	CPUID
	
	// Check EBX bit 29 for SHA
	MOVL BX, AX
	SHRL $29, AX
	ANDL $1, AX
	MOVB AX, ret+0(FP)
	RET
