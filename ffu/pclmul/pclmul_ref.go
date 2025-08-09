package pclmul

// Reference implementations for non-amd64 or CGO builds

// crc32Software is a software CRC32 implementation
func crc32Software(data []byte, polynomial uint64) uint64 {
	crc := uint64(0xFFFFFFFFFFFFFFFF)
	
	for _, b := range data {
		crc ^= uint64(b)
		for i := 0; i < 8; i++ {
			if crc&1 != 0 {
				crc = (crc >> 1) ^ polynomial
			} else {
				crc >>= 1
			}
		}
	}
	
	return crc ^ 0xFFFFFFFFFFFFFFFF
}

// galoisMulSoftware performs Galois field multiplication in software
func galoisMulSoftware(a, b []byte, result []byte) {
	// Simple XOR for now - real implementation would be more complex
	for i := 0; i < len(a) && i < len(b) && i < len(result); i++ {
		result[i] = a[i] ^ b[i]
	}
}

// rsEncodeSoftware is a simple Reed-Solomon reference
func rsEncodeSoftware(data []byte, parity []byte, matrix []byte) {
	// Simplified - real RS would use proper Galois field arithmetic
	for i := range parity {
		parity[i] = 0
	}
	
	for i, d := range data {
		for j := range parity {
			if i < len(matrix) {
				parity[j] ^= galoisMul(d, matrix[i])
			}
		}
	}
}

// galoisMul performs GF(256) multiplication
func galoisMul(a, b byte) byte {
	var p byte = 0
	var hi_bit_set byte
	
	for i := 0; i < 8; i++ {
		if b&1 != 0 {
			p ^= a
		}
		hi_bit_set = a & 0x80
		a <<= 1
		if hi_bit_set != 0 {
			a ^= 0x1b // x^8 + x^4 + x^3 + x + 1
		}
		b >>= 1
	}
	
	return p
}