package amx

import (
	"fmt"
	"unsafe"
)

// TileConfigData represents the 64-byte tile configuration structure
// This must match the hardware layout exactly
type TileConfigData struct {
	Palette   uint8      // Palette ID (must be 1)
	StartRow  uint8      // Starting row (must be 0)
	Reserved1 [14]uint8  // Reserved (must be 0)
	ColsB     [16]uint16 // Bytes per row for each tile (16 tiles max)
	Reserved2 [16]uint8  // Reserved (must be 0)
	Rows      [16]uint8  // Number of rows for each tile
	Reserved3 [16]uint8  // Reserved (must be 0)
}

// AMXTileConfig provides high-level tile configuration
type AMXTileConfig struct {
	// Tile assignments for our standard INT8 GEMM kernel
	TileA1 int // First A tile (16×64 INT8)
	TileA2 int // Second A tile (16×64 INT8)
	TileB1 int // First B tile (64×16 INT8)
	TileB2 int // Second B tile (64×16 INT8)
	TileC1 int // First C accumulator (16×16 INT32)
	TileC2 int // Second C accumulator (16×16 INT32)
	TileC3 int // Third C accumulator (16×16 INT32)
	TileC4 int // Fourth C accumulator (16×16 INT32)
}

// DefaultInt8Config returns the default tile configuration for INT8 GEMM
func DefaultInt8Config() *AMXTileConfig {
	return &AMXTileConfig{
		TileA1: 0, // tmm0: 16×64 INT8
		TileA2: 1, // tmm1: 16×64 INT8
		TileB1: 2, // tmm2: 64×16 INT8
		TileB2: 3, // tmm3: 64×16 INT8
		TileC1: 4, // tmm4: 16×16 INT32 accumulator
		TileC2: 5, // tmm5: 16×16 INT32 accumulator
		TileC3: 6, // tmm6: 16×16 INT32 accumulator
		TileC4: 7, // tmm7: 16×16 INT32 accumulator
	}
}

// ConfigureInt8GEMM creates tile configuration for INT8 matrix multiply
// This configures tiles for optimal 32×32 output using 4 16×16 tiles
func ConfigureInt8GEMM() *TileConfigData {
	cfg := &TileConfigData{
		Palette:  1, // Must be 1
		StartRow: 0, // Must be 0
	}
	
	// Configure A tiles (16×64 INT8 = 16 rows × 64 bytes)
	cfg.Rows[0] = 16
	cfg.ColsB[0] = 64
	cfg.Rows[1] = 16
	cfg.ColsB[1] = 64
	
	// Configure B tiles (16×64 INT8 = 16 rows × 64 bytes)
	// Note: B will be packed in a special way for TDPBSSD
	cfg.Rows[2] = 16
	cfg.ColsB[2] = 64
	cfg.Rows[3] = 16
	cfg.ColsB[3] = 64
	
	// Configure C accumulator tiles (16×16 INT32 = 16 rows × 64 bytes)
	for i := 4; i < 8; i++ {
		cfg.Rows[i] = 16
		cfg.ColsB[i] = 64 // 16 × 4 bytes per INT32
	}
	
	return cfg
}

// ValidateConfig checks if tile configuration is valid
func ValidateConfig(cfg *TileConfigData) error {
	if cfg.Palette != 1 {
		return fmt.Errorf("palette must be 1, got %d", cfg.Palette)
	}
	
	if cfg.StartRow != 0 {
		return fmt.Errorf("start row must be 0, got %d", cfg.StartRow)
	}
	
	// Check tile dimensions
	totalBytes := 0
	for i := 0; i < 8; i++ { // Using first 8 tiles
		if cfg.Rows[i] > 0 {
			if cfg.Rows[i] > 16 {
				return fmt.Errorf("tile %d: rows %d exceeds maximum 16", i, cfg.Rows[i])
			}
			if cfg.ColsB[i] > 64 {
				return fmt.Errorf("tile %d: colsB %d exceeds maximum 64", i, cfg.ColsB[i])
			}
			tileBytes := int(cfg.Rows[i]) * int(cfg.ColsB[i])
			if tileBytes > 1024 {
				return fmt.Errorf("tile %d: size %d exceeds maximum 1024 bytes", i, tileBytes)
			}
			totalBytes += tileBytes
		}
	}
	
	// Check total tile register usage
	if totalBytes > 8192 { // 8KB total tile register space
		return fmt.Errorf("total tile size %d exceeds maximum 8192 bytes", totalBytes)
	}
	
	return nil
}

// GetConfigBytes returns the 64-byte configuration as a byte slice
func GetConfigBytes(cfg *TileConfigData) []byte {
	return (*[64]byte)(unsafe.Pointer(cfg))[:]
}

// TileLayout describes how to pack matrix data for AMX tiles
type TileLayout struct {
	TileRows int // Rows per tile
	TileCols int // Columns per tile (in elements, not bytes)
	ElemSize int // Bytes per element
}

// Int8Layout returns the layout for INT8 tiles
func Int8Layout() TileLayout {
	return TileLayout{
		TileRows: 16,
		TileCols: 64, // 64 INT8 elements = 64 bytes
		ElemSize: 1,
	}
}

// Int32Layout returns the layout for INT32 accumulator tiles
func Int32Layout() TileLayout {
	return TileLayout{
		TileRows: 16,
		TileCols: 16, // 16 INT32 elements = 64 bytes
		ElemSize: 4,
	}
}

// CalculateTiles determines how many tiles are needed for a matrix
func CalculateTiles(rows, cols int, layout TileLayout) (tileRows, tileCols int) {
	tileRows = (rows + layout.TileRows - 1) / layout.TileRows
	tileCols = (cols + layout.TileCols - 1) / layout.TileCols
	return
}