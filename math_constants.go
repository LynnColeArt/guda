package guda

// Mathematical constants and configuration for GUDA computations
const (
	// Layer normalization default epsilon value
	// Standard value used in most neural network frameworks
	DefaultLayerNormEpsilon = 1e-5
	
	// Activation function saturation limits
	DefaultActivationSaturation = 10.0
	
	// Test tolerance levels for different precision requirements
	TestToleranceStrict = 1e-6  // For critical accuracy tests
	TestToleranceNormal = 1e-5  // For standard tests
	TestToleranceRelaxed = 1e-4 // For approximate methods
	
	// Mathematical constants with high precision
	MathE      = 2.7182818284590452354   // e
	MathPi     = 3.1415926535897932385   // π
	MathSqrt2  = 1.4142135623730950488   // √2
	MathLn2    = 0.6931471805599453094   // ln(2)
	MathLn10   = 2.3025850929940456840   // ln(10)
	MathLog2E  = 1.4426950408889634074   // log₂(e)
	MathLog10E = 0.4342944819032518277   // log₁₀(e)
	
	// Reciprocal constants for efficiency
	MathInvSqrt2   = 0.7071067811865475244  // 1/√2
	MathInvSqrtPi  = 0.5641895835477562869  // 1/√π
	MathInvSqrt2Pi = 0.3989422804014326780  // 1/√(2π)
	
	// GELU activation constants from Hendrycks & Gimpel paper
	// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
	GELUSqrt2OverPi = 0.7978845608028653559  // √(2/π)
	GELUCoefficient = 0.044715               // β coefficient
	
	// Error function approximation constants (Abramowitz & Stegun)
	// erf(x) ≈ 1 - exp(-x²) * polynomial(x)
	ErfA1 = 0.254829592   // a₁
	ErfA2 = -0.284496736  // a₂ 
	ErfA3 = 1.421413741   // a₃
	ErfA4 = -1.453152027  // a₄
	ErfA5 = 1.061405429   // a₅
	ErfP  = 0.3275911     // p
)

// MathConfig allows runtime configuration of mathematical parameters
type MathConfig struct {
	// Layer normalization epsilon for numerical stability
	LayerNormEpsilon float32
	
	// Activation function saturation limits
	ActivationSaturation float32
	
	// Test tolerance for numerical comparisons
	TestTolerance float64
	
	// Use high-precision implementations (slower but more accurate)
	HighPrecision bool
}

// DefaultMathConfig returns the default mathematical configuration
func DefaultMathConfig() MathConfig {
	return MathConfig{
		LayerNormEpsilon:     DefaultLayerNormEpsilon,
		ActivationSaturation: DefaultActivationSaturation,
		TestTolerance:        TestToleranceNormal,
		HighPrecision:        false,
	}
}

// StrictMathConfig returns a configuration optimized for accuracy
func StrictMathConfig() MathConfig {
	return MathConfig{
		LayerNormEpsilon:     1e-6,
		ActivationSaturation: 20.0,
		TestTolerance:        TestToleranceStrict,
		HighPrecision:        true,
	}
}

// FastMathConfig returns a configuration optimized for speed
func FastMathConfig() MathConfig {
	return MathConfig{
		LayerNormEpsilon:     1e-4,
		ActivationSaturation: 8.0,
		TestTolerance:        TestToleranceRelaxed,
		HighPrecision:        false,
	}
}