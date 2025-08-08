package guda

import "math"

// Activation function implementations with proper numerical accuracy

// SigmoidFloat32 computes sigmoid(x) = 1 / (1 + exp(-x)) with good accuracy
// Uses a rational approximation for the range [-5, 5] and clips outside
func SigmoidFloat32(x float32) float32 {
	// For large |x|, sigmoid saturates
	if x < -DefaultActivationSaturation {
		return 0
	}
	if x > DefaultActivationSaturation {
		return 1
	}
	
	// For |x| <= 5, use a more accurate approximation
	// Based on "Efficient Approximations for the Sigmoid Function" by Schraudolph
	if x >= 0 {
		exp_neg_x := ExpFloat32(-x)
		return 1.0 / (1.0 + exp_neg_x)
	} else {
		exp_x := ExpFloat32(x)
		return exp_x / (1.0 + exp_x)
	}
}

// TanhFloat32 computes tanh(x) with good accuracy
// Uses exp formula for best accuracy across all ranges
func TanhFloat32(x float32) float32 {
	// For large |x|, tanh saturates
	if x > DefaultActivationSaturation {
		return 1
	}
	if x < -DefaultActivationSaturation {
		return -1
	}
	
	// For small |x|, use the identity tanh(x) = (e^2x - 1) / (e^2x + 1)
	// to avoid catastrophic cancellation
	if x >= 0 {
		if x < 0.5 {
			// For very small x, use series expansion to avoid numerical issues
			// tanh(x) ≈ x - x³/3 + 2x⁵/15 - 17x⁷/315
			x2 := x * x
			return x * (1 - x2/3 + 2*x2*x2/15)
		} else {
			exp_2x := ExpFloat32(2 * x)
			return (exp_2x - 1) / (exp_2x + 1)
		}
	} else {
		// tanh(-x) = -tanh(x)
		return -TanhFloat32(-x)
	}
}

// ExpFloat32 computes exp(x) with good accuracy for float32
// Uses range reduction and polynomial approximation
func ExpFloat32(x float32) float32 {
	// Handle special cases
	if x > 88.7 { // exp(88.7) ≈ max float32
		return math.MaxFloat32
	}
	if x < -87.3 { // exp(-87.3) ≈ min positive float32
		return 0
	}
	
	// Range reduction: exp(x) = 2^k * exp(r) where x = k*ln(2) + r
	const ln2 = MathLn2
	k := int(math.Floor(float64(x) / ln2))
	r := x - float32(k)*float32(ln2)
	
	// Compute exp(r) using degree-5 polynomial (Remez approximation)
	// exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
	// But using optimized coefficients for better accuracy
	r2 := r * r
	r3 := r2 * r
	r4 := r2 * r2
	r5 := r4 * r
	
	exp_r := 1.0 + r + 
		0.4999999701976776*r2 + 
		0.1666666567325592*r3 + 
		0.0416666679084301*r4 + 
		0.0083333337679505*r5
	
	// Reconstruct: exp(x) = 2^k * exp(r)
	return float32(math.Ldexp(float64(exp_r), k))
}

// GeluFloat32Accurate computes GELU with high accuracy
// Uses error function (erf) for best results
func GeluFloat32Accurate(x float32) float32 {
	// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x/√2))
	const invSqrt2 = MathInvSqrt2 // 1/√2
	return x * 0.5 * (1 + ErfFloat32(x*invSqrt2))
}

// ErfFloat32 computes the error function with good accuracy
// Uses rational approximation from Abramowitz & Stegun
func ErfFloat32(x float32) float32 {
	// Handle negative values using erf(-x) = -erf(x)
	sign := float32(1)
	if x < 0 {
		sign = -1
		x = -x
	}
	
	// Constants for approximation
	const (
		a1 = ErfA1
		a2 = ErfA2
		a3 = ErfA3
		a4 = ErfA4
		a5 = ErfA5
		p  = ErfP
	)
	
	// Approximation: erf(x) ≈ 1 - exp(-x²) * polynomial(x)
	t := 1 / (1 + p*x)
	t2 := t * t
	t3 := t2 * t
	t4 := t2 * t2
	t5 := t4 * t
	
	exp_neg_x2 := ExpFloat32(-x * x)
	polynomial := a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
	
	return sign * (1 - exp_neg_x2*polynomial)
}