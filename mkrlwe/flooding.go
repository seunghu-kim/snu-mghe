package mkrlwe

// Flooding (smudging) noise for partial-decryption shares.
//
// This file makes the distribution, magnitude, truncation and validation of the
// noise added by GenShare explicit. It provides infrastructure only: adding
// noise to partial-decryption shares is NOT by itself an implementation of any
// particular security theorem (e.g. the Kluczniak–Santato multikey construction
// additionally modifies Eval).
//
// Four distributions are available:
//
//   - FloodingGaussianLegacy: the Lattigo v2.3.0 ring.GaussianSampler that the
//     experiments have always used (sigma = 3.2, bound = 19). The Lattigo
//     sampler builds its normal deviate from a 31-bit ziggurat index scaled by
//     float32 constants and writes negative values as (q_i - |v|) without
//     reduction, so it is only accepted for small sigma and for bounds below
//     every RNS prime (see LegacyGaussianMaxSigma).
//   - FloodingGaussianWide: rounded Gaussian round(sigma * z) with z from a
//     Box–Muller transform on 53-bit uniforms drawn from a CSPRNG, tail-cut by
//     rejection at Bound, written to every RNS prime by exact signed reduction.
//     It supports magnitudes far beyond the RNS primes, but it is a
//     floating-point sampler: its statistical distance to the ideal discrete
//     Gaussian is NOT bounded by a proof here.
//   - FloodingUniform: exact uniform integers in [-Bound, Bound] by rejection
//     sampling on CSPRNG output (no floating point). Exactly samplable, so the
//     classical smudging lemma applies to it directly, at the price of a larger
//     magnitude than a Gaussian.
//   - FloodingDiscreteGaussianExact: the exact, untruncated discrete Gaussian
//     D_{Z,sigma} for rational sigma^2, sampled with the algorithm of
//     Canonne–Kamath–Steinke (NeurIPS 2020) on crypto/rand output using only
//     integer/rational arithmetic (flooding_exact.go). Its output distribution
//     equals D_{Z,sigma} exactly given ideal uniform bits; there is no tail cut.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"

	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
)

// FloodingDistribution identifies the distribution of partial-decryption noise.
type FloodingDistribution int

const (
	// floodingNone is deliberately unexported: noise-free shares must never be
	// reachable from production code. Internal tests construct it directly.
	floodingNone FloodingDistribution = iota
	FloodingGaussianLegacy
	FloodingGaussianWide
	FloodingUniform
	FloodingDiscreteGaussianExact
)

func (d FloodingDistribution) String() string {
	switch d {
	case floodingNone:
		return "none(internal-test-only)"
	case FloodingGaussianLegacy:
		return "gaussian-legacy-lattigo"
	case FloodingGaussianWide:
		return "gaussian-wide-boxmuller"
	case FloodingUniform:
		return "uniform-exact"
	case FloodingDiscreteGaussianExact:
		return "discrete-gaussian-exact-cks20"
	}
	return fmt.Sprintf("unknown(%d)", int(d))
}

const (
	// LegacySigma and LegacyBound are the values hard-coded in every data-owner
	// GenShare call before this refactoring.
	LegacySigma = 3.2
	LegacyBound = 19

	// LegacyGaussianMaxSigma is the largest sigma accepted for the Lattigo
	// sampler. Its normal deviate has an absolute resolution of roughly
	// 1e-9..2e-7 per strip (31-bit index times float32 widths), so beyond
	// ~2^20 the rounded output no longer covers all integers near the mean
	// (TestLegacySamplerSparseSupport demonstrates the gaps at 2^30 and 2^40).
	LegacyGaussianMaxSigma = float64(1 << 20)

	// WideGaussianMaxBound keeps sigma*z exactly representable as an integer in
	// float64 (53-bit mantissa) and well inside int64.
	WideGaussianMaxBound = uint64(1) << 52

	// UniformMaxBound keeps 2*Bound+1 and the signed result inside int64.
	UniformMaxBound = uint64(1) << 61
)

// FloodingConfig fully specifies the noise that a data owner adds to its share.
type FloodingConfig struct {
	Distribution FloodingDistribution
	// Sigma is the standard deviation of the (untruncated) Gaussian. Ignored
	// for FloodingUniform.
	Sigma float64
	// Bound is the truncation: every sampled coefficient v satisfies |v| <= Bound.
	// For FloodingUniform it is the half-width of the support. It must be 0
	// for FloodingDiscreteGaussianExact, which is untruncated.
	Bound uint64
	// Sigma2 is the exact rational sigma^2 ("a/b" or "a", big.Rat syntax) of
	// FloodingDiscreteGaussianExact; Sigma is then only its float64 square root
	// for logging. Unused by the other distributions.
	Sigma2 string
}

// LegacyFloodingConfig returns exactly the configuration used by the published
// experiments (Lattigo Gaussian, sigma = 3.2, bound = 19).
func LegacyFloodingConfig() FloodingConfig {
	return FloodingConfig{Distribution: FloodingGaussianLegacy, Sigma: LegacySigma, Bound: LegacyBound}
}

// WideGaussianFloodingConfig returns a wide Gaussian with the given log2(sigma)
// and a tail cut of tailCut*sigma.
func WideGaussianFloodingConfig(logSigma, tailCut float64) FloodingConfig {
	sigma := math.Exp2(logSigma)
	return FloodingConfig{Distribution: FloodingGaussianWide, Sigma: sigma, Bound: uint64(math.Ceil(tailCut * sigma))}
}

// UniformFloodingConfig returns exact uniform noise on [-2^logBound, 2^logBound].
func UniformFloodingConfig(logBound uint) FloodingConfig {
	return FloodingConfig{Distribution: FloodingUniform, Bound: uint64(1) << logBound}
}

// String gives a compact description used in logs and reports.
func (c FloodingConfig) String() string {
	switch c.Distribution {
	case FloodingUniform:
		return fmt.Sprintf("%s(B=2^%.2f)", c.Distribution, log2u(c.Bound))
	case FloodingDiscreteGaussianExact:
		return fmt.Sprintf("%s(sigma=2^%.2f, untruncated)", c.Distribution, math.Log2(c.Sigma))
	default:
		return fmt.Sprintf("%s(sigma=2^%.2f, B=2^%.2f)", c.Distribution, math.Log2(c.Sigma), log2u(c.Bound))
	}
}

// Validate checks that the sampler for this configuration represents the
// requested distribution correctly on the RNS basis of params (independent of
// decryption correctness, which is checked by CheckHeadroom).
func (c FloodingConfig) Validate(params Parameters) error {
	minQ := params.RingQ().Modulus[0]
	for _, qi := range params.RingQ().Modulus {
		if qi < minQ {
			minQ = qi
		}
	}

	switch c.Distribution {
	case floodingNone:
		return errors.New("flooding: noise-free partial decryption is not permitted")
	case FloodingGaussianLegacy:
		if err := checkSigma(c.Sigma); err != nil {
			return err
		}
		if c.Sigma > LegacyGaussianMaxSigma {
			return fmt.Errorf("flooding: legacy Lattigo Gaussian sampler does not represent sigma=2^%.2f correctly (max 2^%.0f); use FloodingGaussianWide",
				math.Log2(c.Sigma), math.Log2(LegacyGaussianMaxSigma))
		}
		if c.Bound == 0 || c.Bound > math.MaxInt32 {
			return fmt.Errorf("flooding: legacy bound %d out of range", c.Bound)
		}
		// Lattigo writes negative values as q_i - |v| without reduction.
		if c.Bound >= minQ {
			return fmt.Errorf("flooding: legacy bound 2^%.2f >= smallest RNS prime 2^%.2f", log2u(c.Bound), log2u(minQ))
		}
	case FloodingGaussianWide:
		if err := checkSigma(c.Sigma); err != nil {
			return err
		}
		if c.Bound == 0 || c.Bound > WideGaussianMaxBound {
			return fmt.Errorf("flooding: wide Gaussian bound must be in [1, 2^52], got 2^%.2f", log2u(c.Bound))
		}
		if float64(c.Bound) < c.Sigma {
			return fmt.Errorf("flooding: bound %d below sigma %.3g truncates the bulk of the distribution", c.Bound, c.Sigma)
		}
	case FloodingUniform:
		if c.Bound == 0 || c.Bound > UniformMaxBound {
			return fmt.Errorf("flooding: uniform bound must be in [1, 2^61], got %d", c.Bound)
		}
	case FloodingDiscreteGaussianExact:
		return c.validateExact()
	default:
		return fmt.Errorf("flooding: unknown distribution %d", int(c.Distribution))
	}
	return nil
}

func checkSigma(s float64) error {
	if math.IsNaN(s) || math.IsInf(s, 0) || s <= 0 {
		return fmt.Errorf("flooding: invalid sigma %v", s)
	}
	return nil
}

// CheckHeadroom is a correctness (not security) check: with nShares shares each
// bounded by c.Bound, plus a ciphertext-error-plus-message bound 2^logMsgErr
// (in coefficient units), the aggregated phase must stay below q_level/2,
// where q_level = prod_{i<=level} q_i. It returns the remaining headroom in bits.
//
// The exact discrete Gaussian is untruncated and has no deterministic bound;
// CheckHeadroom returns an error for it instead of inventing a tail cut.
func (c FloodingConfig) CheckHeadroom(params Parameters, level, nShares int, logMsgErr float64) (float64, error) {
	if c.Distribution == FloodingDiscreteGaussianExact {
		return math.NaN(), errors.New("flooding: exact discrete Gaussian is untruncated; no deterministic headroom bound")
	}
	logQ := 0.0
	for i := 0; i <= level; i++ {
		logQ += math.Log2(float64(params.RingQ().Modulus[i]))
	}
	total := math.Log2(float64(nShares)*float64(c.Bound) + math.Exp2(logMsgErr))
	headroom := (logQ - 1) - total
	if headroom <= 0 {
		return headroom, fmt.Errorf("flooding: noise 2^%.2f exceeds q_level/2 = 2^%.2f at level %d", total, logQ-1, level)
	}
	return headroom, nil
}

func log2u(x uint64) float64 { return math.Log2(float64(x)) }

// floodingSampler writes a polynomial of independent flooding coefficients,
// consistently represented on RNS primes 0..level, in coefficient domain.
type floodingSampler interface {
	readLvl(level int, pol *ring.Poly)
}

func newFloodingSampler(params Parameters, cfg FloodingConfig) (floodingSampler, error) {
	if err := cfg.Validate(params); err != nil {
		return nil, err
	}
	prng, err := utils.NewPRNG() // BLAKE2b XOF keyed with 64 bytes from crypto/rand
	if err != nil {
		return nil, err
	}
	return newFloodingSamplerFromPRNG(params, cfg, prng), nil
}

func newFloodingSamplerFromPRNG(params Parameters, cfg FloodingConfig, prng utils.PRNG) floodingSampler {
	ringQ := params.RingQ()
	switch cfg.Distribution {
	case FloodingGaussianLegacy:
		return &legacyGaussian{g: ring.NewGaussianSampler(prng, ringQ, cfg.Sigma, int(cfg.Bound)), ringQ: ringQ}
	case FloodingGaussianWide:
		return &wideGaussian{src: newByteSource(prng), ringQ: ringQ, sigma: cfg.Sigma, bound: cfg.Bound}
	case FloodingUniform:
		return &exactUniform{src: newByteSource(prng), ringQ: ringQ, bound: cfg.Bound}
	case FloodingDiscreteGaussianExact:
		// independent of prng: reads crypto/rand directly
		s2, err := cfg.parseSigma2()
		if err != nil {
			panic(err)
		}
		return newExactDiscreteGaussian(s2, ringQ, exactGaussianWorkers(), func() wordSource { return newCryptoWords() })
	case floodingNone:
		return zeroSampler{}
	}
	panic("unreachable")
}

// --- legacy -----------------------------------------------------------------

type legacyGaussian struct {
	g     *ring.GaussianSampler
	ringQ *ring.Ring
}

// readLvl draws from the Lattigo sampler and canonicalises its output: Lattigo
// writes a sampled 0 with negative sign as q_i (not reduced). The integer is
// unchanged; only its residue representation is fixed to [0, q_i).
func (s *legacyGaussian) readLvl(level int, pol *ring.Poly) {
	s.g.ReadLvl(level, pol)
	for i := 0; i <= level; i++ {
		qi := s.ringQ.Modulus[i]
		for j, c := range pol.Coeffs[i] {
			if c >= qi {
				pol.Coeffs[i][j] = c - qi
			}
		}
	}
}

// --- internal no-noise (tests only; unreachable through Validate) ------------

type zeroSampler struct{}

func (zeroSampler) readLvl(level int, pol *ring.Poly) {
	for i := 0; i <= level; i++ {
		for j := range pol.Coeffs[i] {
			pol.Coeffs[i][j] = 0
		}
	}
}

// --- shared helpers -----------------------------------------------------------

type byteSource struct {
	prng utils.PRNG
	buf  []byte
	ptr  int
}

func newByteSource(prng utils.PRNG) *byteSource {
	b := &byteSource{prng: prng, buf: make([]byte, 8192)}
	b.ptr = len(b.buf)
	return b
}

func (b *byteSource) uint64() uint64 {
	if b.ptr+8 > len(b.buf) {
		b.prng.Clock(b.buf)
		b.ptr = 0
	}
	v := binary.LittleEndian.Uint64(b.buf[b.ptr:])
	b.ptr += 8
	return v
}

// float53 returns a uniform value in (0, 1] with 53 bits of resolution.
func (b *byteSource) float53() float64 {
	return float64((b.uint64()>>11)+1) * (1.0 / (1 << 53))
}

// setSigned writes the signed integer v (|v| < 2^63) into coefficient j of pol
// on primes 0..level, reducing it exactly modulo each prime.
func setSigned(ringQ *ring.Ring, level int, pol *ring.Poly, j int, v int64) {
	if v >= 0 {
		u := uint64(v)
		for i := 0; i <= level; i++ {
			pol.Coeffs[i][j] = u % ringQ.Modulus[i]
		}
		return
	}
	u := uint64(-v)
	for i := 0; i <= level; i++ {
		qi := ringQ.Modulus[i]
		r := u % qi
		if r == 0 {
			pol.Coeffs[i][j] = 0
		} else {
			pol.Coeffs[i][j] = qi - r
		}
	}
}

// --- wide Gaussian -------------------------------------------------------------

type wideGaussian struct {
	src   *byteSource
	ringQ *ring.Ring
	sigma float64
	bound uint64
	spare float64
	has   bool
}

func (s *wideGaussian) normal() float64 {
	if s.has {
		s.has = false
		return s.spare
	}
	u1 := s.src.float53() // (0,1], so log is finite
	u2 := s.src.float53()
	r := math.Sqrt(-2 * math.Log(u1))
	sn, cs := math.Sincos(2 * math.Pi * u2)
	s.spare, s.has = r*sn, true
	return r * cs
}

func (s *wideGaussian) readLvl(level int, pol *ring.Poly) {
	b := float64(s.bound)
	for j := 0; j < s.ringQ.N; j++ {
		var v float64
		for {
			v = math.Round(s.sigma * s.normal())
			if math.Abs(v) <= b {
				break
			}
		}
		setSigned(s.ringQ, level, pol, j, int64(v))
	}
}

// --- exact uniform -------------------------------------------------------------

type exactUniform struct {
	src   *byteSource
	ringQ *ring.Ring
	bound uint64
}

func (s *exactUniform) readLvl(level int, pol *ring.Poly) {
	width := 2*s.bound + 1 // number of values in [-B, B]
	mask := uint64(1)<<bits.Len64(width-1) - 1
	if bits.Len64(width-1) == 64 {
		mask = math.MaxUint64
	}
	for j := 0; j < s.ringQ.N; j++ {
		var r uint64
		for {
			r = s.src.uint64() & mask
			if r < width {
				break
			}
		}
		setSigned(s.ringQ, level, pol, j, int64(r)-int64(s.bound))
	}
}
