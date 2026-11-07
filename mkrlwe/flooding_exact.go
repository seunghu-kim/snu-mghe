package mkrlwe

// Exact discrete Gaussian flooding: the sampler of
//
//	C. L. Canonne, G. Kamath, T. Steinke, "The Discrete Gaussian for
//	Differential Privacy", NeurIPS 2020 (arXiv:2004.00010v5), Algorithms 1-3,
//
// transcribed from the authors' reference implementation
// (github.com/IBM/discrete-gaussian-differential-privacy, discretegauss.py,
// commit cb190d2a990a78eff6e21159203bc888e095f01b). Function names below
// mirror the reference functions one to one.
//
// Output distribution: for a rational sigma2 = sigma^2 > 0, every coefficient is
// an independent sample of
//
//	D_{Z,sigma}(x) = exp(-x^2 / (2 sigma^2)) / sum_{y in Z} exp(-y^2 / (2 sigma^2)),
//
// with NO truncation: the sampler returns an arbitrary integer (math/big) and
// it is reduced exactly modulo every RNS prime. Exactness holds under the
// assumption that the underlying source produces independent uniform bits;
// all arithmetic is on integers and rationals (math/big), no floating point is
// used anywhere on the sampling path.
//
// Randomness: production samplers read crypto/rand (the operating-system
// CSPRNG) through an 8 KiB buffer. Every random decision consumes only the
// primitive uniform(m) below.
//
// Timing: this sampler is NOT constant time; the number of loop iterations and
// the cost of big-integer arithmetic depend on the sampled value.

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/big"
	"math/bits"
	"runtime"
	"sync"

	"github.com/ldsec/lattigo/v2/ring"
)

// ExactGaussianMaxLogSigma is a sanity limit on log2(sigma) for the exact
// sampler (cost only; the algorithm itself has no upper limit).
const ExactGaussianMaxLogSigma = 128

// ExactDiscreteGaussianFloodingConfig returns an untruncated exact discrete
// Gaussian D_{Z,sigma} with the given rational sigma^2.
func ExactDiscreteGaussianFloodingConfig(sigma2 *big.Rat) FloodingConfig {
	f, _ := new(big.Float).SetRat(sigma2).Float64()
	return FloodingConfig{Distribution: FloodingDiscreteGaussianExact, Sigma: math.Sqrt(f), Sigma2: sigma2.RatString()}
}

// ExactDiscreteGaussianFloodingConfigPow2 returns D_{Z,sigma} with sigma = 2^logSigma
// exactly (sigma^2 = 2^(2 logSigma)).
func ExactDiscreteGaussianFloodingConfigPow2(logSigma uint) FloodingConfig {
	return ExactDiscreteGaussianFloodingConfig(new(big.Rat).SetInt(new(big.Int).Lsh(big.NewInt(1), 2*logSigma)))
}

// parseSigma2 returns the exact rational sigma^2 of an exact-Gaussian config.
func (c FloodingConfig) parseSigma2() (*big.Rat, error) {
	if c.Sigma2 == "" {
		return nil, fmt.Errorf("flooding: exact discrete Gaussian requires Sigma2 (exact rational sigma^2)")
	}
	r, ok := new(big.Rat).SetString(c.Sigma2)
	if !ok {
		return nil, fmt.Errorf("flooding: cannot parse Sigma2 %q as a rational", c.Sigma2)
	}
	if r.Sign() <= 0 {
		return nil, fmt.Errorf("flooding: Sigma2 must be positive, got %s", c.Sigma2)
	}
	return r, nil
}

func (c FloodingConfig) validateExact() error {
	s2, err := c.parseSigma2()
	if err != nil {
		return err
	}
	if c.Bound != 0 {
		return fmt.Errorf("flooding: the exact discrete Gaussian is untruncated; Bound must be 0, got %d", c.Bound)
	}
	f, _ := new(big.Float).SetRat(s2).Float64()
	if ls := math.Log2(f) / 2; ls > ExactGaussianMaxLogSigma {
		return fmt.Errorf("flooding: log2(sigma)=%.2f above sanity limit %d", ls, ExactGaussianMaxLogSigma)
	}
	if want := math.Sqrt(f); c.Sigma != 0 && math.Abs(c.Sigma-want) > 1e-9*want {
		return fmt.Errorf("flooding: Sigma=%v inconsistent with Sigma2=%s (sqrt=%v)", c.Sigma, c.Sigma2, want)
	}
	return nil
}

// ---- randomness ------------------------------------------------------------

// wordSource yields independent uniform 64-bit words.
type wordSource interface{ uint64() uint64 }

// cryptoWords reads crypto/rand through a buffer. A read failure panics: the
// sampler never continues with missing randomness.
type cryptoWords struct {
	buf [8192]byte
	ptr int
}

func newCryptoWords() *cryptoWords { c := &cryptoWords{}; c.ptr = len(c.buf); return c }

func (c *cryptoWords) uint64() uint64 {
	if c.ptr+8 > len(c.buf) {
		if _, err := io.ReadFull(rand.Reader, c.buf[:]); err != nil {
			panic(fmt.Sprintf("mkrlwe: crypto/rand failed: %v", err))
		}
		c.ptr = 0
	}
	v := binary.LittleEndian.Uint64(c.buf[c.ptr:])
	c.ptr += 8
	return v
}

// ---- CKS20 sampler ------------------------------------------------------------

type cks20 struct {
	src    wordSource
	sigma2 *big.Rat
	t      *big.Int // floorsqrt(sigma2) + 1
	ratT   *big.Rat // t
	invT   *big.Rat // 1/t   (scale of the discrete Laplace)
	s2divT *big.Rat // sigma2 / t
	twoS2  *big.Rat // 2 sigma2
	// statistics (diagnostic only; never influence sampling)
	nUniform uint64
	nLaplace uint64
}

var (
	ratOne  = big.NewRat(1, 1)
	ratHalf = big.NewRat(1, 2)
	bigOne  = big.NewInt(1)
)

func newCKS20(sigma2 *big.Rat, src wordSource) *cks20 {
	s := &cks20{src: src, sigma2: new(big.Rat).Set(sigma2)}
	s.t = new(big.Int).Add(floorsqrt(s.sigma2), bigOne)
	s.ratT = new(big.Rat).SetInt(s.t)
	s.invT = new(big.Rat).Inv(s.ratT)
	s.s2divT = new(big.Rat).Quo(s.sigma2, s.ratT)
	s.twoS2 = new(big.Rat).Add(s.sigma2, s.sigma2)
	return s
}

// floorsqrt returns floor(sqrt(x)) for rational x >= 0 exactly:
// floor(sqrt(x)) = isqrt(floor(x)).
func floorsqrt(x *big.Rat) *big.Int {
	fl := new(big.Int).Quo(x.Num(), x.Denom()) // x >= 0, so truncation = floor
	return fl.Sqrt(fl)
}

// sample_uniform: uniform integer in [0, m), m >= 1. All randomness of the
// sampler flows through here. Specification (also used by the differential
// test against the reference): if m == 1 return 0 without consuming
// randomness; otherwise let k = bitlen(m-1) and w = ceil(k/64); draw words
// w_0..w_{w-1}, r = (sum_i w_i 2^(64 i)) mod 2^k; return r if r < m, else retry.
func (s *cks20) uniform(m *big.Int) *big.Int {
	if m.Sign() <= 0 {
		panic("mkrlwe: uniform(m) with m <= 0")
	}
	if m.IsUint64() {
		return new(big.Int).SetUint64(s.uniformU64(m.Uint64()))
	}
	k := new(big.Int).Sub(m, bigOne).BitLen()
	w := (k + 63) / 64
	words := make([]uint64, w)
	r, tmp := new(big.Int), new(big.Int)
	for {
		for i := range words {
			words[i] = s.src.uint64()
		}
		s.nUniform++
		r.SetInt64(0)
		for i := w - 1; i >= 0; i-- {
			r.Lsh(r, 64)
			r.Or(r, tmp.SetUint64(words[i]))
		}
		for i := r.BitLen() - 1; i >= k; i-- { // r mod 2^k
			r.SetBit(r, i, 0)
		}
		if r.Cmp(m) < 0 {
			return r
		}
	}
}

func (s *cks20) uniformU64(m uint64) uint64 {
	if m == 1 {
		return 0
	}
	k := bits.Len64(m - 1)
	mask := uint64(math.MaxUint64)
	if k < 64 {
		mask = uint64(1)<<k - 1
	}
	for {
		s.nUniform++
		if r := s.src.uint64() & mask; r < m {
			return r
		}
	}
}

// sample_bernoulli: 1 with probability p, p rational in [0,1] (lowest terms,
// as Fraction in the reference).
func (s *cks20) bernoulli(p *big.Rat) bool {
	return s.uniform(p.Denom()).Cmp(p.Num()) < 0
}

// sample_bernoulli_exp1: Bernoulli(exp(-x)), x rational in [0,1].
func (s *cks20) bernoulliExp1(x *big.Rat) bool {
	k := int64(1)
	p := new(big.Rat)
	for {
		p.Quo(x, new(big.Rat).SetInt64(k))
		if s.bernoulli(p) {
			k++
		} else {
			break
		}
	}
	return k%2 == 1
}

// sample_bernoulli_exp: Bernoulli(exp(-x)), x rational >= 0.
func (s *cks20) bernoulliExp(x *big.Rat) bool {
	x = new(big.Rat).Set(x)
	for x.Cmp(ratOne) > 0 {
		if s.bernoulliExp1(ratOne) {
			x.Sub(x, ratOne)
		} else {
			return false
		}
	}
	return s.bernoulliExp1(x)
}

// sample_geometric_exp_slow with x = 1 (the only use): number of successes of
// Bernoulli(exp(-1)) before the first failure.
func (s *cks20) geometricExpSlow1() int64 {
	k := int64(0)
	for s.bernoulliExp(ratOne) {
		k++
	}
	return k
}

// sample_geometric_exp_fast with x = 1/t (t = x.denominator, x.numerator = 1).
func (s *cks20) geometricExpFastInvT() *big.Int {
	t := s.invT.Denom()
	var u *big.Int
	for {
		u = s.uniform(t)
		if s.bernoulliExp(new(big.Rat).SetFrac(u, t)) {
			break
		}
	}
	v := s.geometricExpSlow1()
	value := new(big.Int).Mul(big.NewInt(v), t)
	value.Add(value, u)
	return value.Quo(value, s.invT.Num()) // numerator is 1; kept to mirror the reference
}

// sample_dlaplace with scale t.
func (s *cks20) dlaplace() *big.Int {
	for {
		s.nLaplace++
		sign := s.bernoulli(ratHalf)
		magnitude := s.geometricExpFastInvT()
		if sign && magnitude.Sign() == 0 {
			continue
		}
		if sign {
			magnitude.Neg(magnitude)
		}
		return magnitude
	}
}

// sample_dgauss: one sample of D_{Z,sigma}, sigma^2 = sigma2.
func (s *cks20) dgauss() *big.Int {
	bias, abs := new(big.Rat), new(big.Rat)
	for {
		candidate := s.dlaplace()
		abs.SetInt(new(big.Int).Abs(candidate))
		bias.Sub(abs, s.s2divT)
		bias.Mul(bias, bias)
		bias.Quo(bias, s.twoS2)
		if s.bernoulliExp(bias) {
			return candidate
		}
	}
}

// ---- flooding sampler adapter ----------------------------------------------------

// ExactGaussianWorkers bounds the number of goroutines one exact flooding
// sampler uses to fill a polynomial (each with its own crypto/rand buffer and
// its own sampler state; coefficients are independent, so the joint
// distribution is unchanged). 0 selects min(GOMAXPROCS, 2): a small bound,
// because several owners may sample concurrently on one machine.
var ExactGaussianWorkers = 0

const exactGaussianMaxAutoWorkers = 2

func exactGaussianWorkers() int {
	if ExactGaussianWorkers > 0 {
		return ExactGaussianWorkers
	}
	w := runtime.GOMAXPROCS(0)
	if w > exactGaussianMaxAutoWorkers {
		w = exactGaussianMaxAutoWorkers
	}
	if w < 1 {
		w = 1
	}
	return w
}

type exactDiscreteGaussian struct {
	ws    []*cks20Fast
	ringQ *ring.Ring
}

func newExactDiscreteGaussian(sigma2 *big.Rat, ringQ *ring.Ring, workers int, newSrc func() wordSource) *exactDiscreteGaussian {
	e := &exactDiscreteGaussian{ringQ: ringQ}
	for i := 0; i < workers; i++ {
		e.ws = append(e.ws, newCKS20Fast(sigma2, newSrc()))
	}
	return e
}

func (s *exactDiscreteGaussian) readLvl(level int, pol *ring.Poly) {
	n, w := s.ringQ.N, len(s.ws)
	if w == 1 {
		s.fill(s.ws[0], level, pol, 0, n)
		return
	}
	var wg sync.WaitGroup
	for k := 0; k < w; k++ {
		lo, hi := k*n/w, (k+1)*n/w
		wg.Add(1)
		go func(g *cks20Fast) {
			defer wg.Done()
			s.fill(g, level, pol, lo, hi)
		}(s.ws[k])
	}
	wg.Wait()
}

// fill writes coefficients [lo, hi) (disjoint ranges per worker).
func (s *exactDiscreteGaussian) fill(g *cks20Fast, level int, pol *ring.Poly, lo, hi int) {
	for j := lo; j < hi; j++ {
		v, bv, isBig := g.dgauss()
		if isBig {
			setSignedBig(s.ringQ, level, pol, j, bv)
		} else {
			setSigned(s.ringQ, level, pol, j, v)
		}
	}
}

func (s *exactDiscreteGaussian) counters() (laplace, uniform uint64) {
	for _, g := range s.ws {
		laplace += g.nLaplace
		uniform += g.nUniform
	}
	return
}

// setSignedBig writes the integer v into coefficient j on primes 0..level by
// exact reduction (int64 fast path, math/big otherwise).
func setSignedBig(ringQ *ring.Ring, level int, pol *ring.Poly, j int, v *big.Int) {
	if v.IsInt64() && v.Int64() != math.MinInt64 {
		setSigned(ringQ, level, pol, j, v.Int64())
		return
	}
	r, q := new(big.Int), new(big.Int)
	for i := 0; i <= level; i++ {
		q.SetUint64(ringQ.Modulus[i])
		pol.Coeffs[i][j] = r.Mod(v, q).Uint64() // Euclidean: result in [0, q)
	}
}
