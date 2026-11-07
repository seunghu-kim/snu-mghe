package mkrlwe

import (
	"math"
	"math/big"
	"sort"
	"testing"

	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// Parameter sets copied verbatim from the experiment runners
// (logistic_regression/go_inference/experiments/runner.go and
// cnn/mnist/go_inference/experiment/runner.go).
var (
	floodTestLR = rlwe.ParametersLiteral{
		LogN: 15,
		Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001,
			0x10000290001, 0xffffc40001, 0x100003e0001,
			0x10000470001, 0x100004b0001, 0xffffb20001},
		P:     []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001},
		Sigma: rlwe.DefaultSigma,
	}
	floodTestMNIST = rlwe.ParametersLiteral{
		LogN: 14,
		Q: []uint64{0x2000000002b0001, 0x800000020001, 0x800000280001,
			0x800000520001, 0x800000770001, 0x800000aa0001, 0x800000ad0001},
		P:     []uint64{0x800000df0001, 0x800000f80001},
		Sigma: rlwe.DefaultSigma,
	}
)

func floodParams(t testing.TB, lit rlwe.ParametersLiteral) Parameters {
	t.Helper()
	p, err := rlwe.NewParametersFromLiteral(lit)
	if err != nil {
		t.Fatal(err)
	}
	return NewParameters(p, 2)
}

// centeredCRT reconstructs coefficient j of pol on primes 0..level as a
// centered integer in (-Q_level/2, Q_level/2].
func centeredCRT(ringQ *ring.Ring, level int, pol *ring.Poly, j int) *big.Int {
	Q := big.NewInt(1)
	for i := 0; i <= level; i++ {
		Q.Mul(Q, new(big.Int).SetUint64(ringQ.Modulus[i]))
	}
	acc := new(big.Int)
	for i := 0; i <= level; i++ {
		qi := new(big.Int).SetUint64(ringQ.Modulus[i])
		Qi := new(big.Int).Quo(Q, qi)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(Qi, qi), qi)
		term := new(big.Int).SetUint64(pol.Coeffs[i][j])
		term.Mul(term, inv).Mod(term, qi).Mul(term, Qi)
		acc.Add(acc, term)
	}
	acc.Mod(acc, Q)
	half := new(big.Int).Rsh(Q, 1)
	if acc.Cmp(half) > 0 {
		acc.Sub(acc, Q)
	}
	return acc
}

func TestFloodingConfigValidate(t *testing.T) {
	params := floodParams(t, floodTestMNIST)
	ok := []FloodingConfig{
		LegacyFloodingConfig(),
		WideGaussianFloodingConfig(10, 6),
		WideGaussianFloodingConfig(40, 6),
		WideGaussianFloodingConfig(49, 6),
		UniformFloodingConfig(20),
		UniformFloodingConfig(61),
	}
	for _, c := range ok {
		if err := c.Validate(params); err != nil {
			t.Errorf("%v rejected: %v", c, err)
		}
	}
	bad := []FloodingConfig{
		{Distribution: floodingNone},
		{},
		{Distribution: FloodingGaussianLegacy, Sigma: math.Exp2(30), Bound: 1 << 31}, // beyond legacy sampler
		{Distribution: FloodingGaussianLegacy, Sigma: 3.2, Bound: 0},
		{Distribution: FloodingGaussianWide, Sigma: -1, Bound: 10},
		{Distribution: FloodingGaussianWide, Sigma: math.NaN(), Bound: 10},
		{Distribution: FloodingGaussianWide, Sigma: math.Inf(1), Bound: 10},
		{Distribution: FloodingGaussianWide, Sigma: 1 << 20, Bound: 1 << 10}, // bound < sigma
		{Distribution: FloodingGaussianWide, Sigma: 1 << 50, Bound: 1 << 53}, // beyond float64 integers
		{Distribution: FloodingUniform, Bound: 0},
		{Distribution: FloodingUniform, Bound: 1 << 62},
		{Distribution: FloodingDistribution(99), Sigma: 1, Bound: 1},
	}
	for _, c := range bad {
		if err := c.Validate(params); err == nil {
			t.Errorf("%+v accepted, want error", c)
		}
	}
	if _, err := NewDecryptorWithFlooding(params, FloodingConfig{Distribution: floodingNone}); err == nil {
		t.Error("NewDecryptorWithFlooding accepted a noise-free configuration")
	}
}

func TestGenShareWithoutFloodingPanics(t *testing.T) {
	params := floodParams(t, floodTestMNIST)
	kgen := NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair("owner0")
	ct := NewCiphertext(params, NewIDSet(), params.MaxLevel())
	ct.Value = map[string]*ring.Poly{"0": params.RingQ().NewPoly(), "owner0": params.RingQ().NewPoly()}
	_ = pk
	defer func() {
		if recover() == nil {
			t.Fatal("GenShare on a decryptor without flooding did not panic")
		}
	}()
	NewDecryptor(params).GenShare(ct, sk)
}

// TestFloodingSamplerRNSConsistency checks that every sampled coefficient is
// the same integer on all active primes, lies within the bound, and that
// primes above the requested level are left untouched.
func TestFloodingSamplerRNSConsistency(t *testing.T) {
	for _, lit := range []rlwe.ParametersLiteral{floodTestMNIST, floodTestLR} {
		params := floodParams(t, lit)
		ringQ := params.RingQ()
		for _, cfg := range []FloodingConfig{
			LegacyFloodingConfig(),
			WideGaussianFloodingConfig(20, 6),
			WideGaussianFloodingConfig(45, 6),  // exceeds every 40/47-bit prime
			WideGaussianFloodingConfig(49, 12), // bound 2^52.6 > 2^52 -> invalid, skipped below
			UniformFloodingConfig(30),
			UniformFloodingConfig(61),
		} {
			if cfg.Validate(params) != nil {
				continue
			}
			s, err := newFloodingSampler(params, cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, level := range []int{0, 1, params.MaxLevel()} {
				pol := ringQ.NewPoly()
				const sentinel = 0xDEADBEEF
				for i := level + 1; i < len(ringQ.Modulus); i++ {
					for j := range pol.Coeffs[i] {
						pol.Coeffs[i][j] = sentinel
					}
				}
				s.readLvl(level, pol)
				Qlog := 0.0
				for i := 0; i <= level; i++ {
					Qlog += math.Log2(float64(ringQ.Modulus[i]))
				}
				bound := new(big.Int).SetUint64(cfg.Bound)
				for j := 0; j < ringQ.N; j += 7 {
					for i := 0; i <= level; i++ {
						if pol.Coeffs[i][j] >= ringQ.Modulus[i] {
							t.Fatalf("%v level %d: non-canonical residue on prime %d", cfg, level, i)
						}
					}
					if float64(cfg.Bound) >= math.Exp2(Qlog-1) {
						continue // integer not uniquely determined at this level; skip
					}
					v := centeredCRT(ringQ, level, pol, j)
					if new(big.Int).Abs(v).Cmp(bound) > 0 {
						t.Fatalf("%v level %d coeff %d: |v|=%v exceeds bound", cfg, level, j, v)
					}
					// re-reduce v on each prime and compare with stored residue
					for i := 0; i <= level; i++ {
						qi := new(big.Int).SetUint64(ringQ.Modulus[i])
						r := new(big.Int).Mod(v, qi).Uint64()
						if r != pol.Coeffs[i][j] {
							t.Fatalf("%v level %d coeff %d: inconsistent residue on prime %d", cfg, level, j, i)
						}
					}
				}
				for i := level + 1; i < len(ringQ.Modulus); i++ {
					if pol.Coeffs[i][0] != sentinel || pol.Coeffs[i][ringQ.N-1] != sentinel {
						t.Fatalf("%v level %d: prime %d above level was overwritten", cfg, level, i)
					}
				}
			}
		}
	}
}

// sampleIntegers draws n coefficients at level 0 of the LR ring (q0 ~ 2^50),
// returning centered integers. Valid only for bounds < q0/2.
func sampleIntegers(t *testing.T, params Parameters, s floodingSampler, rounds int) []int64 {
	ringQ := params.RingQ()
	out := make([]int64, 0, rounds*ringQ.N)
	q0 := ringQ.Modulus[0]
	pol := ringQ.NewPoly()
	for r := 0; r < rounds; r++ {
		s.readLvl(0, pol)
		for _, c := range pol.Coeffs[0] {
			c %= q0 // the Lattigo sampler may emit q0 for zero
			if c > q0/2 {
				out = append(out, -int64(q0-c))
			} else {
				out = append(out, int64(c))
			}
		}
	}
	return out
}

func moments(xs []int64) (mean, std, maxAbs float64) {
	var s, s2 float64
	for _, x := range xs {
		f := float64(x)
		s += f
		s2 += f * f
		if math.Abs(f) > maxAbs {
			maxAbs = math.Abs(f)
		}
	}
	n := float64(len(xs))
	mean = s / n
	std = math.Sqrt(s2/n - mean*mean)
	return
}

// lowByteChi2 returns the chi-square statistic of (x mod 256) against uniform
// (255 degrees of freedom: mean 255, sd ~22.6).
func lowByteChi2(xs []int64) float64 {
	var cnt [256]float64
	for _, x := range xs {
		cnt[uint8(x)]++
	}
	e := float64(len(xs)) / 256
	chi := 0.0
	for _, c := range cnt {
		chi += (c - e) * (c - e) / e
	}
	return chi
}

func TestWideGaussianMoments(t *testing.T) {
	params := floodParams(t, floodTestLR)
	for _, logSigma := range []float64{3, 10, 20, 30, 40, 45} {
		cfg := WideGaussianFloodingConfig(logSigma, 6)
		s, err := newFloodingSampler(params, cfg)
		if err != nil {
			t.Fatal(err)
		}
		xs := sampleIntegers(t, params, s, 4) // 131072 samples
		mean, std, maxAbs := moments(xs)
		sigma := cfg.Sigma
		if math.Abs(std/sigma-1) > 0.02 {
			t.Errorf("sigma=2^%v: empirical std %.4g (ratio %.4f)", logSigma, std, std/sigma)
		}
		if math.Abs(mean) > 5*sigma/math.Sqrt(float64(len(xs))) {
			t.Errorf("sigma=2^%v: mean %.4g too large", logSigma, mean)
		}
		if maxAbs > float64(cfg.Bound) {
			t.Errorf("sigma=2^%v: max %.4g exceeds bound", logSigma, maxAbs)
		}
		chi := ""
		if logSigma >= 10 {
			c := lowByteChi2(xs)
			chi = "  lowbyte-chi2=" + ftoa(c)
			if c > 400 { // p << 1e-6 under uniformity
				t.Errorf("sigma=2^%v: low byte not uniform (chi2=%.1f)", logSigma, c)
			}
		}
		t.Logf("wide sigma=2^%-2v std/sigma=%.4f max=2^%.2f%s", logSigma, std/sigma, math.Log2(maxAbs), chi)
	}
}

func TestUniformExact(t *testing.T) {
	params := floodParams(t, floodTestLR)
	for _, logB := range []uint{3, 20, 40} {
		cfg := UniformFloodingConfig(logB)
		s, _ := newFloodingSampler(params, cfg)
		xs := sampleIntegers(t, params, s, 4)
		mean, std, maxAbs := moments(xs)
		B := float64(cfg.Bound)
		want := math.Sqrt(((2*B+1)*(2*B+1) - 1) / 12)
		if math.Abs(std/want-1) > 0.02 || maxAbs > B || math.Abs(mean) > 5*want/math.Sqrt(float64(len(xs))) {
			t.Errorf("uniform B=2^%d: mean=%.3g std/want=%.4f max=%.4g", logB, mean, std/want, maxAbs)
		}
		if logB == 3 { // every value in [-8,8] must appear
			seen := map[int64]bool{}
			for _, x := range xs {
				seen[x] = true
			}
			if len(seen) != 17 {
				t.Errorf("uniform B=8: %d distinct values, want 17", len(seen))
			}
		}
	}
}

// collisions returns the number of equal pairs among xs (sort-based).
func collisions(xs []int64) int {
	ys := append([]int64(nil), xs...)
	sort.Slice(ys, func(a, b int) bool { return ys[a] < ys[b] })
	c := 0
	for i := 1; i < len(ys); i++ {
		if ys[i] == ys[i-1] {
			c++
		}
	}
	return c
}

// idealCollisions is the expected number of equal pairs among m draws of an
// ideal rounded Gaussian of large sigma: C(m,2) * sum_x p(x)^2 ~ C(m,2)/(2 sigma sqrt(pi)).
func idealCollisions(m int, sigma float64) float64 {
	return float64(m) * float64(m-1) / 2 / (2 * sigma * math.Sqrt(math.Pi))
}

// TestLegacySamplerSparseSupport characterises the Lattigo v2.3.0 Gaussian
// sampler at large sigma. Its bulk deviate is x = j*wn[i] with a 31-bit index
// j, so the support has at most ~2^32 points regardless of sigma. A birthday
// (collision) test exposes this once sigma*sqrt(2*pi) >> 2^32: the ideal
// rounded Gaussian then has far fewer collisions than the legacy sampler.
// The test calls ring.NewGaussianSampler directly (bypassing Validate) and
// reads only prime q0 (~2^50) so the bound stays below the prime.
func TestLegacySamplerSparseSupport(t *testing.T) {
	params := floodParams(t, floodTestLR)
	const rounds = 32 // 2^20 samples
	for _, logSigma := range []float64{20, 30, 36, 40} {
		sigma := math.Exp2(logSigma)
		bound := int(math.Min(6*sigma, math.Exp2(48)))
		prng, _ := utils.NewPRNG()
		leg := &legacyGaussian{g: ring.NewGaussianSampler(prng, params.RingQ(), sigma, bound), ringQ: params.RingQ()}
		wide, _ := newFloodingSampler(params, WideGaussianFloodingConfig(logSigma, 6))
		xl := sampleIntegers(t, params, leg, rounds)
		xw := sampleIntegers(t, params, wide, rounds)
		cl, cw, ci := collisions(xl), collisions(xw), idealCollisions(len(xl), sigma)
		t.Logf("sigma=2^%v  equal pairs among 2^20 draws: legacy=%d  wide=%d  ideal~%.1f", logSigma, cl, cw, ci)
		if logSigma >= 36 {
			// Poisson(ci): ci < 1 here; >= 20 collisions is astronomically unlikely for a correct sampler
			if cl < 20 {
				t.Errorf("sigma=2^%v: expected excess collisions from the legacy sampler, got %d", logSigma, cl)
			}
			if float64(cw) > ci+10*math.Sqrt(ci)+10 {
				t.Errorf("sigma=2^%v: wide sampler has %d collisions (ideal %.1f)", logSigma, cw, ci)
			}
		}
	}
}

func ftoa(f float64) string { return big.NewFloat(f).Text('f', 1) }

func newShareFixture(t *testing.T, lit rlwe.ParametersLiteral) (Parameters, *SecretKey, *Ciphertext) {
	params := floodParams(t, lit)
	kgen := NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPair("owner0")
	pt := rlwe.NewPlaintext(params.Parameters, params.MaxLevel())
	ct := NewCiphertext(params, NewIDSet(), params.MaxLevel())
	ct.Value["owner0"] = params.RingQ().NewPoly()
	NewEncryptor(params).Encrypt(pt, pk, ct)
	return params, sk, ct
}

// expectedProduct returns c_id * s_id in coefficient domain, computed
// independently of GenShare.
func expectedProduct(params Parameters, ct *Ciphertext, sk *SecretKey) *ring.Poly {
	ringQ := params.RingQ()
	level := ct.Level()
	c := ct.Value[sk.ID].CopyNew()
	if !c.IsNTT {
		ringQ.NTTLvl(level, c, c)
	}
	ringQ.MulCoeffsMontgomeryLvl(level, c, sk.Value.Q, c)
	ringQ.InvNTTLvl(level, c, c)
	return c
}

func maxDiffLog2(params Parameters, level int, a, b *ring.Poly) float64 {
	ringQ := params.RingQ()
	d := ringQ.NewPoly()
	ringQ.SubLvl(level, a, b, d)
	maxAbs := 0.0
	for j := 0; j < ringQ.N; j += 3 {
		v, _ := new(big.Float).SetInt(centeredCRT(ringQ, level, d, j)).Float64()
		maxAbs = math.Max(maxAbs, math.Abs(v))
	}
	return math.Log2(maxAbs)
}

// TestGenShareNoiseDomain checks that the flooding noise is added in the
// correct domain for both coefficient-domain ciphertexts (the form produced
// by the MK-CKKS encryptor/evaluator) and NTT-domain ciphertexts.
func TestGenShareNoiseDomain(t *testing.T) {
	params, sk, ct := newShareFixture(t, floodTestMNIST)
	ringQ := params.RingQ()
	if ct.Value["0"].IsNTT || ct.Value[sk.ID].IsNTT {
		t.Fatal("expected coefficient-domain ciphertext from mkrlwe encryptor")
	}
	for _, cfg := range []FloodingConfig{LegacyFloodingConfig(), WideGaussianFloodingConfig(30, 6), WideGaussianFloodingConfig(45, 6), UniformFloodingConfig(40)} {
		dec, err := NewDecryptorWithFlooding(params, cfg)
		if err != nil {
			t.Fatal(err)
		}
		want := expectedProduct(params, ct, sk)

		// coefficient domain
		share := dec.GenShare(ct, sk)
		got := maxDiffLog2(params, ct.Level(), share, want)
		if got > log2u(cfg.Bound)+1e-9 || got < log2u(cfg.Bound)-8 {
			t.Errorf("%v coeff-domain: |share - c*s|_inf = 2^%.2f, want <= 2^%.2f and near it", cfg, got, log2u(cfg.Bound))
		}

		// NTT domain
		ctN := ct.CopyNew()
		for id := range ctN.Value {
			ringQ.NTTLvl(ctN.Level(), ctN.Value[id], ctN.Value[id])
			ctN.Value[id].IsNTT = true
		}
		shareN := dec.GenShare(ctN, sk)
		ringQ.InvNTTLvl(ctN.Level(), shareN, shareN)
		gotN := maxDiffLog2(params, ct.Level(), shareN, want)
		if gotN > log2u(cfg.Bound)+1e-9 || gotN < log2u(cfg.Bound)-8 {
			t.Errorf("%v NTT-domain: |share - c*s|_inf = 2^%.2f, want <= 2^%.2f and near it", cfg, gotN, log2u(cfg.Bound))
		}
		t.Logf("%v: coeff 2^%.2f, ntt 2^%.2f", cfg, got, gotN)
	}
}

// TestGenShareFreshRandomness: shares of the same ciphertext must differ both
// across calls on one decryptor and across decryptor instances.
func TestGenShareFreshRandomness(t *testing.T) {
	params, sk, ct := newShareFixture(t, floodTestMNIST)
	cfg := WideGaussianFloodingConfig(20, 6)
	d1, _ := NewDecryptorWithFlooding(params, cfg)
	d2, _ := NewDecryptorWithFlooding(params, cfg)
	a, b, c := d1.GenShare(ct, sk), d1.GenShare(ct, sk), d2.GenShare(ct, sk)
	same := func(x, y *ring.Poly) int {
		n := 0
		for j := range x.Coeffs[0] {
			if x.Coeffs[0][j] == y.Coeffs[0][j] {
				n++
			}
		}
		return n
	}
	// with sigma = 2^20 a coincidence has probability ~ 1/(2.5*2^20) per coefficient
	if n := same(a, b); n > 8 {
		t.Errorf("two calls on one decryptor share %d equal coefficients", n)
	}
	if n := same(a, c); n > 8 {
		t.Errorf("two decryptors share %d equal coefficients", n)
	}
	// GenShare must not mutate the input ciphertext
	want := expectedProduct(params, ct, sk)
	if l := maxDiffLog2(params, ct.Level(), d1.GenShare(ct, sk), want); l > log2u(cfg.Bound) {
		t.Errorf("ciphertext modified by GenShare")
	}
}
