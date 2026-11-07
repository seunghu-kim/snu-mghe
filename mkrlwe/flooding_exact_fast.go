package mkrlwe

// Allocation-free implementation of the CKS20 exact discrete Gaussian sampler
// (flooding_exact.go holds the line-by-line transcription `cks20`, which is
// kept as the executable specification).
//
// cks20Fast performs EXACTLY the same sequence of uniform(m) calls, with the
// same arguments m, as cks20 (and hence as the reference discretegauss.py):
// given the same word stream it returns identical samples and consumes an
// identical number of words. Only the representation of the intermediate
// rationals changes: instead of big.Rat (which re-normalises with a gcd and
// allocates on every operation), every rational is kept as an explicit
// (numerator, denominator) pair of integers that is in lowest terms by
// construction, using these exact identities:
//
//	(E1) x = n/d lowest, k >= 1:  x/k = (n/g) / (d*(k/g)) lowest, g = gcd(n mod k, k)
//	     (gcd(n, d*k) = gcd(n, k) because gcd(n, d) = 1).
//	(E2) x = n/d lowest:  x - 1 = (n-d)/d is lowest (gcd(n-d, d) = gcd(n, d)).
//	(E3) U < n/g  <=>  U*g < n   for integers U >= 0, g >= 1, g | n.
//	(E4) bias = (|Y| - sigma2/t)^2 / (2 sigma2). With sigma2/t = sn/sd lowest and
//	     2 sigma2 = tn/td lowest, D = |Y| sd - sn:
//	         bias = D^2 td / (sd^2 tn),  gcd(D, sd) = gcd(sn, sd) = 1,
//	     and therefore (gcd(xy, zw) = gcd(x, w) gcd(y, z) when gcd(x,z) = gcd(y,w) = 1)
//	         gcd(D^2 td, sd^2 tn) = g1 * g2,  g1 = gcd(D^2, tn),  g2 = gcd(td, sd^2),
//	     g2 is a constant; D = 0 gives bias = 0/1.
//
// Integers are uint64 where the value provably fits (overflow checked with
// math/bits) and reusable big.Int scratch otherwise; no floating point.

import (
	"math"
	"math/big"
	"math/bits"
)

const wordIs64 = bits.UintSize == 64

type cks20Fast struct {
	src wordSource

	// constants (all exact)
	t      *big.Int // floorsqrt(sigma2) + 1
	tU     uint64   // t, valid when tIsU64
	tIsU64 bool
	sn, sd *big.Int // sigma2/t = sn/sd, lowest terms
	tn     *big.Int // 2 sigma2 = tn/td, lowest terms
	tnTz   int      // trailing zeros of tn if tn is a power of two, else -1
	tdp    *big.Int // td / g2
	sd2p   *big.Int // sd^2 / g2

	// scratch (reused; never shared between goroutines)
	words                 []big.Word
	ub, a, d, d2, bn, bd  big.Int
	den, tmp, tmp2, g1, u big.Int
	mag                   big.Int

	nUniform, nLaplace uint64 // diagnostics only
}

func newCKS20Fast(sigma2 *big.Rat, src wordSource) *cks20Fast {
	if !wordIs64 {
		panic("mkrlwe: cks20Fast requires 64-bit big.Word")
	}
	f := &cks20Fast{src: src}
	s2 := new(big.Rat).Set(sigma2) // lowest terms a/b
	f.t = new(big.Int).Add(floorsqrt(s2), bigOne)
	f.tIsU64 = f.t.IsUint64()
	if f.tIsU64 {
		f.tU = f.t.Uint64()
	}
	q := new(big.Rat).Quo(s2, new(big.Rat).SetInt(f.t))
	f.sn, f.sd = new(big.Int).Set(q.Num()), new(big.Int).Set(q.Denom())
	two := new(big.Rat).Add(s2, s2)
	f.tn = new(big.Int).Set(two.Num())
	td := new(big.Int).Set(two.Denom())
	f.tnTz = -1
	if tz := int(f.tn.TrailingZeroBits()); tz == f.tn.BitLen()-1 {
		f.tnTz = tz
	}
	sd2 := new(big.Int).Mul(f.sd, f.sd)
	g2 := new(big.Int).GCD(nil, nil, td, sd2)
	f.tdp = new(big.Int).Quo(td, g2)
	f.sd2p = new(big.Int).Quo(sd2, g2)
	return f
}

func gcd64(a, b uint64) uint64 { // gcd(0, b) = b
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// modSmall returns x mod m for x >= 0, m >= 1.
func modSmall(x *big.Int, m uint64) uint64 {
	ws := x.Bits()
	var r uint64
	for i := len(ws) - 1; i >= 0; i-- {
		_, r = bits.Div64(r, uint64(ws[i]), m)
	}
	return r
}

// ---- uniform(m): same specification and consumption as cks20.uniform -------

func (f *cks20Fast) uniformU64(m uint64) uint64 {
	if m == 1 {
		return 0
	}
	k := bits.Len64(m - 1)
	mask := uint64(math.MaxUint64)
	if k < 64 {
		mask = uint64(1)<<k - 1
	}
	for {
		f.nUniform++
		if r := f.src.uint64() & mask; r < m {
			return r
		}
	}
}

// uniformBig returns a uniform integer in [0, m) for m >= 2^64 (callers use
// uniformU64 otherwise). The result aliases internal scratch and is valid only
// until the next uniformBig call.
func (f *cks20Fast) uniformBig(m *big.Int) *big.Int {
	bl := m.BitLen()
	k := bl // bitlen(m-1)
	if int(m.TrailingZeroBits()) == bl-1 {
		k = bl - 1
	}
	w := (k + 63) / 64
	if cap(f.words) < w {
		f.words = make([]big.Word, w)
	}
	ws := f.words[:w]
	for {
		for i := range ws { // word 0 least significant
			ws[i] = big.Word(f.src.uint64())
		}
		f.nUniform++
		if r := k % 64; r != 0 {
			ws[w-1] &= big.Word(uint64(1)<<r - 1)
		}
		f.ub.SetBits(ws)
		if f.ub.Cmp(m) < 0 {
			return &f.ub
		}
	}
}

// ---- Bernoulli(exp(-x)) ------------------------------------------------------------

// bernoulliExp1One = sample_bernoulli_exp1(1): p_k = 1/k.
func (f *cks20Fast) bernoulliExp1One() bool {
	k := uint64(1)
	for f.uniformU64(k) < 1 {
		k++
	}
	return k%2 == 1
}

// bernoulliExp1U64 = sample_bernoulli_exp1(n/d), n/d in [0,1] lowest terms.
func (f *cks20Fast) bernoulliExp1U64(n, d uint64) bool {
	k := uint64(1)
	for {
		g := gcd64(n%k, k) // (E1)
		hi, den := bits.Mul64(d, k/g)
		if hi != 0 { // denominator leaves uint64: continue exactly in math/big
			f.bn.SetUint64(n)
			f.bd.SetUint64(d)
			return f.bernoulliExp1BigFrom(&f.bn, &f.bd, k)
		}
		if f.uniformU64(den) >= n/g {
			break
		}
		k++
	}
	return k%2 == 1
}

// bernoulliExp1Big = sample_bernoulli_exp1(n/d), n/d in [0,1] lowest terms.
// n and d are read only.
func (f *cks20Fast) bernoulliExp1Big(n, d *big.Int) bool { return f.bernoulliExp1BigFrom(n, d, 1) }

func (f *cks20Fast) bernoulliExp1BigFrom(n, d *big.Int, k uint64) bool {
	for {
		g := gcd64(modSmall(n, k), k) // (E1)
		f.tmp.SetUint64(k / g)
		f.den.Mul(d, &f.tmp)
		if f.den.IsUint64() {
			u := f.uniformU64(f.den.Uint64())
			f.tmp.SetUint64(u)
		} else {
			f.tmp.Set(f.uniformBig(&f.den))
		}
		if g != 1 { // (E3)
			f.g1.SetUint64(g)
			f.tmp2.Mul(&f.tmp, &f.g1)
			f.tmp.Set(&f.tmp2)
		}
		if f.tmp.Cmp(n) >= 0 {
			break
		}
		k++
	}
	return k%2 == 1
}

// bernoulliExpBig = sample_bernoulli_exp(n/d), n/d >= 0 lowest terms.
// n is consumed (modified in place); d is read only.
func (f *cks20Fast) bernoulliExpBig(n, d *big.Int) bool {
	for n.Cmp(d) > 0 {
		if !f.bernoulliExp1One() {
			return false
		}
		n.Sub(n, d) // (E2)
	}
	if n.IsUint64() && d.IsUint64() {
		return f.bernoulliExp1U64(n.Uint64(), d.Uint64())
	}
	return f.bernoulliExp1Big(n, d)
}

// sample_geometric_exp_slow(1).
func (f *cks20Fast) geometricExpSlow1() uint64 {
	k := uint64(0)
	for f.bernoulliExp1One() { // bernoulli_exp(1) = bernoulli_exp1(1)
		k++
	}
	return k
}

// geometricExpFastInvT = sample_geometric_exp_fast(1/t). Returns the
// magnitude: in small if it fits below 2^63 (isBig = false), else in f.mag.
func (f *cks20Fast) geometricExpFastInvT() (small uint64, isBig bool) {
	if f.tIsU64 {
		t := f.tU
		var u uint64
		for {
			u = f.uniformU64(t)
			g := gcd64(u, t) // Fraction(U, t) in lowest terms; U < t so bernoulli_exp = bernoulli_exp1
			if f.bernoulliExp1U64(u/g, t/g) {
				break
			}
		}
		v := f.geometricExpSlow1()
		hi, lo := bits.Mul64(v, t)
		sum, carry := bits.Add64(lo, u, 0)
		if hi == 0 && carry == 0 && sum <= math.MaxInt64 {
			return sum, false
		}
		f.tmp2.SetUint64(v)
		f.mag.Mul(&f.tmp2, f.t)
		f.tmp.SetUint64(u)
		f.mag.Add(&f.mag, &f.tmp)
		return 0, true
	}
	for { // t >= 2^64
		f.u.Set(f.uniformBig(f.t))
		f.g1.GCD(nil, nil, &f.u, f.t)
		f.bn.Quo(&f.u, &f.g1)
		f.bd.Quo(f.t, &f.g1)
		if f.bernoulliExp1Big(&f.bn, &f.bd) {
			break
		}
	}
	v := f.geometricExpSlow1()
	f.tmp2.SetUint64(v)
	f.mag.Mul(&f.tmp2, f.t)
	f.mag.Add(&f.mag, &f.u)
	return 0, true
}

// dlaplace returns sign and magnitude (small or f.mag).
func (f *cks20Fast) dlaplace() (neg bool, small uint64, isBig bool) {
	for {
		f.nLaplace++
		sign := f.uniformU64(2) < 1 // bernoulli(1/2)
		small, isBig = f.geometricExpFastInvT()
		if sign && !isBig && small == 0 {
			continue
		}
		return sign, small, isBig
	}
}

// dgauss returns one sample of D_{Z,sigma}: v = +-small (|v| <= 2^63-1) when
// isBig is false, otherwise the value is in *big.Int bv (freshly allocated).
func (f *cks20Fast) dgauss() (v int64, bv *big.Int, isBig bool) {
	for {
		neg, small, big_ := f.dlaplace()
		if big_ {
			f.a.Set(&f.mag)
		} else {
			f.a.SetUint64(small)
		}
		// (E4)
		f.d.Mul(&f.a, f.sd)
		f.d.Sub(&f.d, f.sn)
		if f.d.Sign() == 0 {
			f.bn.SetInt64(0)
			f.bd.SetInt64(1)
		} else {
			f.d2.Mul(&f.d, &f.d)
			switch {
			case f.tnTz >= 0:
				e := int(f.d2.TrailingZeroBits())
				if e > f.tnTz {
					e = f.tnTz
				}
				f.tmp.Rsh(&f.d2, uint(e))
				f.tmp2.Rsh(f.tn, uint(e))
			default:
				f.g1.GCD(nil, nil, &f.d2, f.tn)
				f.tmp.Quo(&f.d2, &f.g1)
				f.tmp2.Quo(f.tn, &f.g1)
			}
			// big.Int.Mul allocates when the receiver aliases an operand
			f.bn.Mul(&f.tmp, f.tdp)
			f.bd.Mul(&f.tmp2, f.sd2p)
		}
		if f.bernoulliExpBig(&f.bn, &f.bd) {
			if !big_ {
				if neg {
					return -int64(small), nil, false
				}
				return int64(small), nil, false
			}
			out := new(big.Int).Set(&f.mag)
			if neg {
				out.Neg(out)
			}
			return 0, out, true
		}
	}
}

// dgaussBig returns the sample as a big.Int (tests).
func (f *cks20Fast) dgaussBig() *big.Int {
	v, bv, isBig := f.dgauss()
	if isBig {
		return bv
	}
	return big.NewInt(v)
}
