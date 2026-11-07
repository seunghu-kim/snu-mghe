package mkckks

// R1-3 parameter audit (diagnostic only): exact, coefficient-domain error that
// each production primitive adds, for the runner and D60 parameter sets.
//
//	PARAM_AUDIT=1 go test -run TestParamAuditOps -v -timeout 0
//
// Every error below is an exact centered integer polynomial obtained by CRT
// reconstruction over the active RNS primes:
//
//	encode   : float64 Encode  - 256-bit Encode (deterministic, same values)
//	fresh    : phase(Enc_pk(pt)) - pt                    (u*e_pk + e0 + e1*s)
//	relin    : phase(MulAndRelinHoisted(a,b)) - phase(a)*phase(b)   (no rescale)
//	rescale  : phase(Rescale(x)) - round(phase(x)/q_l)
//	rotate   : phase(Rotate(x,k)) - sigma_k(phase(x))
//
// "sub" statistics are restricted to the 2*slots coefficients that
// ckks.Encoder.Decode reads (indices that are multiples of N/(2*slots)).
// Measurements on one random instance; not worst-case bounds.

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"os"
	"testing"

	"github.com/seunghu-kim/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

var auditProfiles = []struct {
	name string
	lit  ckks.ParametersLiteral
}{
	{"LR-runner", ckks.ParametersLiteral{LogN: 15, LogSlots: 10, Scale: 1 << 40, Sigma: rlwe.DefaultSigma,
		Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001, 0x10000290001, 0xffffc40001, 0x100003e0001, 0x10000470001, 0x100004b0001, 0xffffb20001},
		P: []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001}}},
	{"LR-D60", ckks.ParametersLiteral{LogN: 15, LogSlots: 10, Scale: 1 << 60, Sigma: rlwe.DefaultSigma,
		Q: []uint64{0x1fffffffffc80001, 0xffffffffffc0001, 0x10000000001d0001, 0x10000000006e0001, 0xfffffffff840001, 0x1000000000860001, 0x1000000000870001, 0x1000000000930001, 0xfffffffff6a0001},
		P: []uint64{0x1fffffffffe10001, 0x1fffffffffe00001, 0x1fffffffffdd0001}}},
	{"MNIST-runner", ckks.ParametersLiteral{LogN: 14, LogSlots: 13, Scale: 1 << 47, Sigma: rlwe.DefaultSigma,
		Q: []uint64{0x2000000002b0001, 0x800000020001, 0x800000280001, 0x800000520001, 0x800000770001, 0x800000aa0001, 0x800000ad0001},
		P: []uint64{0x800000df0001, 0x800000f80001}}},
	{"MNIST-N15-D60", ckks.ParametersLiteral{LogN: 15, LogSlots: 13, Scale: 1 << 60, Sigma: rlwe.DefaultSigma,
		Q: []uint64{0x1fffffffffdd0001, 0xffffffffffc0001, 0x10000000001d0001, 0x10000000006e0001, 0xfffffffff840001, 0x1000000000860001, 0x1000000000870001},
		P: []uint64{0x1fffffffffe10001, 0x1fffffffffe00001}}},
}

// centeredBig returns the coefficients of p (rows 0..level) as centered integers mod Q_level.
func centeredBig(ringQ *ring.Ring, level int, p *ring.Poly) []*big.Int {
	Q := big.NewInt(1)
	for i := 0; i <= level; i++ {
		Q.Mul(Q, new(big.Int).SetUint64(ringQ.Modulus[i]))
	}
	half := new(big.Int).Rsh(Q, 1)
	crt := make([]*big.Int, level+1)
	for i := 0; i <= level; i++ {
		qi := new(big.Int).SetUint64(ringQ.Modulus[i])
		Qi := new(big.Int).Quo(Q, qi)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(Qi, qi), qi)
		crt[i] = inv.Mul(inv, Qi)
	}
	out := make([]*big.Int, ringQ.N)
	term := new(big.Int)
	for j := range out {
		acc := new(big.Int)
		for i := 0; i <= level; i++ {
			term.SetUint64(p.Coeffs[i][j] % ringQ.Modulus[i])
			term.Mul(term, crt[i])
			acc.Add(acc, term)
		}
		acc.Mod(acc, Q)
		if acc.Cmp(half) > 0 {
			acc.Sub(acc, Q)
		}
		out[j] = acc
	}
	return out
}

type cstat struct{ max, l2, rms, maxSub, l2Sub float64 } // log2 of |.|

func (s cstat) String() string {
	return fmt.Sprintf("max=2^%.2f l2=2^%.2f rms=2^%.2f | sub: max=2^%.2f l2=2^%.2f", s.max, s.l2, s.rms, s.maxSub, s.l2Sub)
}

func alg(x float64) float64 {
	if x == 0 {
		return math.Inf(-1)
	}
	return math.Log2(x)
}

func statsBig(v []*big.Int, gap int) cstat {
	var mx, ss, mxs, sss float64
	for j, x := range v {
		f, _ := new(big.Float).SetInt(x).Float64()
		f = math.Abs(f)
		mx = math.Max(mx, f)
		ss += f * f
		if j%gap == 0 {
			mxs = math.Max(mxs, f)
			sss += f * f
		}
	}
	return cstat{alg(mx), alg(math.Sqrt(ss)), alg(math.Sqrt(ss / float64(len(v)))), alg(mxs), alg(math.Sqrt(sss))}
}

func subBig(a, b []*big.Int) []*big.Int {
	out := make([]*big.Int, len(a))
	for i := range a {
		out[i] = new(big.Int).Sub(a[i], b[i])
	}
	return out
}

// automorphism X -> X^g on centered integer coefficients (negacyclic).
func permuteBig(v []*big.Int, g uint64) []*big.Int {
	N := uint64(len(v))
	out := make([]*big.Int, N)
	for i := uint64(0); i < N; i++ {
		e := i * g
		idx := e & (N - 1)
		if (e/N)&1 == 1 {
			out[idx] = new(big.Int).Neg(v[i])
		} else {
			out[idx] = new(big.Int).Set(v[i])
		}
	}
	return out
}

func phaseAll(p Parameters, ct *Ciphertext, sks *mkrlwe.SecretKeySet) *ring.Poly {
	return NewDecryptor(p).DecryptToPtxt(ct, sks)
}

func TestParamAuditOps(t *testing.T) {
	if os.Getenv("PARAM_AUDIT") != "1" {
		t.Skip("set PARAM_AUDIT=1")
	}
	rng := rand.New(rand.NewSource(1))
	const rot = 64 // a production LR rotation (1<<j)*SamplePad, CRS present by default
	t.Logf("CSV,profile,quantity,keys,level_in,level_out,logscale_in,logscale_out,max_log2,l2_log2,rms_log2,sub_max_log2,sub_l2_log2,note")
	for _, pr := range auditProfiles {
		cp, err := ckks.NewParametersFromLiteral(pr.lit)
		if err != nil {
			t.Fatal(err)
		}
		p := NewParameters(cp)
		ringQ := p.RingQ()
		N, slots := p.N(), p.Slots()
		gap := N / 2 / slots
		L := p.MaxLevel()
		row := func(q string, k, lin, lout int, sin, sout float64, s cstat, note string) {
			t.Logf("CSV,%s,%s,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%s", pr.name, q, k, lin, lout, alg(sin), alg(sout), s.max, s.l2, s.rms, s.maxSub, s.l2Sub, note)
		}

		ids := []string{"client", "owner0", "owner1", "owner2", "owner3", "owner4"}
		kgen := NewKeyGenerator(p)
		sks := map[string]*mkrlwe.SecretKey{}
		pks := map[string]*mkrlwe.PublicKey{}
		rlkSet := mkrlwe.NewRelinearizationKeySet(p.Parameters)
		rtkSet := mkrlwe.NewRotationKeySet()
		for _, id := range ids {
			sk, pk := kgen.GenKeyPair(id)
			sks[id], pks[id] = sk, pk
			rlkSet.AddRelinearizationKey(kgen.GenRelinearizationKey(sk))
			rtkSet.AddRotationKey(kgen.GenRotationKey(rot, sk))
		}
		skAll := func(k int) *mkrlwe.SecretKeySet {
			s := mkrlwe.NewSecretKeySet()
			for _, id := range ids[:k] {
				s.AddSecretKey(sks[id])
			}
			return s
		}

		msg := func() []complex128 {
			v := make([]complex128, slots)
			for i := range v {
				v[i] = complex(2*rng.Float64()-1, 0)
			}
			return v
		}
		enc := NewEncryptor(p)
		eval := NewEvaluator(p)
		encoder := ckks.NewEncoder(cp)
		bigEnc := ckks.NewEncoderBigComplex(cp, 256)

		// --- encoding: float64 vs 256-bit, at the top level and default scale
		m1 := msg()
		pt64 := ckks.NewPlaintext(cp, L, p.Scale())
		encoder.Encode(pt64, m1, p.LogSlots())
		ptB := ckks.NewPlaintext(cp, L, p.Scale())
		bv := make([]*ring.Complex, slots)
		for i, x := range m1 {
			bv[i] = ring.NewComplex(ring.NewFloat(real(x), 256), ring.NewFloat(imag(x), 256))
		}
		bigEnc.Encode(ptB, bv, p.LogSlots())
		row("encode_float64_minus_exact", 0, L, L, p.Scale(), p.Scale(),
			statsBig(subBig(centeredBig(ringQ, L, pt64.Value), centeredBig(ringQ, L, ptB.Value)), gap), "deterministic;|m|<=1 uniform")

		// --- fresh encryption error under one key (exact)
		fresh := func(id string, m []complex128) *Ciphertext {
			pt := ckks.NewPlaintext(cp, L, p.Scale())
			encoder.Encode(pt, m, p.LogSlots())
			idset := mkrlwe.NewIDSet()
			idset.Add(id)
			ct := NewCiphertext(p, idset, L, p.Scale())
			enc.EncryptPtxt(pt, pks[id], ct)
			s := mkrlwe.NewSecretKeySet()
			s.AddSecretKey(sks[id])
			e := subBig(centeredBig(ringQ, L, phaseAll(p, ct, s)), centeredBig(ringQ, L, pt.Value))
			if id == "client" {
				row("fresh_encryption", 1, L, L, p.Scale(), p.Scale(), statsBig(e, gap),
					fmt.Sprintf("analytic rms sigma*sqrt(1+N/2+N/2)=2^%.3f", alg(3.2*math.Sqrt(1+float64(N)))))
			}
			return ct
		}
		// k-key ciphertext: Enc_client(m) + sum_{i<k-1} Enc_owner_i(m_i)
		ctK := func(k int) *Ciphertext {
			ct := fresh("client", msg())
			for _, id := range ids[1:k] {
				ct = eval.AddNew(ct, fresh(id, msg()))
			}
			return ct
		}

		// --- relinearized product without rescale (production MulAndRelinHoisted)
		relin := func(a, b *Ciphertext, k int, note string) *Ciphertext {
			lvl := a.Level()
			out := eval.newCiphertextBinary(a, b)
			out.Scale = a.Scale * b.Scale
			var ha, hb *mkrlwe.HoistedCiphertext
			ha = eval.HoistedForm(a)
			if a == b {
				hb = ha
			} else {
				hb = eval.HoistedForm(b)
			}
			eval.ksw.MulAndRelinHoisted(a.Ciphertext, b.Ciphertext, ha, hb, rlkSet, out.Ciphertext)
			s := skAll(k)
			pa, pb := phaseAll(p, a, s), phaseAll(p, b, s)
			ringQ.MFormLvl(lvl, pa, pa)
			prod := ringQ.NewPolyLvl(lvl)
			ringQ.NTTLvl(lvl, pa, pa)
			ringQ.NTTLvl(lvl, pb, pb)
			ringQ.MulCoeffsMontgomeryLvl(lvl, pa, pb, prod)
			ringQ.InvNTTLvl(lvl, prod, prod)
			e := subBig(centeredBig(ringQ, lvl, phaseAll(p, out, s)), centeredBig(ringQ, lvl, prod))
			row("relin_added", k, lvl, out.Level(), a.Scale, out.Scale, statsBig(e, gap), note)
			return out
		}
		// --- rescale rounding error
		rescale := func(x *Ciphertext, k int, note string) *Ciphertext {
			lvl := x.Level()
			s := skAll(k)
			in := centeredBig(ringQ, lvl, phaseAll(p, x, s))
			out := x.CopyNew()
			if err := eval.Rescale(x, p.Scale(), out); err != nil {
				t.Fatal(err)
			}
			div := big.NewInt(1)
			for i := out.Level() + 1; i <= lvl; i++ {
				div.Mul(div, new(big.Int).SetUint64(ringQ.Modulus[i]))
			}
			halfDiv := new(big.Int).Rsh(div, 1)
			want := make([]*big.Int, N)
			for j, v := range in {
				// round(v/div), ties away from zero (exact)
				a := new(big.Int).Abs(v)
				a.Add(a, halfDiv)
				a.Quo(a, div)
				if v.Sign() < 0 {
					a.Neg(a)
				}
				want[j] = a
			}
			e := subBig(centeredBig(ringQ, out.Level(), phaseAll(p, out, s)), want)
			row("rescale_added", k, lvl, out.Level(), x.Scale, out.Scale, statsBig(e, gap),
				fmt.Sprintf("analytic rms sqrt((1+k*N/2)/12)=2^%.3f;%s", alg(math.Sqrt((1+float64(k*N)/2)/12)), note))
			return out
		}
		// --- rotation (non-hoisted Rotate and hoisted RotateHoisted)
		rotate := func(x *Ciphertext, k int, hoisted bool) {
			lvl := x.Level()
			s := skAll(k)
			var out *Ciphertext
			name := "rotate_added"
			if hoisted {
				out = eval.RotateHoistedNew(x, rot, eval.HoistedForm(x), rtkSet)
				name = "rotate_hoisted_added"
			} else {
				out = eval.RotateNew(x, rot, rtkSet)
			}
			g := p.GaloisElementForColumnRotationBy(rot)
			want := permuteBig(centeredBig(ringQ, lvl, phaseAll(p, x, s)), g)
			e := subBig(centeredBig(ringQ, lvl, phaseAll(p, out, s)), want)
			row(name, k, lvl, out.Level(), x.Scale, out.Scale, statsBig(e, gap), fmt.Sprintf("rot=%d", rot))
		}

		// 1-key x 1-key with different keys (LR: EncWeights(owner) x encTestData(client))
		a := fresh("client", msg())
		b := fresh("owner0", msg())
		prod := relin(a, b, 2, "client x owner0 at top level")
		rescale(prod, 2, "after product")
		for _, k := range []int{1, 2, 3, 6} {
			x := ctK(k)
			rotate(x, k, false)
			if k == 2 || k == 3 {
				rotate(x, k, true)
			}
			if k >= 2 {
				sq := relin(x, x, k, "square at top level")
				rescale(sq, k, "after square")
			}
		}
		// rescale of a MultByConst(1/5) output (LR applyLogistic path): exact constant quantization
		x := ctK(2)
		c := 1.0 / 5
		// production: scale = float64(q_l); integer K = floor(float64(c*scale) + 0.5) (ckks.scaleUpExact)
		qlInt := ringQ.Modulus[x.Level()]
		Kf := new(big.Float).SetPrec(200).SetFloat64(c * float64(qlInt))
		Kf.Add(Kf, big.NewFloat(0.5))
		K, _ := Kf.Int(nil)
		eff := new(big.Float).SetPrec(200).Quo(new(big.Float).SetPrec(200).SetInt(K), new(big.Float).SetPrec(200).SetUint64(qlInt))
		effF, _ := eff.Float64()
		qExact := new(big.Float).SetPrec(200).SetUint64(qlInt)
		qFlt := new(big.Float).SetPrec(200).SetFloat64(float64(qlInt))
		rel := new(big.Float).SetPrec(200).Quo(new(big.Float).SetPrec(200).Sub(qFlt, qExact), qExact)
		relF, _ := rel.Float64()
		t.Logf("CONST %s: c=1/5 at level %d (q_l=%d): K=%s, K/q_l - c = %.3g (relative %.3g); float64(q_l)/q_l - 1 = %.3g (scale bookkeeping); deterministic",
			pr.name, x.Level(), qlInt, K.String(), effF-c, (effF-c)/c, relF)
		mc := x.CopyNew()
		eval.MultByConst(x, complex(c, 0), mc)
		rescale(mc, 2, "after MultByConst(1/5)")
	}
}
