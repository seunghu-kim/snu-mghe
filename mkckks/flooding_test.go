package mkckks

// Distributed-decryption correctness and flooding-noise diagnostics.
//
// These are EMPIRICAL correctness measurements. They are not evidence of any
// formal security property: they show how much flooding noise each parameter
// set tolerates before decoded values lose precision or the phase wraps
// around the active modulus.

import (
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/seunghu-kim/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// CKKS parameter literals copied verbatim from the experiment runners.
var runnerParams = map[string]ckks.ParametersLiteral{
	// logistic_regression/go_inference/experiments/runner.go (LogSlots there
	// depends on the dataset; 13 = heart_disease 16x64 -> 2^10, wdbc 2^12, pima 2^11;
	// 13 is used here only for the synthetic circuit)
	"LR": {
		LogN: 15, LogSlots: 13,
		Q: []uint64{0x4000000120001, 0x10000140001, 0xffffe80001,
			0x10000290001, 0xffffc40001, 0x100003e0001,
			0x10000470001, 0x100004b0001, 0xffffb20001},
		P:     []uint64{0x40000001b0001, 0x3ffffffdf0001, 0x4000000270001},
		Scale: 1 << 40, Sigma: rlwe.DefaultSigma,
	},
	// cnn/mnist/go_inference/experiment/runner.go
	"MNIST": {
		LogN: 14, LogSlots: 13,
		Q: []uint64{0x2000000002b0001, 0x800000020001, 0x800000280001,
			0x800000520001, 0x800000770001, 0x800000aa0001, 0x800000ad0001},
		P:     []uint64{0x800000df0001, 0x800000f80001},
		Scale: 1 << 47, Sigma: rlwe.DefaultSigma,
	},
	// cnn/fmnist/go_inference/experiement/runner.go
	"FMNIST": {
		LogN: 14, LogSlots: 13,
		Q: []uint64{0x200000008001, 0x400018001, 0x3fffd0001, 0x400060001,
			0x400068001, 0x3fff90001, 0x400080001, 0x4000a8001, 0x400108001},
		P:     []uint64{0x7fffffd8001, 0x7fffffc8001},
		Scale: 1 << 34, Sigma: rlwe.DefaultSigma,
	},
	// cnn/cifar10 and cnn/svhn runner.go (identical)
	"CIFAR10/SVHN": {
		LogN: 16, LogSlots: 15,
		Q: []uint64{0x80000000080001, 0x2000000a0001, 0x2000000e0001, 0x1fffffc20001,
			0x200000440001, 0x200000500001, 0x200000620001, 0x1fffff980001,
			0x2000006a0001, 0x1fffff7e0001, 0x200000860001, 0x200000a60001},
		P:     []uint64{0x80000000440001, 0x7fffffffba0001, 0x80000000500001, 0x7fffffffaa0001},
		Scale: 1 << 45, Sigma: rlwe.DefaultSigma,
	},
}

// Candidate (NOT production) parameter sets for the feasibility study:
// larger scales so that heavier flooding leaves usable precision.
var candidateParams = func() map[string]ckks.ParametersLiteral {
	n2 := 1 << 16 // 2N for LogN = 15
	q60 := ring.GenerateNTTPrimesQ(60, n2, 8)
	p61 := ring.GenerateNTTPrimesP(61, n2, 5)
	q40 := ring.GenerateNTTPrimesQ(40, n2, 8)
	return map[string]ckks.ParametersLiteral{
		// one 60-bit prime per level, Delta = 2^60; logQP ~ 61+480+183 = 724
		"CAND-N15-D60": {LogN: 15, LogSlots: 14, Q: append([]uint64{p61[0]}, q60...), P: p61[1:4], Scale: 1 << 60, Sigma: rlwe.DefaultSigma},
		// two 40-bit primes per rescale, Delta = 2^80, base modulus 2x61 bits; logQP ~ 122+320+183 = 625
		"CAND-N15-D80": {LogN: 15, LogSlots: 14, Q: append([]uint64{p61[0], p61[4]}, q40...), P: p61[1:4], Scale: 1 << 80, Sigma: rlwe.DefaultSigma},
	}
}()

func mkParamsFrom(t testing.TB, name string) Parameters {
	t.Helper()
	lit, ok := runnerParams[name]
	if !ok {
		lit = candidateParams[name]
	}
	p, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return NewParameters(p)
}

// TestRunnerParameterReport prints the static facts of every runner parameter set.
func TestRunnerParameterReport(t *testing.T) {
	for _, name := range []string{"LR", "MNIST", "FMNIST", "CIFAR10/SVHN"} {
		lit := runnerParams[name]
		p, err := ckks.NewParametersFromLiteral(lit)
		if err != nil {
			t.Errorf("%s: lattigo rejects literal: %v", name, err)
			continue
		}
		qs := make([]string, len(lit.Q))
		for i, q := range lit.Q {
			qs[i] = fmt.Sprintf("%.1f", math.Log2(float64(q)))
		}
		ps := make([]string, len(lit.P))
		for i, q := range lit.P {
			ps[i] = fmt.Sprintf("%.1f", math.Log2(float64(q)))
		}
		lq, lp := 0.0, 0.0
		for _, q := range lit.Q {
			lq += math.Log2(float64(q))
		}
		for _, q := range lit.P {
			lp += math.Log2(float64(q))
		}
		t.Logf("%-13s logN=%d logQ=%.2f logP=%.2f logQP=%.2f maxLevel=%d logScale=%.0f\n    Q bits=[%s]\n    P bits=[%s]",
			name, p.LogN(), lq, lp, lq+lp, p.MaxLevel(), math.Log2(p.Scale()),
			strings.Join(qs, " "), strings.Join(ps, " "))
	}
}

// ---------------------------------------------------------------------------

type mkFixture struct {
	params Parameters
	ids    []string // "client", owner0..owner{n-1}
	sks    map[string]*mkrlwe.SecretKey
	pks    map[string]*mkrlwe.PublicKey
	rlkSet *mkrlwe.RelinearizationKeySet
	enc    *Encryptor
	eval   *Evaluator
}

func newMKFixture(t testing.TB, params Parameters, nOwners int) *mkFixture {
	f := &mkFixture{params: params, sks: map[string]*mkrlwe.SecretKey{}, pks: map[string]*mkrlwe.PublicKey{}}
	f.ids = append(f.ids, "client")
	for i := 0; i < nOwners; i++ {
		f.ids = append(f.ids, fmt.Sprintf("owner%d", i))
	}
	kgen := NewKeyGenerator(params)
	f.rlkSet = mkrlwe.NewRelinearizationKeySet(params.Parameters)
	for _, id := range f.ids { // independent keys per party
		sk, pk := kgen.GenKeyPair(id)
		f.sks[id], f.pks[id] = sk, pk
		f.rlkSet.AddRelinearizationKey(kgen.GenRelinearizationKey(sk))
	}
	f.enc = NewEncryptor(params)
	f.eval = NewEvaluator(params)
	return f
}

func randomMsg(params Parameters, r *rand.Rand) *Message {
	m := NewMessage(params)
	for i := range m.Value {
		m.Value[i] = complex(2*r.Float64()-1, 0)
	}
	return m
}

// circuit: y = sum_i x * w_i (client input x, owner models w_i), rescale,
// z = y^2, rescale. Multiplicative depth 2, every party's key involved.
func (f *mkFixture) circuit(t testing.TB, r *rand.Rand) (*Ciphertext, []complex128) {
	x := randomMsg(f.params, r)
	ctX := f.enc.EncryptMsgNew(x, f.pks["client"])
	want := make([]complex128, len(x.Value))
	var acc *Ciphertext
	for _, id := range f.ids[1:] {
		w := randomMsg(f.params, r)
		ctW := f.enc.EncryptMsgNew(w, f.pks[id])
		prod := f.eval.MulRelinNew(ctX, ctW, f.rlkSet)
		if acc == nil {
			acc = prod
		} else {
			acc = f.eval.AddNew(acc, prod)
		}
		for i := range want {
			want[i] += x.Value[i] * w.Value[i]
		}
	}
	if err := f.eval.Rescale(acc, f.params.Scale(), acc); err != nil {
		t.Fatal(err)
	}
	sq := f.eval.MulRelinNew(acc, acc, f.rlkSet)
	if err := f.eval.Rescale(sq, f.params.Scale(), sq); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		want[i] *= want[i]
	}
	return sq, want
}

// distributedDecrypt runs the protocol path: every owner produces a flooded
// share with its own decryptor, the CSP aggregates, the client decrypts the
// remaining client component with its own key.
func (f *mkFixture) distributedDecrypt(t testing.TB, ct *Ciphertext, cfg mkrlwe.FloodingConfig) (*Message, *ring.Poly) {
	shares := map[string]*ring.Poly{}
	for _, id := range f.ids[1:] {
		dec, err := NewDecryptorWithFlooding(f.params, cfg)
		if err != nil {
			t.Fatal(err)
		}
		shares[id] = dec.GenShare(ct, f.sks[id])
	}
	agg := ct.CopyNew()
	NewDecryptor(f.params).AggregateSharesAndDrop(agg, shares)
	if len(agg.Value) != 2 {
		t.Fatalf("after aggregation expected {0, client}, got %d components", len(agg.Value))
	}
	skC := mkrlwe.NewSecretKeySet()
	skC.AddSecretKey(f.sks["client"])
	clientDec := NewDecryptor(f.params)
	msg := clientDec.Decrypt(agg, skC)
	phase := clientDec.DecryptToPtxt(agg, skC)
	return msg, phase
}

// noiseFreeDecrypt is the INTERNAL diagnostic baseline: one decryptor holding
// every secret key computes the exact phase. Never reachable from the protocol.
func (f *mkFixture) noiseFreeDecrypt(ct *Ciphertext) (*Message, *ring.Poly) {
	all := mkrlwe.NewSecretKeySet()
	for _, id := range f.ids {
		all.AddSecretKey(f.sks[id])
	}
	d := NewDecryptor(f.params)
	return d.Decrypt(ct, all), d.DecryptToPtxt(ct, all)
}

func centered(ringQ *ring.Ring, level int, p *ring.Poly, j int, Q *big.Int) *big.Int {
	acc := new(big.Int)
	for i := 0; i <= level; i++ {
		qi := new(big.Int).SetUint64(ringQ.Modulus[i])
		Qi := new(big.Int).Quo(Q, qi)
		inv := new(big.Int).ModInverse(new(big.Int).Mod(Qi, qi), qi)
		term := new(big.Int).SetUint64(p.Coeffs[i][j] % ringQ.Modulus[i])
		term.Mul(term, inv).Mod(term, qi).Mul(term, Qi)
		acc.Add(acc, term)
	}
	acc.Mod(acc, Q)
	if acc.Cmp(new(big.Int).Rsh(Q, 1)) > 0 {
		acc.Sub(acc, Q)
	}
	return acc
}

func maxLog2Centered(ringQ *ring.Ring, level int, p *ring.Poly) float64 {
	Q := big.NewInt(1)
	for i := 0; i <= level; i++ {
		Q.Mul(Q, new(big.Int).SetUint64(ringQ.Modulus[i]))
	}
	m := 0
	for j := 0; j < ringQ.N; j++ {
		if b := new(big.Int).Abs(centered(ringQ, level, p, j, Q)).BitLen(); b > m {
			m = b
		}
	}
	return float64(m)
}

func logQLevel(params Parameters, level int) float64 {
	s := 0.0
	for i := 0; i <= level; i++ {
		s += math.Log2(float64(params.RingQ().Modulus[i]))
	}
	return s
}

type errStats struct {
	max, rms  float64
	nonFinite int
}

func slotErr(got, want []complex128, n int) errStats {
	var s errStats
	for i := 0; i < n; i++ {
		if cmplx.IsNaN(got[i]) || cmplx.IsInf(got[i]) {
			s.nonFinite++
			continue
		}
		e := cmplx.Abs(got[i] - want[i])
		s.max = math.Max(s.max, e)
		s.rms += e * e
	}
	s.rms = math.Sqrt(s.rms / float64(n))
	return s
}

func lg(x float64) float64 {
	if x == 0 {
		return math.Inf(-1)
	}
	return math.Log2(x)
}

// TestDistributedDecryptionCorrectness: n = 2 and n = 5 owners, independent
// keys, multikey products; flooded distributed decryption must match the
// expected values up to CKKS precision.
func TestDistributedDecryptionCorrectness(t *testing.T) {
	params := mkParamsFrom(t, "MNIST")
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{2, 5} {
		f := newMKFixture(t, params, n)
		ct, want := f.circuit(t, r)
		if got := len(ct.Value); got != n+2 {
			t.Fatalf("n=%d: ciphertext has %d components, want %d", n, got, n+2)
		}
		for _, cfg := range []mkrlwe.FloodingConfig{mkrlwe.LegacyFloodingConfig(), mkrlwe.WideGaussianFloodingConfig(20, 6)} {
			msg, _ := f.distributedDecrypt(t, ct, cfg)
			e := slotErr(msg.Value, want, params.Slots())
			t.Logf("n=%d %v: level=%d max err=2^%.1f rms=2^%.1f", n, cfg, ct.Level(), lg(e.max), lg(e.rms))
			if e.nonFinite > 0 || e.max > 1e-3 {
				t.Errorf("n=%d %v: max err %.3g (non-finite %d)", n, cfg, e.max, e.nonFinite)
			}
		}
		// share aggregation is order-independent and drops exactly the owner components
		base, _ := f.noiseFreeDecrypt(ct)
		e := slotErr(base.Value, want, params.Slots())
		if e.max > 1e-3 {
			t.Errorf("n=%d noise-free baseline err %.3g", n, e.max)
		}
	}
}

// TestFloodingNoiseSweep is the diagnostic sweep (B). It is skipped unless
// FLOOD_SWEEP=1 because it is slow for LogN=16. Set FLOOD_SWEEP_PARAMS to a
// comma-separated subset of LR,MNIST,FMNIST,CIFAR10/SVHN.
func TestFloodingNoiseSweep(t *testing.T) {
	if os.Getenv("FLOOD_SWEEP") != "1" {
		t.Skip("set FLOOD_SWEEP=1")
	}
	names := []string{"LR", "MNIST", "FMNIST"}
	if v := os.Getenv("FLOOD_SWEEP_PARAMS"); v != "" {
		names = strings.Split(v, ",")
	}
	nOwners := 2
	r := rand.New(rand.NewSource(7))
	for _, name := range names {
		params := mkParamsFrom(t, name)
		f := newMKFixture(t, params, nOwners)
		ct0, want := f.circuit(t, r)
		for _, drop := range []bool{false, true} {
			ct := ct0
			if drop { // worst case: decrypt at level 0 (only q0 left)
				ct = f.eval.DropLevelNew(ct0, ct0.Level())
			}
			level := ct.Level()
			ringQ := params.RingQ()
			_, basePhase := f.noiseFreeDecrypt(ct)
			baseMsg, _ := f.noiseFreeDecrypt(ct)
			be := slotErr(baseMsg.Value, want, params.Slots())
			t.Logf("== %s n=%d level=%d logScale=%.2f logQ_level=%.1f | no-flood baseline: max=2^%.1f rms=2^%.1f phase=2^%.0f",
				name, nOwners, level, math.Log2(ct.Scale), logQLevel(params, level), lg(be.max), lg(be.rms), maxLog2Centered(ringQ, level, basePhase))

			cfgs := []mkrlwe.FloodingConfig{mkrlwe.LegacyFloodingConfig()}
			for _, ls := range []float64{10, 20, 25, 30, 35, 40, 45, 49} {
				cfgs = append(cfgs, mkrlwe.WideGaussianFloodingConfig(ls, 6))
			}
			if strings.HasPrefix(name, "CAND") { // exact uniform reaches beyond the 2^52 Gaussian bound
				cfgs = append(cfgs, mkrlwe.UniformFloodingConfig(55), mkrlwe.UniformFloodingConfig(61))
			}
			for _, cfg := range cfgs {
				if err := cfg.Validate(params.Parameters); err != nil {
					t.Logf("   %-48v rejected: %v", cfg, err)
					continue
				}
				head, herr := cfg.CheckHeadroom(params.Parameters, level, nOwners, maxLog2Centered(ringQ, level, basePhase))
				msg, phase := f.distributedDecrypt(t, ct, cfg)
				diff := ringQ.NewPoly()
				ringQ.SubLvl(level, phase, basePhase, diff)
				noiseBits := maxLog2Centered(ringQ, level, diff)
				e := slotErr(msg.Value, want, params.Slots())
				eb := slotErr(msg.Value, baseMsg.Value, params.Slots())
				wrap := ""
				if herr != nil {
					wrap = "  WRAP-RISK(" + fmt.Sprintf("%.1f", head) + " bits)"
				}
				t.Logf("   %-48v noise=2^%-4.0f phase=2^%-4.0f | vs exact max=2^%-6.1f rms=2^%-6.1f | vs no-flood max=2^%-6.1f nonfinite=%d%s",
					cfg, noiseBits, maxLog2Centered(ringQ, level, phase), lg(e.max), lg(e.rms), lg(eb.max), e.nonFinite, wrap)
			}
		}
	}
}
