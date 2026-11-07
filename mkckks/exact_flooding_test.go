package mkckks

// R1-3 (final): encrypted-share reconstruction with the exact discrete
// Gaussian flooding (CKS20) on the test-only D60 parameters
// (CAND-N15-D60 = the LR-D60 primes). Implementation-correctness test only.

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/seunghu-kim/snu-mghe/mkrlwe"
)

func TestEncryptedShareExactGaussian(t *testing.T) {
	params := mkParamsFrom(t, "CAND-N15-D60")
	for _, tc := range []struct {
		n        int
		logSigma uint
	}{{2, 45}, {5, 45}, {2, 49}} {
		f := newMKFixture(t, params, tc.n)
		ringQ := params.RingQ()
		ct, want := f.circuit(t, rand.New(rand.NewSource(int64(tc.n))))
		level := ct.Level()
		cfg := mkrlwe.ExactDiscreteGaussianFloodingConfigPow2(tc.logSigma)
		pkC, skC := f.pks["client"], f.sks["client"]
		Q := big.NewInt(1)
		for i := 0; i <= level; i++ {
			Q.Mul(Q, new(big.Int).SetUint64(ringQ.Modulus[i]))
		}

		_, full := f.noiseFreeDecrypt(ct)
		shares := map[string]*mkrlwe.EncryptedShare{}
		sumPhi := make([]*big.Int, ringQ.N) // integer (non-modular) sums of the centered parts
		sumEps := make([]*big.Int, ringQ.N)
		for j := range sumPhi {
			sumPhi[j], sumEps[j] = new(big.Int), new(big.Int)
		}
		phiSS := 0.0
		for _, id := range f.ids[1:] {
			dec, err := NewDecryptorWithFlooding(params, cfg)
			if err != nil {
				t.Fatal(err)
			}
			d := dec.GenShare(ct, f.sks[id])
			dCopy := cloneLvl(level, d)
			sh, err := dec.EncryptShareForClient(ct, d, pkC)
			if err != nil {
				t.Fatal(err)
			}
			// phi_i = d_i - c_i s_i
			cs := ringQ.NewPolyLvl(level)
			ringQ.NTTLvl(level, ct.Value[id], cs)
			ringQ.MulCoeffsMontgomeryLvl(level, cs, f.sks[id].Value.Q, cs)
			ringQ.InvNTTLvl(level, cs, cs)
			phi := ringQ.NewPolyLvl(level)
			ringQ.SubLvl(level, dCopy, cs, phi)
			r0 := ringQ.NewPolyLvl(level)
			ringQ.SubLvl(level, sh.H0, dCopy, r0)
			eps := f.phaseUnder(level, r0, sh.H1, skC)
			for j := 0; j < ringQ.N; j++ {
				p := centered(ringQ, level, phi, j, Q)
				ff, _ := new(big.Float).SetInt(p).Float64()
				phiSS += ff * ff
				sumPhi[j].Add(sumPhi[j], p)
				sumEps[j].Add(sumEps[j], centered(ringQ, level, eps, j, Q))
			}
			shares[id] = sh
		}
		out, err := NewDecryptor(params).AggregateEncryptedShares(ct, shares, "client")
		if err != nil {
			t.Fatal(err)
		}
		skSet := mkrlwe.NewSecretKeySet()
		skSet.AddSecretKey(skC)
		phC := NewDecryptor(params).DecryptToPtxt(out, skSet)
		// integer identity without modular reduction: centered(phase_C) == centered(full) + sum phi + sum eps
		wraps, maxBits := 0, 0
		for j := 0; j < ringQ.N; j++ {
			sum := new(big.Int).Add(centered(ringQ, level, full, j, Q), sumPhi[j])
			sum.Add(sum, sumEps[j])
			if centered(ringQ, level, phC, j, Q).Cmp(sum) != 0 {
				wraps++
			}
			if b := sum.BitLen(); b > maxBits {
				maxBits = b
			}
		}
		got := NewDecryptor(params).Decrypt(out, skSet)
		e := slotErr(got.Value, want, len(want))
		vr := phiSS / float64(tc.n*ringQ.N) / math.Exp2(2*float64(tc.logSigma))
		t.Logf("n=%d sigma=2^%d level=%d logQ_l=%.1f: var(phi)/sigma^2=%.4f, max|phase_C|<2^%d, wrap-around coefficients=%d, slot err max=2^%.2f rms=2^%.2f",
			tc.n, tc.logSigma, level, logQLevel(params, level), vr, maxBits, wraps, lg(e.max), lg(e.rms))
		if wraps != 0 {
			t.Errorf("%d coefficients wrapped around modulo Q_l", wraps)
		}
		if math.Abs(vr-1) > 5*math.Sqrt(2/float64(tc.n*ringQ.N)) {
			t.Errorf("phi variance ratio %.4f", vr)
		}
		// flooding dominates: |slot error| rms ~= sigma*sqrt(n*N)/scale (coefficient
		// noise of variance n*sigma^2 through the canonical embedding)
		pred := math.Exp2(float64(tc.logSigma)) * math.Sqrt(float64(tc.n*ringQ.N)) / ct.Scale
		t.Logf("    slot err rms / (sigma*sqrt(nN)/scale) = %.3f", e.rms/pred)
		if e.nonFinite > 0 || e.rms > 1.2*pred || e.rms < 0.8*pred {
			t.Errorf("client decryption error rms %v, flooding prediction %v", e.rms, pred)
		}
	}
}

