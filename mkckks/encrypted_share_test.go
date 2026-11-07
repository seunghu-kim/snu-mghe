package mkckks

// R1-3 v4: unit tests of the encrypted-share response
//   h_i0 = c_i*s_i + phi_i + r_i0,  h_i1 = r_i1,  (r_i0, r_i1) <- Enc_pkC(0)
// and of the CSP aggregation c0' = c0 + sum h_i0, cC' = cC + sum h_i1.
// These are implementation-correctness and regression tests. They are not
// evidence of any simulation or IND-CPA property.

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/seunghu-kim/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ring"
)

func polyEqualLvl(level int, a, b *ring.Poly) bool {
	for i := 0; i <= level; i++ {
		for j := range a.Coeffs[i] {
			if a.Coeffs[i][j] != b.Coeffs[i][j] {
				return false
			}
		}
	}
	return true
}

func cloneLvl(level int, p *ring.Poly) *ring.Poly {
	q := &ring.Poly{Coeffs: make([][]uint64, level+1), IsNTT: p.IsNTT, IsMForm: p.IsMForm}
	for i := 0; i <= level; i++ {
		q.Coeffs[i] = append([]uint64(nil), p.Coeffs[i]...)
	}
	return q
}

func cloneCt(ct *Ciphertext) map[string]*ring.Poly {
	m := map[string]*ring.Poly{}
	for id, p := range ct.Value {
		m[id] = cloneLvl(ct.Level(), p)
	}
	return m
}

// phaseUnder returns p0 + p1*s for the two-component ciphertext (p0, p1) of
// party sk.ID, computed by the library decryptor (coefficient domain).
func (f *mkFixture) phaseUnder(level int, p0, p1 *ring.Poly, sk *mkrlwe.SecretKey) *ring.Poly {
	ids := mkrlwe.NewIDSet()
	ids.Add(sk.ID)
	tmp := NewCiphertext(f.params, ids, level, f.params.Scale())
	ring.CopyValuesLvl(level, p0, tmp.Value["0"])
	ring.CopyValuesLvl(level, p1, tmp.Value[sk.ID])
	tmp.Value["0"].IsNTT, tmp.Value[sk.ID].IsNTT = p0.IsNTT, p1.IsNTT
	skSet := mkrlwe.NewSecretKeySet()
	skSet.AddSecretKey(sk)
	return NewDecryptor(f.params).DecryptToPtxt(tmp, skSet)
}

func TestEncryptedShareIdentityAndCorrectness(t *testing.T) {
	for _, n := range []int{2, 5} {
		f := newMKFixture(t, mkParamsFrom(t, "MNIST"), n)
		ringQ := f.params.RingQ()
		ct, want := f.circuit(t, rand.New(rand.NewSource(int64(n))))
		level := ct.Level()
		before := cloneCt(ct)
		cfg := mkrlwe.LegacyFloodingConfig()
		pkC, skC := f.pks["client"], f.sks["client"]

		shares := map[string]*mkrlwe.EncryptedShare{}
		sumD := ringQ.NewPolyLvl(level)
		sumEps := ringQ.NewPolyLvl(level)
		maxEps := 0.0
		for _, id := range f.ids[1:] {
			dec, err := NewDecryptorWithFlooding(f.params, cfg)
			if err != nil {
				t.Fatal(err)
			}
			d := dec.GenShare(ct, f.sks[id])
			dCopy := cloneLvl(level, d)
			sh, err := dec.EncryptShareForClient(ct, d, pkC)
			if err != nil {
				t.Fatal(err)
			}
			if sh.Level() != level || len(sh.H1.Coeffs) != level+1 {
				t.Fatalf("share rows %d/%d, want %d", len(sh.H0.Coeffs), len(sh.H1.Coeffs), level+1)
			}
			if sh.H0.IsNTT != ct.Value["0"].IsNTT || sh.H1.IsNTT != ct.Value["0"].IsNTT || sh.H0.IsMForm || sh.H1.IsMForm {
				t.Fatal("share domain differs from ciphertext domain")
			}
			// eps_zero_i = r_i0 + r_i1*sC with r_i0 = h_i0 - d_i, r_i1 = h_i1
			r0 := ringQ.NewPolyLvl(level)
			ringQ.SubLvl(level, sh.H0, dCopy, r0)
			eps := f.phaseUnder(level, r0, sh.H1, skC)
			if b := maxLog2Centered(ringQ, level, eps); b > maxEps {
				maxEps = b
			}
			ringQ.AddLvl(level, sumD, dCopy, sumD)
			ringQ.AddLvl(level, sumEps, eps, sumEps)
			shares[id] = sh
		}

		out, err := NewDecryptor(f.params).AggregateEncryptedShares(ct, shares, "client")
		if err != nil {
			t.Fatal(err)
		}
		if !polyEqualLvl(level, before["0"], ct.Value["0"]) || len(ct.Value) != n+2 {
			t.Fatal("input ciphertext was modified")
		}
		for id, p := range before {
			if !polyEqualLvl(level, p, ct.Value[id]) {
				t.Fatalf("input component %s modified", id)
			}
		}
		if len(out.Value) != 2 || out.Value["client"] == nil || out.Level() != level || out.Scale != ct.Scale {
			t.Fatalf("aggregate has %d components, level %d", len(out.Value), out.Level())
		}
		// cC is preserved: cC' - cC = sum h_i1 (not cC' = sum h_i1)
		sumH1 := ringQ.NewPolyLvl(level)
		for _, sh := range shares {
			ringQ.AddLvl(level, sumH1, sh.H1, sumH1)
		}
		diffC := ringQ.NewPolyLvl(level)
		ringQ.SubLvl(level, out.Value["client"], ct.Value["client"], diffC)
		if !polyEqualLvl(level, diffC, sumH1) || polyEqualLvl(level, out.Value["client"], sumH1) {
			t.Fatal("client component not preserved")
		}

		// exact ring identity:
		// c0' + cC'*sC == (c0 + cC*sC) + sum_i d_i + sum_i eps_zero_i  (mod Q_l)
		lhs := f.phaseUnder(level, out.Value["0"], out.Value["client"], skC)
		base := f.phaseUnder(level, ct.Value["0"], ct.Value["client"], skC)
		rhs := ringQ.NewPolyLvl(level)
		ringQ.AddLvl(level, base, sumD, rhs)
		ringQ.AddLvl(level, rhs, sumEps, rhs)
		if !polyEqualLvl(level, lhs, rhs) {
			t.Fatal("ring identity violated")
		}
		// sum_i d_i - sum_i c_i*s_i = sum_i phi_i (full multi-key phase as reference)
		_, full := f.noiseFreeDecrypt(ct)
		sumPhi := ringQ.NewPolyLvl(level)
		ringQ.AddLvl(level, base, sumD, sumPhi)
		ringQ.SubLvl(level, sumPhi, full, sumPhi)
		phiBits := maxLog2Centered(ringQ, level, sumPhi)
		if phiBits > lg(float64(uint64(n)*cfg.Bound))+1 {
			t.Fatalf("sum phi has %v bits, bound n*B = %d", phiBits, uint64(n)*cfg.Bound)
		}
		if maxEps > 30 {
			t.Fatalf("zero-encryption phase has %v bits: not an encryption of 0 under pkC", maxEps)
		}

		// client decryption with its own key only
		skSet := mkrlwe.NewSecretKeySet()
		skSet.AddSecretKey(skC)
		got := NewDecryptor(f.params).Decrypt(out, skSet)
		rawMsg, _ := f.distributedDecrypt(t, ct, cfg)
		eEnc, eRaw := slotErr(got.Value, want, len(want)), slotErr(rawMsg.Value, want, len(want))
		t.Logf("n=%d level=%d: |sum phi|<=2^%.0f, max_i |eps_zero_i|<=2^%.0f, |sum eps|<=2^%.0f; slot max err enc=2^%.1f raw=2^%.1f",
			n, level, phiBits, maxEps, maxLog2Centered(ringQ, level, sumEps), lg(eEnc.max), lg(eRaw.max))
		if eEnc.nonFinite > 0 || eEnc.max > 1e-6 {
			t.Fatalf("client decryption error %v", eEnc.max)
		}
	}
}

func TestEncryptedShareFreshness(t *testing.T) {
	f := newMKFixture(t, mkParamsFrom(t, "MNIST"), 2)
	ct, _ := f.circuit(t, rand.New(rand.NewSource(7)))
	level := ct.Level()
	dec, err := NewDecryptorWithFlooding(f.params, mkrlwe.LegacyFloodingConfig())
	if err != nil {
		t.Fatal(err)
	}
	// same d_i twice: the zero encryptions must differ
	d := dec.GenShare(ct, f.sks["owner0"])
	d2 := cloneLvl(level, d)
	a, err := dec.EncryptShareForClient(ct, d, f.pks["client"])
	if err != nil {
		t.Fatal(err)
	}
	b, err := dec.EncryptShareForClient(ct, d2, f.pks["client"])
	if err != nil {
		t.Fatal(err)
	}
	if polyEqualLvl(level, a.H1, b.H1) || polyEqualLvl(level, a.H0, b.H0) {
		t.Fatal("zero encryption reused for the same share")
	}
	// two full requests
	c1, _ := dec.GenEncryptedShare(ct, f.sks["owner0"], f.pks["client"])
	c2, _ := dec.GenEncryptedShare(ct, f.sks["owner0"], f.pks["client"])
	if polyEqualLvl(level, c1.H1, c2.H1) || polyEqualLvl(level, c1.H0, c2.H0) {
		t.Fatal("two requests produced identical responses")
	}
}

func TestEncryptedShareRejects(t *testing.T) {
	f := newMKFixture(t, mkParamsFrom(t, "MNIST"), 2)
	ct, _ := f.circuit(t, rand.New(rand.NewSource(9)))
	dec, _ := NewDecryptorWithFlooding(f.params, mkrlwe.LegacyFloodingConfig())
	if _, err := NewDecryptor(f.params).GenEncryptedShare(ct, f.sks["owner0"], f.pks["client"]); err == nil {
		t.Error("decryptor without flooding accepted")
	}
	if _, err := dec.GenEncryptedShare(ct, f.sks["owner0"], f.pks["owner0"]); err == nil {
		t.Error("client key equal to owner key accepted")
	}
	stranger, pkX := NewKeyGenerator(f.params).GenKeyPair("stranger")
	_ = stranger
	if _, err := dec.GenEncryptedShare(ct, f.sks["owner0"], pkX); err == nil {
		t.Error("client key absent from ciphertext accepted")
	}
	s0, _ := dec.GenEncryptedShare(ct, f.sks["owner0"], f.pks["client"])
	s1, _ := dec.GenEncryptedShare(ct, f.sks["owner1"], f.pks["client"])
	agg := NewDecryptor(f.params)
	if _, err := agg.AggregateEncryptedShares(ct, map[string]*mkrlwe.EncryptedShare{"owner0": s0}, "client"); err == nil {
		t.Error("missing owner share accepted")
	}
	if _, err := agg.AggregateEncryptedShares(ct, map[string]*mkrlwe.EncryptedShare{"owner0": s0, "owner1": s1, "client": s1}, "client"); err == nil {
		t.Error("share for the client component accepted")
	}
	short := &mkrlwe.EncryptedShare{H0: &ring.Poly{Coeffs: s1.H0.Coeffs[:ct.Level()]}, H1: &ring.Poly{Coeffs: s1.H1.Coeffs[:ct.Level()]}}
	if _, err := agg.AggregateEncryptedShares(ct, map[string]*mkrlwe.EncryptedShare{"owner0": s0, "owner1": short}, "client"); err == nil {
		t.Error("share at wrong level accepted")
	}
	if _, err := agg.AggregateEncryptedShares(ct, map[string]*mkrlwe.EncryptedShare{"owner0": s0, "owner1": s1}, "owner0"); err == nil {
		t.Error("aggregation for a non-client ID accepted")
	}
}

// Concurrent owner requests, each with its own decryptor (as in the data-owner
// RPC handlers); run with -race to check that no sampler or buffer is shared.
func TestEncryptedShareConcurrent(t *testing.T) {
	f := newMKFixture(t, mkParamsFrom(t, "MNIST"), 2)
	ct, want := f.circuit(t, rand.New(rand.NewSource(11)))
	var wg sync.WaitGroup
	var mu sync.Mutex
	shares := map[string]*mkrlwe.EncryptedShare{}
	for _, id := range f.ids[1:] {
		for k := 0; k < 2; k++ {
			wg.Add(1)
			go func(id string, k int) {
				defer wg.Done()
				dec, err := NewDecryptorWithFlooding(f.params, mkrlwe.LegacyFloodingConfig())
				if err != nil {
					t.Error(err)
					return
				}
				sh, err := dec.GenEncryptedShare(ct, f.sks[id], f.pks["client"])
				if err != nil {
					t.Error(err)
					return
				}
				if k == 0 {
					mu.Lock()
					shares[id] = sh
					mu.Unlock()
				}
			}(id, k)
		}
	}
	wg.Wait()
	out, err := NewDecryptor(f.params).AggregateEncryptedShares(ct, shares, "client")
	if err != nil {
		t.Fatal(err)
	}
	skSet := mkrlwe.NewSecretKeySet()
	skSet.AddSecretKey(f.sks["client"])
	if e := slotErr(NewDecryptor(f.params).Decrypt(out, skSet).Value, want, len(want)); e.max > 1e-6 {
		t.Fatalf("decryption error %v", e.max)
	}
}
