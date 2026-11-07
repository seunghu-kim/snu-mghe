package mkckks

// R1-3 follow-up v2, Section 5.1/5.2: what can a SEMI-HONEST CSP compute from
// its own view in the raw-share distributed decryption?
//
// Run with: CSP_VIEW=1 go test -run TestCSPViewSeparability -v -timeout 0 -args -n=2
//
// The CSP distinguisher (cspGuess) is given ONLY values in the CSP's view:
// the owners' input ciphertexts, the client's input ciphertext, the public
// evaluation/rotation keys, the evaluated ciphertext and the owners' shares.
// It re-evaluates the public circuit on the owner-only branch and decodes
// c0(branch_i) + d_i. Secret keys, encryption coins and sampled flooding
// noise are used ONLY in the separate ground-truth harness (groundTruthErr).
//
// Diagnostic only: a "no distinguisher found" row is NOT a security proof.

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/seunghu-kim/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
)

// decodeAt decodes a coefficient-domain polynomial at (level, scale).
func decodeAt(params Parameters, p *ring.Poly, level int, scale float64) []complex128 {
	ckksParams, _ := ckks.NewParameters(params.Parameters.Parameters, params.LogSlots(), params.Scale())
	encd := ckks.NewEncoder(ckksParams)
	pt := ckks.NewPlaintext(ckksParams, level, scale)
	for i := 0; i <= level; i++ {
		copy(pt.Value.Coeffs[i], p.Coeffs[i])
	}
	pt.Value.IsNTT = false
	return encd.Decode(pt, params.LogSlots())
}

func constMsg(params Parameters, v float64) *Message {
	m := NewMessage(params)
	for i := range m.Value {
		m.Value[i] = complex(v, 0)
	}
	return m
}

// cspView: exactly the objects the CSP holds in the protocol (no secrets).
type cspView struct {
	ownerCt map[string]*Ciphertext // [[W_i]]_{D_i} (or the world-dependent value)
	biasCt  map[string]*Ciphertext // optional owner-encrypted bias
	ctX     *Ciphertext            // [[X]]_C
	final   *Ciphertext            // ciphertext sent to the owners
	shares  map[string]*ring.Poly  // d_i = c_i s_i + phi_i
}

type viewCase struct {
	name  string
	note  string
	build func(f *mkFixture, rtk *mkrlwe.RotationKeySet, worldMsg map[string]float64, x *Message, w map[string]*Message) (*cspView, error)
	// branch: the CSP's recomputation of owner i's owner-only branch, from public data only
	branch func(f *mkFixture, rtk *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext
}

func TestCSPViewSeparability(t *testing.T) {
	if os.Getenv("CSP_VIEW") != "1" {
		t.Skip("set CSP_VIEW=1")
	}
	params := mkParamsFrom(t, "LR")
	params.AddCRS(1)
	f := newMKFixture(t, params, 2)
	kgen := NewKeyGenerator(params)
	rtk := mkrlwe.NewRotationKeySet()
	for _, id := range f.ids {
		rtk.AddRotationKey(kgen.GenRotationKey(1, f.sks[id]))
	}
	o0, o1 := f.ids[1], f.ids[2]
	r := rand.New(rand.NewSource(7))
	x := randomMsg(params, r)
	w := map[string]*Message{o0: randomMsg(params, r), o1: randomMsg(params, r)}

	rescale := func(ct *Ciphertext) *Ciphertext {
		if err := f.eval.Rescale(ct, params.Scale(), ct); err != nil {
			t.Fatal(err)
		}
		return ct
	}

	cases := []viewCase{
		{
			name: "A owner-only sum",
			note: "final = [[m0]]_D0 + [[m1]]_D1 (no client term)",
			build: func(f *mkFixture, _ *mkrlwe.RotationKeySet, m map[string]float64, _ *Message, _ map[string]*Message) (*cspView, error) {
				v := &cspView{ownerCt: map[string]*Ciphertext{}}
				for _, id := range []string{o0, o1} {
					v.ownerCt[id] = f.enc.EncryptMsgNew(constMsg(params, m[id]), f.pks[id])
				}
				v.final = f.eval.AddNew(v.ownerCt[o0], v.ownerCt[o1])
				return v, nil
			},
			branch: func(_ *mkFixture, _ *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext { return v.ownerCt[id] },
		},
		{
			name: "B owner+client sum",
			note: "final = [[m0]]_D0 + [[m1]]_D1 + [[x]]_C",
			build: func(f *mkFixture, _ *mkrlwe.RotationKeySet, m map[string]float64, x *Message, _ map[string]*Message) (*cspView, error) {
				v := &cspView{ownerCt: map[string]*Ciphertext{}}
				for _, id := range []string{o0, o1} {
					v.ownerCt[id] = f.enc.EncryptMsgNew(constMsg(params, m[id]), f.pks[id])
				}
				v.ctX = f.enc.EncryptMsgNew(x, f.pks["client"])
				v.final = f.eval.AddNew(f.eval.AddNew(v.ownerCt[o0], v.ownerCt[o1]), v.ctX)
				return v, nil
			},
			branch: func(_ *mkFixture, _ *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext { return v.ownerCt[id] },
		},
		{
			name: "C affine: Rot+const",
			note: "final = sum_i Rot_1(3*[[m_i]]) + [[x]]_C",
			build: func(f *mkFixture, rtk *mkrlwe.RotationKeySet, m map[string]float64, x *Message, _ map[string]*Message) (*cspView, error) {
				v := &cspView{ownerCt: map[string]*Ciphertext{}}
				var acc *Ciphertext
				for _, id := range []string{o0, o1} {
					v.ownerCt[id] = f.enc.EncryptMsgNew(constMsg(params, m[id]), f.pks[id])
					b := v.ownerCt[id].CopyNew()
					f.eval.MultByConst(b, 3, b)
					b = f.eval.RotateNew(b, 1, rtk)
					if acc == nil {
						acc = b
					} else {
						acc = f.eval.AddNew(acc, b)
					}
				}
				v.ctX = f.enc.EncryptMsgNew(x, f.pks["client"])
				v.final = f.eval.AddNew(acc, v.ctX)
				return v, nil
			},
			branch: func(f *mkFixture, rtk *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext {
				b := v.ownerCt[id].CopyNew()
				f.eval.MultByConst(b, 3, b)
				return f.eval.RotateNew(b, 1, rtk)
			},
		},
		{
			name: "D product w_i*x",
			note: "final = sum_i Rescale([[m_i]]_Di * [[x]]_C), m_i = world constant",
			build: func(f *mkFixture, _ *mkrlwe.RotationKeySet, m map[string]float64, x *Message, _ map[string]*Message) (*cspView, error) {
				v := &cspView{ownerCt: map[string]*Ciphertext{}}
				v.ctX = f.enc.EncryptMsgNew(x, f.pks["client"])
				var acc *Ciphertext
				for _, id := range []string{o0, o1} {
					v.ownerCt[id] = f.enc.EncryptMsgNew(constMsg(params, m[id]), f.pks[id])
					p := rescale(f.eval.MulRelinNew(v.ownerCt[id], v.ctX, f.rlkSet))
					if acc == nil {
						acc = p
					} else {
						acc = f.eval.AddNew(acc, p)
					}
				}
				v.final = acc
				return v, nil
			},
			branch: func(f *mkFixture, _ *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext {
				return rescale(f.eval.MulRelinNew(v.ownerCt[id], v.ctX, f.rlkSet))
			},
		},
		{
			name: "E product + owner bias",
			note: "final = sum_i (Rescale([[w_i]]*[[x]]) + [[b_i]]_Di), b_i = world constant (MNIST-like final bias)",
			build: func(f *mkFixture, _ *mkrlwe.RotationKeySet, m map[string]float64, x *Message, w map[string]*Message) (*cspView, error) {
				v := &cspView{ownerCt: map[string]*Ciphertext{}, biasCt: map[string]*Ciphertext{}}
				v.ctX = f.enc.EncryptMsgNew(x, f.pks["client"])
				var acc *Ciphertext
				for _, id := range []string{o0, o1} {
					v.ownerCt[id] = f.enc.EncryptMsgNew(w[id], f.pks[id])
					v.biasCt[id] = f.enc.EncryptMsgNew(constMsg(params, m[id]), f.pks[id])
					p := rescale(f.eval.MulRelinNew(v.ownerCt[id], v.ctX, f.rlkSet))
					p = f.eval.AddNew(p, v.biasCt[id])
					if acc == nil {
						acc = p
					} else {
						acc = f.eval.AddNew(acc, p)
					}
				}
				v.final = acc
				return v, nil
			},
			// the only owner-only branch is the bias ciphertext itself
			branch: func(_ *mkFixture, _ *mkrlwe.RotationKeySet, v *cspView, id string) *Ciphertext { return v.biasCt[id] },
		},
	}

	cfgs := []mkrlwe.FloodingConfig{
		mkrlwe.LegacyFloodingConfig(),
		mkrlwe.WideGaussianFloodingConfig(30, 6),
		mkrlwe.WideGaussianFloodingConfig(40, 6),
		mkrlwe.WideGaussianFloodingConfig(45, 6),
	}
	trials := 12 // per world; CSP_VIEW_TRIALS overrides
	if v := os.Getenv("CSP_VIEW_TRIALS"); v != "" {
		fmt.Sscanf(v, "%d", &trials)
	}
	if only := os.Getenv("CSP_VIEW_CASES"); only != "" { // e.g. "D,E"
		var keep []viewCase
		for _, c := range cases {
			if strings.Contains(only, c.name[:1]) {
				keep = append(keep, c)
			}
		}
		cases = keep
	}
	if os.Getenv("CSP_VIEW_LEGACY_ONLY") == "1" {
		cfgs = cfgs[:1]
	}
	ringQ := params.RingQ()
	t.Logf("params LR: logN=%d logSlots=%d logScale=%.0f; owners %v; %d trials per world, fresh encryption + flooding randomness each trial",
		params.LogN(), params.LogSlots(), math.Log2(params.Scale()), []string{o0, o1}, trials)
	t.Logf("CSV,case,flooding,trials,csp_accuracy,mean_abs_est_err_log2,max_abs_est_mean,identity_ok")
	for _, c := range cases {
		for _, cfg := range cfgs {
			correct, total := 0, 0
			errSum, maxAbsMean := 0.0, 0.0
			identityOK := true
			for b := 0; b < 2; b++ {
				world := map[string]float64{o0: 0, o1: 1} // aggregate m0+m1 = 1 in both worlds
				if b == 1 {
					world = map[string]float64{o0: 1, o1: 0}
				}
				for tr := 0; tr < trials; tr++ {
					v, err := c.build(f, rtk, world, x, w)
					if err != nil {
						t.Fatal(err)
					}
					v.shares = map[string]*ring.Poly{}
					for _, id := range []string{o0, o1} {
						dec, err := NewDecryptorWithFlooding(params, cfg)
						if err != nil {
							t.Fatal(err)
						}
						v.shares[id] = dec.GenShare(v.final, f.sks[id])
					}
					// ---- CSP side: public data only ----
					guess, est := cspGuess(f, rtk, c, v, o0)
					if guess == b {
						correct++
					}
					total++
					// ---- ground truth (separate harness) ----
					errSum += math.Abs(est - world[o0])
					maxAbsMean = math.Max(maxAbsMean, math.Abs(est))
					if !groundTruthIdentity(f, c, rtk, v, o0) {
						identityOK = false
					}
				}
			}
			_ = ringQ
			t.Logf("CSV,%s,%v,%d,%.3f,%.1f,%.3g,%v", c.name, cfg, total, float64(correct)/float64(total),
				lg(errSum/float64(total)), maxAbsMean, identityOK)
		}
		t.Logf("   (%s: %s)", c.name, c.note)
	}
}

// cspGuess: CSP-side distinguisher. Decodes c0(branch) + d_i at the final
// level/scale and guesses world 1 (m_o0 = 1) if the slot mean exceeds 1/2.
func cspGuess(f *mkFixture, rtk *mkrlwe.RotationKeySet, c viewCase, v *cspView, id string) (int, float64) {
	br := c.branch(f, rtk, v, id)
	level := v.final.Level()
	u := f.params.RingQ().NewPolyLvl(level)
	f.params.RingQ().AddLvl(level, br.Value["0"], v.shares[id], u)
	vals := decodeAt(f.params, u, level, br.Scale)
	s := 0.0
	for _, z := range vals {
		s += real(z)
	}
	mean := s / float64(len(vals))
	if math.IsNaN(mean) || math.IsInf(mean, 0) {
		return 0, mean
	}
	if mean > 0.5 {
		return 1, mean
	}
	return 0, mean
}

// groundTruthIdentity (uses the owner's SECRET key; harness only) checks that
// owner id's component of the final ciphertext equals that of the CSP-recomputed
// branch, i.e. that c0(branch)+d_i = c0(branch)+c_i(branch)*s_i+phi_i holds.
func groundTruthIdentity(f *mkFixture, c viewCase, rtk *mkrlwe.RotationKeySet, v *cspView, id string) bool {
	br := c.branch(f, rtk, v, id)
	level := v.final.Level()
	a, b := v.final.Value[id], br.Value[id]
	for i := 0; i <= level; i++ {
		for j := range a.Coeffs[i] {
			if a.Coeffs[i][j] != b.Coeffs[i][j] {
				return false
			}
		}
	}
	return true
}

var _ = fmt.Sprintf
