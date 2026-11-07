package mkrlwe

import (
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

type Decryptor struct {
	params   Parameters
	ringQ    *ring.Ring
	pool     *ring.Poly
	sk       *SecretKey
	sampler  floodingSampler
	flooding FloodingConfig
}

// NewDecryptor instantiates a new generic RLWE Decryptor.
func NewDecryptor(params Parameters) *Decryptor {

	return &Decryptor{
		params: params,
		ringQ:  params.RingQ(),
		pool:   params.RingQ().NewPoly(),
	}
}

// NewDecryptorWithFlooding returns a Decryptor whose GenShare adds noise drawn
// from cfg. cfg is validated against params; noise-free configurations are rejected.
func NewDecryptorWithFlooding(params Parameters, cfg FloodingConfig) (*Decryptor, error) {
	sampler, err := newFloodingSampler(params, cfg)
	if err != nil {
		return nil, err
	}
	return &Decryptor{
		params:   params,
		ringQ:    params.RingQ(),
		pool:     params.RingQ().NewPoly(),
		sampler:  sampler,
		flooding: cfg,
	}, nil
}

// NewDecryptorWithGaussianNoise is kept for source compatibility with the
// experiment code. It uses the legacy Lattigo Gaussian sampler and panics if
// (sigma, bound) is outside the range that sampler represents correctly.
func NewDecryptorWithGaussianNoise(params Parameters, sigma float64, bound int) *Decryptor {
	if bound <= 0 {
		panic("mkrlwe: flooding bound must be positive")
	}
	dec, err := NewDecryptorWithFlooding(params, FloodingConfig{Distribution: FloodingGaussianLegacy, Sigma: sigma, Bound: uint64(bound)})
	if err != nil {
		panic(err)
	}
	return dec
}

// FloodingConfig returns the flooding configuration of this decryptor
// (zero value if it was built without flooding).
func (decryptor *Decryptor) FloodingConfig() FloodingConfig { return decryptor.flooding }

// PartialDecrypt partially decrypts the ct with single secretkey sk and update result inplace
func (decryptor *Decryptor) PartialDecrypt(ct *Ciphertext, sk *SecretKey) {
	ringQ := decryptor.ringQ
	id := sk.ID
	level := ct.Level()

	if !ct.Value[id].IsNTT {
		ringQ.NTTLvl(level, ct.Value[id], ct.Value[id])
	}

	ringQ.MulCoeffsMontgomeryLvl(level, ct.Value[id], sk.Value.Q, ct.Value[id])

	if !ct.Value[id].IsNTT {
		ringQ.InvNTTLvl(level, ct.Value[id], ct.Value[id])
	}

	ringQ.AddLvl(level, ct.Value["0"], ct.Value[id], ct.Value["0"])
	delete(ct.Value, id)
}

// GenShare computes the noisy partial decryption share c_id*s_id + e_flood for
// one party, in the same domain (NTT or coefficient) as ct.Value[sk.ID].
// It panics if the decryptor has no flooding sampler: noise-free shares are
// not available outside internal tests.
func (decryptor *Decryptor) GenShare(ct *Ciphertext, sk *SecretKey) *ring.Poly {
	if decryptor.sampler == nil {
		panic("mkrlwe: GenShare requires a decryptor built with NewDecryptorWithFlooding")
	}
	if _, ok := ct.Value[sk.ID]; !ok {
		panic("mkrlwe: ciphertext has no component for party " + sk.ID)
	}
	ringQ := decryptor.ringQ
	level := ct.Level()

	tmp := ringQ.NewPoly()
	if ct.Value[sk.ID].IsNTT {
		ringQ.MulCoeffsMontgomeryLvl(level, ct.Value[sk.ID], sk.Value.Q, tmp)
	} else {
		ringQ.NTTLvl(level, ct.Value[sk.ID], tmp)
		ringQ.MulCoeffsMontgomeryLvl(level, tmp, sk.Value.Q, tmp)
		ringQ.InvNTTLvl(level, tmp, tmp)
	}

	// coefficient-domain noise, identical integer on every RNS prime 0..level
	noise := ringQ.NewPoly()
	decryptor.sampler.readLvl(level, noise)

	if ct.Value[sk.ID].IsNTT {
		ringQ.NTTLvl(level, noise, noise)
	}

	ringQ.AddLvl(level, tmp, noise, tmp)

	return tmp
}

func (decryptor *Decryptor) AggregateSharesAndDrop(ct *Ciphertext, shares map[string]*ring.Poly) {
	ringQ := decryptor.ringQ
	level := ct.Level()

	agg := ringQ.NewPoly()
	for _, sh := range shares {
		ringQ.AddLvl(level, agg, sh, agg)
	}
	// add aggregated share to c0
	ringQ.AddLvl(level, ct.Value["0"], agg, ct.Value["0"])

	// drop only the terms for which we added c_i*s_i (i.e., keys in shares)
	for id := range shares {
		if id != "0" {
			delete(ct.Value, id)
		}
	}
}

// Decrypt decrypts the ciphertext with given secretkey set and write the result in ptOut.
// The level of the output plaintext is min(ciphertext.Level(), plaintext.Level())
// Output domain will match plaintext.Value.IsNTT value.
func (decryptor *Decryptor) Decrypt(ciphertext *Ciphertext, skSet *SecretKeySet, plaintext *rlwe.Plaintext) {
	ringQ := decryptor.ringQ
	level := utils.MinInt(ciphertext.Level(), plaintext.Level())
	plaintext.Value.Coeffs = plaintext.Value.Coeffs[:level+1]

	ctTmp := ciphertext.CopyNew()
	idset := ctTmp.IDSet()
	for _, sk := range skSet.Value {
		if idset.Has(sk.ID) {
			decryptor.PartialDecrypt(ctTmp, sk)
		}
	}

	if len(ctTmp.Value) > 1 {
		panic("Cannot Decrypt: there is a missing secretkey")
	}

	ringQ.ReduceLvl(level, ctTmp.Value["0"], plaintext.Value)
}
