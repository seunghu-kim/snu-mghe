package mkckks

import (
	"github.com/seunghu-kim/snu-mghe/mkrlwe"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
)

type Decryptor struct {
	*mkrlwe.Decryptor
	encoder  ckks.Encoder
	params   Parameters
	ptxtPool *ckks.Plaintext
}

// NewDecryptor instantiates a Decryptor for the CKKS scheme.
func NewDecryptor(params Parameters) *Decryptor {
	ckksParams, _ := ckks.NewParameters(params.Parameters.Parameters, params.LogSlots(), params.Scale())

	ret := new(Decryptor)
	ret.Decryptor = mkrlwe.NewDecryptor(params.Parameters)
	ret.encoder = ckks.NewEncoder(ckksParams)
	ret.params = params
	ret.ptxtPool = ckks.NewPlaintext(ckksParams, params.MaxLevel(), params.Scale())
	ret.ptxtPool.Value.IsNTT = false
	return ret
}

func NewDecryptorWithGaussianNoise(params Parameters, sigma float64, bound int) *Decryptor {
	ckksParams, _ := ckks.NewParameters(params.Parameters.Parameters, params.LogSlots(), params.Scale())

	ret := new(Decryptor)
	ret.Decryptor = mkrlwe.NewDecryptorWithGaussianNoise(params.Parameters, sigma, bound)
	ret.encoder = ckks.NewEncoder(ckksParams)
	ret.params = params
	ret.ptxtPool = ckks.NewPlaintext(ckksParams, params.MaxLevel(), params.Scale())
	ret.ptxtPool.Value.IsNTT = false
	return ret
}

// NewDecryptorWithFlooding instantiates a CKKS Decryptor whose GenShare adds
// noise drawn from cfg (validated against params).
func NewDecryptorWithFlooding(params Parameters, cfg mkrlwe.FloodingConfig) (*Decryptor, error) {
	ckksParams, err := ckks.NewParameters(params.Parameters.Parameters, params.LogSlots(), params.Scale())
	if err != nil {
		return nil, err
	}
	inner, err := mkrlwe.NewDecryptorWithFlooding(params.Parameters, cfg)
	if err != nil {
		return nil, err
	}
	ret := new(Decryptor)
	ret.Decryptor = inner
	ret.encoder = ckks.NewEncoder(ckksParams)
	ret.params = params
	ret.ptxtPool = ckks.NewPlaintext(ckksParams, params.MaxLevel(), params.Scale())
	ret.ptxtPool.Value.IsNTT = false
	return ret, nil
}

// PartialDecrypt partially decrypts the ct with single secretkey sk and update result inplace
func (dec *Decryptor) PartialDecrypt(ct *Ciphertext, sk *mkrlwe.SecretKey) {
	dec.Decryptor.PartialDecrypt(ct.Ciphertext, sk)
}

func (dec *Decryptor) GenShare(ct *Ciphertext, sk *mkrlwe.SecretKey) *ring.Poly {
	return dec.Decryptor.GenShare(ct.Ciphertext, sk)
}

func (dec *Decryptor) AggregateSharesAndDrop(ct *Ciphertext, shares map[string]*ring.Poly) {
	dec.Decryptor.AggregateSharesAndDrop(ct.Ciphertext, shares)
}

// Decrypt decrypts the ciphertext with given secretkey set and write the result in ptOut.
// The level of the output plaintext is min(ciphertext.Level(), plaintext.Level())
// Output domain will match plaintext.Value.IsNTT value.
func (dec *Decryptor) Decrypt(ciphertext *Ciphertext, skSet *mkrlwe.SecretKeySet) (msg *Message) {
	ctTmp := ciphertext.CopyNew()

	dec.Decryptor.Decrypt(ctTmp.Ciphertext, skSet, dec.ptxtPool.Plaintext)
	dec.ptxtPool.Scale = ctTmp.Scale
	msg = new(Message)
	msg.Value = dec.encoder.Decode(dec.ptxtPool, dec.params.logSlots)

	return
}

// Decrypt decrypts the ciphertext with given secretkey set and write the result in ptOut.
// The level of the output plaintext is min(ciphertext.Level(), plaintext.Level())
// Output domain will match plaintext.Value.IsNTT value.
func (dec *Decryptor) DecryptToPtxt(ciphertext *Ciphertext, skSet *mkrlwe.SecretKeySet) *ring.Poly {
	ctTmp := ciphertext.CopyNew()

	idset := ctTmp.IDSet()
	for _, sk := range skSet.Value {
		if idset.Has(sk.ID) {
			dec.PartialDecrypt(ctTmp, sk)
		}
	}

	return ctTmp.Value["0"]
}

// GenEncryptedShare returns the encrypted share (h_i0, h_i1) = (c_i*s_i + phi_i + r_i0, r_i1)
// of party sk.ID, with (r_i0, r_i1) a fresh encryption of zero under pkC.
// See mkrlwe/encrypted_share.go. ct is not modified.
func (dec *Decryptor) GenEncryptedShare(ct *Ciphertext, sk *mkrlwe.SecretKey, pkC *mkrlwe.PublicKey) (*mkrlwe.EncryptedShare, error) {
	return dec.Decryptor.GenEncryptedShare(ct.Ciphertext, sk, pkC)
}

// EncryptShareForClient re-encrypts an already computed share d (from GenShare
// on ct) as (d + r0, r1) with a fresh zero encryption under pkC.
func (dec *Decryptor) EncryptShareForClient(ct *Ciphertext, d *ring.Poly, pkC *mkrlwe.PublicKey) (*mkrlwe.EncryptedShare, error) {
	return dec.Decryptor.EncryptShareForClient(ct.Ciphertext, d, pkC)
}

// AggregateEncryptedShares returns a new ciphertext (c0 + sum h_i0, cC + sum h_i1)
// with components "0" and clientID only, carrying the scale of ct. ct is not modified.
func (dec *Decryptor) AggregateEncryptedShares(ct *Ciphertext, shares map[string]*mkrlwe.EncryptedShare, clientID string) (*Ciphertext, error) {
	out, err := dec.Decryptor.AggregateEncryptedShares(ct.Ciphertext, shares, clientID)
	if err != nil {
		return nil, err
	}
	return &Ciphertext{Ciphertext: out, Scale: ct.Scale}, nil
}
