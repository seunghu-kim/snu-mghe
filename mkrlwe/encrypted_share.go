package mkrlwe

// Encrypted partial-decryption share (owner response re-encrypted under the
// client's public key), agreed for Information Sciences R1-3 (v4).
//
// Notation (code name in brackets). The evaluated ciphertext at level l is
//   ct = (c0 [ct.Value["0"]], cC [ct.Value[clientID]], c1..cn [ct.Value[owner_i]]).
// Owner i's raw share is d_i = c_i*s_i + phi_i mod Q_l (GenShare; phi_i is the
// flooding noise of FloodingConfig). For EVERY request owner i draws a fresh
// zero encryption under the client key, with its own encryptor/PRNG,
//   (r_i0, r_i1) <- Enc_pkC(0)        (ordinary RLWE encryption noise, sigma = params.Sigma())
// and sends only
//   h_i0 = d_i + r_i0 mod Q_l,   h_i1 = r_i1 mod Q_l.
// The CSP outputs
//   c0' = c0 + sum_i h_i0,   cC' = cC + sum_i h_i1   (cC is kept, not replaced),
// so that c0' + cC'*sC = c0 + cC*sC + sum_i c_i*s_i + sum_i phi_i + sum_i eps_zero_i
// with eps_zero_i = r_i0 + r_i1*sC (exact identity mod Q_l).
//
// Representation: mkckks ciphertexts are kept in the COEFFICIENT domain
// (IsNTT = false), non-Montgomery, on the RNS primes q_0..q_l. GenShare maps c_i
// to NTT, multiplies by sk (stored NTT+Montgomery) with a Montgomery product and
// maps back, so d_i is coefficient-domain, non-Montgomery. Encrypt computes u*pk
// in NTT (u in Montgomery form, pk NTT non-Montgomery), maps back, and adds the
// Gaussian errors in the coefficient domain whenever the output ciphertext is
// non-NTT. The zero encryption is generated in the domain of ct, so d_i and r_i0
// are residues of the same integers on the same primes and are added directly;
// nothing is decoded to slots or rescaled. Output polynomials carry exactly l+1
// RNS rows (the active level of ct; no level is consumed).

import (
	"fmt"
	"sort"

	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
)

// EncryptedShare is the two-polynomial response (h_i0, h_i1) of one owner.
type EncryptedShare struct {
	H0 *ring.Poly
	H1 *ring.Poly
}

// Level returns the level of the share (number of RNS rows - 1).
func (es *EncryptedShare) Level() int { return len(es.H0.Coeffs) - 1 }

// GenEncryptedShare returns (h_i0, h_i1) for party sk.ID, using the flooding
// configuration of this decryptor for phi_i and a fresh zero encryption under
// pkC. ct is not modified.
func (decryptor *Decryptor) GenEncryptedShare(ct *Ciphertext, sk *SecretKey, pkC *PublicKey) (*EncryptedShare, error) {
	if decryptor.sampler == nil {
		return nil, fmt.Errorf("mkrlwe: GenEncryptedShare requires a decryptor built with NewDecryptorWithFlooding")
	}
	if err := checkShareInputs(ct, sk.ID, pkC); err != nil {
		return nil, err
	}
	return decryptor.EncryptShareForClient(ct, decryptor.GenShare(ct, sk), pkC)
}

// EncryptShareForClient adds a fresh zero encryption under pkC to an already
// computed share d (as returned by GenShare for ct) and returns (d + r0, r1).
// d is consumed: its first l+1 rows become H0. Exposed so that test harnesses
// can pair the same d_i (same phi_i) with and without re-encryption; it adds
// no option for deterministic or noise-free output.
func (decryptor *Decryptor) EncryptShareForClient(ct *Ciphertext, d *ring.Poly, pkC *PublicKey) (*EncryptedShare, error) {
	if pkC == nil {
		return nil, fmt.Errorf("mkrlwe: nil client public key")
	}
	if _, ok := ct.Value[pkC.ID]; !ok || pkC.ID == "0" {
		return nil, fmt.Errorf("mkrlwe: ciphertext has no component for client %q", pkC.ID)
	}
	level := ct.Level()
	if len(d.Coeffs) < level+1 {
		return nil, fmt.Errorf("mkrlwe: share has %d RNS rows, ciphertext level %d", len(d.Coeffs), level)
	}
	isNTT := ct.Value["0"].IsNTT
	if ct.Value[pkC.ID].IsNTT != isNTT {
		return nil, fmt.Errorf("mkrlwe: inconsistent NTT flags in ciphertext")
	}

	z := decryptor.encryptZeroLvl(level, isNTT, pkC)

	ringQ := decryptor.ringQ
	h0 := &ring.Poly{Coeffs: d.Coeffs[:level+1], IsNTT: isNTT}
	ringQ.AddLvl(level, h0, z.Value["0"], h0)
	h1 := z.Value[pkC.ID]
	h1.Coeffs = h1.Coeffs[:level+1]
	h1.IsNTT, h1.IsMForm = isNTT, false

	// r_i0 is no longer needed and must not leave this function
	for i := range z.Value["0"].Coeffs {
		for j := range z.Value["0"].Coeffs[i] {
			z.Value["0"].Coeffs[i][j] = 0
		}
	}
	return &EncryptedShare{H0: h0, H1: h1}, nil
}

// encryptZeroLvl returns a fresh public-key encryption of 0 under pkC at the
// given level, in the requested domain. A new Encryptor (own CSPRNG-keyed
// samplers and buffers) is built per call: zero encryptions are never cached
// and no mutable sampler is shared between concurrent requests.
func (decryptor *Decryptor) encryptZeroLvl(level int, isNTT bool, pkC *PublicKey) *Ciphertext {
	ids := NewIDSet()
	ids.Add(pkC.ID)
	z := NewCiphertext(decryptor.params, ids, level)
	z.Value["0"].IsNTT = isNTT
	z.Value[pkC.ID].IsNTT = isNTT
	pt := &rlwe.Plaintext{Value: ring.NewPoly(decryptor.params.N(), level+1)}
	pt.Value.IsNTT = isNTT
	NewEncryptor(decryptor.params).Encrypt(pt, pkC, z)
	return z
}

func checkShareInputs(ct *Ciphertext, ownerID string, pkC *PublicKey) error {
	if pkC == nil {
		return fmt.Errorf("mkrlwe: nil client public key")
	}
	if ownerID == pkC.ID {
		return fmt.Errorf("mkrlwe: client key ID %q equals owner ID", pkC.ID)
	}
	if _, ok := ct.Value[ownerID]; !ok {
		return fmt.Errorf("mkrlwe: ciphertext has no component for party %q", ownerID)
	}
	if ct.Value[ownerID].IsNTT != ct.Value["0"].IsNTT {
		return fmt.Errorf("mkrlwe: inconsistent NTT flags in ciphertext")
	}
	return nil
}

// AggregateEncryptedShares returns a new ciphertext (c0 + sum h_i0, cC + sum h_i1)
// whose only components are "0" and clientID. ct is not modified. Every
// component of ct other than "0" and clientID must have exactly one share, and
// every share must be at the ciphertext level and in its domain.
func (decryptor *Decryptor) AggregateEncryptedShares(ct *Ciphertext, shares map[string]*EncryptedShare, clientID string) (*Ciphertext, error) {
	level := ct.Level()
	if _, ok := ct.Value[clientID]; !ok || clientID == "0" {
		return nil, fmt.Errorf("mkrlwe: ciphertext has no component for client %q", clientID)
	}
	isNTT := ct.Value["0"].IsNTT
	for id := range ct.Value {
		if id == "0" || id == clientID {
			continue
		}
		if _, ok := shares[id]; !ok {
			return nil, fmt.Errorf("mkrlwe: missing encrypted share for party %q", id)
		}
	}
	ids := make([]string, 0, len(shares))
	for id, sh := range shares {
		if _, ok := ct.Value[id]; !ok || id == "0" || id == clientID {
			return nil, fmt.Errorf("mkrlwe: unexpected encrypted share for %q", id)
		}
		if sh == nil || sh.H0 == nil || sh.H1 == nil {
			return nil, fmt.Errorf("mkrlwe: incomplete encrypted share for %q", id)
		}
		if len(sh.H0.Coeffs) != level+1 || len(sh.H1.Coeffs) != level+1 {
			return nil, fmt.Errorf("mkrlwe: share of %q has %d/%d RNS rows, expected %d", id, len(sh.H0.Coeffs), len(sh.H1.Coeffs), level+1)
		}
		if sh.H0.IsNTT != isNTT || sh.H1.IsNTT != isNTT || sh.H0.IsMForm || sh.H1.IsMForm {
			return nil, fmt.Errorf("mkrlwe: share of %q is not in the ciphertext domain", id)
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)

	ringQ := decryptor.ringQ
	keep := NewIDSet()
	keep.Add(clientID)
	out := NewCiphertext(decryptor.params, keep, level)
	ring.CopyValuesLvl(level, ct.Value["0"], out.Value["0"])
	ring.CopyValuesLvl(level, ct.Value[clientID], out.Value[clientID])
	for _, id := range ids {
		ringQ.AddLvl(level, out.Value["0"], shares[id].H0, out.Value["0"])
		ringQ.AddLvl(level, out.Value[clientID], shares[id].H1, out.Value[clientID])
	}
	out.Value["0"].IsNTT = isNTT
	out.Value[clientID].IsNTT = isNTT
	return out, nil
}
