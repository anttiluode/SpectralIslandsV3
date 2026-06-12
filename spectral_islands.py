"""
spectral_islands.py  —  the islands were the spectrum all along
================================================================
One claim, tested three ways:

  Across the whole Geometric-Neuron line the "spectral islands" / directed
  edges were HAND-ASSIGNED:  z_k = <P_k,s> + i<P_{k+1},s>,  L_k = Im(z z*_lag).
  v8 tried to make them emergent with a Ky Fan flow on the SYMMETRIC lag
  covariance C and stalled at 0.50 captured energy ("nearest-target coverage
  did not rise").

  The reason: direction is not in the symmetric part. Split the lag covariance
        C_tau = E[ r(t) r(t-tau)^T ],   r_k = <P_k, s>   (the overlaps)
  into
        S = (C+C^T)/2   (symmetric: Wiener-Khinchin power, phase-blind)
        A = (C-C^T)/2   (skew: rotation, chirality, direction).
  A is real antisymmetric:  eigenvalues +/- i*omega_j, eigenvectors = 2D
  rotation planes. Diagonalizing A gives the islands for free:

    * each conjugate eigenpair IS a spectral island (a Koopman rotation mode);
    * sign(omega_j) is that island's chirality = its arrow of time;
    * the hand-built per-edge L_k is one basis chart of this same operator
      (net L = - cyclic flux of A, ratio exactly -1, verified);
    * Ky Fan on S is blind to all of it (the v8 ceiling explained).

This unifies, with one diagonalization: the v5 chiral readout, the v3 Takens
orbit, the v8 coverage objective, the Chiral-Eye/Ear EMD detector, and the
IslandNet poles (a rotation rate omega_j + a decay = a complex pole rho_j).

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import numpy as np


def make_patterns(N, K, seed=0):
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((N, K)))[0]
    return Q[:, :K].T                                   # (K,N) orthonormal


def tour_field(P, direction=+1, steps=24000, dwell=60, leak=0.96,
               inject=0.18, noise=0.02, seed=1):
    """Shared ephaptic field s(t) toured through patterns; norm-stabilized."""
    r = np.random.default_rng(seed); N = P.shape[1]
    s = P[0].copy(); S = np.zeros((steps, N))
    for t in range(steps):
        k = (direction * (t // dwell)) % len(P)
        s = leak * s + inject * P[k] + noise * r.standard_normal(N)
        s /= np.linalg.norm(s) + 1e-9
        S[t] = s
    return S


def lag_covariance(S, P, tau):
    R = S @ P.T                                         # overlaps r_k(t)  (T,K)
    return R[tau:].T @ R[:-tau] / (len(R) - tau)        # C_tau in pattern basis


def extract_islands(C):
    """Diagonalize the skew part. Returns conjugate-paired (omega, plane)."""
    A = 0.5 * (C - C.T)
    w, V = np.linalg.eig(A)
    om = w.imag
    keep = om > 1e-9                                    # one of each +/- pair
    om, V = om[keep], V[:, keep]
    order = np.argsort(-om)
    return om[order], V[:, order], A


def island_chirality(S, P, plane, tau):
    """Project field onto one rotation plane, read L = Im(z z*_lag)."""
    R = S @ P.T
    z = R @ np.conj(plane)                              # complex coordinate in the plane
    L = (z[tau:] * np.conj(z[:-tau])).imag
    return float(L.mean())


def kyfan(M, Q):
    """captured energy tr(Q^H M Q) for orthonormal columns Q (the v8 metric)."""
    return float(np.real(np.einsum("ik,ij,jk->", Q.conj(), M, Q)))


if __name__ == "__main__":
    N, K, tau = 64, 8, 60
    P = make_patterns(N, K)

    print("=" * 70)
    print("SPECTRAL ISLANDS AS THE SKEW LAG-OPERATOR SPECTRUM")
    print("=" * 70)

    Sf = tour_field(P, +1); Sr = tour_field(P, -1)
    Cf = lag_covariance(Sf, P, tau); Cr = lag_covariance(Sr, P, tau)
    omf, Vf, Af = extract_islands(Cf)
    omr, Vr, Ar = extract_islands(Cr)

    print(f"\n[1] islands EMERGE as conjugate eigenpairs of the skew operator")
    print(f"    forward rotation rates omega_j = {np.round(omf, 4)}")
    print(f"    (each is one spectral island; no edges were assigned)")

    print(f"\n[2] chirality = sign(omega) = arrow of time, per island")
    v = Vf[:, 0]
    rf = (v.conj() @ Af @ v / (v.conj() @ v)).imag
    rr = (v.conj() @ Ar @ v / (v.conj() @ v)).imag
    print(f"    leading island, same plane read on both tours:")
    print(f"      forward rate {rf:+.4f}   reverse rate {rr:+.4f}   sign flips: {np.sign(rf)!=np.sign(rr)}")
    Lf = island_chirality(Sf, P, v, tau)
    Lr = island_chirality(Sr, P, v, tau)
    print(f"    L = Im(z z*_lag) on that island:  fwd {Lf:+.4f}  rev {Lr:+.4f}")

    print(f"\n[3] the hand-built v5 edges are one basis of this same operator")
    def v5_netL(S):
        R = S @ P.T; z = R + 1j*np.roll(R, -1, axis=1)
        return (z[tau:]*np.conj(z[:-tau])).imag.mean(0).sum()
    flux = lambda A: sum(A[k,(k+1)%K]-A[(k+1)%K,k] for k in range(K))
    nf, nr = v5_netL(Sf), v5_netL(Sr)
    ff, fr = flux(Af), flux(Ar)
    print(f"    v5 net angular momentum:   fwd {nf:+.4f}  rev {nr:+.4f}")
    print(f"    skew-op cyclic flux:       fwd {ff:+.4f}  rev {fr:+.4f}")
    print(f"    ratio (constant => identical object): fwd {nf/ff:+.3f}  rev {nr/fr:+.3f}")

    print(f"\n[4] why v8's symmetric Ky Fan was blind (the explained ceiling)")
    Sf_sym = 0.5*(Cf+Cf.T); Sr_sym = 0.5*(Cr+Cr.T)
    # symmetric part is identical forward vs reverse (time-reversal preserves it)
    print(f"    || S_fwd - S_rev || = {np.linalg.norm(Sf_sym-Sr_sym):.2e}   (symmetric part: direction-blind)")
    print(f"    || A_fwd - A_rev || = {np.linalg.norm(Af-Ar):.4f}   (skew part: carries the reversal)")
    print(f"    => coverage on S can never see direction; it lives entirely in A.")

    print(f"\n[5] bridge to IslandNet: each island -> a complex pole")
    print(f"    rotation rate omega_j (skew spectrum) + field leak (decay) = pole rho_j")
    leak_decay = -np.log(0.96)                          # from the field leak above
    print(f"    rho_j = {leak_decay:.4f} + i*omega_j  for omega_j in {np.round(omf[:4],3)} ...")
    print(f"    archiving = deepening Re(rho); chirality is carried in Im(rho).")
