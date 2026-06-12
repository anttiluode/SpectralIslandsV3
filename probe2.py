"""
Rigorous version: track the SAME rotation mode across forward/reverse and
confirm its rate flips sign. Also recover per-edge L_k from the eigenmodes
and show the hand-built v5 edges are a basis-dependent shadow of these.
"""
import numpy as np
rng = np.random.default_rng(0)
N, K = 64, 8
P = np.linalg.qr(rng.standard_normal((N, K)))[0][:, :K].T

def run_tour(direction=+1, steps=24000, dwell=60, leak=0.96, inject=0.18, noise=0.02, seed=1):
    r = np.random.default_rng(seed)
    s = P[0].copy(); S = np.zeros((steps, N))
    for t in range(steps):
        k = (direction*(t//dwell)) % K
        s = leak*s + inject*P[k] + noise*r.standard_normal(N)
        s /= np.linalg.norm(s)+1e-9
        S[t] = s
    return S

def lag_cov(S, tau=60):
    return S[tau:].T @ S[:-tau] / (len(S)-tau)

# Work entirely in the K-dim PATTERN basis (project field onto P).
# This is what the cell actually "sees": its overlaps r_k = <P_k, s>.
def pattern_lagcov(S, tau=60):
    R = S @ P.T                      # (T,K) overlaps
    return R[tau:].T @ R[:-tau] / (len(R)-tau)

Sf = run_tour(+1); Sr = run_tour(-1)
Cf = pattern_lagcov(Sf); Cr = pattern_lagcov(Sr)
Af = 0.5*(Cf-Cf.T); Ar = 0.5*(Cr-Cr.T)

# eigvecs of forward skew op; reuse the SAME basis to read forward & reverse
wf, Vf = np.linalg.eig(Af)
# pair up +/- i*omega; take the plane with largest |omega|
idx = np.argsort(-np.abs(wf.imag))
print("forward skew op rotation rates (imag eigenvalues):")
print(" ", np.round(np.sort(wf.imag)[::-1][:K], 4))
print("reverse skew op rotation rates:")
print(" ", np.round(np.sort(np.linalg.eigvals(Ar).imag)[::-1][:K], 4))

# Project the reverse operator onto the forward eigenmodes -> same mode, flipped rate
v = Vf[:, idx[0]]                                  # leading forward rotation plane
rate_f = (v.conj() @ Af @ v) / (v.conj() @ v)
rate_r = (v.conj() @ Ar @ v) / (v.conj() @ v)
print(f"\nSAME eigen-rotation-plane read on both tours:")
print(f"  rotation rate forward = {rate_f.imag:+.4f}")
print(f"  rotation rate reverse = {rate_r.imag:+.4f}   -> sign flips: {np.sign(rate_f.imag)!=np.sign(rate_r.imag)}")

# Now: the hand-built v5 per-edge angular momentum L_k = Im<z_k(t) conj z_k(t-lag)>,
# z_k = r_k + i r_{k+1}. Show net Sum_k L_k EQUALS 2*tr-like sum of the skew op,
# i.e. the assigned edges are just one basis for the same skew operator.
def v5_netL(S, tau=60):
    R = S @ P.T
    z = R + 1j*np.roll(R, -1, axis=1)
    L = (z[tau:]*np.conj(z[:-tau])).imag
    return L.mean(0).sum()                          # net angular momentum
netL_f = v5_netL(Sf); netL_r = v5_netL(Sr)
# skew-operator invariant: sum of |omega| with sign = "total circulation"
circ_f = np.sum(np.sort(wf.imag))                   # ~0 (antisym), use directed trace instead
# directed circulation = sum over edges of A[k,k+1]-A[k+1,k] (the cyclic flux)
flux_f = sum(Af[k,(k+1)%K]-Af[(k+1)%K,k] for k in range(K))
flux_r = sum(Ar[k,(k+1)%K]-Ar[(k+1)%K,k] for k in range(K))
print(f"\nhand-built v5 net angular momentum:  fwd {netL_f:+.4f}  rev {netL_r:+.4f}")
print(f"skew-operator cyclic flux (basis-free): fwd {flux_f:+.4f}  rev {flux_r:+.4f}")
print(f"  ratio v5/flux fwd = {netL_f/flux_f:.3f}  rev = {netL_r/flux_r:.3f}  (constant => same object)")
