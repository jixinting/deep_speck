#!/usr/bin/env python3
"""
Gohr-style Bayesian key recovery on Speck32/64
  – distinguisher  : truncated differential (one-round peel)
  – wrong-key profile : BCT algebraic transfer-matrix

Truncated differential class:
  left  word alpha : A=0x0109,  free-bit mask AM=0x6A74
  right word beta  : B=0x8003,  free-bit mask BM=0x6AFC
Input difference to Speck: (0x0040, 0x0000)

Attack parameters (user-specified):
  p_t       = 2^{-11}   truncated differential probability
  threshold = 2^{-5}    chosen from BCT WKP
  D         = p_t * threshold = 2^{-16}   score normalisation
  score(k') = #{hits(k')} / D    (beam-search ranking)
  E[score | wrong key offset g] = P_RAND + p_t * pself[g]
"""

import numpy as np
import speck as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os   import urandom
from math import sqrt, log2, log
from time import time

WORD_SIZE = sp.WORD_SIZE()   # 16

# ═══════════════════════════════════════════════════════════════════════════
#  §1  Attack parameters (user-specified)
# ═══════════════════════════════════════════════════════════════════════════

P_T    = 2.0 ** (-11)    # truncated differential probability
THRESH = 2.0 ** (-5)     # threshold parameter (from BCT WKP)
D      = P_T * THRESH    # = 2^{-16}, score normalisation constant
# score(k') = #{hits(k')} / D   → E[score | correct key] = P_T/D = 2^5 = 32/N_pairs
# v(k')     = log2(#{hits}/D)   → used for beam-search ranking

# ═══════════════════════════════════════════════════════════════════════════
#  §2  Truncated differential class
# ═══════════════════════════════════════════════════════════════════════════

A,  AM  = 0x0109, 0x6A74
B,  BM  = 0x8003, 0x6AFC

A_FIXED = (~AM) & 0xFFFF
B_FIXED = (~BM) & 0xFFFF

N_FREE_A = bin(AM).count('1')
N_FREE_B = bin(BM).count('1')
N_FREE   = N_FREE_A + N_FREE_B   # = 18
N_FIXED  = 32 - N_FREE           # = 14
# P(random 32-bit difference falls in class) = 2^{N_FREE} / 2^{32} = 2^{-N_FIXED}
P_RAND   = 2.0 ** (-N_FIXED)     # = 2^{-14}

print(f"[trunc-diff]  A=0x{A:04X}/AM=0x{AM:04X}  B=0x{B:04X}/BM=0x{BM:04X}")
print(f"[trunc-diff]  N_free={N_FREE}  N_fixed={N_FIXED}  "
      f"P_rand=2^(-{N_FIXED})={P_RAND:.2e}")
print(f"[attack]      P_T=2^-11={P_T:.2e}  "
      f"threshold=2^-5={THRESH:.4f}  D=2^-16={D:.2e}")
print(f"[attack]      SNR = P_T/P_RAND = {P_T/P_RAND:.1f}x = 2^{log2(P_T/P_RAND):.1f}")

# ═══════════════════════════════════════════════════════════════════════════
#  §3  BCT algebraic wrong-key profile
# ═══════════════════════════════════════════════════════════════════════════

def _carry(x, y, c):
    return (x & y) ^ (x & c) ^ (y & c)

def _two_carry(x, y, c1, c2, g):
    return _carry(x ^ y ^ g ^ c1, y ^ 1, c2)

def _build_M8():
    M = np.zeros((8, 16, 16), dtype=np.float64)
    for i in range(8):
        a  = i & 1;  b  = (i >> 1) & 1;  gv = (i >> 2) & 1
        for j in range(16):
            c1  = j & 1;       c2  = (j >> 1) & 1
            c1d = (j >> 2) & 1; c2d = (j >> 3) & 1
            for z in range(4):
                x = z & 1;  y = (z >> 1) & 1
                if (c1 ^ c2 ^ c1d ^ c2d) == 0:
                    idx = (  _carry(x, y, c1)
                           + 2 * _two_carry(x, y, c1, c2, gv)
                           + 4 * _carry(x ^ a, y ^ b, c1d)
                           + 8 * _two_carry(x ^ a, y ^ b, c1d, c2d, gv))
                    M[i, idx, j] += 0.25
    return M

_M8 = _build_M8()

def compute_pself_all_g(a, am, b, bm):
    """
    pself[g] = P(diff stays in class T after one Speck round with key-offset g
                 | diff was already in T)
    Exact BCT-algebraic computation for all g in [0, 2^16).
    """
    g_idx = np.arange(65536, dtype=np.int32)
    L     = np.zeros((65536, 16), dtype=np.float64)
    L[:, 5] = 1.0
    for i in range(WORD_SIZE):
        a_fix = (a >> i) & 1;  fa = (am >> i) & 1
        b_fix = (b >> i) & 1;  fb = (bm >> i) & 1
        n_free_local = fa + fb
        w = 1.0 / (1 << n_free_local)
        Meff = np.zeros((2, 16, 16), dtype=np.float64)
        for combo in range(1 << n_free_local):
            pos = 0
            ai = ((combo >> pos) & 1) if fa else a_fix;  pos += fa
            bi = ((combo >> pos) & 1) if fb else b_fix
            for gv in range(2):
                Meff[gv] += w * _M8[ai | (bi << 1) | (gv << 2)]
        gv_k  = (g_idx >> i) & 1
        L_new = np.empty_like(L)
        m0    = (gv_k == 0)
        L_new[ m0] = L[ m0] @ Meff[0].T
        L_new[~m0] = L[~m0] @ Meff[1].T
        L = L_new
    return L.sum(axis=1)

print("\nComputing BCT wrong-key profile pself[g] for all 2^16 offsets …",
      flush=True)
_t0    = time()
PSELF  = compute_pself_all_g(A, AM, B, BM)
print(f"  done in {time()-_t0:.1f}s   "
      f"pself[0]={PSELF[0]:.6f}  max={PSELF.max():.6f}  "
      f"mean={PSELF.mean():.6f}  min={PSELF.min():.6f}")

# ── Wrong-key profile arrays ─────────────────────────────────────────────
# E[hit_rate | wrong key offset g] = P_RAND + P_T * pself[g]
#   g=0  → P_RAND + P_T ≈ P_T  (correct-key expected hit rate)
#   g≠0  → P_RAND + P_T*pself[g] (most ≈ P_RAND since pself[g] small)
_EPS  = 1e-12
M_BCT = P_RAND + P_T * PSELF                                  # (65536,)
S_BCT = 1.0 / np.sqrt(M_BCT * (1.0 - M_BCT) + _EPS)         # (65536,)

print(f"[WKP]  M[0]={M_BCT[0]:.2e}  M_mean={M_BCT.mean():.2e}  "
      f"P_T={P_T:.2e}  P_RAND={P_RAND:.2e}")

# ═══════════════════════════════════════════════════════════════════════════
#  §4  Wrong-key profile plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_wkp(outpath='wkp_bct_trunc_0109.png'):
    def _fmt(val, mask):
        s = ''
        for i in range(WORD_SIZE - 1, -1, -1):
            s += '?' if (mask >> i) & 1 else ('1' if (val >> i) & 1 else '0')
            if i in (12, 8, 4): s += ' '
        return s.strip()

    g_arr = np.arange(65536)
    fig, (ax_s, ax_l) = plt.subplots(2, 1, figsize=(14, 8), dpi=150,
                                      gridspec_kw={'hspace': 0.35})
    title = (
        f'BCT-algebraic WKP  $M_{{\\rm BCT}}[g] = P_{{\\rm RAND}} + p_t '
        f'\\cdot p^{{\\rm self}}_{{\\rm trunc}}(g)$\n'
        f'$\\alpha={_fmt(A,AM)}$  $\\beta={_fmt(B,BM)}$  '
        f'$p_t=2^{{-11}}$  $P_{{\\rm RAND}}=2^{{-14}}$  '
        f'$D=p_t\\cdot 2^{{-5}}=2^{{-16}}$'
    )
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    for ax, style in [(ax_s, dict(marker=',', linestyle='None',
                                   color='#2E8B57', alpha=0.8)),
                       (ax_l, dict(lw=0.4, color='#2E8B57', alpha=0.8))]:
        ax.plot(g_arr, M_BCT, **style)
        ax.axhline(M_BCT[0], color='steelblue', lw=1.0, ls='-', alpha=0.8,
                   label=f'$M[0]=p_t={P_T:.2e}$  (correct key)')
        ax.axhline(P_RAND,   color='salmon',    lw=0.8, ls=':', alpha=0.9,
                   label=f'$P_{{\\rm RAND}}=2^{{-14}}={P_RAND:.2e}$')
        ax.axhline(M_BCT.mean(), color='gray', lw=0.7, ls='--', alpha=0.6,
                   label=f'mean={M_BCT.mean():.2e}')
        ax.set_xlim(0, 65536); ax.set_ylim(-0.0001, M_BCT[0] * 1.1)
        ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.15)
    ax_s.set_ylabel(r'$M_{\rm BCT}[g]$', fontsize=11)
    ax_l.set_xlabel(r'$g = k_r \oplus k_g$', fontsize=11)
    ax_l.set_ylabel(r'$M_{\rm BCT}[g]$', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, bbox_inches='tight', facecolor='white')
    print(f"[plot] WKP saved → {outpath}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  §5  Truncated differential scoring
# ═══════════════════════════════════════════════════════════════════════════

def trunc_hit(d0, d1):
    """Boolean array: True where (d0,d1) ∈ class T."""
    return ((d0 & A_FIXED) == (A & A_FIXED)) & \
           ((d1 & B_FIXED) == (B & B_FIXED))

# ═══════════════════════════════════════════════════════════════════════════
#  §6  Core Bayesian key-recovery framework
# ═══════════════════════════════════════════════════════════════════════════

def hw(v):
    r = np.zeros(v.shape, dtype=np.uint8)
    for i in range(16): r += (v >> i) & 1
    return r

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]   # hw ≤ 2 neighbours

tmp_br = np.arange(2**14, dtype=np.uint16)
tmp_br = np.repeat(tmp_br, 32).reshape(-1, 32)

# ── Bayesian rank ─────────────────────────────────────────────────────────

def bayesian_rank_kr(cand, emp_hit_rate, m=M_BCT, s=S_BCT):
    """
    For each candidate true-key low-14-bits b in [0, 2^14):
      residual[j] = (emp_hit_rate[j] - M_BCT[b XOR cand[j]]) * S_BCT[b XOR cand[j]]
      score[b]    = L2_norm(residual)    (smaller = better match = more likely b is correct)
    """
    global tmp_br
    n = len(cand)
    if tmp_br.shape[1] != n:
        tmp_br = np.arange(2**14, dtype=np.uint16)
        tmp_br = np.repeat(tmp_br, n).reshape(-1, n)
    tmp    = tmp_br ^ cand           # key-offset lookup indices  (2^14, n)
    v      = (emp_hit_rate - m[tmp]) * s[tmp]
    v      = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return scores

# ── Beam search (bayesian_key_recovery) ───────────────────────────────────

def bayesian_key_recovery(cts, m=M_BCT, s=S_BCT, num_cand=32, num_iter=5,
                           seed=None):
    """
    Beam search over the 16-bit subkey space.

    Scoring:
      hits(k') = #{pairs : Dec(C,k') XOR Dec(C',k') ∈ class T}
      emp_hit_rate(k') = hits(k') / N                   → fed to bayesian_rank
      v(k')            = log2(hits(k') / D)             → beam ranking score

    E[emp_hit_rate | wrong key offset g] = M_BCT[g] = P_RAND + P_T * pself[g]
    """
    n    = len(cts[0])
    keys = (np.random.choice(2**(WORD_SIZE - 2), num_cand, replace=False)
            if seed is None else np.copy(seed))

    ct0a = np.tile(cts[0], num_cand)
    ct1a = np.tile(cts[1], num_cand)
    ct0b = np.tile(cts[2], num_cand)
    ct1b = np.tile(cts[3], num_cand)

    used     = np.zeros(2**(WORD_SIZE - 2))
    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
    all_v    = np.zeros(num_cand * num_iter)

    for i in range(num_iter):
        k = np.repeat(keys, n)

        # ── one-round peel ──────────────────────────────────────────
        c0a, c1a = sp.dec_one_round((ct0a, ct1a), k)
        c0b, c1b = sp.dec_one_round((ct0b, ct1b), k)

        # ── truncated differential scoring ──────────────────────────
        d0   = (c0a ^ c0b).reshape(num_cand, n)   # XOR differences
        d1   = (c1a ^ c1b).reshape(num_cand, n)
        hit  = trunc_hit(d0, d1).astype(np.float64)   # (num_cand, n), 0/1

        # emp_hit_rate per candidate  → compared to M_BCT in bayesian_rank
        emp_hit_rate = hit.mean(axis=1)             # (num_cand,)

        # v = log2(hits / D)  → beam ranking score
        hits_count   = hit.sum(axis=1)              # (num_cand,)
        v = np.where(hits_count > 0,
                     np.log2(hits_count / D),
                     np.log2(0.5 / D))              # floor for zero hits

        all_v   [i * num_cand:(i+1) * num_cand] = v
        all_keys[i * num_cand:(i+1) * num_cand] = keys.copy()

        # ── BCT-guided beam update ───────────────────────────────────
        scores = bayesian_rank_kr(keys, emp_hit_rate, m=m, s=s)
        keys   = np.argpartition(scores + used, num_cand)[:num_cand]
        r      = np.random.randint(0, 4, num_cand, dtype=np.uint16) << 14
        keys   = (keys ^ r).astype(np.uint16)

    return all_keys, all_v

# ── Verifier: exhaustive Hamming-≤2 neighbourhood refinement ─────────────

def verifier_search(cts, best_guess, use_n=128):
    """
    Two-round peel + truncated hit score over the Hamming-≤2 neighbourhood
    of best_guess = (k1, k2).
    """
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight
    n   = len(ck1)                     # ≈ 137

    ck1 = np.repeat(ck1, n);  keys1 = ck1.copy()
    ck2 = np.tile(ck2,   n);  keys2 = ck2.copy()
    ck1 = np.repeat(ck1, use_n)
    ck2 = np.repeat(ck2, use_n)

    ct0a = np.tile(cts[0][:use_n], n * n)
    ct1a = np.tile(cts[1][:use_n], n * n)
    ct0b = np.tile(cts[2][:use_n], n * n)
    ct1b = np.tile(cts[3][:use_n], n * n)

    p0a, p1a = sp.dec_one_round((ct0a, ct1a), ck1)
    p0b, p1b = sp.dec_one_round((ct0b, ct1b), ck1)
    p0a, p1a = sp.dec_one_round((p0a,  p1a),  ck2)
    p0b, p1b = sp.dec_one_round((p0b,  p1b),  ck2)

    d0  = p0a ^ p0b;  d1 = p1a ^ p1b
    hit = trunc_hit(d0, d1).astype(np.float64).reshape(-1, use_n)

    hits_count = hit.sum(axis=1)
    v = np.where(hits_count > 0,
                 np.log2(hits_count / D) * len(cts[0]) / use_n,
                 np.log2(0.5 / D) * len(cts[0]) / use_n)

    m_idx  = int(np.argmax(v))
    return keys1[m_idx], keys2[m_idx], float(v[m_idx])

# ═══════════════════════════════════════════════════════════════════════════
#  §7  Challenge generation
#
#  Layout:
#    cts[j] has shape (num_structures, N_per_struct)
#    cts[j][s] is the s-th structure's N_per_struct ciphertexts (1-D array)
#
#  All structures share the SAME key schedule (we are recovering that key).
#  Each structure is an independent set of N_per_struct plaintext pairs
#  generated with the target input difference diff = (0x0040, 0x0000).
# ═══════════════════════════════════════════════════════════════════════════

def gen_key(nr):
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    return sp.expand_key(key, nr)

def gen_challenge(num_structures, N_per_struct, nr,
                  diff=(0x0040, 0x0)):
    """
    Generate `num_structures` independent ciphertext-pair batches,
    each containing `N_per_struct` pairs, all encrypted under the same
    random key schedule of length nr.

    Returns
    -------
    cts : list of 4 arrays each of shape (num_structures, N_per_struct)
    key : round-key schedule (length nr)
    """
    key = gen_key(nr)

    ct0a = np.empty((num_structures, N_per_struct), dtype=np.uint16)
    ct1a = np.empty_like(ct0a)
    ct0b = np.empty_like(ct0a)
    ct1b = np.empty_like(ct0a)

    for s in range(num_structures):
        pt0 = np.frombuffer(urandom(2 * N_per_struct), dtype=np.uint16)
        pt1 = np.frombuffer(urandom(2 * N_per_struct), dtype=np.uint16)
        # apply one extra round of decryption to extend the distinguisher depth
        pt0a, pt1a = sp.dec_one_round((pt0,          pt1         ), 0)
        pt0b, pt1b = sp.dec_one_round((pt0 ^ diff[0], pt1 ^ diff[1]), 0)
        ct0a[s], ct1a[s] = sp.encrypt((pt0a, pt1a), key)
        ct0b[s], ct1b[s] = sp.encrypt((pt0b, pt1b), key)

    return [ct0a, ct1a, ct0b, ct1b], key

# ═══════════════════════════════════════════════════════════════════════════
#  §8  UCB outer loop + two-layer attack  (test_bayes)
# ═══════════════════════════════════════════════════════════════════════════

def test_bayes(cts, it=500, cutoff1=18.0, cutoff2=22.0,
               m=M_BCT, s=S_BCT, verify_breadth=128):
    """
    UCB-guided two-layer Bayesian key recovery.
    Uses truncated differential scoring and BCT-algebraic WKP throughout.

    v thresholds (for N=2^14 pairs):
      E[v | correct key] ≈ log2(N*P_T/D) = log2(2^14 * 2^-11 / 2^-16) = 19
      E[v | noise key  ] ≈ log2(N*P_RAND/D) = log2(2^14 * 2^-14 / 2^-16) = 16
      cutoff1 = 18  (trigger layer-2 search)
      cutoff2 = 22  (trigger verifier + early exit)
    """
    n   = len(cts[0])
    alpha      = sqrt(n)
    best_val   = -1e9
    best_key   = (np.uint16(0), np.uint16(0))
    best_pod   = 0
    eps        = 1e-3
    local_best = np.full(n, -1e9)
    num_visits = np.full(n, eps)

    for j in range(it):

        # UCB structure selection
        priority = local_best + alpha * np.sqrt(log2(j + 1) / num_visits)
        i        = int(np.argmax(priority))
        num_visits[i] += 1

        # Early exit: first layer best already above cutoff2 → refine
        if best_val > cutoff2:
            improvement = True
            while improvement:
                k1, k2, val = verifier_search(
                    [cts[0][best_pod], cts[1][best_pod],
                     cts[2][best_pod], cts[3][best_pod]],
                    best_key, use_n=verify_breadth)
                improvement = (val > best_val)
                if improvement:
                    best_key = (k1, k2);  best_val = val
            return best_key, j

        # ── Layer 1: recover k_{n-1} ──────────────────────────────────
        keys, v = bayesian_key_recovery(
            [cts[0][i], cts[1][i], cts[2][i], cts[3][i]],
            m=m, s=s, num_cand=32, num_iter=5)

        vtmp = float(np.max(v))
        if vtmp > local_best[i]:
            local_best[i] = vtmp

        # ── Layer 2: if layer-1 has a hit, recover k_{n-2} ───────────
        if vtmp > cutoff1:
            promising = [idx for idx in range(len(keys)) if v[idx] > cutoff1]
            for idx in promising:
                k1_cand = np.uint16(keys[idx])
                c0a, c1a = sp.dec_one_round(
                    (cts[0][i], cts[1][i]), k1_cand)
                c0b, c1b = sp.dec_one_round(
                    (cts[2][i], cts[3][i]), k1_cand)
                keys2, v2 = bayesian_key_recovery(
                    [c0a, c1a, c0b, c1b],
                    m=m, s=s, num_cand=32, num_iter=5)
                vtmp2 = float(np.max(v2))
                if vtmp2 > best_val:
                    best_val = vtmp2
                    best_key = (k1_cand, np.uint16(keys2[int(np.argmax(v2))]))
                    best_pod = i

    # End of UCB iterations → final verifier pass
    improvement = True
    while improvement:
        k1, k2, val = verifier_search(
            [cts[0][best_pod], cts[1][best_pod],
             cts[2][best_pod], cts[3][best_pod]],
            best_key, use_n=verify_breadth)
        improvement = (val > best_val)
        if improvement:
            best_key = (k1, k2);  best_val = val

    return best_key, it

# ═══════════════════════════════════════════════════════════════════════════
#  §9  Outer evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

def test(n_trials=50, nr=7,
         num_structures=32, N_per_struct=2**14,
         it=300, cutoff1=18.0, cutoff2=22.0,
         diff=(0x0040, 0x0), verify_breadth=128):
    """
    Run n_trials independent attacks on nr-round Speck32/64.

    num_structures : number of independent structure batches
    N_per_struct   : ciphertext pairs per structure
    """
    print(f"\n{'='*65}")
    print(f"Attack on {nr}-round Speck32/64")
    print(f"N per structure = {N_per_struct} = 2^{log2(N_per_struct):.1f}")
    print(f"Structures      = {num_structures}")
    print(f"Trials          = {n_trials}")
    E_v_corr = log2(max(N_per_struct * P_T, 0.5) / D)
    E_v_noise = log2(max(N_per_struct * P_RAND, 0.5) / D)
    print(f"E[v|correct key] ≈ {E_v_corr:.1f}   E[v|noise] ≈ {E_v_noise:.1f}")
    print(f"cutoff1={cutoff1}  cutoff2={cutoff2}")
    print(f"{'='*65}")

    if not sp.check_testvector():
        print("Speck test vector FAILED – aborting.")
        return None, None

    arr1 = np.zeros(n_trials, dtype=np.uint16)  # err on k_{n-1}
    arr2 = np.zeros(n_trials, dtype=np.uint16)  # err on k_{n-2}
    t0   = time()

    for trial in range(n_trials):
        print(f"  Trial {trial+1:3d}/{n_trials} … ", end='', flush=True)
        ct, key = gen_challenge(num_structures, N_per_struct, nr, diff=diff)
        guess, n_used = test_bayes(
            ct, it=it, cutoff1=cutoff1, cutoff2=cutoff2,
            verify_breadth=verify_breadth)

        arr1[trial] = np.uint16(guess[0]) ^ key[nr - 1]
        arr2[trial] = np.uint16(guess[1]) ^ key[nr - 2]
        ok = (arr1[trial] == 0) and (arr2[trial] == 0)
        print(f"err k_{{n-1}}=0x{arr1[trial]:04X}  "
              f"err k_{{n-2}}=0x{arr2[trial]:04X}  "
              f"{'✓' if ok else '✗'}")

    t1    = time()
    succ  = int(np.sum((arr1 == 0) & (arr2 == 0)))
    print(f"\n{'='*65}")
    print(f"Success : {succ}/{n_trials}  ({100*succ/n_trials:.1f}%)")
    print(f"Avg time: {(t1-t0)/n_trials:.1f}s per trial")
    print(f"{'='*65}")
    return arr1, arr2

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. Plot BCT wrong-key profile
    plot_wkp('wkp_bct_trunc_0109.png')

    # 2. Run experiment
    #    nr=7 : 7-round Speck32/64
    #    n_base * 2^|neutral_bits| = 2^12 * 2^2 = 2^14 pairs per structure
    #    cutoff1/cutoff2 calibrated to E[v] = 19 (correct key, N=2^14)
    arr1, arr2 = test(
        n_trials       = 50,
        nr             = 7,
        num_structures = 32,
        N_per_struct   = 2**14,
        it             = 300,
        cutoff1        = 18.0,
        cutoff2        = 22.0,
        diff           = (0x0040, 0x0),
        verify_breadth = 128,
    )

    if arr1 is not None:
        np.save('trunc_bct_err1.npy', arr1)
        np.save('trunc_bct_err2.npy', arr2)
        print("Saved → trunc_bct_err1/2.npy")
