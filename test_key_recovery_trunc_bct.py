#!/usr/bin/env python3
"""
Gohr-style Bayesian key recovery on Speck32/64
  – upper distinguisher : truncated differential (one-round peel)
  – wrong-key profile   : BCT algebraic transfer-matrix

Truncated differential class (one word each):
  left  word  alpha : A  = 0x0109,  free-bit mask AM = 0x6A74
  right word  beta  : B  = 0x8003,  free-bit mask BM = 0x6AFC

Input difference to Speck: (0x0040, 0x0000)
"""

import numpy as np
import speck as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os   import urandom
from math import sqrt, log2, log
from time import time
from random import randint

WORD_SIZE = sp.WORD_SIZE()   # 16

# ═══════════════════════════════════════════════════════════════════════════
#  §1  Truncated differential parameters
# ═══════════════════════════════════════════════════════════════════════════

# The class is shared by both the upper (k_{n-1}) and lower (k_{n-2})
# recovery layers.  If you want two different truncations, define a second
# (A_LO, AM_LO, B_LO, BM_LO) set and pass it explicitly.
A,  AM  = 0x0109, 0x6A74   # left-word  fixed value / free-bit mask
B,  BM  = 0x8003, 0x6AFC   # right-word fixed value / free-bit mask

A_FIXED = (~AM) & 0xFFFF    # bits that must equal A
B_FIXED = (~BM) & 0xFFFF    # bits that must equal B

N_FREE_A = bin(AM).count('1')
N_FREE_B = bin(BM).count('1')
N_FREE   = N_FREE_A + N_FREE_B          # free bits in the 32-bit class
N_FIXED  = 32 - N_FREE                  # fixed bits
CLASS_SIZE = 1 << N_FREE                # |{(d0,d1) ∈ class}| = 2^N_FREE
# P(random 32-bit diff falls in class) = 2^N_FREE / 2^32 = 2^{-N_FIXED}
P_RAND   = 2.0 ** (-N_FIXED)

print(f"[trunc-diff]  A=0x{A:04X} AM=0x{AM:04X}  B=0x{B:04X} BM=0x{BM:04X}")
print(f"[trunc-diff]  N_free={N_FREE}  N_fixed={N_FIXED}  "
      f"class_size=2^{N_FREE}  P_rand=2^(-{N_FIXED})")

# ═══════════════════════════════════════════════════════════════════════════
#  §2  BCT algebraic wrong-key profile
#      Computes pself[g] for all g in [0, 2^16) via a carry-propagation
#      transfer matrix (adapted from plot_truncated_bct_self_0109.py).
# ═══════════════════════════════════════════════════════════════════════════

def _carry(x, y, c):
    return (x & y) ^ (x & c) ^ (y & c)

def _two_carry(x, y, c1, c2, g):
    return _carry(x ^ y ^ g ^ c1, y ^ 1, c2)

def _build_M8():
    """
    Build the 8×16×16 base transfer tensor.
    Axis 0 indexes (a, b, g_v) ∈ {0,1}^3 (= 8 combinations).
    Axes 1,2 index the (output carry-state, input carry-state) transition
    probabilities averaged uniformly over the plaintext bits (x, y).
    The carry-state index j encodes (c1, c2, c1', c2') as a 4-bit integer.
    """
    M = np.zeros((8, 16, 16), dtype=np.float64)
    for i in range(8):
        a  = i & 1;  b  = (i >> 1) & 1;  gv = (i >> 2) & 1
        for j in range(16):
            c1  = j & 1;       c2  = (j >> 1) & 1
            c1d = (j >> 2) & 1; c2d = (j >> 3) & 1
            for z in range(4):
                x = z & 1;  y = (z >> 1) & 1
                # parity constraint that keeps the two carry-chains consistent
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
    Return pself[g] for all g ∈ [0, 2^16), shape (65536,).

    pself[g] = probability that a pair whose left/right word differences
    already lie in the truncated class (a, am, b, bm) maps – after one
    round of Speck with key offset g – BACK into the same class.

    Interpretation as wrong-key profile:
        g = k_real XOR k_guess
        pself[g=0]   ≈ P(hit | correct key)
        pself[g≠0]   = P(hit | wrong key with offset g)
    """
    g_idx = np.arange(65536, dtype=np.int32)
    # L[g, j] = probability of being in carry-state j for key-offset g
    # Initial state: carry-pair (c1,c2,c1',c2') = (1,0,1,0) → index 5
    L = np.zeros((65536, 16), dtype=np.float64)
    L[:, 5] = 1.0
    for i in range(WORD_SIZE):
        a_fix = (a >> i) & 1;  fa = (am >> i) & 1
        b_fix = (b >> i) & 1;  fb = (bm >> i) & 1
        n_free_local = fa + fb
        w = 1.0 / (1 << n_free_local)
        # Effective transfer matrix averaged over free bits of (a, b)
        Meff = np.zeros((2, 16, 16), dtype=np.float64)
        for combo in range(1 << n_free_local):
            pos = 0
            ai = ((combo >> pos) & 1) if fa else a_fix;  pos += fa
            bi = ((combo >> pos) & 1) if fb else b_fix
            for gv in range(2):
                Meff[gv] += w * _M8[ai | (bi << 1) | (gv << 2)]
        # Apply the matrix corresponding to the g-bit at position i
        gv_k = (g_idx >> i) & 1
        L_new = np.empty_like(L)
        m0 = (gv_k == 0)
        L_new[ m0] = L[ m0] @ Meff[0].T
        L_new[~m0] = L[~m0] @ Meff[1].T
        L = L_new
    return L.sum(axis=1)   # marginalise over final carry state

print("Computing BCT wrong-key profile pself[g] for all 2^16 offsets …",
      flush=True)
_t0 = time()
PSELF = compute_pself_all_g(A, AM, B, BM)   # shape (65536,)
print(f"  done in {time()-_t0:.1f}s   "
      f"pself[0]={PSELF[0]:.6f}  max={PSELF.max():.6f}  "
      f"mean={PSELF.mean():.6f}  min={PSELF.min():.6f}")

# Wrong-key profile arrays fed to bayesian_rank_kr
_EPS  = 1e-12
M_BCT = PSELF.copy()                                          # expected hit rate
S_BCT = 1.0 / np.sqrt(PSELF * (1.0 - PSELF) + _EPS)         # 1 / Bernoulli-std

# Per-pair LLR constants (using pself[0] as the "correct key" probability)
_P0       = float(PSELF[0])
LLR_HIT   = log2(max(_P0,       _EPS) / max(P_RAND,       _EPS))
LLR_MISS  = log2(max(1.0 - _P0, _EPS) / max(1.0 - P_RAND, _EPS))
print(f"[LLR]  hit={LLR_HIT:.3f} bits   miss={LLR_MISS:.3f} bits")


# ═══════════════════════════════════════════════════════════════════════════
#  §3  Wrong-key profile plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_wkp(pself=PSELF, outpath='wkp_bct_trunc_0109.png'):
    """Plot the BCT-algebraic wrong-key profile pself[g]."""
    def _fmt_bits(val, mask):
        s = ''
        for i in range(WORD_SIZE - 1, -1, -1):
            if (mask >> i) & 1:
                s += '?'
            elif (val >> i) & 1:
                s += '1'
            else:
                s += '0'
            if i in (12, 8, 4):
                s += ' '
        return s.strip()

    g_arr = np.arange(65536)
    alpha_pattern = _fmt_bits(A, AM)
    beta_pattern  = _fmt_bits(B, BM)

    fig, (ax_s, ax_l) = plt.subplots(
        2, 1, figsize=(14, 8), dpi=150,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})

    title = (
        'BCT-algebraic wrong-key profile  '
        r'$p^{\rm self}_{\rm trunc}(g)$  '
        r'for all $g = k_r \oplus k_g \in [0,\,2^{16})$' + '\n'
        r'$\alpha = \mathtt{' + alpha_pattern + r'}$   '
        r'$\beta  = \mathtt{' + beta_pattern  + r'}$   '
        f'(? = free,  class size = $2^{{{N_FREE}}}$,  '
        f'$p_0={_P0:.5f}$,  $p_{{\\rm rand}}=2^{{-{N_FIXED}}}$)'
    )
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.99)

    for ax, style, lbl in [
            (ax_s, dict(marker=',', color='#2E8B57', alpha=0.85, markersize=0.5,
                        linestyle='None'), 'scatter'),
            (ax_l, dict(color='#2E8B57', lw=0.4, alpha=0.85), 'line')]:
        ax.plot(g_arr, pself, **style)
        ax.set_xlim(0, 65536)
        ax.set_ylim(-0.02, max(pself.max() * 1.05, 0.1))
        ax.axhline(_P0,    color='steelblue', lw=0.9, ls='-',  alpha=0.7,
                   label=f'$p_0$ (correct key) = {_P0:.5f}')
        ax.axhline(pself.mean(), color='gray', lw=0.7, ls='--', alpha=0.5,
                   label=f'mean = {pself.mean():.5f}')
        ax.axhline(P_RAND, color='salmon',    lw=0.7, ls=':',  alpha=0.8,
                   label=f'$p_{{\\rm rand}}$ = $2^{{-{N_FIXED}}}$')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.15)

    ax_s.set_ylabel(r'$p^{\rm self}_{\rm trunc}(g)$', fontsize=11)
    ax_l.set_xlabel(r'$g = k_r \oplus k_g$', fontsize=11)
    ax_l.set_ylabel(r'$p^{\rm self}_{\rm trunc}(g)$', fontsize=11)
    ax_s.text(0.01, 0.97,
              f'alpha fixed: 0x{A & A_FIXED:04X}\n'
              f'beta  fixed: 0x{B & B_FIXED:04X}',
              transform=ax_s.transAxes, fontsize=9, va='top', ha='left',
              family='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', alpha=0.9, ec='gray'))
    ax_s.text(0.99, 0.97,
              f'max  = {pself.max():.5f}\n'
              f'mean = {pself.mean():.5f}\n'
              f'min  = {pself.min():.5f}',
              transform=ax_s.transAxes, fontsize=9, va='top', ha='right',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='gray'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, bbox_inches='tight', facecolor='white')
    print(f"[plot] WKP saved → {outpath}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  §4  Truncated differential scoring (replaces net.predict)
# ═══════════════════════════════════════════════════════════════════════════

def trunc_hit(d0, d1):
    """
    Boolean array, True where (d0, d1) lies in the truncated class.
    d0, d1 : uint16 arrays of left / right word XOR-differences.
    """
    return ((d0 & A_FIXED) == (A & A_FIXED)) & \
           ((d1 & B_FIXED) == (B & B_FIXED))

def trunc_llr(d0, d1):
    """
    Per-pair log-likelihood ratio (bits).
    Returns a float64 array of the same shape as d0/d1.
    """
    hit = trunc_hit(d0, d1).astype(np.float64)
    return hit * LLR_HIT + (1.0 - hit) * LLR_MISS


# ═══════════════════════════════════════════════════════════════════════════
#  §5  Gohr framework – modified for truncated distinguisher + BCT WKP
# ═══════════════════════════════════════════════════════════════════════════

def hw(v):
    res = np.zeros(v.shape, dtype=np.uint8)
    for i in range(16):
        res += (v >> i) & 1
    return res

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]

# --- Bayesian rank (identical formula to Gohr, different M/S source) ------

tmp_br = np.arange(2**14, dtype=np.uint16)
tmp_br = np.repeat(tmp_br, 32).reshape(-1, 32)

def bayesian_rank_kr(cand, emp_mean, m=M_BCT, s=S_BCT):
    """
    Score all 2^14 low-bit candidates by comparing the empirical per-candidate
    hit rate (emp_mean) to the BCT-algebraic expected hit rate m[g] using an
    L2 residual normalised by the BCT-algebraic inverse standard deviation s[g].

    Lower score → better match → higher posterior probability of being correct.
    """
    global tmp_br
    n = len(cand)
    if tmp_br.shape[1] != n:
        tmp_br = np.arange(2**14, dtype=np.uint16)
        tmp_br = np.repeat(tmp_br, n).reshape(-1, n)
    tmp    = tmp_br ^ cand                          # key-offset lookup index
    v      = (emp_mean - m[tmp]) * s[tmp]           # normalised residual
    v      = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return scores

# --- Core beam-search (replaces neural-network calls) ---------------------

def bayesian_key_recovery(cts, m=M_BCT, s=S_BCT,
                           num_cand=32, num_iter=5, seed=None):
    """
    Beam search over the 16-bit subkey space.

    Differences from Gohr:
      • net.predict(X)       → trunc_hit / trunc_llr  (§4)
      • m7 / s7              → M_BCT / S_BCT           (§2)
      • The 'means' fed to bayesian_rank_kr are the per-candidate
        empirical hit rates (∈ [0,1]) instead of neural-network outputs.
    Everything else (beam width, iteration count, random high-bit
    injection, argpartition selection) is unchanged.

    Returns
    -------
    all_keys : (num_cand * num_iter,) uint16 – all evaluated subkeys
    all_v    : (num_cand * num_iter,) float  – their LLR scores
    """
    n    = len(cts[0])
    keys = (np.random.choice(2**(WORD_SIZE - 2), num_cand, replace=False)
            if seed is None else np.copy(seed))

    # Tile ciphertexts once (num_cand copies) for vectorised batch decryption
    ct0a = np.tile(cts[0], num_cand)
    ct1a = np.tile(cts[1], num_cand)
    ct0b = np.tile(cts[2], num_cand)
    ct1b = np.tile(cts[3], num_cand)

    used     = np.zeros(2**(WORD_SIZE - 2))
    all_keys = np.zeros(num_cand * num_iter, dtype=np.uint16)
    all_v    = np.zeros(num_cand * num_iter)

    for i in range(num_iter):
        k = np.repeat(keys, n)

        # --- one-round peel with each candidate subkey ---
        c0a, c1a = sp.dec_one_round((ct0a, ct1a), k)
        c0b, c1b = sp.dec_one_round((ct0b, ct1b), k)

        # --- truncated differential scoring ---
        d0   = (c0a ^ c0b).reshape(num_cand, n)
        d1   = (c1a ^ c1b).reshape(num_cand, n)
        hit  = trunc_hit(d0, d1).astype(np.float64)        # (num_cand, n)
        llr  = hit * LLR_HIT + (1.0 - hit) * LLR_MISS      # (num_cand, n)

        means = hit.mean(axis=1)       # empirical hit rate per candidate
        v     = llr.sum(axis=1)        # accumulated LLR per candidate

        all_v   [i * num_cand:(i+1) * num_cand] = v
        all_keys[i * num_cand:(i+1) * num_cand] = keys.copy()

        # --- BCT-guided beam update ---
        scores = bayesian_rank_kr(keys, means, m=m, s=s)
        keys   = np.argpartition(scores + used, num_cand)[:num_cand]
        # Randomise the top 2 bits to ensure full 16-bit coverage
        r      = np.random.randint(0, 4, num_cand, dtype=np.uint16) << 14
        keys   = (keys ^ r).astype(np.uint16)

    return all_keys, all_v

# --- Verifier (replaces net.predict in the local refinement step) ---------

def verifier_search(cts, best_guess, use_n=64):
    """
    Exhaustively search the Hamming-≤2 neighbourhood of best_guess = (k1, k2)
    using two-round peel + truncated LLR scoring.
    Returns (k1_best, k2_best, score).
    """
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight
    n   = len(ck1)                       # ≈ 137

    ck1 = np.repeat(ck1, n);  keys1 = ck1.copy()
    ck2 = np.tile(ck2,   n);  keys2 = ck2.copy()
    ck1 = np.repeat(ck1, use_n)
    ck2 = np.repeat(ck2, use_n)

    ct0a = np.tile(cts[0][:use_n], n * n)
    ct1a = np.tile(cts[1][:use_n], n * n)
    ct0b = np.tile(cts[2][:use_n], n * n)
    ct1b = np.tile(cts[3][:use_n], n * n)

    # Two-round peel: last round (k1) then second-to-last round (k2)
    p0a, p1a = sp.dec_one_round((ct0a, ct1a), ck1)
    p0b, p1b = sp.dec_one_round((ct0b, ct1b), ck1)
    p0a, p1a = sp.dec_one_round((p0a, p1a),   ck2)
    p0b, p1b = sp.dec_one_round((p0b, p1b),   ck2)

    d0  = p0a ^ p0b
    d1  = p1a ^ p1b
    llr = trunc_llr(d0, d1).reshape(-1, use_n)

    # Score = mean LLR × total pair count (matches Gohr's normalisation)
    v   = llr.mean(axis=1) * len(cts[0])
    m   = int(np.argmax(v))
    return keys1[m], keys2[m], float(v[m])

# ─── challenge generation (input diff (0x0040, 0x0)) ─────────────────────

def gen_key(nr):
    key = np.frombuffer(urandom(8), dtype=np.uint16)
    return sp.expand_key(key, nr)

def gen_plain(n):
    pt0 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    pt1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    return pt0, pt1

def make_structure(pt0, pt1, diff=(0x0040, 0x0), neutral_bits=None):
    """Build a plaintext structure from base points pt0, pt1."""
    p0 = pt0.reshape(-1, 1).copy()
    p1 = pt1.reshape(-1, 1).copy()
    if neutral_bits:
        for i in neutral_bits:
            d = 1 << i;  d0 = d >> 16;  d1 = d & 0xffff
            p0 = np.concatenate([p0, p0 ^ d0], axis=1)
            p1 = np.concatenate([p1, p1 ^ d1], axis=1)
    p0b = p0 ^ diff[0]
    p1b = p1 ^ diff[1]
    return p0, p1, p0b, p1b

def gen_challenge(n, nr, diff=(0x0040, 0x0), neutral_bits=None):
    """
    Generate n plaintext structures, encrypt with a random key.
    Returns ciphertext arrays [ct0a, ct1a, ct0b, ct1b] and the key schedule.
    """
    pt0, pt1 = gen_plain(n)
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff,
                                             neutral_bits=neutral_bits)
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)
    key = gen_key(nr)
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), key)
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), key)
    return [ct0a, ct1a, ct0b, ct1b], key

# ─── top-level attack (UCB + two-layer beam search) ───────────────────────

def test_bayes(cts, it=500, cutoff1=0.0, cutoff2=0.0,
               m=M_BCT, s=S_BCT, verify_breadth=None):
    """
    Gohr's UCB-guided two-layer Bayesian key recovery,
    rewritten to use the truncated differential distinguisher
    and the BCT-algebraic wrong-key profile.

    Parameters
    ----------
    cts         : list [ct0a, ct1a, ct0b, ct1b] where each entry is a
                  2-D array (num_structures × n_pairs_per_structure).
    it          : maximum number of UCB iterations.
    cutoff1     : LLR threshold to trigger the second (lower) key layer.
    cutoff2     : LLR threshold for early exit + verifier.
    m, s        : wrong-key profile mean / inverse-std (default: BCT).
    verify_breadth : number of pairs used in verifier_search.

    Returns
    -------
    (k1_guess, k2_guess), num_iterations_used
    """
    n = len(cts[0])
    if verify_breadth is None:
        verify_breadth = len(cts[0][0])

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

        # Early-exit: if second-layer best already exceeds cutoff2, refine
        if best_val > cutoff2:
            improvement = (verify_breadth > 0)
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

        # ── Layer 2: recover k_{n-2} if layer-1 score is promising ───
        if vtmp > cutoff1:
            promising = [idx for idx in range(len(keys)) if v[idx] > cutoff1]
            for idx in promising:
                k1_cand = keys[idx]
                c0a, c1a = sp.dec_one_round((cts[0][i], cts[1][i]), k1_cand)
                c0b, c1b = sp.dec_one_round((cts[2][i], cts[3][i]), k1_cand)
                keys2, v2 = bayesian_key_recovery(
                    [c0a, c1a, c0b, c1b],
                    m=m, s=s, num_cand=32, num_iter=5)
                vtmp2 = float(np.max(v2))
                if vtmp2 > best_val:
                    best_val = vtmp2
                    best_key = (k1_cand, keys2[int(np.argmax(v2))])
                    best_pod = i

    # End of iterations – run verifier on current best
    improvement = (verify_breadth > 0)
    while improvement:
        k1, k2, val = verifier_search(
            [cts[0][best_pod], cts[1][best_pod],
             cts[2][best_pod], cts[3][best_pod]],
            best_key, use_n=verify_breadth)
        improvement = (val > best_val)
        if improvement:
            best_key = (k1, k2);  best_val = val

    return best_key, it

# ─── outer evaluation loop ────────────────────────────────────────────────

def test(n_trials=100, nr=9, num_structures=32, it=500,
         cutoff1=0.0, cutoff2=50.0,
         diff=(0x0040, 0x0), neutral_bits=None,
         verify_breadth=None):
    """
    Run n_trials independent key-recovery experiments and report statistics.
    """
    print("Checking Speck32/64 test vector …")
    if not sp.check_testvector():
        print("Test vector FAILED – aborting.")
        return None, None

    arr1 = np.zeros(n_trials, dtype=np.uint16)
    arr2 = np.zeros(n_trials, dtype=np.uint16)
    t0   = time()

    for i in range(n_trials):
        print(f"Trial {i+1}/{n_trials}", flush=True)
        ct, key = gen_challenge(num_structures, nr,
                                diff=diff, neutral_bits=neutral_bits)
        guess, n_used = test_bayes(
            ct, it=it, cutoff1=cutoff1, cutoff2=cutoff2,
            verify_breadth=verify_breadth)

        arr1[i] = np.uint16(guess[0]) ^ key[nr - 1]
        arr2[i] = np.uint16(guess[1]) ^ key[nr - 2]
        print(f"  err k_{{n-1}}=0x{arr1[i]:04X}  "
              f"err k_{{n-2}}=0x{arr2[i]:04X}")

    t1 = time()
    succ = int(np.sum((arr1 == 0) & (arr2 == 0)))
    print(f"\n{'='*60}")
    print(f"Trials : {n_trials}")
    print(f"Success: {succ}/{n_trials}  ({100*succ/n_trials:.1f}%)")
    print(f"Wall time per trial (avg): {(t1-t0)/n_trials:.1f}s")
    return arr1, arr2

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. Plot the BCT wrong-key profile
    plot_wkp(PSELF, outpath='wkp_bct_trunc_0109.png')

    # 2. Run a small batch of key-recovery trials on 9-round Speck32/64.
    #    Adjust nr, num_structures, it, cutoff1/cutoff2 to match the
    #    desired attack depth and available compute budget.
    arr1, arr2 = test(
        n_trials      = 20,
        nr            = 9,
        num_structures= 32,
        it            = 200,
        cutoff1       = 0.0,    # tune based on observed LLR distribution
        cutoff2       = 50.0,   # tune based on observed LLR distribution
        diff          = (0x0040, 0x0),
        neutral_bits  = None,
        verify_breadth= 64,
    )

    if arr1 is not None:
        np.save('trunc_bct_run_err1.npy', arr1)
        np.save('trunc_bct_run_err2.npy', arr2)
        print("Results saved to trunc_bct_run_err1/2.npy")
