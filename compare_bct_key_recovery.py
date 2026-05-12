"""
Compare key recovery effectiveness in Gohr's BKS using:
  (A) Original empirical wrong key profile  (m_orig, s_orig)
  (B) BCT-computed wrong key profile        (m_bct,  s_bct)

BCT gives p(g) = probability that truncated differential
(alpha=0x181E / beta=0x1C1E, with masks 0xE7E1 / 0xE3E1) passes
through one Speck round when the key difference is g.

Mapping to network-output scale (0-1):
    m_bct[g] = 0.5 + (m_orig[0] - 0.5) * p_bct[g] / p_bct[0]

std is kept from the original profile; alternatively a flat std equal
to the mean of s_orig is tried as well.

Key-rank comparison uses two methods:
  1. Profile-only (synthetic data): samples g ~ Uniform, simulates
     emp_mean = m_true[g] + noise, ranks correct key (g=0) with each profile.
  2. If keras/TensorFlow is available: runs actual network-based key rank
     using bayesian_rank_kr().

Usage:
    python compare_bct_key_recovery.py          # 7-round (default)
    python compare_bct_key_recovery.py --rounds 6
"""

import sys, argparse, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ── 0.  CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--rounds', type=int, default=7,
                    help='rounds for wrong-key profile (6, 7, or 8)')
parser.add_argument('--n_rank_trials', type=int, default=2000,
                    help='number of synthetic key-rank trials')
parser.add_argument('--n_cand', type=int, default=32,
                    help='number of key candidates per trial in synthetic test')
parser.add_argument('--no_keras', action='store_true',
                    help='skip keras-based test even if keras is available')
parser.add_argument('--outdir', type=str, default='.',
                    help='directory for output figures / npy files')
args = parser.parse_args()

NR = args.rounds
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# ── 1.  Load original wrong-key profiles ─────────────────────────────────────
profile_files = {
    6: ('data_wrong_key_mean_6r.npy', 'data_wrong_key_std_6r.npy'),
    7: ('data_wrong_key_mean_7r.npy', 'data_wrong_key_std_7r.npy'),
    8: ('data_wrong_key_8r_mean_1e6.npy', 'data_wrong_key_8r_std_1e6.npy'),
}
if NR not in profile_files:
    sys.exit(f"No profile available for {NR} rounds. Choose 6, 7, or 8.")

mf, sf = profile_files[NR]
if not (os.path.exists(mf) and os.path.exists(sf)):
    sys.exit(f"Profile files not found: {mf}, {sf}")

m_orig = np.load(mf).astype(np.float64)
s_orig = np.load(sf).astype(np.float64)   # already 1/std in test_key_recovery.py
print(f"\n[ORIG {NR}r] m: min={m_orig.min():.4f}  mean={m_orig.mean():.4f}"
      f"  max={m_orig.max():.4f}  (g=0: {m_orig[0]:.4f})")
print(f"[ORIG {NR}r] 1/s: min={s_orig.min():.4f}  mean={s_orig.mean():.4f}"
      f"  max={s_orig.max():.4f}")

# ── 2.  Compute BCT profile ───────────────────────────────────────────────────
# Identical to plot_truncated_bct_self_181E.py
WORD_SIZE = 16

def _carry(x, y, c):
    return (x & y) ^ (x & c) ^ (y & c)

def _two_carry(x, y, c1, c2, g):
    return _carry(x ^ y ^ g ^ c1, y ^ 1, c2)

def build_M8():
    M = np.zeros((8, 16, 16), dtype=np.float64)
    for i in range(8):
        a = i & 1;  b = (i >> 1) & 1;  gv = (i >> 2) & 1
        for j in range(16):
            c1 = j&1; c2 = (j>>1)&1; c1d = (j>>2)&1; c2d = (j>>3)&1
            for z in range(4):
                x = z & 1;  y = (z >> 1) & 1
                if (c1 ^ c2 ^ c1d ^ c2d) == 0:
                    idx = (_carry(x,y,c1)  + 2*_two_carry(x,y,c1,c2,gv)
                         + 4*_carry(x^a,y^b,c1d) + 8*_two_carry(x^a,y^b,c1d,c2d,gv))
                    M[i, idx, j] += 0.25
    return M

M8 = build_M8()

def compute_all_g_self(a, a_mask, b, b_mask):
    g_idx = np.arange(65536, dtype=np.int32)
    L     = np.zeros((65536, 16), dtype=np.float64)
    L[:, 5] = 1.0
    for i in range(WORD_SIZE):
        a_fix = (a >> i) & 1;  fa = (a_mask >> i) & 1
        b_fix = (b >> i) & 1;  fb = (b_mask >> i) & 1
        n_free = fa + fb
        w      = 1.0 / (1 << n_free)
        Meff = np.zeros((2, 16, 16), dtype=np.float64)
        for combo in range(1 << n_free):
            pos = 0
            ai = ((combo >> pos) & 1) if fa else a_fix;  pos += fa
            bi = ((combo >> pos) & 1) if fb else b_fix
            for gv in range(2):
                Meff[gv] += w * M8[ai | (bi << 1) | (gv << 2)]
        gv_k  = (g_idx >> i) & 1
        L_new = np.empty_like(L)
        m0 = gv_k == 0
        L_new[ m0] = L[ m0] @ Meff[0].T
        L_new[~m0] = L[~m0] @ Meff[1].T
        L = L_new
    return L.sum(axis=1)

A, AM = 0x181E, 0xE7E1
B, BM = 0x1C1E, 0xE3E1

print("\nComputing BCT profile for all 65536 g values ...", flush=True)
p_bct = compute_all_g_self(A, AM, B, BM)
print(f"[BCT] p(0)={p_bct[0]:.6f}  max={p_bct.max():.6f}"
      f"  mean={p_bct.mean():.6f}  min={p_bct.min():.6f}")

# ── 3.  Build BCT-scaled mean profile ────────────────────────────────────────
# Linear mapping: p(g=0) → m_orig[0],  p=0 → 0.5
# m_bct[g] = 0.5 + (m_orig[0] - 0.5) * p_bct[g] / p_bct[0]
m_correct = m_orig[0]
p0        = p_bct[0]

if p0 < 1e-12:
    sys.exit("p_bct[0] ≈ 0, cannot normalise. Check BCT parameters.")

m_bct = 0.5 + (m_correct - 0.5) * (p_bct / p0)
m_bct = np.clip(m_bct, 0.0, 1.0)

# std: keep original inverse-std array; also prepare flat-std variant
s_bct_orig_std = s_orig.copy()                          # original std unchanged
s_bct_flat     = np.full(65536, s_orig.mean())          # constant std

print(f"\n[BCT mean] min={m_bct.min():.4f}  mean={m_bct.mean():.4f}"
      f"  max={m_bct.max():.4f}  (g=0: {m_bct[0]:.4f})")

# Save BCT profiles for use in BKS
np.save(os.path.join(OUTDIR, f'data_wrong_key_mean_{NR}r_bct.npy'), m_bct.astype(np.float32))
np.save(os.path.join(OUTDIR, f'data_wrong_key_std_{NR}r_bct.npy'),  s_bct_orig_std.astype(np.float32))
print(f"Saved BCT profiles to {OUTDIR}/")

# ── 4.  Profile comparison plots ─────────────────────────────────────────────
g_arr = np.arange(65536)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=130)
fig.suptitle(f'Wrong-key profile comparison ({NR}-round Speck32/64)',
             fontsize=13, fontweight='bold')

# 4a – scatter: m_orig vs g
ax = axes[0, 0]
ax.plot(g_arr, m_orig, ',', color='steelblue', alpha=0.6, markersize=0.5)
ax.set_title('Original empirical profile', fontsize=10)
ax.set_xlabel('key difference g'); ax.set_ylabel('mean net output')
ax.axhline(0.5, color='gray', lw=0.7, ls='--')
ax.text(0.98, 0.97, f'g=0: {m_orig[0]:.4f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, family='monospace')

# 4b – scatter: m_bct vs g
ax = axes[0, 1]
ax.plot(g_arr, m_bct, ',', color='darkorange', alpha=0.6, markersize=0.5)
ax.set_title('BCT-derived profile', fontsize=10)
ax.set_xlabel('key difference g'); ax.set_ylabel('mean net output')
ax.axhline(0.5, color='gray', lw=0.7, ls='--')
ax.text(0.98, 0.97, f'g=0: {m_bct[0]:.4f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, family='monospace')

# 4c – scatter: m_bct vs m_orig
ax = axes[1, 0]
ax.plot(m_orig, m_bct, ',', color='purple', alpha=0.4, markersize=0.8)
r_p, _ = pearsonr(m_orig, m_bct)
r_s, _ = spearmanr(m_orig, m_bct)
ax.set_xlabel('m_orig'); ax.set_ylabel('m_bct')
ax.set_title('BCT vs Original (scatter)', fontsize=10)
ax.text(0.02, 0.97,
        f'Pearson r = {r_p:.4f}\nSpearman ρ = {r_s:.4f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='gray'))

# 4d – histogram of difference
ax = axes[1, 1]
delta = m_bct - m_orig
ax.hist(delta, bins=80, color='teal', alpha=0.8, edgecolor='none')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('m_bct − m_orig'); ax.set_ylabel('count')
ax.set_title(f'Profile difference (mean={delta.mean():.4f}  std={delta.std():.4f})',
             fontsize=10)

plt.tight_layout()
fig_path = os.path.join(OUTDIR, f'bct_profile_compare_{NR}r.png')
plt.savefig(fig_path, bbox_inches='tight', facecolor='white')
print(f"Saved profile comparison → {fig_path}")
plt.close()

# ── 5.  Synthetic key-rank comparison ─────────────────────────────────────────
# For each trial:
#   - pick a random correct key k_true in [0, 2^16)
#   - generate n_cand candidate keys (including k_true at index 0)
#   - simulate emp_mean[k] = m_true[k XOR k_true] + noise
#     where m_true = original profile (ground truth)
#   - score candidates with bayesian_rank_kr using (a) m_orig, (b) m_bct
#   - record rank of k_true under each scoring
# Repeated at multiple noise levels to reveal where profiles diverge.

def bayesian_rank_scores(cand_keys, emp_mean, m, s):
    """Score key candidates; lower score = better match to the profile."""
    n = len(cand_keys)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        g_vec = np.uint16(cand_keys[i]) ^ cand_keys.astype(np.uint16)
        v = (emp_mean - m[g_vec]) * s[g_vec]
        scores[i] = np.linalg.norm(v)
    return scores

def print_rank_stats(name, r, n_cand):
    top1 = np.mean(r == 0) * 100
    top3 = np.mean(r  < 3) * 100
    top5 = np.mean(r  < 5) * 100
    med  = np.median(r)
    print(f"  {name:<35s}  top-1={top1:5.1f}%  top-3={top3:5.1f}%"
          f"  top-5={top5:5.1f}%  median={med:.1f}")

rng = np.random.default_rng(42)

# Noise levels: σ = 0.005 (easy), 0.02 (medium), 0.05 (hard)
noise_levels = [0.005, 0.02, 0.05]
all_results  = {}   # noise_sigma → dict of profile_name → rank array

for NOISE_SCALE in noise_levels:
    ranks_orig     = []
    ranks_bct      = []
    ranks_bct_flat = []

    for trial in range(args.n_rank_trials):
        k_true  = int(rng.integers(0, 65536))
        others  = rng.integers(0, 65536, size=args.n_cand - 1).astype(np.uint16)
        cands   = np.concatenate([[k_true], others]).astype(np.uint16)

        g_vec    = cands.astype(np.uint32) ^ np.uint32(k_true)
        noise    = rng.normal(0.0, NOISE_SCALE, args.n_cand)
        emp_mean = m_orig[g_vec] + noise

        sc = bayesian_rank_scores(cands, emp_mean, m_orig, s_orig)
        ranks_orig.append(int(np.sum(sc < sc[0])))

        sc = bayesian_rank_scores(cands, emp_mean, m_bct, s_bct_orig_std)
        ranks_bct.append(int(np.sum(sc < sc[0])))

        sc = bayesian_rank_scores(cands, emp_mean, m_bct, s_bct_flat)
        ranks_bct_flat.append(int(np.sum(sc < sc[0])))

    all_results[NOISE_SCALE] = {
        'orig':     np.array(ranks_orig),
        'bct':      np.array(ranks_bct),
        'bct_flat': np.array(ranks_bct_flat),
    }

    print(f"\nKey-rank results  noise σ={NOISE_SCALE}  "
          f"(n_cand={args.n_cand}, n_trials={args.n_rank_trials})")
    print(f"  Ground truth = ORIGINAL profile")
    print("-" * 80)
    print_rank_stats(f'Original profile ({NR}r)',        all_results[NOISE_SCALE]['orig'], args.n_cand)
    print_rank_stats(f'BCT profile ({NR}r, orig std)',   all_results[NOISE_SCALE]['bct'], args.n_cand)
    print_rank_stats(f'BCT profile ({NR}r, flat std)',   all_results[NOISE_SCALE]['bct_flat'], args.n_cand)

# use the medium noise level for the rank plot
ranks_orig     = all_results[0.02]['orig']
ranks_bct      = all_results[0.02]['bct']
ranks_bct_flat = all_results[0.02]['bct_flat']

# ── 6.  Rank distribution plot ────────────────────────────────────────────────
bins = np.arange(-0.5, args.n_cand + 0.5, 1)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130, sharey=True)
fig.suptitle(f'Key-rank distribution — {NR}-round synthetic test'
             f'  ({args.n_rank_trials} trials, {args.n_cand} candidates)',
             fontsize=12, fontweight='bold')

configs = [
    (ranks_orig,     'Original profile\n(empirical m, s)',   'steelblue'),
    (ranks_bct,      'BCT profile\n(BCT m, orig s)',         'darkorange'),
    (ranks_bct_flat, 'BCT profile\n(BCT m, flat s)',         'green'),
]
for ax, (r, lbl, col) in zip(axes, configs):
    ax.hist(r, bins=bins, color=col, alpha=0.8, edgecolor='none', density=True)
    top1 = np.mean(r == 0) * 100
    med  = np.median(r)
    ax.set_title(f'{lbl}\ntop-1={top1:.1f}%  median={med:.1f}', fontsize=9)
    ax.set_xlabel('key rank')
    ax.axvline(0, color='red', lw=1.2, ls='--', label='correct key')
    ax.set_xlim(-1, min(30, args.n_cand))
axes[0].set_ylabel('density')
plt.tight_layout()
rank_fig_path = os.path.join(OUTDIR, f'bct_key_rank_compare_{NR}r.png')
plt.savefig(rank_fig_path, bbox_inches='tight', facecolor='white')
print(f"\nSaved rank distribution → {rank_fig_path}")
plt.close()

# ── 7.  Optional keras-based actual key rank test ────────────────────────────
if args.no_keras:
    print("\n[keras test skipped via --no_keras flag]")
else:
    try:
        warnings.filterwarnings('ignore')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        from keras.models import model_from_json
        import speck as sp
        from os import urandom

        net_file  = f'net{NR}_small.h5'
        json_file = 'single_block_resnet.json'
        if not (os.path.exists(net_file) and os.path.exists(json_file)):
            raise FileNotFoundError(f"Network files not found ({net_file})")

        print(f"\n[Keras key-rank test]  Loading {net_file} …", flush=True)
        with open(json_file) as fh:
            net = model_from_json(fh.read())
        net.load_weights(net_file)

        DIFF   = (0x0040, 0x0)
        N_NET  = 50          # number of ciphertext pairs per trial
        N_NET_TRIALS = 200   # number of trials

        # tmp_br helper (from test_key_recovery.py)
        tmp_br_global = np.arange(2**14, dtype=np.uint16)
        tmp_br_global = np.repeat(tmp_br_global, N_NET).reshape(-1, N_NET)

        def bayesian_rank_kr_net(cand, emp_mean, m, s):
            """Direct port of bayesian_rank_kr from test_key_recovery.py."""
            global tmp_br_global
            n = len(cand)
            if tmp_br_global.shape[1] != n:
                tmp_br_global = np.arange(2**14, dtype=np.uint16)
                tmp_br_global = np.repeat(tmp_br_global, n).reshape(-1, n)
            tmp = tmp_br_global ^ cand
            v   = (emp_mean - m[tmp]) * s[tmp]
            v   = v.reshape(-1, n)
            return np.linalg.norm(v, axis=1)

        def one_keras_trial(net, m, s, n_blocks=N_NET, diff=DIFF, nr=NR):
            """Run one key-rank trial with actual network output."""
            from os import urandom
            pt0a = np.frombuffer(urandom(2*n_blocks), dtype=np.uint16).reshape(n_blocks,-1)
            pt1a = np.frombuffer(urandom(2*n_blocks), dtype=np.uint16).reshape(n_blocks,-1)
            pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1]
            pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0)
            pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0)
            key = np.frombuffer(urandom(8), dtype=np.uint16)
            ks  = sp.expand_key(key, nr); k_true = ks[nr-1]
            ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks)
            ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks)
            trial_keys = np.arange(2**16, dtype=np.uint16)
            c0a, c1a   = sp.dec_one_round((ct0a, ct1a), trial_keys)
            c0b, c1b   = sp.dec_one_round((ct0b, ct1b), trial_keys)
            c1a = np.tile(c1a, 2**16); c1b = np.tile(c1b, 2**16)
            X   = sp.convert_to_binary([c0a.flatten(), c1a.flatten(),
                                        c0b.flatten(), c1b.flatten()])
            Z   = net.predict(X, batch_size=10000, verbose=0)
            Z   = Z.flatten()
            emp_mean_full = Z.reshape(n_blocks, -1).mean(axis=0)  # shape (65536,)
            # subkey candidates = 14 lower bits enumerated, 2 upper bits random
            cand = np.arange(2**14, dtype=np.uint16)
            r_bits = np.random.randint(0, 4, 2**14, dtype=np.uint16) << 14
            cand = cand ^ r_bits
            scores_orig = bayesian_rank_kr_net(cand, emp_mean_full[cand], m, s)
            rank_orig   = int(np.sum(scores_orig < scores_orig[k_true & 0x3FFF]))
            return rank_orig, k_true

        print(f"  Running {N_NET_TRIALS} trials with {N_NET} pairs each …", flush=True)
        net_ranks_orig = []
        net_ranks_bct  = []

        for t in range(N_NET_TRIALS):
            # same trial with orig profile
            r_orig, k_true = one_keras_trial(net, m_orig, s_orig)
            net_ranks_orig.append(r_orig)
            # same trial with bct profile (re-use precomputed emp_mean by
            # running once more – independent trial for fairness)
            r_bct, _       = one_keras_trial(net, m_bct, s_bct_orig_std)
            net_ranks_bct.append(r_bct)
            if (t+1) % 20 == 0:
                print(f"  {t+1}/{N_NET_TRIALS}  orig top-1={np.mean(np.array(net_ranks_orig)==0)*100:.1f}%"
                      f"  bct top-1={np.mean(np.array(net_ranks_bct)==0)*100:.1f}%", flush=True)

        net_ranks_orig = np.array(net_ranks_orig)
        net_ranks_bct  = np.array(net_ranks_bct)
        print("\n[Keras network key-rank results]")
        print("-" * 60)
        print_rank_stats(f'Original ({NR}r, network)', net_ranks_orig, 2**14)
        print_rank_stats(f'BCT      ({NR}r, network)', net_ranks_bct,  2**14)

        # save keras rank results
        np.save(os.path.join(OUTDIR, f'net_ranks_orig_{NR}r.npy'), net_ranks_orig)
        np.save(os.path.join(OUTDIR, f'net_ranks_bct_{NR}r.npy'),  net_ranks_bct)

    except Exception as e:
        print(f"\n[keras test skipped: {e}]")

# ── 8.  Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"BCT differential:  alpha=0x{A:04X}/mask=0x{AM:04X},"
      f"  beta=0x{B:04X}/mask=0x{BM:04X}")
print(f"Original profile:  m[0]={m_orig[0]:.4f}  mean={m_orig.mean():.4f}")
print(f"BCT profile:       m[0]={m_bct[0]:.4f}  mean={m_bct.mean():.4f}")
print(f"Pearson r(orig,bct)  = {r_p:.4f}")
print(f"Spearman ρ(orig,bct) = {r_s:.4f}")
print(f"\nSynthetic key-rank test (n={args.n_rank_trials}, n_cand={args.n_cand}):")
for ns, res in all_results.items():
    print(f"  noise σ={ns}:")
    print(f"    Original:        top-1={np.mean(res['orig']==0)*100:.1f}%"
          f"  median={np.median(res['orig']):.1f}")
    print(f"    BCT (orig std):  top-1={np.mean(res['bct']==0)*100:.1f}%"
          f"  median={np.median(res['bct']):.1f}")
    print(f"    BCT (flat std):  top-1={np.mean(res['bct_flat']==0)*100:.1f}%"
          f"  median={np.median(res['bct_flat']):.1f}")
