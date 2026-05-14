#!/usr/bin/env python3
"""
BCT_truncated_self scatter plot for the specific truncated differential:
  a=0xC6A5, a_mask=0xE7E1, b=0x1C1E, b_mask=0xE3E1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------------------
A, AM = 0xC6A5, 0x2152
B, BM = 0x0000, 0x0000

n_free_a = bin(AM).count('1')
n_free_b = bin(BM).count('1')
class_size = 2 ** (n_free_a + n_free_b)

print(f"alpha = 0x{A:04X}, a_mask = 0x{AM:04X}  ({n_free_a} free bits)")
print(f"beta  = 0x{B:04X}, b_mask = 0x{BM:04X}  ({n_free_b} free bits)")
print(f"Truncated class size: 2^{n_free_a+n_free_b} = {class_size:,}")
print("Computing BCT_truncated_self for all 65536 g values ...", flush=True)

g_arr = np.arange(65536)
p = compute_all_g_self(A, AM, B, BM)

print(f"  max={p.max():.6f}  mean={p.mean():.6f}  min={p.min():.6f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Bit pattern strings for axis labels
def fmt_bits(val, mask):
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

alpha_pattern = fmt_bits(A, AM)
beta_pattern  = fmt_bits(B, BM)

fig, (ax_scatter, ax_line) = plt.subplots(
    2, 1, figsize=(14, 8), dpi=150,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35})

title = (
    f'Self-returning Truncated BCT:  '
    r'$p^{\rm self}_{\rm trunc}(g)$  for all $g \in [0,\,2^{16})$' + '\n' +
    r'$\alpha = \mathtt{' + f'{alpha_pattern}' + r'}$  '
    r'$\beta = \mathtt{' + f'{beta_pattern}' + r'}$  '
    f'   (? = free,  class size = $2^{{{n_free_a+n_free_b}}}$)'
)
fig.suptitle(title, fontsize=12, fontweight='bold', y=0.99)

# --- scatter plot ---
ax_scatter.plot(g_arr, p, ',', color='#2E8B57', alpha=0.85, markersize=0.5)
ax_scatter.set_ylabel(r'$p^{\rm self}_{\rm trunc}(g)$', fontsize=11)
ax_scatter.set_xlim(0, 65536)
ax_scatter.set_ylim(-0.02, 1.05)
ax_scatter.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.5)
ax_scatter.grid(True, alpha=0.15)
ax_scatter.text(0.99, 0.97,
        f'max  = {p.max():.5f}\nmean = {p.mean():.5f}\nmin  = {p.min():.5f}',
        transform=ax_scatter.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='gray'))
ax_scatter.text(0.01, 0.97,
        f'alpha fixed: 0x{A & ~AM & 0xFFFF:04X}\n'
        f'beta  fixed: 0x{B & ~BM & 0xFFFF:04X}',
        transform=ax_scatter.transAxes, fontsize=9, va='top', ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', alpha=0.9, ec='gray'))

# --- line plot ---
ax_line.plot(g_arr, p, color='#2E8B57', lw=0.4, alpha=0.85)
ax_line.set_xlabel(r'$g = k_r \oplus k_g$', fontsize=11)
ax_line.set_ylabel(r'$p^{\rm self}_{\rm trunc}(g)$', fontsize=11)
ax_line.set_xlim(0, 65536)
ax_line.set_ylim(-0.02, 1.05)
ax_line.axhline(0.5, color='gray', lw=0.7, ls='--', alpha=0.5)
ax_line.grid(True, alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
outpath = '/home/user/wkrh_bct/truncated_bct_self_C6A5.png'
plt.savefig(outpath, bbox_inches='tight', facecolor='white')
print(f"Saved -> {outpath}")
