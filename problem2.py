"""HW3 Problem 2 (Newman 7.4): Fourier filtering and smoothing of the Dow Jones.

data/dow.txt holds the daily closing value of the Dow Jones Industrial Average
for 1024 business days from late 2006 to the end of 2010.  Its real FFT has
N/2 + 1 = 513 coefficients.  Keeping only the first 10% (or 2%) of them and
transforming back is a low-pass filter: it removes every variation faster than
N/k_cut days and leaves a smoothed curve.  The filtered curve misbehaves at the
two ends because the DFT treats the series as periodic, and the last value is
not equal to the first.  A discrete cosine transform, which mirrors the series
before transforming it, removes that wrap-around pull (shown for comparison).
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from scipy.fft import dct, idct

from plotstyle import use_science

use_science()
PCT = r"\%" if plt.rcParams["text.usetex"] else "%"   # % starts a LaTeX comment

dow = np.loadtxt("data/dow.txt")
N = len(dow)
day = np.arange(N)
print(f"(a) {N} business days, close from {dow[0]:.2f} to {dow[-1]:.2f} "
      f"(min {dow.min():.2f} on day {dow.argmin()}, max {dow.max():.2f} on day {dow.argmax()})")

# --- (b) DFT ------------------------------------------------------------------
c = rfft(dow)
print(f"(b) rfft gives {len(c)} = N/2 + 1 complex coefficients")


def lowpass(y, fraction):
    """Zero all but the first `fraction` of the rfft coefficients and invert."""
    c = rfft(y)
    keep = int(fraction * len(c))
    c[keep:] = 0.0
    return irfft(c, n=len(y)), keep


def lowpass_dct(y, fraction):
    """Same, with the type-II DCT (the series is mirrored, not wrapped)."""
    a = dct(y, norm="ortho")
    keep = int(fraction * len(a))
    a[keep:] = 0.0
    return idct(a, norm="ortho"), keep


results = {}
for frac, label in [(0.10, "(c, d)"), (0.02, "(e)")]:
    smooth, keep = lowpass(dow, frac)
    resid = dow - smooth
    sm_dct, keep_dct = lowpass_dct(dow, frac)
    resid_dct = dow - sm_dct
    results[frac] = (smooth, keep, sm_dct)
    # The highest frequency kept, k = keep - 1, has period N/(keep - 1) days.
    print(f"{label} keep the first {frac:.0%}: k = 0..{keep - 1} "
          f"({keep} of {len(c)} coefficients)")
    print(f"    shortest period kept N/k = {N / (keep - 1):.1f} business days "
          f"(~{N / (keep - 1) / 21:.1f} months)")
    print(f"    rms residual |data - filtered| = {np.sqrt(np.mean(resid ** 2)):.1f}")
    print(f"    filtered end points: {smooth[0]:.0f} and {smooth[-1]:.0f} "
          f"(data {dow[0]:.0f} and {dow[-1]:.0f}; both pulled towards the "
          f"midpoint {0.5 * (dow[0] + dow[-1]):.0f})")
    print(f"    DCT with the same fraction ({keep_dct} of {N} coefficients): "
          f"rms residual {np.sqrt(np.mean(resid_dct ** 2)):.1f}, "
          f"end points {sm_dct[0]:.0f} and {sm_dct[-1]:.0f}")

print(f"    the series jumps by {dow[-1] - dow[0]:+.0f} when wrapped from the "
      f"last day to the first")

# --- figure -------------------------------------------------------------------
fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.35, wspace=0.28)

ax = fig.add_subplot(gs[0, :])
ax.plot(day, dow, lw=0.6, color="0.55", label="daily close")
for frac, col in [(0.10, "C0"), (0.02, "C3")]:
    smooth, keep, _ = results[frac]
    ax.plot(day, smooth, lw=1.3, color=col,
            label=rf"first {frac * 100:.0f}{PCT} of $c_k$ kept ($k < {keep}$)")
ax.set_xlim(0, N - 1)
ax.set_xlabel("business day (late 2006 to end of 2010)")
ax.set_ylabel("Dow Jones Industrial Average")
ax.set_title("(a, d, e) Daily closing value and its Fourier low-pass filtered versions",
             fontsize=8)
ax.legend(fontsize=7, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
k = np.arange(len(c))
ax.semilogy(k[1:], np.abs(c[1:]), lw=0.6, color="0.3")
for frac, col in [(0.10, "C0"), (0.02, "C3")]:
    keep = results[frac][1]
    ax.axvline(keep - 0.5, color=col, ls="--", lw=0.9)
    ax.text(keep + 6, 2e5 if frac == 0.10 else 6e5, f"{frac * 100:.0f}{PCT}", color=col, fontsize=7)
ax.semilogy(k[1:], np.abs(dow[-1] - dow[0]) / (2 * np.sin(np.pi * k[1:] / N)),
            "k:", lw=0.8, label=r"$|y_{N-1} - y_0| / 2\sin(\pi k/N)$")
ax.set_xlim(0, len(c))
ax.set_xlabel("$k$")
ax.set_ylabel(r"$|c_k|$")
ax.set_title("(b) Fourier coefficients", fontsize=8)
ax.legend(fontsize=6, loc="upper right")

for j, (lo, hi, title) in enumerate([(0, 120, "first 120 days"),
                                     (N - 120, N, "last 120 days")]):
    ax = fig.add_subplot(gs[1, 1 + j])
    ax.plot(day[lo:hi], dow[lo:hi], lw=0.7, color="0.55")
    for frac, col in [(0.10, "C0"), (0.02, "C3")]:
        smooth, _, sm_dct = results[frac]
        ax.plot(day[lo:hi], smooth[lo:hi], lw=1.2, color=col)
        ax.plot(day[lo:hi], sm_dct[lo:hi], lw=1.0, ls="--", color=col)
    ax.plot([], [], "k-", lw=1.0, label="DFT (rfft)")
    ax.plot([], [], "k--", lw=1.0, label="DCT")
    ax.set_xlim(lo, hi - 1)
    ax.set_xlabel("business day")
    ax.set_title(f"End effects: {title}", fontsize=8)
    ax.legend(fontsize=6, loc="lower left" if j == 0 else "upper left")

fig.savefig("figs/problem2_dow.png")
