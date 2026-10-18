"""HW3 Problem 1 (Newman 7.2): detecting the periodicity of the sunspot record.

data/sunspots.txt holds the monthly sunspot number since January 1749 (column 1:
month index, column 2: sunspot number).  (a) The cycle length is estimated
directly from the time series, by locating the maxima of the 13-month smoothed
series.  (b) The power spectrum |c_k|^2 of the discrete Fourier transform shows
a peak at k ~ 24.  (c) Its period N/k is compared with the estimate from (a).
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.signal import find_peaks

from plotstyle import use_science

use_science()

month, spots = np.loadtxt("data/sunspots.txt", unpack=True)
N = len(spots)
year = 1749 + month / 12.0
print(f"{N} months of data, {year[0]:.0f} to {year[-1]:.2f}")

# --- (a) cycle length from the time series ----------------------------------
# The standard 13-month running mean (half weight on the two end months) removes
# the month-to-month scatter; its maxima are the solar maxima.
w = np.ones(13)
w[[0, -1]] = 0.5
w /= w.sum()
smooth = np.convolve(spots, w, mode="same")
smooth[:6] = smooth[-6:] = np.nan
# Solar maxima are at least ~7 years apart and stand well above the minima.
peaks, _ = find_peaks(np.nan_to_num(smooth), distance=84, prominence=40)
gaps = np.diff(peaks)
T_a = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
print(f"(a) {len(peaks)} maxima of the smoothed series, first in {year[peaks[0]]:.1f}, "
      f"last in {year[peaks[-1]]:.1f}")
print(f"    spacing between maxima: {gaps.min()} to {gaps.max()} months, "
      f"standard deviation {gaps.std(ddof=1):.0f} months")
print(f"    mean cycle length (last - first) / {len(peaks) - 1} = {T_a:.1f} months "
      f"= {T_a / 12:.2f} years")

# --- (b) power spectrum -------------------------------------------------------
c = rfft(spots)                 # c_k = sum_n y_n exp(-2 pi i k n / N), k = 0..N/2
power = np.abs(c) ** 2
k = np.arange(len(c))

# --- (c) the peak -------------------------------------------------------------
kpk = np.argmax(power[1:]) + 1  # skip k = 0, which is just N times the mean
T_c = N / kpk
top = np.argsort(power[1:])[::-1][:5] + 1
print(f"(b) |c_0|^2 = {power[0]:.3e} (the mean, N <y> = {N * spots.mean():.4g} squared)")
print(f"    largest |c_k|^2 for k > 0: k = {', '.join(map(str, top))}")
print(f"(c) peak at k = {kpk}: |c_k|^2 = {power[kpk]:.3e}, "
      f"{power[kpk] / np.median(power[1:]):.0f} times the median of |c_k|^2")
print(f"    period N/k = {N}/{kpk} = {T_c:.1f} months = {T_c / 12:.2f} years")

# The peak is several bins wide because the cycle length varies from cycle to
# cycle; quote the power-weighted mean frequency over the bins of the peak.
band = (k >= 20) & (k <= 28)
kbar = np.sum(k[band] * power[band]) / np.sum(power[band])
print(f"    peak bins k = 20..28 cover periods {N / 28:.0f}-{N / 20:.0f} months; "
      f"power-weighted mean k = {kbar:.2f} -> period {N / kbar:.1f} months")
print(f"    difference from (a): {abs(T_c - T_a):.1f} months "
      f"({abs(T_c - T_a) / T_a * 100:.1f}%), well inside the {gaps.std(ddof=1):.0f}-month "
      f"cycle-to-cycle scatter")

# --- figure -------------------------------------------------------------------
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.6, 1],
                      hspace=0.38, wspace=0.22)

ax = fig.add_subplot(gs[0, :])
ax.plot(year, spots, lw=0.4, color="0.6", label="monthly sunspot number")
ax.plot(year, smooth, lw=1.0, color="C0", label="13-month running mean")
ax.plot(year[peaks], smooth[peaks], "v", ms=4, color="C3",
        label=f"{len(peaks)} solar maxima")
ax.set_xlim(year[0], year[-1])
ax.set_ylim(0, 300)
ax.set_xlabel("year")
ax.set_ylabel("sunspot number")
ax.set_title(f"(a) Sunspot record: mean spacing of maxima {T_a:.0f} months "
             f"({T_a / 12:.1f} yr)", fontsize=8)
ax.legend(fontsize=6.5, loc="upper left", ncol=3)

ax = fig.add_subplot(gs[1, 0])
ax.plot(k[1:], power[1:], lw=0.8)
ax.plot(kpk, power[kpk], "o", ms=3.5, color="C3")
ax.annotate(f"$k = {kpk}$\n$N/k = {T_c:.0f}$ months", xy=(kpk, power[kpk]),
            xytext=(kpk + 25, power[kpk] * 0.75), fontsize=7, color="C3",
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
ax.set_xlim(0, 300)
ax.set_ylim(0, power[kpk] * 1.08)
ax.set_xlabel("$k$")
ax.set_ylabel(r"$|c_k|^2$")
ax.set_title(r"(b) Power spectrum ($k = 0$ omitted)", fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.semilogy(k[1:], power[1:], lw=0.8)
ax.axvspan(19.5, 28.5, color="C1", alpha=0.15, lw=0)
ax.plot(kpk, power[kpk], "o", ms=3.5, color="C3")
ax.axhline(np.median(power[1:]), color="0.4", ls="--", lw=0.7)
ax.text(52, np.median(power[1:]) * 0.4, "median", fontsize=6.5, color="0.3")
ax.text(40, power[kpk] * 0.12, "peak bins\n$k = 20$--$28$", fontsize=6.5, color="C1")
ax.set_xlim(0, 60)
ax.set_xlabel("$k$")
ax.set_ylabel(r"$|c_k|^2$")
ax.set_title("(c) Close-up of the peak", fontsize=8)

fig.savefig("figs/problem1_sunspots.png")
