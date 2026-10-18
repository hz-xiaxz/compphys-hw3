"""HW3 Problem 3 (Newman 7.9): image deconvolution.

data/blur.txt is a 1024 x 1024 black-and-white photo blurred with a Gaussian
point spread function (PSF) of width sigma = 25 pixels.  A periodic blur is a
circular convolution, b = a * f, so in Fourier space B = A F and the sharp photo
is a = irfft2(B / F).  F falls off as a Gaussian, exp(-2 pi^2 sigma^2 q^2), so
Fourier modes with |F| < eps are left undivided (Newman's prescription).

The PSF is normalised to unit sum, so F(0) = 1 and |F(q)| is the factor by which
the blur attenuates the mode q; eps is then a relative attenuation threshold.
(For the unnormalised PSF exp(-r^2 / 2 sigma^2), whose sum is 2 pi sigma^2, the
same cut would be at eps * 2 pi sigma^2.)

Part (d): the data are stored to five decimals, so they carry a rounding noise
of rms 1e-5 / sqrt(12).  Dividing by F multiplies that noise by 1/|F|, which
sets how small eps can be, and the Gaussian fall-off of F means that the
resolution grows only as sqrt(ln(1/eps)).
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq

from plotstyle import use_science

use_science()

SIGMA = 25.0
EPS_BOOK = 1e-3
EPS_SCAN = [1e-3, 1e-6, 1e-8, 1e-9, 1e-10]
DIGITS = 5                        # blur.txt is written with five decimals

# --- (a) the blurred photo ----------------------------------------------------
blur = np.loadtxt("data/blur.txt")
K, L = blur.shape
print(f"(a) blurred photo: {K} x {L}, brightness {blur.min():.2f} to {blur.max():.2f}")
# Row 0 of the file is the top of the picture: imshow's default origin="upper"
# puts the bright sky at the top and the dark ground at the bottom.


# --- (b) the point spread function ----------------------------------------------
def gaussian_psf(shape, sigma):
    """Periodic Gaussian centred on pixel (0, 0), normalised to unit sum.

    Pixel j stands for the displacement min(j, K - j): negative displacements
    wrap around to the end of each axis, which puts a quarter of the Gaussian
    in each corner of the array.
    """
    K, L = shape
    x = np.minimum(np.arange(K), K - np.arange(K))
    y = np.minimum(np.arange(L), L - np.arange(L))
    f = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / (2 * sigma ** 2))
    return f / f.sum()


psf = gaussian_psf(blur.shape, SIGMA)
print(f"(b) PSF: peak-1 Gaussian sums to 2 pi sigma^2 = {2 * np.pi * SIGMA ** 2:.2f}; "
      f"normalised to unit sum, so the peak is {psf.max():.3e}")


# --- (c) deconvolution ----------------------------------------------------------
def deconvolve(b, f, eps):
    """Divide rfft2(b) by rfft2(f) wherever |rfft2(f)| > eps, then invert.

    Coefficients where the PSF transform is below eps are left as they are.
    Returns the deblurred image and the fraction of modes that were divided.
    """
    B = rfft2(b)
    F = rfft2(f)
    use = np.abs(F) > eps
    A = B.copy()
    A[use] = B[use] / F[use]
    return irfft2(A, s=b.shape), use.mean()


F = rfft2(psf)
q = np.hypot(fftfreq(K)[:, None], rfftfreq(L)[None, :])     # cycles per pixel
F_model = np.exp(-2 * np.pi ** 2 * SIGMA ** 2 * q ** 2)
print(f"    |F - exp(-2 pi^2 sigma^2 q^2)| <= {np.abs(F - F_model).max():.1e} "
      f"(F is real and positive to {np.abs(F.imag).max():.0e})")


def q_cut(eps):
    """Spatial frequency where the Gaussian transfer function drops to eps."""
    return np.sqrt(np.log(1 / eps) / (2 * np.pi ** 2 * SIGMA ** 2))


# Rounding noise alone, pushed through the same deconvolution.
rng = np.random.default_rng(1)
noise = rng.uniform(-0.5, 0.5, blur.shape) * 10.0 ** -DIGITS
sigma_round = 10.0 ** -DIGITS / np.sqrt(12)

images = {}
print(f"(c) deconvolution, keeping modes with |F| > eps "
      f"(image dynamic range {np.ptp(blur):.0f}, blur FWHM {2.3548 * SIGMA:.0f} px)")
print("    eps     modes divided  q_c [1/px]  1/q_c [px]  noise rms  reblur residual (max)")
for eps in sorted(set(EPS_SCAN + [EPS_BOOK, 1e-2, 1e-12]), reverse=True):
    img, frac = deconvolve(blur, psf, eps)
    noise_out, _ = deconvolve(noise, psf, eps)
    reblur = irfft2(rfft2(img) * F, s=blur.shape)
    images[eps] = img
    print(f"    {eps:.0e}  {frac:12.2e}  {q_cut(eps):10.4f}  {1 / q_cut(eps):10.1f}  "
          f"{noise_out.std():9.2e}  {np.abs(reblur - blur).max():.1e}")
print(f"    rounding noise in the data: rms {sigma_round:.2e}; the reblurred image "
      f"matches the data to that level once eps <~ 1e-7")

# Radial profile of |B(q)|: signal at low q, a flat rounding-noise floor beyond.
B = rfft2(blur)
qbins = np.linspace(0, 0.5, 201)
qc = 0.5 * (qbins[1:] + qbins[:-1])
which = np.digitize(q.ravel(), qbins) - 1
absB = np.abs(B).ravel()
B_med = np.array([np.median(absB[which == i]) for i in range(len(qc))])
floor_rms = np.sqrt(K * L) * sigma_round              # rms of a noise coefficient
floor_med = B_med[(qc > 0.1) & (qc < 0.5)].mean()
print(f"(d) |B(q)| levels off at q > {qc[np.argmax(B_med < 2 * floor_med)]:.3f}: "
      f"median plateau {floor_med:.2e} vs predicted rms {floor_rms:.2e} "
      f"(ratio {floor_med / floor_rms:.2f}; sqrt(ln 2) = {np.sqrt(np.log(2)):.2f} for "
      f"complex Gaussian noise)")
# Output noise ~ C sigma_in / eps (C from the eps = 1e-8 run); requiring it to
# stay below 1% of the brightness range gives the smallest usable eps.
noise_8, _ = deconvolve(noise, psf, 1e-8)
C = noise_8.std() * 1e-8 / sigma_round
eps_min = C * sigma_round / (0.01 * np.ptp(blur))
print(f"    output noise rms ~ C sigma_in / eps with C = {C:.3f}")
print(f"    noise < 1% of the range needs eps > C sigma_in / (0.01 range) = {eps_min:.1e} "
      f"here (5-decimal data), 1/q_c = {1 / q_cut(eps_min):.0f} px")
eps_8bit = C * (1 / 256 / np.sqrt(12)) / 0.01
print(f"    for an 8-bit photo (sigma_in = range / (256 sqrt 12)) the same rule gives "
      f"eps = {eps_8bit:.1e}, 1/q_c = {1 / q_cut(eps_8bit):.0f} px")
print(f"    eps 1e-3 -> 1e-8 multiplies the noise by 1e5 but the resolution 1/q_c "
      f"only by {q_cut(1e-8) / q_cut(1e-3):.2f}")

# --- figure 1: the photo, the PSF, and two deblurred versions ------------------
EPS_BEST = 1e-8
fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.6), gridspec_kw=dict(hspace=0.14, wspace=0.06))
lo, hi = np.percentile(images[EPS_BEST], [0.2, 99.8])
panels = [(blur, "(a) blurred photo (\\texttt{blur.txt})" if plt.rcParams["text.usetex"]
           else "(a) blurred photo (blur.txt)", (blur.min(), blur.max())),
          (psf, rf"(b) periodic Gaussian PSF, $\sigma = {SIGMA:.0f}$ px", (0, psf.max())),
          (images[EPS_BOOK], rf"(c) deblurred, $\epsilon = 10^{{{np.log10(EPS_BOOK):.0f}}}$",
           (lo, hi)),
          (images[EPS_BEST], rf"(c) deblurred, $\epsilon = 10^{{{np.log10(EPS_BEST):.0f}}}$",
           (lo, hi))]
for ax, (img, title, (vmin, vmax)) in zip(axes.flat, panels):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="lanczos")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
fig.savefig("figs/problem3_deblur.png", dpi=200)

# --- figure 2: how far can eps be pushed? ----------------------------------------
fig = plt.figure(figsize=(11, 5.9))
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.75], hspace=0.0)
axes = top.subplots(1, len(EPS_SCAN), gridspec_kw=dict(wspace=0.06))
r0, c0, n = 230, 40, 400                     # crop: the left gable
for ax, eps in zip(axes, EPS_SCAN):
    crop = images[eps][r0:r0 + n, c0:c0 + n]
    ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="lanczos")
    ax.set_title(rf"$\epsilon = 10^{{{np.log10(eps):.0f}}}$, $1/q_c = {1 / q_cut(eps):.0f}$ px",
                 fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

ax, axn = bottom.subplots(1, 2, gridspec_kw=dict(wspace=0.32))
Fq = np.exp(-2 * np.pi ** 2 * SIGMA ** 2 * qc ** 2)
ax.semilogy(qc, B_med, lw=1.2, label=r"blurred, $|\tilde b(q)|$ (ring median)")
ok = Fq > 1e-14
ax.semilogy(qc[ok], B_med[ok] / Fq[ok], lw=1.0, color="C4",
            label=r"deblurred, $|\tilde b(q)| / \tilde f(q)$")
ax.axhline(floor_rms, color="C3", ls="--", lw=0.9,
           label=r"rounding noise, $\sqrt{KL}\,10^{-5}/\sqrt{12}$")
for eps, col in [(EPS_BOOK, "C1"), (EPS_BEST, "C2")]:
    ax.axvline(q_cut(eps), color=col, lw=0.9)
    ax.text(q_cut(eps) - 0.0015, 1e-3, rf"$q_c(10^{{{np.log10(eps):.0f}}})$",
            color=col, fontsize=6.5, rotation=90, va="bottom", ha="right")
ax.set_xlim(0, 0.08)
ax.set_ylim(1e-4, 1e10)
ax.set_xlabel(r"spatial frequency $q$ [cycles / pixel]")
ax.set_ylabel("Fourier amplitude")
ax.set_title("(d) Spectrum before and after dividing by the PSF", fontsize=8)
ax.legend(fontsize=6, loc="upper right")

eps_grid = np.logspace(-12, -1, 23)
noise_rms = []
for eps in eps_grid:
    out, _ = deconvolve(noise, psf, eps)
    noise_rms.append(out.std())
axn.loglog(eps_grid, noise_rms, "o-", ms=2.5, lw=1.0, color="C3")
axn.axhline(np.ptp(blur) / 100, color="C3", ls=":", lw=0.8)
axn.text(5e-2, np.ptp(blur) / 100 * 2, r"1\% of the brightness range"
         if plt.rcParams["text.usetex"] else "1% of the brightness range",
         fontsize=6.5, color="C3")
axn.set_xlabel(r"threshold $\epsilon$")
axn.set_ylabel("rms of rounding noise after deconvolution", color="C3")
axn.invert_xaxis()
ax2 = axn.twinx()
ax2.semilogx(eps_grid, 1 / q_cut(eps_grid), "s-", ms=2.5, lw=1.0, color="C0")
ax2.set_ylabel(r"resolution $1/q_c$ [px]", color="C0")
ax2.grid(False)
axn.set_title(r"(d) Noise grows as $1/\epsilon$, resolution only as $\sqrt{\ln(1/\epsilon)}$",
              fontsize=8)

fig.savefig("figs/problem3_limits.png", dpi=170)
