#!/usr/bin/env python3
"""
yinyang_contour_gui.py
======================

Interactive GUI (Tkinter + Matplotlib) for contour plots of a 2-D scalar field
defined on both panels of a Yin-Yang grid (Kageyama & Sato 2004).

Grid convention
---------------
Each panel is a latitude-longitude patch in its *own* spherical coordinates:

    colatitude  theta in [45 deg - d, 135 deg + d]
    longitude   phi   in [-135 deg - d, 135 deg + d]     (d = optional overlap margin)

The Yang panel's Cartesian frame is related to the Yin (global) frame by

    (x_yang, y_yang, z_yang) = (-x_yin, z_yin, y_yin)      (an involution)

GEM model output
----------------
Point the viewer at a GEM output folder (e.g.
~/gem-macos/.mnt/GEMrun/RUNMOD/output/cfg_0000/laststep_0000000008/001-001)
or at one of its RPN standard files. The files are decoded in pure Python
(ported from ECCC's librmn: XDF directory, datyp 1/5/133/134 packings), so no
librmn or rpnpy installation is needed. All Yin-Yang ('U' grid) records are
listed by variable, level and time; the grid's '^>' record supplies the panel
coordinates and the grid rotation, so the global maps, globe and spectrum use
true geographic latitude/longitude. In the native-panel view, panel longitude
phi is GEM's rotated longitude minus 180 degrees.

Other input
-----------
A single 2-D array holding BOTH panels, stacked either

    rows    : Yin above Yang   -> shape (2*n_theta, n_phi)
    columns : Yin left of Yang -> shape (n_theta, 2*n_phi)

Each panel is indexed F[i_theta, j_phi]. The "Panel layout" setting chooses
the split; "Auto-detect" picks the one giving panels with n_phi ~ 3 n_theta
(or uses theta/phi when the file provides them). If the panels are stored
as (n_phi, n_theta), tick "Transpose panels" (guessed automatically).

File formats:
  * .npy                 the 2-D array
  * .npz                 the 2-D array (named 'field' or 'data', or the only 2-D
                         array in the file), optionally with 1-D 'theta' and 'phi'
                         (panel coordinates, radians or degrees; auto-detected)
  * .txt / .dat / .csv   the 2-D array as text
If the file has no theta/phi, the grid extent typed in the GUI is used
(uniform spacing, endpoints included). A built-in demo field is also available.

Comparing two fields
--------------------
Load a second field (B) in the same format; it must split into panels of the
same shape (the layout/transpose settings and grid of field A are applied).
"Show" then selects Field A, Field B or the difference A - B, and every view
(native panels, global maps, sphere, power spectrum) displays that choice.

Views
-----
  1. Native panels  - Yin above Yang, each in its own (phi, latitude) coords
  2. Global map     - both panels merged onto a regular lon-lat grid (equirectangular)
  3. Global map     - same, Mollweide projection
  4. Sphere         - filled contours (contourf) on an orthographic globe drawn with
                      Basemap (view centre adjustable, optional far-side globe);
                      without Basemap the same view is drawn with Matplotlib's contourf;
                      click and drag a globe to rotate it
  5. Power spectrum - spherical-harmonic power per degree S(l) of the merged field,
                      computed with numpy (Gauss-Legendre quadrature, no extra packages);
                      S(l) = (1/4pi) sum_m |f_lm|^2, so that sum_l S(l) = <f^2>
                      (the 4pi-normalised convention used by pyshtools)

Usage
-----
    python yinyang_contour_gui.py                       # GEM default folder if present, else demo
    python yinyang_contour_gui.py path/to/001-001 --var TT
    python yinyang_contour_gui.py runA/001-001 runB/001-001 --var TT --diff   # compare runs
    python yinyang_contour_gui.py field.npy
    python yinyang_contour_gui.py field.txt --layout rows     # Yin above Yang
    python yinyang_contour_gui.py run1.npy run2.npy --diff    # compare: show run1 - run2

or from Python:
    from yinyang_contour_gui import show
    show(field_2d, theta=None, phi=None, layout="auto")
    show(field_a, field_b=field_b, mode="diff")               # A - B

Requires: numpy, matplotlib (Tkinter ships with most Python installs).
Optional: basemap  (pip install basemap  or  conda install -c conda-forge basemap)
          for the sphere view.
"""

import argparse
import datetime as _dt
import glob as _glob
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
from matplotlib import cm as mcm

DEG = np.pi / 180.0
# Where the GEM run's output lives on this Mac (used as the default folder).
GEM_DEFAULT_DIR = os.path.expanduser(
    "~/gem-macos/.mnt/GEMrun/RUNMOD/output/cfg_0000/laststep_0000000008/001-001")
GEM_PREFERRED_VARS = ("TT", "GZ", "P0", "PN", "UU", "VV", "HU", "TD", "WW", "PR")
THETA_NOM = (45.0 * DEG, 135.0 * DEG)     # nominal (non-overlapping) extents
PHI_NOM = (-135.0 * DEG, 135.0 * DEG)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def sph2cart(theta, phi):
    st = np.sin(theta)
    return st * np.cos(phi), st * np.sin(phi), np.cos(theta)


def cart2sph(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    return np.arccos(np.clip(z / r, -1.0, 1.0)), np.arctan2(y, x)


def yang_to_yin_xyz(x, y, z):
    """Cartesian transform between panel frames (it is its own inverse)."""
    return -x, z, y


def swap_panel(theta, phi):
    """(theta, phi) in one panel's coordinates -> the other panel's coordinates."""
    return cart2sph(*yang_to_yin_xyz(*sph2cart(theta, phi)))


def nominal_boundary(n=181):
    """Closed curve (theta, phi) tracing the nominal panel rectangle."""
    t0, t1 = THETA_NOM
    p0, p1 = PHI_NOM
    s = np.linspace(0.0, 1.0, n)
    th = np.concatenate([np.full(n, t0), t0 + (t1 - t0) * s,
                         np.full(n, t1), t1 + (t0 - t1) * s])
    ph = np.concatenate([p0 + (p1 - p0) * s, np.full(n, p1),
                         p1 + (p0 - p1) * s, np.full(n, p0)])
    return th, ph


def break_wraps(lon, lat, jump=np.pi):
    """Insert NaNs where longitude jumps across the dateline (for line plots)."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    idx = np.where(np.abs(np.diff(lon)) > jump)[0] + 1
    return np.insert(lon, idx, np.nan), np.insert(lat, idx, np.nan)


# ---------------------------------------------------------------------------
# Data container and interpolation
# ---------------------------------------------------------------------------
class YYData:
    """Field on both Yin-Yang panels sharing the same (theta, phi) grid (radians).

    rot (optional 3x3): rotation taking geographic Cartesian coordinates to the Yin
    panel's frame (x_yin = rot @ x_geo); None means the Yin frame is geographic.
    GEM files supply it from the grid's (xlat1, xlon1, xlat2, xlon2).
    """

    def __init__(self, yin, yang, theta, phi, name="field", rot=None):
        yin = np.asarray(yin, dtype=float)
        yang = np.asarray(yang, dtype=float)
        theta = np.asarray(theta, dtype=float).ravel()
        phi = np.asarray(phi, dtype=float).ravel()
        if yin.ndim != 2 or yang.ndim != 2:
            raise ValueError("Yin and Yang fields must be 2-D arrays.")
        if yin.shape != yang.shape:
            raise ValueError(f"Yin {yin.shape} and Yang {yang.shape} shapes differ.")
        if yin.shape != (theta.size, phi.size):
            raise ValueError(f"Field shape {yin.shape} does not match grid "
                             f"(n_theta={theta.size}, n_phi={phi.size}).")
        if theta.size < 2 or phi.size < 2:
            raise ValueError("Need at least 2 points in each direction.")
        if theta[0] > theta[-1]:
            theta, yin, yang = theta[::-1], yin[::-1], yang[::-1]
        if phi[0] > phi[-1]:
            phi, yin, yang = phi[::-1], yin[:, ::-1], yang[:, ::-1]
        self.yin, self.yang, self.theta, self.phi, self.name = yin, yang, theta, phi, name
        self.rot = None if rot is None else np.asarray(rot, float).reshape(3, 3)

    @property
    def shape(self):
        return self.yin.shape

    def with_values(self, yin, yang, name):
        """Same grid and rotation, new values (e.g. a difference field)."""
        return YYData(yin, yang, self.theta, self.phi, name, rot=self.rot)


def geo_to_yin(data, TH, PH):
    """Geographic (colatitude, longitude) [rad] -> Yin-panel (theta, phi) [rad]."""
    if getattr(data, "rot", None) is None:
        return TH, PH
    x = np.stack(sph2cart(TH, PH))
    y = np.tensordot(data.rot, x, axes=1)
    return cart2sph(*y)


def yin_to_geo(data, th, ph):
    """Yin-panel (theta, phi) [rad] -> geographic (colatitude, longitude) [rad]."""
    if getattr(data, "rot", None) is None:
        return th, ph
    x = np.stack(sph2cart(th, ph))
    g = np.tensordot(data.rot.T, x, axes=1)
    return cart2sph(*g)


def panel_edges_geo(data):
    """Nominal Yin and Yang panel edges as geographic (lon, lat) in degrees."""
    th, ph = nominal_boundary()
    th2, ph2 = swap_panel(th, ph)
    out = []
    for t, p in ((th, ph), (th2, ph2)):
        gt, gp = yin_to_geo(data, t, p)
        out.append((gp / DEG, 90 - gt / DEG))
    return out


def bilinear(theta, phi, F, tq, pq):
    """Bilinear interpolation of F(theta, phi) at query points; NaN outside."""
    tq = np.asarray(tq, float)
    pq = np.asarray(pq, float)
    nt, nph = theta.size, phi.size
    i = np.clip(np.searchsorted(theta, tq) - 1, 0, nt - 2)
    j = np.clip(np.searchsorted(phi, pq) - 1, 0, nph - 2)
    wt = (tq - theta[i]) / (theta[i + 1] - theta[i])
    wp = (pq - phi[j]) / (phi[j + 1] - phi[j])
    f = ((1 - wt) * (1 - wp) * F[i, j] + wt * (1 - wp) * F[i + 1, j]
         + (1 - wt) * wp * F[i, j + 1] + wt * wp * F[i + 1, j + 1])
    eps = 1e-9
    outside = ((tq < theta[0] - eps) | (tq > theta[-1] + eps)
               | (pq < phi[0] - eps) | (pq > phi[-1] + eps))
    return np.where(outside, np.nan, f)


def to_global(data, nlat=181, nlon=361):
    """Merge both panels onto a regular geographic lon-lat grid (degrees).

    The Yin panel is used inside its nominal region, the Yang panel elsewhere.
    """
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(-180.0, 180.0, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    return lon, lat, eval_global(data, (90.0 - LAT) * DEG, LON * DEG)


def eval_global(data, TH, PH):
    """Evaluate the Yin-Yang field at geographic (colatitude, longitude) in radians.

    The Yin panel is used inside its nominal region, the Yang panel elsewhere.
    """
    TH, PH = geo_to_yin(data, np.asarray(TH, float), np.asarray(PH, float))
    PH = np.mod(PH + np.pi, 2 * np.pi) - np.pi                      # wrap to [-pi, pi)
    f_yin = bilinear(data.theta, data.phi, data.yin, TH, PH)
    th2, ph2 = swap_panel(TH, PH)
    f_yang = bilinear(data.theta, data.phi, data.yang, th2, ph2)
    eps = 1e-9
    in_yin = ((TH >= THETA_NOM[0] - eps) & (TH <= THETA_NOM[1] + eps)
              & (np.abs(PH) <= PHI_NOM[1] + eps))
    G = np.where(in_yin & np.isfinite(f_yin), f_yin, f_yang)
    return np.where(np.isfinite(G), G, f_yin)


# ---------------------------------------------------------------------------
# Spherical-harmonic analysis
# ---------------------------------------------------------------------------
def auto_lmax(data, cap=512):
    """Highest degree resolvable by the panel grid (~ 180 deg / grid spacing)."""
    dth = np.median(np.diff(data.theta)) / DEG
    dph = np.median(np.diff(data.phi)) / DEG
    return int(max(2, min(cap, np.floor(180.0 / max(dth, dph)) - 1)))


def sh_coefficients(data, lmax):
    """Complex orthonormal SH coefficients f_lm (m >= 0) of the merged field.

    Y_lm = Pbar_lm(cos theta) exp(i m phi) with  integral |Y_lm|^2 dOmega = 1.
    The field is sampled on a Gauss-Legendre grid (exact quadrature for
    band-limited fields up to lmax) by bilinear interpolation from the panels.
    Returns an (lmax+1, lmax+1) complex array indexed [l, m] (zero for m > l).
    """
    L = int(lmax)
    nlat, nlon = L + 2, 2 * L + 2
    x, w = np.polynomial.legendre.leggauss(nlat)          # x = cos(theta)
    theta = np.arccos(x)
    phi = 2 * np.pi * np.arange(nlon) / nlon
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    f = eval_global(data, TH, PH)
    if not np.all(np.isfinite(f)):
        bad = ~np.isfinite(f)
        raise ValueError(f"{bad.sum()} sample points fall outside both panels "
                         "(check the panel grid extent).")
    # Longitude integral  F_m(theta) = int f exp(-i m phi) dphi
    Fm = np.fft.rfft(f, axis=1)[:, :L + 1] * (2 * np.pi / nlon)
    s = np.sqrt(np.clip(1 - x * x, 0, None))
    coeffs = np.zeros((L + 1, L + 1), complex)
    pmm = np.full(nlat, np.sqrt(1.0 / (4 * np.pi)))        # Pbar_00
    for m in range(L + 1):
        if m > 0:
            pmm = -np.sqrt((2 * m + 1) / (2.0 * m)) * s * pmm
        wf = w * Fm[:, m]
        p_lm2, p_lm1 = None, pmm
        coeffs[m, m] = np.dot(p_lm1, wf)
        for l in range(m + 1, L + 1):
            a = np.sqrt((4.0 * l * l - 1) / (l * l - m * m))
            if p_lm2 is None:
                p_l = a * x * p_lm1
            else:
                b = np.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1) ** 2 - 1))
                p_l = a * (x * p_lm1 - b * p_lm2)
            coeffs[l, m] = np.dot(p_l, wf)
            p_lm2, p_lm1 = p_lm1, p_l
    return coeffs


def power_spectrum(data, lmax=None):
    """Power per spherical-harmonic degree, S(l) = (1/4pi) sum_m |f_lm|^2.

    Normalised so that sum_l S(l) equals the mean of f^2 over the sphere
    (the "4pi-normalised" power convention, as in pyshtools).
    Returns (l, S, mean_square) where mean_square is computed directly by
    quadrature for a Parseval check.
    """
    L = auto_lmax(data) if lmax is None else int(lmax)
    c = sh_coefficients(data, L)
    mag2 = np.abs(c) ** 2
    S = (mag2[:, 0] + 2.0 * mag2[:, 1:].sum(axis=1)) / (4 * np.pi)
    x, w = np.polynomial.legendre.leggauss(L + 2)
    nlon = 2 * L + 2
    TH, PH = np.meshgrid(np.arccos(x), 2 * np.pi * np.arange(nlon) / nlon, indexing="ij")
    f = eval_global(data, TH, PH)
    mean_sq = np.sum(w[:, None] * f * f) * (2 * np.pi / nlon) / (4 * np.pi)
    return np.arange(L + 1), S, mean_sq


# ---------------------------------------------------------------------------
# Demo field
# ---------------------------------------------------------------------------
def demo_field(nt=48, nph=140, margin_deg=2.0, variant="A"):
    """Smooth global field evaluated on both panels (should join seamlessly).

    variant "B" moves the blob slightly and weakens it, for trying differences.
    """
    theta = np.linspace(THETA_NOM[0] - margin_deg * DEG, THETA_NOM[1] + margin_deg * DEG, nt)
    phi = np.linspace(PHI_NOM[0] - margin_deg * DEG, PHI_NOM[1] + margin_deg * DEG, nph)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    cx, cy, cz, amp = (0.55, 0.6, 0.58, 1.5) if variant == "A" else (0.45, 0.7, 0.55, 1.3)

    def f(X, Y, Z):
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / 0.04)
        return 3.0 * Z * (X ** 2 - Y ** 2) + Y + amp * blob

    x, y, z = sph2cart(TH, PH)
    yin = f(x, y, z)
    yang = f(*yang_to_yin_xyz(x, y, z))
    return YYData(yin, yang, theta, phi, name=f"demo field {variant}")


# ---------------------------------------------------------------------------
# File loading: one 2-D array holding both panels
# ---------------------------------------------------------------------------
_FIELD_KEYS = ("field", "data", "f", "yinyang", "yy", "values")
_THETA_KEYS = ("theta", "th", "colat", "colatitude")
_PHI_KEYS = ("phi", "ph", "lon", "longitude")

SHOW_MODES = {
    "A": "Field A",
    "B": "Field B",
    "diff": "Difference A − B",
}

LAYOUTS = {
    "auto": "Auto-detect",
    "rows": "Yin above Yang (stacked rows)",
    "cols": "Yin left of Yang (stacked columns)",
}


def _pick(d, keys):
    low = {k.lower(): k for k in d}
    for k in keys:
        if k in low:
            return np.asarray(d[low[k]])
    return None


def _to_radians(a):
    if a is None:
        return None
    a = np.asarray(a, float).ravel()
    return a * DEG if np.nanmax(np.abs(a)) > 2 * np.pi + 0.1 else a


def load_field(path):
    """Load one 2-D array containing both panels.

    Returns (field, theta, phi); theta/phi are 1-D panel coordinates in radians,
    or None when the file does not provide them (.npz only).
    Supported: .npy, .npz (array named 'field'/'data' or the only 2-D array,
    optional 'theta'/'phi'), and text .txt/.dat/.csv.
    """
    ext = os.path.splitext(path)[1].lower()
    theta = phi = None
    if ext == ".npy":
        a = np.load(path)
    elif ext == ".npz":
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        theta, phi = _pick(d, _THETA_KEYS), _pick(d, _PHI_KEYS)
        a = _pick(d, _FIELD_KEYS)
        if a is None:
            two_d = [v for v in d.values() if np.ndim(v) == 2]
            if len(two_d) != 1:
                raise ValueError("The .npz file must contain exactly one 2-D field "
                                 "(or name it 'field'); found "
                                 f"{len(two_d)} 2-D arrays: {sorted(d)}.")
            a = two_d[0]
    elif ext == ".csv":
        a = np.loadtxt(path, delimiter=",", comments="#")
    else:
        a = np.loadtxt(path, comments="#")
    a = np.asarray(a, float)
    if a.ndim != 2:
        raise ValueError(f"Expected a single 2-D field holding both Yin and Yang; got shape {a.shape}.")
    return a, _to_radians(theta), _to_radians(phi)


def split_panels(field, layout="auto", theta=None, phi=None):
    """Split one 2-D array into (yin, yang, axis), axis = 0 (rows) or 1 (columns).

    'auto' uses the grid sizes when theta/phi are known; otherwise it picks the
    split whose half-panels have the Yin-Yang aspect ratio n_phi/n_theta ~ 3
    (either orientation, so transposed storage is also recognised).
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError(f"Field must be 2-D; got shape {field.shape}.")
    candidates = [ax for ax in (0, 1) if field.shape[ax] % 2 == 0 and field.shape[ax] >= 4]
    if layout in ("rows", "cols"):
        ax = 0 if layout == "rows" else 1
        if ax not in candidates:
            raise ValueError(f"Cannot split shape {field.shape} into two equal panels along "
                             f"{'rows' if ax == 0 else 'columns'} (odd or too small size).")
    else:
        if not candidates:
            raise ValueError(f"Shape {field.shape} cannot be split into two equal panels.")

        def half(ax):
            s = list(field.shape)
            s[ax] //= 2
            return tuple(s)

        if theta is not None and phi is not None:
            want = {(theta.size, phi.size), (phi.size, theta.size)}
            match = [ax for ax in candidates if half(ax) in want]
            if not match:
                raise ValueError(f"Shape {field.shape} does not split into panels matching the "
                                 f"grid (n_theta={theta.size}, n_phi={phi.size}).")
            ax = match[0]
        else:
            def score(ax):
                a, b = half(ax)
                r = abs(np.log(b / a))
                return abs(r - np.log(3.0))
            ax = min(candidates, key=score)
    yin, yang = np.split(field, 2, axis=ax)
    return yin, yang, ax


# ---------------------------------------------------------------------------
# GEM output: pure-Python reader for RPN standard files (XDF "STDR" format)
# ---------------------------------------------------------------------------
# Ported from ECCC librmn (fst98.c, xdf98.c, c_zfstlib.c, armn_compress_32.c,
# float_packer.c, compact.tmplc, convip_plus.f90), so no librmn/rpnpy install
# is needed. Supported data types (datyp, ignoring the +64 "missing" flag):
#   1   packed float (legacy)          5   IEEE float (8..32, 64 bits)
#   133 compressed IEEE (GEM, >16 bits) 134 compressed packed float (GEM, <=16 bits)
#   2/4 integers, 130 compressed integers
# GEM Yin-Yang output (gemdyn OUTs_fstecr/OUTs_wrtref): data records on grid
# type 'U' with nj = 2*njs (Yin rows first, then Yang), described by a '^>'
# record whose ip1..ip3 equal the data record's ig1..ig3. '^>' holds
# [family, version, 2, 1, 1, then per panel: nis, njs, x0, x1, y0, y1,
#  xlat1, xlon1, xlat2, xlon2, posx(nis), posy(njs)]; posx/posy are the
# panel's rotated longitudes (~45..315) and latitudes (~-45..45) in degrees.
class RPNError(Exception):
    pass


class _BitStream:
    """MSB-first bit reader over a big-endian 32-bit word stream (as librmn's extract())."""

    def __init__(self, words_be):
        b = np.frombuffer(np.ascontiguousarray(words_be, dtype=">u4").tobytes(), dtype=np.uint8)
        self.b = np.concatenate([b, np.zeros(16, np.uint8)])
        self.raw = self.b.tobytes()
        self.nbits = 8 * b.size

    def get(self, pos, width):
        if width == 0:
            return 0
        k = pos >> 3
        v = int.from_bytes(self.raw[k:k + 5], "big")
        return (v >> (40 - (pos & 7) - width)) & ((1 << width) - 1)

    def gather(self, pos, width):
        """Vectorised extract of 'width'-bit tokens (width <= 32) at bit positions 'pos'."""
        pos = np.asarray(pos, np.int64)
        if pos.size == 0 or width == 0:
            return np.zeros(pos.size, np.uint64)
        if int(pos.max()) + width > self.nbits + 64:
            raise RPNError("corrupt compressed stream (read past end)")
        k = pos >> 3
        v = np.zeros(pos.size, np.uint64)
        for i in range(5):
            v = (v << np.uint64(8)) | self.b[k + i].astype(np.uint64)
        sh = (np.int64(40) - (pos & 7) - width).astype(np.uint64)
        return (v >> sh) & np.uint64((1 << width) - 1)


def _tile_order(ni, nj, step, i0):
    """Flat indices (row-major (nj, ni)) of the cells covered by step x step tiles
    starting at 0-based row/col i0, in the stream order of librmn (tiles j-major then
    i; inside a tile rows then columns), plus each tile's cell count."""
    r = np.arange(i0, nj)
    c = np.arange(i0, ni)
    R, Cc = np.meshgrid(r, c, indexing="ij")
    tj, ti = (R - i0) // step, (Cc - i0) // step
    n, m = (R - i0) % step, (Cc - i0) % step
    order = np.lexsort((m.ravel(), n.ravel(), ti.ravel(), tj.ravel()))
    flat = (R * ni + Cc).ravel()[order]
    nti = (ni - i0 + step - 1) // step
    ntj = (nj - i0 + step - 1) // step
    tile_id = (tj * nti + ti).ravel()[order]
    counts = np.bincount(tile_id, minlength=nti * ntj)
    return flat, counts


def _read_tiles(bs, pos, counts, hdr_width, width_of):
    """Walk the per-tile headers sequentially; return per-tile (code, start, width) and end pos.
    width_of(code) -> (pre_bits_to_skip_before_values, value_width)."""
    nt = counts.size
    codes = np.empty(nt, np.int64)
    starts = np.empty(nt, np.int64)
    widths = np.empty(nt, np.int64)
    pre = np.empty(nt, np.int64)
    get = bs.get
    for t in range(nt):
        code = get(pos, hdr_width)
        pos += hdr_width
        p, w = width_of(code)
        codes[t], pre[t] = code, pos
        pos += p
        starts[t], widths[t] = pos, w
        pos += int(counts[t]) * w
    return codes, pre, starts, widths, pos


def _expand(starts, widths, counts):
    """Per-value bit positions and widths from per-tile (start, width, count)."""
    rep_start = np.repeat(starts, counts)
    rep_w = np.repeat(widths, counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    k = np.arange(rep_start.size) - first
    return rep_start + k * rep_w, rep_w


def _gather_signed(bs, pos, w):
    out = np.zeros(pos.size, np.int64)
    for width in np.unique(w):
        sel = w == width
        if width == 0:
            continue
        v = bs.gather(pos[sel], int(width)).astype(np.int64)
        sign = np.int64(1) << np.int64(width - 1)
        out[sel] = (v ^ sign) - sign                     # sign-extend 'width'-bit value
    return out


def _unpack_parallelogram(bs, pos, ni, nj, nbits, step, mask, special_16):
    """librmn unpackTokensParallelogram{,32,_8}: Lorenzo predictor, tiles of step x step.
    Returns the (nj, ni) unsigned field (masked to 'mask') and the end position."""
    req = bs.get(pos, 3)
    pos += 3
    first_row = bs.gather(pos + nbits * np.arange(ni), nbits).astype(np.int64)
    pos += nbits * ni
    first_col = bs.gather(pos + nbits * np.arange(nj - 1), nbits).astype(np.int64)
    pos += nbits * (nj - 1)
    flat, counts = _tile_order(ni, nj, step, 1)

    def width_of(code):
        if code == 0:
            return 0, 0
        if special_16 and code in (15, 16):
            return 0, 17
        return 0, code + 1

    _, _, starts, widths, pos = _read_tiles(bs, pos, counts, req, width_of)
    vpos, vw = _expand(starts, widths, counts)
    E = np.zeros(nj * ni, np.int64)
    E[flat] = _gather_signed(bs, vpos, vw)
    E = E.reshape(nj, ni)
    # Boundary row/column given as values: convert to first differences, then a
    # 2-D cumulative sum reproduces F[r,c] = tmp + F[r,c-1] + F[r-1,c] - F[r-1,c-1].
    E[0, 0] = first_row[0]
    E[0, 1:] = np.diff(first_row)
    E[1:, 0] = np.diff(np.concatenate([[first_row[0]], first_col]))
    F = np.cumsum(np.cumsum(E, axis=0), axis=1)
    return F & mask, pos


def _unpack_minimum(bs, pos, ni, nj, nbits, step):
    """librmn unpackTokensMinimum (tile minimum + offsets)."""
    flat, counts = _tile_order(ni, nj, step, 0)

    def width_of(code):
        if code == 0:
            return nbits, 0
        if code in (15, 16):
            return 0, 16
        return nbits, code

    codes, pre, starts, widths, pos = _read_tiles(bs, pos, counts, 4, width_of)
    has_min = ~np.isin(codes, (15, 16))
    mins = np.zeros(codes.size, np.int64)
    if has_min.any():
        mins[has_min] = bs.gather(pre[has_min], nbits).astype(np.int64)
    vpos, vw = _expand(starts, widths, counts)
    vals = np.zeros(vpos.size, np.int64)
    for width in np.unique(vw):
        if width:
            sel = vw == width
            vals[sel] = bs.gather(vpos[sel], int(width)).astype(np.int64)
    vals += np.repeat(mins, counts)
    out = np.zeros(nj * ni, np.int64)
    out[flat] = vals
    return out.reshape(nj, ni) & 0xFFFF, pos


def _armn_uncompress16(words, ni, nj):
    """librmn armn_compress(..., UNCOMPRESS): -> (nj, ni) uint16 tokens."""
    hdr = int(words[0])
    predictor, step, nbits = hdr & 0xF, (hdr >> 7) & 0x7, (hdr >> 10) & 0x1F
    bs = _BitStream(words[1:])
    if predictor == 4:                                    # PARALLELOGRAM
        f, _ = _unpack_parallelogram(bs, 0, ni, nj, nbits, step, 0xFFFF, True)
    elif predictor == 3:                                  # MINIMUM
        f, _ = _unpack_minimum(bs, 0, ni, nj, nbits, step)
    else:
        raise RPNError(f"unsupported compression predictor {predictor}")
    return f.astype(np.uint16)


def _unpack_rle_1bit(bs, pos, npts):
    """librmn unpack1bitRLE (sign bits of datyp 133)."""
    out = np.zeros(npts, np.uint8)
    i, last = 0, 0xFF
    get = bs.get
    while i < npts:
        if get(pos, 1) == 0:                              # SEQUENCE of up to 7 raw bits
            pos += 1
            lim = min(7, npts - i)
            for j in range(lim):
                out[i + j] = get(pos, 1)
                pos += 1
            i += lim
        else:                                             # COUNT
            val = get(pos + 1, 1)
            cnt = get(pos + 2, 6)
            pos += 8
            if cnt == 63:
                n = min(255, npts - i)
                out[i:i + n] = last & 1
                i += 255
            else:
                n = min(cnt, npts - i)
                out[i:i + n] = val
                i += cnt
                last = val
    return out


def _armn_uncompress32(words, ni, nj):
    """librmn c_armn_uncompress32 (datyp 133)."""
    npts = ni * nj
    w = [int(x) for x in words[:2]]
    nbits = (w[0] >> 10) & 0x1F
    info = w[1]
    exp_min, nbits_exp, codes = info >> 16, (info >> 8) & 0xFF, info & 0xFF
    code_sign, code_exp, code_mant = codes & 0x30, codes & 0x0C, codes & 0x03
    cur = 2
    if code_sign in (0x20, 0x30):
        lng = int(words[cur]); cur += 1
        sign = _unpack_rle_1bit(_BitStream(words[cur:]), 0, npts).astype(np.uint32)
        cur += lng >> 2
    else:
        sign = np.full(npts, 0 if code_sign == 0x00 else 1, np.uint32)
    if code_exp in (0x08, 0x0C):
        lng = int(words[cur]); cur += 1
        e, _ = _unpack_parallelogram(_BitStream(words[cur:]), 0, ni, nj, nbits_exp, 3, 0xFF, False)
        expo = (e.ravel() + exp_min) & 0xFF
        cur += lng >> 2
    else:
        expo = np.full(npts, exp_min, np.int64)
    cur += 1                                              # lng_mantisse
    bs = _BitStream(words[cur:])
    if code_mant == 0x00:
        mant, _ = _unpack_parallelogram(bs, 0, ni, nj, nbits, 3, 0xFFFFFFFF, False)
        mant = mant.ravel()
    else:
        mant = bs.gather(nbits * np.arange(npts, dtype=np.int64), nbits).astype(np.int64)
    bits = ((sign.astype(np.uint64) << np.uint64(31)) | (expo.astype(np.uint64) << np.uint64(23))
            | ((mant.astype(np.uint64) << np.uint64(23 - nbits)) & np.uint64(0x7FFFFF)))
    return bits.astype(np.uint32).view(np.float32).reshape(nj, ni)


def _float_unpack16(header, tokens):
    """librmn float_unpacker_1 (datyp 6/134): 16-bit tokens -> float32."""
    h0, minimum, npts = (int(np.int32(np.uint32(x))) for x in header[:3])
    if (h0 >> 20) & 0xFFF != 0xEFF:
        raise RPNError("invalid float-packer header")
    maxexp, shift2 = (h0 >> 8) & 0xFF, h0 & 0xFF
    if maxexp == 0:
        return np.zeros(npts, np.float32)
    mant = (tokens[:npts].astype(np.int64) << shift2) + minimum
    mant = ((mant + 2 ** 31) % 2 ** 32) - 2 ** 31         # int32 wrap-around as in C
    sgn = (mant < 0).astype(np.int64)
    mant = np.minimum(np.abs(mant), 0xFFFFFF)
    t = ((mant & 0x7FFFFF) | (maxexp << 23) | (sgn << 31)).astype(np.uint32).view(np.float32)
    hidden = (mant & (1 << 23)) != 0
    t2 = ((maxexp << 23) | (sgn << 31)).astype(np.uint32).view(np.float32)
    return np.where(hidden, t, t - t2).astype(np.float32)


def _compact_float_unpack(words, n):
    """librmn compact_u_float (datyp 1, legacy packing)."""
    w = [int(x) for x in words[:4]]
    marker, counter = w[0] >> 20, w[0] & 0xFFFFF
    range_expo, min_expo, min_sign = w[1] >> 16, (w[1] >> 4) & 0xFFF, w[1] & 0xF
    min_m32 = w[2]
    token = (w[3] >> 8) & 0xFF
    count = n if marker in (0x7FF, 0x7EF) else ((w[3] & 0xFF) << 20 | counter)
    if min_m32 == 0 or min_expo < 849:
        fmin = 0.0
    else:
        bits = ((min_sign & 1) << 31) | (((min_expo + 127 - 1024 + 48) & 0xFF) << 23) | ((min_m32 >> 8) & 0x7FFFFF)
        fmin = float(np.uint32(bits).view(np.float32))
    mul = 2.0 ** (range_expo - 4096)
    bs = _BitStream(words[3:])
    width = min(token, 32)
    pk = bs.gather(24 + token * np.arange(count, dtype=np.int64), width).astype(np.float64)
    out = np.where(pk == 0, fmin, pk * mul * 1.0000000000001 + fmin)
    return out.astype(np.float32)


_KIND_UNITS = {0: "m", 1: "sg", 2: "mb", 3: "", 4: "M", 5: "hy", 6: "th", 7: "m-",
               10: "H", 15: "", 17: "[]", 21: "hh"}
_KIND_LIMITS = {0: (-20000., 1e5), 1: (0., 1.), 2: (0., 1100.), 3: (-4.8e8, 1e10), 4: (-20000., 1e5),
                5: (0., 1.), 6: (1., 2e5), 7: (0., 2e4), 10: (0., 1e10), 15: (-4.8e8, 1e10),
                17: (1., 1e10), 21: (0., 1e6)}
_KIND_FACT = {17: -1.0, 21: 1.0e4}


def decode_ip(ip):
    """librmn convip (decode): ip -> (value, kind, label)."""
    ip = int(ip)
    if ip > 32767:
        kind = (ip >> 24) & 0xF
        iexp = (ip >> 20) & 0xF
        m = ip & 0xFFFFF
        if m > 1000000:
            m = -(m - 1000000)
        p = m * 10.0 ** (4 - iexp)
        lo, hi = _KIND_LIMITS.get(kind, (-4.8e8, 1e10))
        if (p < lo or p > hi) and (kind + 16) in _KIND_LIMITS:
            kind += 16
        p = p / _KIND_FACT.get(kind, 1.0)
    elif 12000 < ip <= 32000:
        kind, p = 0, 5.0 * (ip - 12001)
    elif 2000 <= ip <= 12000:
        kind, p = 1, (ip - 2000) / 10000.0
    elif 0 <= ip < 1100:
        kind, p = 2, float(ip)
    elif 1200 < ip < 2000:
        kind = 2
        p = ((ip - 1200) / 20000.0 if ip < 1400 else (ip - 1400) / 2000.0 if ip < 1600
             else (ip - 1600) / 200.0 if ip < 1800 else (ip - 1800) / 20.0)
    else:
        kind, p = 3, 1200.0 - ip if 1100 <= ip <= 1200 else float(ip)
    unit = _KIND_UNITS.get(kind, "")
    return p, kind, (f"{p:g} {unit}".strip())


def cmc_stamp_to_datetime(stamp):
    """CMC date-time stamp (new or old style) -> datetime (None if invalid)."""
    try:
        stamp = int(stamp)
        if stamp >= 123200000:
            s = stamp - 123200000
            tdate = (s // 10) * 8 + s % 10            # 5-second intervals since 1980-01-01
            return _dt.datetime(1980, 1, 1) + _dt.timedelta(seconds=5 * tdate)
        if stamp > 0:                                 # old style MMDDYYZZR
            zulu = (stamp // 10) % 100
            year = (stamp // 1000) % 100 + 1900
            day = (stamp // 100000) % 100
            month = (stamp // 10000000) % 100
            return _dt.datetime(year, month, day, zulu)
    except (ValueError, OverflowError):
        pass
    return None


def _ascii6(v, n):
    return "".join(chr(((v >> (6 * (n - 1 - i))) & 0x3F) + 32) for i in range(n))


class RPNFile:
    """Read-only access to one RPN standard file (XDF 'STDR' random-access format)."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as fh:
            head = fh.read(16)
            if len(head) < 16 or head[8:12] not in (b"XDF0", b"xdf0"):
                if head[:4] == b"\x55\x55\x55\x55":
                    raise RPNError(f"{os.path.basename(path)}: old (1989) standard file format is not supported")
                raise RPNError(f"{os.path.basename(path)}: not an RPN standard (XDF) file")
            if head[12:16] not in (b"STDR", b"STDS"):
                raise RPNError(f"{os.path.basename(path)}: XDF file but not a standard file")
            hdr_lng = int.from_bytes(head[0:4], "big") & 0xFFFFFF
            fh.seek(0)
            H = np.frombuffer(fh.read(8 * hdr_lng), ">u4")
            nbd = int(H[7])
            self.lprm = int(H[10]) & 0xFFFF
            self.laux = int(H[11]) & 0xFFFF
            if nbd == 0:
                raise RPNError(f"{os.path.basename(path)}: sequential/empty standard file is not supported")
            entries = []
            addr = 8 * hdr_lng
            page_words = 2 * (self.lprm * 256 + 4)
            for _ in range(nbd):
                fh.seek(addr)
                P = np.frombuffer(fh.read(4 * page_words), ">u4")
                if P.size < 8:
                    raise RPNError(f"{os.path.basename(path)}: truncated directory")
                nxt, nent = int(P[4]), int(P[5])
                E = P[8:8 + nent * 2 * self.lprm].reshape(nent, 2 * self.lprm)
                entries.append(E)
                if nxt == 0:
                    break
                addr = 8 * (nxt - 1)
        E = np.concatenate(entries) if entries else np.zeros((0, 18), ">u4")
        self.records = []
        for row in E.astype(np.uint64):
            w = [int(x) for x in row[:18]]
            idtyp = w[0] >> 24
            if idtyp == 0 or (idtyp & 0x7E) == 0x7E or (w[0] >> 31):
                continue                                   # special or deleted record
            ds = w[17]
            rec = dict(
                lng=w[0] & 0xFFFFFF, addr=w[1],
                deet=w[2] >> 8, nbits=w[2] & 0xFF,
                ni=w[3] >> 8, grtyp=chr(w[3] & 0xFF),
                nj=w[4] >> 8, datyp=w[4] & 0xFF,
                nk=w[5] >> 12, npas=w[6] >> 6,
                ig4=w[7] >> 8, ig1=w[8] >> 8, ig3=w[9] >> 8,
                ig2=((w[7] & 0xFF) << 16) | ((w[8] & 0xFF) << 8) | (w[9] & 0xFF),
                etiket=(_ascii6(w[10] >> 2, 5) + _ascii6(w[11] >> 2, 5) + _ascii6(w[12] >> 20, 2)).rstrip(),
                typvar=_ascii6((w[12] >> 8) & 0xFFF, 2).rstrip(),
                nomvar=_ascii6(w[13] >> 8, 4).rstrip(),
                ip1=w[14] >> 4, ip2=w[15] >> 4, ip3=w[16] >> 4,
                datev=(ds >> 3) * 10 + (ds & 7),
                file=path)
            self.records.append(rec)

    def read(self, rec):
        """Decode one record -> float32 array of shape (nj, ni) (Fortran (ni, nj) order)."""
        ni, nj, nk = rec["ni"], rec["nj"], max(rec["nk"], 1)
        n = ni * nj * nk
        off = 8 * (rec["addr"] - 1) + 8 * (self.lprm + self.laux)
        nwords = 2 * (rec["lng"] - self.lprm - self.laux)
        with open(self.path, "rb") as fh:
            fh.seek(off)
            raw = fh.read(4 * nwords)
        W = np.frombuffer(raw, ">u4").astype(np.uint32)
        datyp = rec["datyp"] & ~64
        nbits = rec["nbits"]
        if datyp in (6, 134):
            if datyp == 134:
                tok = _armn_uncompress16(W[4:], ni, nj * nk).ravel()
            else:
                stream = W[3:3 + (n + 1) // 2]
                tok = np.stack([stream >> 16, stream & 0xFFFF], axis=1).ravel()
            out = _float_unpack16(W[(1 if datyp == 134 else 0):][:3], tok)
        elif datyp == 133:
            out = _armn_uncompress32(W[1:], ni, nj * nk).ravel()
        elif datyp == 5:
            if nbits == 64:
                out = np.frombuffer(W[:2 * n].astype(">u4").tobytes(), ">f8").astype(np.float64)
            elif nbits == 32:
                out = W[:n].view(np.float32)
            else:
                v = _BitStream(W).gather(nbits * np.arange(n, dtype=np.int64), nbits)
                out = (v.astype(np.uint32) << np.uint32(32 - nbits)).view(np.float32)
        elif datyp == 1:
            out = _compact_float_unpack(W, n)
        elif datyp in (2, 4, 130, 132):
            if datyp >= 128:
                tok = _armn_uncompress16(W[1:], ni, nj * nk).ravel()
                out = tok.astype(np.float32)
            else:
                v = _BitStream(W).gather(nbits * np.arange(n, dtype=np.int64), min(nbits, 32))
                if datyp == 4:
                    sign = np.uint64(1) << np.uint64(nbits - 1)
                    v = (v.astype(np.int64) ^ int(sign)) - int(sign)
                out = v.astype(np.float32)
        else:
            raise RPNError(f"{rec['nomvar']}: unsupported data type datyp={rec['datyp']}")
        return np.asarray(out[:n], dtype=np.float32).reshape(nk * nj, ni) if nk == 1 else \
            np.asarray(out[:n], dtype=np.float32).reshape(nk, nj, ni)


# ---- GEM Yin-Yang layer --------------------------------------------------------
def _lac(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def erot_matrix(xlat1, xlon1, xlat2, xlon2):
    """librmn ez_crot: rotation R with  rotated_xyz = R @ geographic_xyz  for an 'E' grid."""
    p1, p2 = _lac(xlon1, xlat1), _lac(xlon2, xlat2)
    a = float(p1 @ p2)
    cr = np.cross(p1, p2)
    b = np.linalg.norm(cr)
    r2 = (a * p1 - p2) / b
    return np.array([-p1 / np.linalg.norm(p1), r2 / np.linalg.norm(r2), cr / b])


# The viewer's panel coordinates use phi = GEM rotated longitude - 180 deg, i.e. a
# 180-degree turn about the rotated z axis.
_SHIFT180 = np.diag([-1.0, -1.0, 1.0])
_YY_INVOLUTION = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])   # (x,y,z) -> (-x,z,y)


def is_rpn_file(path):
    try:
        with open(path, "rb") as fh:
            h = fh.read(16)
        return len(h) == 16 and h[8:12] in (b"XDF0", b"xdf0") and h[12:16] in (b"STDR", b"STDS")
    except OSError:
        return False


class GemDataset:
    """Index of the Yin-Yang ('U' grid) records in GEM output file(s) or a directory."""

    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            cands = sorted(p for p in _glob.glob(os.path.join(path, "*")) if os.path.isfile(p))
            files = [p for p in cands if is_rpn_file(p)]
            if not files:
                raise RPNError(f"No RPN standard files found in {path}")
        else:
            files = [path]
        self.files = {}
        self.records = []
        self.skipped = []
        problems = []
        for f in files:
            try:
                rf = RPNFile(f)
            except RPNError as exc:
                problems.append(str(exc))
                continue
            self.files[f] = rf
            grids = {(r["ip1"], r["ip2"], r["ip3"]): r for r in rf.records if r["nomvar"] == "^>"}
            for r in rf.records:
                if r["nomvar"] in ("^>", ">>", "^^", "!!", "HY", "^^^^", ">>>>", "META"):
                    continue
                if r["grtyp"] == "U" and (r["ig1"], r["ig2"], r["ig3"]) in grids:
                    r = dict(r, grid=grids[(r["ig1"], r["ig2"], r["ig3"])])
                    self.records.append(r)
                else:
                    self.skipped.append(r)
        if not self.records:
            msg = f"No Yin-Yang ('U' grid) fields found in {path}."
            if self.skipped:
                kinds = sorted({f"{r['nomvar']}({r['grtyp']})" for r in self.skipped})[:12]
                msg += " Other records present: " + ", ".join(kinds)
            if problems:
                msg += "\n" + "\n".join(problems[:5])
            raise RPNError(msg)
        self.problems = problems
        self._grid_cache = {}

    # -- choices for the GUI
    def variables(self):
        return sorted({r["nomvar"] for r in self.records})

    def levels(self, nomvar):
        recs = [r for r in self.records if r["nomvar"] == nomvar]
        out = {}
        for r in recs:
            p, kind, label = decode_ip(r["ip1"])
            out[r["ip1"]] = (kind, p, label)
        # sort by kind then value (hybrid/sigma/pressure: top of atmosphere first)
        return [ip for ip, _ in sorted(out.items(), key=lambda kv: (kv[1][0], kv[1][1]))]

    def times(self, nomvar, ip1=None):
        recs = [r for r in self.records if r["nomvar"] == nomvar and (ip1 is None or r["ip1"] == ip1)]
        return sorted({(r["datev"], r["ip2"]) for r in recs})

    @staticmethod
    def time_label(datev, ip2):
        d = cmc_stamp_to_datetime(datev)
        p, kind, _ = decode_ip(ip2)
        lead = f"+{p:g} h" if kind in (10, 2, 3) else f"ip2={ip2}"
        return (d.strftime("%Y-%m-%d %H:%M UTC") if d else f"datev {datev}") + f"  ({lead})"

    @staticmethod
    def level_label(ip1):
        return decode_ip(ip1)[2] or f"ip1={ip1}"

    def find(self, nomvar, ip1=None, time=None):
        cands = [r for r in self.records if r["nomvar"] == nomvar
                 and (ip1 is None or r["ip1"] == ip1)
                 and (time is None or (r["datev"], r["ip2"]) == tuple(time))]
        if not cands:
            raise RPNError(f"No record {nomvar} at the requested level/time.")
        return cands[0]

    # -- grid decoding
    def _grid(self, grec):
        key = (grec["file"], grec["ip1"], grec["ip2"], grec["ip3"])
        if key in self._grid_cache:
            return self._grid_cache[key]
        yy = self.files[grec["file"]].read(grec).ravel().astype(np.float64)
        if yy.size < 5 or int(round(yy[2])) != 2:
            raise RPNError("Unrecognised '^>' grid record (expected 2 sub-grids).")
        panels, k = [], 5
        for _ in range(2):
            nis, njs = int(round(yy[k])), int(round(yy[k + 1]))
            xlat1, xlon1, xlat2, xlon2 = yy[k + 6:k + 10]
            posx = yy[k + 10:k + 10 + nis]
            posy = yy[k + 10 + nis:k + 10 + nis + njs]
            panels.append(dict(nis=nis, njs=njs, rot=(xlat1, xlon1, xlat2, xlon2), posx=posx, posy=posy))
            k += 10 + nis + njs
        yin, yang = panels
        if (yin["nis"], yin["njs"]) != (yang["nis"], yang["njs"]) or not (
                np.allclose(yin["posx"], yang["posx"]) and np.allclose(yin["posy"], yang["posy"])):
            raise RPNError("Yin and Yang sub-grids differ; only identical panel grids are supported.")
        R_yin = _SHIFT180 @ erot_matrix(*yin["rot"])        # geographic -> viewer Yin frame
        R_yang = _SHIFT180 @ erot_matrix(*yang["rot"])
        mismatch = float(np.max(np.abs(R_yang - _YY_INVOLUTION @ R_yin)))
        if mismatch > 1e-3:
            raise RPNError(f"Yang rotation is not the standard Yin-Yang complement (mismatch {mismatch:.2g}).")
        theta = (90.0 - yin["posy"]) * DEG
        phi = (yin["posx"] - 180.0) * DEG
        out = (yin["nis"], yin["njs"], theta, phi, R_yin, yin["rot"])
        self._grid_cache[key] = out
        return out

    def load(self, rec):
        """Decode one Yin-Yang record -> (field_2d with Yin rows then Yang rows, theta, phi, rot, name)."""
        nis, njs, theta, phi, R, _ = self._grid(rec["grid"])
        arr = self.files[rec["file"]].read(rec)
        if arr.shape != (2 * njs, nis):
            raise RPNError(f"{rec['nomvar']}: shape {arr.shape} does not match its grid ({2 * njs}, {nis}).")
        name = f"{rec['nomvar']} {self.level_label(rec['ip1'])}, {self.time_label(rec['datev'], rec['ip2'])}"
        return arr.astype(np.float64), theta, phi, R, name


# ---------------------------------------------------------------------------
# Plotting (pure Matplotlib, usable without the GUI)
# ---------------------------------------------------------------------------
DEFAULT_OPTS = dict(nlevels=20, cmap="RdBu_r", filled=True, lines=True, clabels=False,
                    symmetric=False, vmin=None, vmax=None, boundaries=True,
                    nlat=181, nlon=361, title="", lon0=0.0, lat0=30.0, antipode=True,
                    lmax=None, spec_norm="degree", logl=False, spectra=None)


def compute_levels(arrays, n, vmin=None, vmax=None, symmetric=False):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Field contains no finite values.")
    lo = float(vals.min()) if vmin is None else float(vmin)
    hi = float(vals.max()) if vmax is None else float(vmax)
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if not hi > lo:
        pad = 0.5 * abs(lo) if lo != 0 else 0.5
        lo, hi = lo - pad, hi + pad
    n = max(int(n), 2)
    if symmetric and n % 2 == 0:
        # Odd number of bands: zero sits in the middle of a band instead of on a
        # level, so near-zero round-off (e.g. in differences) draws no jagged 0-contour.
        n += 1
    return np.linspace(lo, hi, n + 1)


def _extend(levels, arrays):
    vals = np.concatenate([np.asarray(a).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min() < levels[0], vals.max() > levels[-1]
    return {(False, False): "neither", (True, False): "min",
            (False, True): "max", (True, True): "both"}[(lo, hi)]


def _contour(ax, X, Y, F, levels, o, extend, transform=None):
    cs = None
    if o["filled"]:
        cs = ax.contourf(X, Y, F, levels=levels, cmap=o["cmap"], extend=extend)
    if o["lines"] or not o["filled"]:
        kw = dict(colors="k", linewidths=0.5) if o["filled"] else dict(cmap=o["cmap"], linewidths=1.0)
        cl = ax.contour(X, Y, F, levels=levels, **kw)
        if o["clabels"]:
            ax.clabel(cl, inline=True, fontsize=7, fmt="%.3g")
        if cs is None:
            cs = cl
    return cs


def plot_native(fig, data, o):
    levels = compute_levels([data.yin, data.yang], o["nlevels"], o["vmin"], o["vmax"], o["symmetric"])
    extend = _extend(levels, [data.yin, data.yang])
    axs = fig.subplots(2, 1, sharex=True, sharey=True)   # Yin on top, Yang below
    X = data.phi / DEG
    Y = 90.0 - data.theta / DEG
    cs = None
    for ax, F, name in zip(axs, (data.yin, data.yang), ("Yin panel", "Yang panel")):
        cs = _contour(ax, X, Y, F, levels, o, extend)
        if o["boundaries"]:
            th, ph = nominal_boundary()
            ax.plot(ph / DEG, 90 - th / DEG, color="0.2", ls=":", lw=1.0, label="nominal edge")
            th2, ph2 = swap_panel(th, ph)
            lx, ly = break_wraps(ph2 / DEG, 90 - th2 / DEG, jump=180)
            ax.plot(lx, ly, color="m", ls="--", lw=1.2, label="other panel's edge")
        ax.set_title(name)
        ax.set_ylabel("panel latitude 90°−θ [deg]")
        ax.set_aspect("equal")
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(Y.min(), Y.max())
    axs[-1].set_xlabel("panel longitude φ [deg]" + ("  (GEM rotated longitude − 180°)" if data.rot is not None else ""))
    if o["boundaries"]:
        axs[0].legend(loc="lower left", fontsize=7, framealpha=0.8)
    fig.colorbar(cs, ax=list(axs), orientation="vertical", fraction=0.04,
                 pad=0.02, label=data.name)
    if o["title"]:
        fig.suptitle(o["title"])
    return axs


def mollweide_xy(lon_deg, lat_deg):
    """Forward Mollweide projection (unit sphere): degrees -> (x, y).

    x in [-2*sqrt(2), 2*sqrt(2)], y in [-sqrt(2), sqrt(2)].  NaNs pass through.
    """
    lam = np.asarray(lon_deg, float) * DEG
    phi = np.asarray(lat_deg, float) * DEG
    target = np.pi * np.sin(phi)
    t = phi.copy()                       # auxiliary angle, solve 2t + sin2t = pi*sin(phi)
    for _ in range(60):
        den = 2.0 + 2.0 * np.cos(2.0 * t)
        step = np.where(den > 1e-12, (2.0 * t + np.sin(2.0 * t) - target) / np.where(den > 1e-12, den, 1.0), 0.0)
        t = t - step
        if np.nanmax(np.abs(step)) < 1e-12:
            break
    t = np.where(np.abs(np.abs(phi) - np.pi / 2) < 1e-12, phi, t)   # exact at the poles
    return 2.0 * np.sqrt(2.0) / np.pi * lam * np.cos(t), np.sqrt(2.0) * np.sin(t)


def _mollweide_frame(ax):
    """Outline, graticule and labels for a hand-built Mollweide map."""
    s = np.linspace(-90, 90, 361)
    for lon0 in range(-180, 181, 30):
        x, y = mollweide_xy(np.full_like(s, lon0), s)
        edge = lon0 in (-180, 180)
        ax.plot(x, y, color="k" if edge else "0.5", lw=1.0 if edge else 0.4,
                alpha=1.0 if edge else 0.5, zorder=3)
    l = np.linspace(-180, 180, 361)
    for lat0 in range(-60, 61, 30):
        x, y = mollweide_xy(l, np.full_like(l, lat0))
        ax.plot(x, y, color="0.5", lw=0.4, alpha=0.5, zorder=3)
        if lat0:
            xl, yl = mollweide_xy(-180.0, lat0)
            ax.text(xl - 0.06, yl, f"{lat0:+d}°", ha="right", va="center", fontsize=7)
    for lon0 in range(-120, 121, 60):
        xl, yl = mollweide_xy(lon0, 0.0)
        ax.text(xl, yl - 0.04, f"{lon0}°", ha="center", va="top", fontsize=7, color="0.25")
    ax.set_xlim(-2.0 * np.sqrt(2) - 0.35, 2.0 * np.sqrt(2) + 0.05)
    ax.set_ylim(-np.sqrt(2) - 0.05, np.sqrt(2) + 0.05)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_global(fig, data, o, projection="equirect"):
    lon, lat, G = to_global(data, o["nlat"], o["nlon"])
    levels = compute_levels([data.yin, data.yang], o["nlevels"], o["vmin"], o["vmax"], o["symmetric"])
    extend = _extend(levels, [G])
    ax = fig.add_subplot(1, 1, 1)
    moll = projection == "mollweide"
    if moll:
        proj = mollweide_xy
        LON, LAT = np.meshgrid(lon, lat)
        X, Y = proj(LON, LAT)
    else:
        proj = lambda a, b: (np.asarray(a, float), np.asarray(b, float))
        X, Y = lon, lat
    cs = _contour(ax, X, Y, G, levels, o, extend)
    if o["boundaries"]:
        (ylon, ylat), (glon, glat) = panel_edges_geo(data)
        lx, ly = break_wraps(ylon, ylat, jump=180)
        ax.plot(*proj(lx, ly), color="0.15", lw=1.2, label="Yin nominal edge", zorder=4)
        lx, ly = break_wraps(glon, glat, jump=180)
        ax.plot(*proj(lx, ly), color="m", ls="--", lw=1.2, label="Yang nominal edge", zorder=4)
        if moll:   # below the map, clear of the ellipse
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=7, framealpha=0.8)
        else:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    if moll:
        _mollweide_frame(ax)
    else:
        ax.set_aspect("equal")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.set_xlabel("longitude [deg]")
        ax.set_ylabel("latitude [deg]")
    title = o["title"] or ("Merged Yin-Yang field (true geographic lat/lon)" if data.rot is not None
                           else "Merged Yin-Yang field (geographic = Yin frame)")
    if moll:
        fig.suptitle(title)   # axes title gets clipped when the axis frame is hidden
    else:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.06, pad=0.1, label=data.name)
    return ax


def ortho_xy(lon_deg, lat_deg, lon0, lat0, R=1.0):
    """Orthographic projection about (lon0, lat0), centred on (0, 0).

    Returns x, y and cos(c), the cosine of the angular distance from the
    centre; points with cos(c) < 0 are on the far side of the globe.
    """
    lam = np.asarray(lon_deg, float) * DEG
    phi = np.asarray(lat_deg, float) * DEG
    l0, p0 = lon0 * DEG, lat0 * DEG
    dl = lam - l0
    cosc = np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(dl)
    x = R * np.cos(phi) * np.sin(dl)
    y = R * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(dl))
    return x, y, cosc


def _ortho_graticule(ax, l0, p0, R, xc, yc):
    """Globe outline, parallels and meridians (used when Basemap is unavailable)."""
    t = np.linspace(0, 2 * np.pi, 361)
    ax.plot(xc + R * np.cos(t), yc + R * np.sin(t), color="k", lw=1.0, zorder=4)
    s = np.linspace(-180, 180, 361)
    for lat0 in np.arange(-60.0, 61.0, 30.0):
        gx, gy, gc = ortho_xy(s, np.full_like(s, lat0), l0, p0, R)
        ax.plot(np.where(gc >= 0, gx + xc, np.nan), np.where(gc >= 0, gy + yc, np.nan),
                color="0.45", lw=0.4, zorder=3)
    s = np.linspace(-80, 80, 161)
    for lon0 in np.arange(-180.0, 180.0, 30.0):
        gx, gy, gc = ortho_xy(np.full_like(s, lon0), s, l0, p0, R)
        ax.plot(np.where(gc >= 0, gx + xc, np.nan), np.where(gc >= 0, gy + yc, np.nan),
                color="0.45", lw=0.4, zorder=3)


def plot_sphere(fig, data, o):
    """Filled contours (contourf) of the merged field on orthographic globe(s).

    Uses Basemap's ``m.contourf`` / ``m.contour`` and graticule when Basemap is
    installed; otherwise the same projection is drawn with Matplotlib's contourf.
    """
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError:
        Basemap = None
    from matplotlib.patches import Circle

    lon, lat, G = to_global(data, o["nlat"], o["nlon"])
    levels = compute_levels([data.yin, data.yang], o["nlevels"], o["vmin"], o["vmax"], o["symmetric"])
    extend = _extend(levels, [G])
    LON, LAT = np.meshgrid(lon, lat)

    lon0, lat0 = float(o["lon0"]), float(o["lat0"])
    centres = [(lon0, lat0)]
    if o["antipode"]:
        centres.append((((lon0 + 360.0) % 360.0) - 180.0, -lat0))
    axs = fig.subplots(1, len(centres), squeeze=False)[0]

    curves = []
    if o["boundaries"]:
        (ylon, ylat), (glon, glat) = panel_edges_geo(data)
        curves = [(ylon, ylat, dict(color="0.15", lw=1.2), "Yin nominal edge"),
                  (glon, glat, dict(color="m", ls="--", lw=1.2), "Yang nominal edge")]

    cs = None
    for k, (ax, (l0, p0)) in enumerate(zip(axs, centres)):
        if Basemap is not None:
            m = Basemap(projection="ortho", lon_0=l0, lat_0=p0, resolution=None, ax=ax)
            R = m.rmajor
            xc, yc = m(l0, p0)             # Basemap map coordinates of the globe centre
            fill_fn, line_fn = m.contourf, m.contour
        else:
            m, R, xc, yc = None, 1.0, 0.0, 0.0
            fill_fn, line_fn = ax.contourf, ax.contour

        # Projected grid. Points just behind the limb are pinned onto it so the
        # filled contours reach the edge cleanly; the rest of the far side is masked.
        x, y, cosc = ortho_xy(LON, LAT, l0, p0, R)
        rho = np.hypot(x, y)
        scale = np.where((cosc < 0) & (rho > 0), R / np.where(rho > 0, rho, 1.0), 1.0)
        x, y = x * scale + xc, y * scale + yc
        F = np.ma.masked_where((cosc < -0.05) | ~np.isfinite(G), G)

        n_before = len(ax.collections)
        cf = cl = None
        if o["filled"]:
            cf = fill_fn(x, y, F, levels, cmap=o["cmap"], extend=extend)
        if o["lines"] or not o["filled"]:
            kw = dict(colors="k", linewidths=0.5) if o["filled"] else dict(cmap=o["cmap"], linewidths=1.0)
            cl = line_fn(x, y, F, levels, **kw)
            if o["clabels"]:
                ax.clabel(cl, inline=True, fontsize=7, fmt="%.3g")
        cs = cf if cf is not None else cl
        limb = Circle((xc, yc), R, transform=ax.transData)
        for coll in ax.collections[n_before:]:
            coll.set_clip_path(limb)

        if m is not None:
            m.drawmapboundary(color="k", linewidth=1.0, fill_color="none")
            m.drawparallels(np.arange(-60.0, 61.0, 30.0), color="0.45", linewidth=0.4, dashes=[1, 0])
            m.drawmeridians(np.arange(-180.0, 180.0, 30.0), color="0.45", linewidth=0.4, dashes=[1, 0])
        else:
            _ortho_graticule(ax, l0, p0, R, xc, yc)

        for clon, clat, style, label in curves:
            bx, by, bc = ortho_xy(clon, clat, l0, p0, R)
            ax.plot(np.where(bc >= 0, bx + xc, np.nan), np.where(bc >= 0, by + yc, np.nan),
                    label=label, zorder=5, **style)

        ax.set_xlim(xc - 1.02 * R, xc + 1.02 * R)
        ax.set_ylim(yc - 1.02 * R, yc + 1.02 * R)
        ax.set_aspect("equal")
        ax.set_axis_off()
        side = "far side" if k else "view"
        ax.set_title(f"{side}: centre {l0:g}°E, {p0:g}°N", fontsize=9)

    if curves:
        axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=7, framealpha=0.8)
    engine = "Basemap" if Basemap is not None else "Matplotlib – install basemap for Basemap rendering"
    fig.suptitle(o["title"] or f"Yin-Yang field on the sphere (orthographic contourf, {engine})")
    fig.colorbar(cs, ax=list(axs), orientation="horizontal", fraction=0.06, pad=0.08, label=data.name)
    return axs


def plot_spectrum(fig, data, o):
    """Spherical-harmonic power spectrum of the merged field.

    If o["spectra"] is a list of (YYData, label) pairs, all of them are drawn
    on the same axes (e.g. fields A and B, and optionally A - B).
    """
    series = o.get("spectra") or [(data, data.name)]
    lmax = auto_lmax(series[0][0]) if o["lmax"] is None else int(o["lmax"])
    per_coeff = o["spec_norm"] == "coeff"
    ax = fig.add_subplot(1, 1, 1)
    styles = [dict(color="C0", ls="-"), dict(color="C1", ls="-"), dict(color="C2", ls="--"),
              dict(color="C3", ls=":")]
    notes, lp = [], None
    for k, (d, label) in enumerate(series):
        l, S, mean_sq = power_spectrum(d, lmax)
        P = S / (2 * l + 1) if per_coeff else S
        mask = l >= (1 if o["logl"] else 0)
        lp, Pp = l[mask], np.where(P[mask] > 0, P[mask], np.nan)
        floor = np.nanmax(Pp) * 1e-30 if np.any(np.isfinite(Pp)) else 1.0
        Pp = np.where(Pp > floor, Pp, np.nan)         # hide exact zeros / round-off
        st = styles[k % len(styles)]
        ax.plot(lp, Pp, marker="o", ms=3 if lp.size < 120 else 0, lw=1.2, label=label, **st)
        tag = f"{label}: " if len(series) > 1 else ""
        notes.append(f"{tag}Σ S(l) = {S.sum():.5g}   ·   ⟨f²⟩ = {mean_sq:.5g}")
    ax.set_yscale("log")
    if o["logl"]:
        ax.set_xscale("log")
    ax.set_xlim(lp[0] - (0 if o["logl"] else 0.5), lp[-1] * (1.05 if o["logl"] else 1) + (0 if o["logl"] else 0.5))
    ax.set_xlabel("spherical harmonic degree  l")
    ax.set_ylabel("power per coefficient  S(l)/(2l+1)" if per_coeff else "power per degree  S(l)")
    ax.grid(True, which="both", alpha=0.3)
    note = "\n".join([f"lmax = {lmax}"] + notes +
                      ["S(l) = (1/4π) Σₘ |f_lm|²  (4π-normalised; Σ S(l) = mean square of f)"])
    ax.text(0.99, 0.99, note, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    if len(series) > 1:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        default = "Spherical-harmonic power spectra"
    else:
        default = f"Spherical-harmonic power spectrum — {series[0][1]}"
    import textwrap
    ax.set_title("\n".join(textwrap.wrap(o["title"] or default, 90)), fontsize=10)
    return [ax]


VIEWS = {
    "Native panels (Yin | Yang)": lambda fig, d, o: plot_native(fig, d, o),
    "Global map – equirectangular": lambda fig, d, o: plot_global(fig, d, o, "equirect"),
    "Global map – Mollweide": lambda fig, d, o: plot_global(fig, d, o, "mollweide"),
    "Sphere – orthographic (Basemap)": lambda fig, d, o: plot_sphere(fig, d, o),
    "Power spectrum (spherical harmonics)": lambda fig, d, o: plot_spectrum(fig, d, o),
}


def draw(fig, data, view, **opts):
    o = dict(DEFAULT_OPTS)
    o.update(opts)
    fig.clf()
    return VIEWS[view](fig, data, o)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
CMAPS = ["RdBu_r", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm",
         "seismic", "jet", "turbo", "Spectral_r", "gist_earth", "twilight", "gray"]


class YinYangApp:
    def __init__(self, root, data=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("Yin-Yang contour viewer")
        root.geometry("1250x760")

        self.raw = None          # field A: (field_2d, theta_or_None, phi_or_None, name, rot_or_None)
        self.raw_b = None        # field B, same structure (optional)
        self.gem = None          # GemDataset currently open in the GEM panel
        self._gem_sel = []       # parallel lists behind the GEM comboboxes
        self.data_a = self.data_b = None
        self.data = None         # what is currently displayed (A, B or A - B)

        V = tk.StringVar
        self.v_view = V(value=list(VIEWS)[0])
        self.v_nlev = tk.IntVar(value=20)
        self.v_cmap = V(value="RdBu_r")
        self.v_filled = tk.BooleanVar(value=True)
        self.v_lines = tk.BooleanVar(value=True)
        self.v_clabels = tk.BooleanVar(value=False)
        self.v_sym = tk.BooleanVar(value=False)
        self.v_bounds = tk.BooleanVar(value=True)
        self.v_transpose = tk.BooleanVar(value=False)
        self.v_layout = V(value=LAYOUTS["auto"])
        self.v_show = V(value=SHOW_MODES["A"])
        self.v_vmin, self.v_vmax = V(value=""), V(value="")
        self.v_tmin, self.v_tmax = V(value="45"), V(value="135")
        self.v_pmin, self.v_pmax = V(value="-135"), V(value="135")
        self.v_nlat, self.v_nlon = V(value="181"), V(value="361")
        self.v_title = V(value="")
        self.v_lon0, self.v_lat0 = V(value="0"), V(value="30")
        self.v_antipode = tk.BooleanVar(value=True)
        self.v_lmax = V(value="")
        self.v_percoeff = tk.BooleanVar(value=False)
        self.v_logl = tk.BooleanVar(value=False)
        self.v_overlay = tk.BooleanVar(value=False)
        self.v_overlay_diff = tk.BooleanVar(value=False)
        self.v_status = V(value="Load a field or click 'Demo field'.")

        # ---- layout
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        # Scrollable control column (the options no longer fit a normal window height)
        side = ttk.Frame(main)
        side.pack(side="left", fill="y")
        bg = ttk.Style().lookup("TFrame", "background") or None
        ctrl_canvas = tk.Canvas(side, highlightthickness=0, width=300, background=bg)
        ctrl_sb = ttk.Scrollbar(side, orient="vertical", command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb.set)
        ctrl_sb.pack(side="right", fill="y")
        ctrl_canvas.pack(side="left", fill="y", expand=True)
        ctrl = ttk.Frame(ctrl_canvas, padding=6)
        ctrl_canvas.create_window((0, 0), window=ctrl, anchor="nw")
        ctrl.bind("<Configure>", lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox("all"), width=ctrl.winfo_reqwidth()))

        def _wheel(ev):
            if getattr(ev, "num", None) == 4 or getattr(ev, "delta", 0) > 0:
                ctrl_canvas.yview_scroll(-1, "units")
            else:
                ctrl_canvas.yview_scroll(1, "units")

        def _bind_wheel(_):
            ctrl_canvas.bind_all("<MouseWheel>", _wheel)
            ctrl_canvas.bind_all("<Button-4>", _wheel)
            ctrl_canvas.bind_all("<Button-5>", _wheel)

        def _unbind_wheel(_):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                ctrl_canvas.unbind_all(seq)

        ctrl_canvas.bind("<Enter>", _bind_wheel)
        ctrl_canvas.bind("<Leave>", _unbind_wheel)
        plot = ttk.Frame(main)
        plot.pack(side="right", fill="both", expand=True)

        f = ttk.LabelFrame(ctrl, text="Input", padding=4)
        f.pack(fill="x", pady=3)
        ttk.Button(f, text="Load field A…", command=self.load_file).pack(fill="x")
        r = ttk.Frame(f); r.pack(fill="x", pady=2)
        ttk.Button(r, text="Load field B (compare)…", command=self.load_file_b).pack(
            side="left", expand=True, fill="x")
        ttk.Button(r, text="Clear B", command=self.clear_b).pack(side="left")
        ttk.Button(f, text="Demo fields (A and B)", command=self.load_demo).pack(fill="x")
        r = ttk.Frame(f); r.pack(fill="x", pady=3)
        ttk.Label(r, text="Show").pack(side="left")
        cb = ttk.Combobox(r, textvariable=self.v_show, values=list(SHOW_MODES.values()),
                          state="readonly", width=22)
        cb.pack(side="left", padx=2, fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_show_change())
        r = ttk.Frame(f); r.pack(fill="x")
        ttk.Label(r, text="Panel layout").pack(side="left")
        cb = ttk.Combobox(r, textvariable=self.v_layout, values=list(LAYOUTS.values()),
                          state="readonly", width=26)
        cb.pack(side="left", padx=2, fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_layout_change())
        ttk.Checkbutton(f, text="Transpose panels (n_phi, n_theta)", variable=self.v_transpose,
                        command=self.rebuild).pack(anchor="w")

        f = ttk.LabelFrame(ctrl, text="GEM model output (RPN standard files)", padding=4)
        f.pack(fill="x", pady=3)
        r = ttk.Frame(f); r.pack(fill="x")
        ttk.Button(r, text="Open GEM folder…", command=self.open_gem_dir).pack(side="left", expand=True, fill="x")
        ttk.Button(r, text="File…", command=self.open_gem_file).pack(side="left")
        self.lbl_gem = ttk.Label(f, text="no GEM output open", wraplength=270, justify="left",
                                 foreground="gray30")
        self.lbl_gem.pack(anchor="w", pady=2)
        g = ttk.Frame(f); g.pack(fill="x")
        self.cb_gem = {}
        for row, (key, label, width) in enumerate((("var", "Variable", 10), ("lev", "Level", 18),
                                                  ("time", "Time", 26))):
            ttk.Label(g, text=label).grid(row=row, column=0, sticky="w")
            v = tk.StringVar(value="")
            cb = ttk.Combobox(g, textvariable=v, state="readonly", width=width)
            cb.grid(row=row, column=1, sticky="ew", padx=2, pady=1)
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self.on_gem_select(k))
            self.cb_gem[key] = (cb, v)
        g.columnconfigure(1, weight=1)
        self.v_gem_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Selection updates field A", variable=self.v_gem_auto).pack(anchor="w")
        r = ttk.Frame(f); r.pack(fill="x", pady=2)
        ttk.Button(r, text="Show as A", command=lambda: self.load_gem_selected("A")).pack(
            side="left", expand=True, fill="x")
        ttk.Button(r, text="Load as B", command=lambda: self.load_gem_selected("B")).pack(
            side="left", expand=True, fill="x")
        ttk.Label(f, text="To compare runs: untick the box, open the other run's\n"
                          "folder, pick the record, then 'Load as B'.",
                  foreground="gray30").pack(anchor="w")
        self.lbl_info = ttk.Label(f, text="no data", wraplength=250, justify="left")
        self.lbl_info.pack(anchor="w", pady=2)

        f = ttk.LabelFrame(ctrl, text="Panel grid extent [deg] (if not in file)", padding=4)
        f.pack(fill="x", pady=3)
        self.grid_entries = []
        for row, (lab, va, vb) in enumerate((("θ (colat)", self.v_tmin, self.v_tmax),
                                             ("φ (lon)", self.v_pmin, self.v_pmax))):
            ttk.Label(f, text=lab).grid(row=row, column=0, sticky="w")
            for col, v in ((1, va), (2, vb)):
                e = ttk.Entry(f, textvariable=v, width=8)
                e.grid(row=row, column=col, padx=2)
                e.bind("<Return>", lambda ev: self.rebuild())
                self.grid_entries.append(e)

        f = ttk.LabelFrame(ctrl, text="Plot", padding=4)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="View").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(f, textvariable=self.v_view, values=list(VIEWS), state="readonly", width=28)
        cb.grid(row=0, column=1, columnspan=2, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        ttk.Label(f, text="Colormap").grid(row=1, column=0, sticky="w")
        cb = ttk.Combobox(f, textvariable=self.v_cmap, values=CMAPS, width=14)
        cb.grid(row=1, column=1, columnspan=2, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        cb.bind("<Return>", lambda e: self.redraw())
        ttk.Label(f, text="Levels").grid(row=2, column=0, sticky="w")
        sp = ttk.Spinbox(f, from_=2, to=200, textvariable=self.v_nlev, width=6, command=self.redraw)
        sp.grid(row=2, column=1, sticky="w")
        sp.bind("<Return>", lambda e: self.redraw())
        ttk.Label(f, text="vmin / vmax").grid(row=3, column=0, sticky="w")
        for col, v in ((1, self.v_vmin), (2, self.v_vmax)):
            e = ttk.Entry(f, textvariable=v, width=9)
            e.grid(row=3, column=col, padx=2)
            e.bind("<Return>", lambda ev: self.redraw())
        ttk.Label(f, text="(blank = auto)").grid(row=4, column=1, columnspan=2, sticky="w")
        for row, (txt, var) in enumerate((("Filled contours", self.v_filled),
                                          ("Contour lines", self.v_lines),
                                          ("Line labels", self.v_clabels),
                                          ("Symmetric about 0", self.v_sym),
                                          ("Show panel boundaries", self.v_bounds)), start=5):
            ttk.Checkbutton(f, text=txt, variable=var, command=self.redraw).grid(
                row=row, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="Global grid nlat×nlon").grid(row=10, column=0, sticky="w")
        for col, v in ((1, self.v_nlat), (2, self.v_nlon)):
            e = ttk.Entry(f, textvariable=v, width=6)
            e.grid(row=10, column=col, padx=2)
            e.bind("<Return>", lambda ev: self.redraw())
        ttk.Label(f, text="Title").grid(row=11, column=0, sticky="w")
        e = ttk.Entry(f, textvariable=self.v_title, width=20)
        e.grid(row=11, column=1, columnspan=2, sticky="ew")
        e.bind("<Return>", lambda ev: self.redraw())

        f = ttk.LabelFrame(ctrl, text="Sphere view (Basemap orthographic)", padding=4)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="Centre lon / lat").grid(row=0, column=0, sticky="w")
        for col, v in ((1, self.v_lon0), (2, self.v_lat0)):
            e = ttk.Entry(f, textvariable=v, width=6)
            e.grid(row=0, column=col, padx=2)
            e.bind("<Return>", lambda ev: self.redraw())
        ttk.Checkbutton(f, text="Also show far side", variable=self.v_antipode,
                        command=self.redraw).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(f, text="Tip: click and drag a globe to rotate it\n(turn off the toolbar's pan/zoom first)",
                  foreground="gray30").grid(row=2, column=0, columnspan=3, sticky="w")

        f = ttk.LabelFrame(ctrl, text="Power spectrum", padding=4)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text="lmax (blank = auto)").grid(row=0, column=0, sticky="w")
        e = ttk.Entry(f, textvariable=self.v_lmax, width=6)
        e.grid(row=0, column=1, padx=2, sticky="w")
        e.bind("<Return>", lambda ev: self.redraw())
        ttk.Checkbutton(f, text="Per coefficient  S(l)/(2l+1)", variable=self.v_percoeff,
                        command=self.redraw).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f, text="Log degree axis", variable=self.v_logl,
                        command=self.redraw).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f, text="Overlay spectra of A and B", variable=self.v_overlay,
                        command=self.on_overlay_change).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f, text="   … and of A − B", variable=self.v_overlay_diff,
                        command=self.on_overlay_change).grid(row=4, column=0, columnspan=2, sticky="w")

        f = ttk.Frame(ctrl, padding=4)
        f.pack(fill="x", pady=3)
        ttk.Button(f, text="Plot / refresh", command=self.redraw).pack(fill="x")
        ttk.Button(f, text="Save figure…", command=self.save_fig).pack(fill="x", pady=2)
        ttk.Button(f, text="Export merged lon-lat grid…", command=self.export_global).pack(fill="x")
        ttk.Button(f, text="Export power spectrum…", command=self.export_spectrum).pack(fill="x", pady=2)

        self.fig = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Mouse-drag rotation of the globe in the sphere view
        self.sphere_axes = []
        self._drag = None
        self._drag_job = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        ttk.Label(root, textvariable=self.v_status, relief="sunken", anchor="w").pack(fill="x", side="bottom")

        if data is not None:
            self.set_field(np.concatenate([data.yin, data.yang], axis=0), data.theta, data.phi, data.name,
                           data.rot)

    # ---- helpers
    def status(self, msg):
        self.v_status.set(msg)

    def error(self, msg):
        from tkinter import messagebox
        self.status("Error: " + msg)
        messagebox.showerror("Yin-Yang viewer", msg)

    @staticmethod
    def _float_or_none(s):
        s = s.strip()
        return float(s) if s else None

    def options(self):
        return dict(nlevels=int(self.v_nlev.get()), cmap=self.v_cmap.get(),
                    filled=self.v_filled.get(), lines=self.v_lines.get(),
                    clabels=self.v_clabels.get(), symmetric=self.v_sym.get(),
                    vmin=self._float_or_none(self.v_vmin.get()),
                    vmax=self._float_or_none(self.v_vmax.get()),
                    boundaries=self.v_bounds.get(), nlat=int(self.v_nlat.get()),
                    nlon=int(self.v_nlon.get()), title=self.v_title.get(),
                    lon0=float(self.v_lon0.get() or 0), lat0=float(self.v_lat0.get() or 0),
                    antipode=self.v_antipode.get(),
                    lmax=int(self.v_lmax.get()) if self.v_lmax.get().strip() else None,
                    spec_norm="coeff" if self.v_percoeff.get() else "degree",
                    logl=self.v_logl.get(),
                    spectra=self._spectra_series())

    # ---- data handling
    def _layout_key(self):
        label = self.v_layout.get()
        return next((k for k, v in LAYOUTS.items() if v == label), "auto")

    def _show_key(self):
        label = self.v_show.get()
        return next((k for k, v in SHOW_MODES.items() if v == label), "A")

    def _guess_transpose(self):
        """Tick 'Transpose' if field A's split panels look like (n_phi, n_theta)."""
        field, theta, phi, _, _ = self.raw
        try:
            yin, _, _ = split_panels(field, self._layout_key(), theta, phi)
        except Exception:
            return
        if theta is not None and phi is not None:
            guess = yin.shape == (phi.size, theta.size) and phi.size != theta.size
        else:
            guess = yin.shape[0] > yin.shape[1]       # Yin-Yang panels have n_phi ~ 3 n_theta
        self.v_transpose.set(bool(guess))

    def _build(self, raw, ref=None):
        """Split/transposed YYData from a raw tuple. With ref (field A), the
        result must have the same panel shape and grid as ref."""
        field, theta, phi, name, rot = raw
        yin, yang, axis = split_panels(field, self._layout_key(),
                                       theta if theta is not None else (ref.theta if ref else None),
                                       phi if phi is not None else (ref.phi if ref else None))
        if self.v_transpose.get():
            yin, yang = yin.T, yang.T
        if ref is not None:
            if yin.shape != ref.shape:
                raise ValueError(f"Field B panels {yin.shape} do not match field A panels {ref.shape} "
                                 "(same layout and transpose settings are applied to both).")
            if theta is not None and phi is not None:
                YYData(yin, yang, theta, phi)        # validates sizes/ordering
                if not (np.allclose(np.sort(theta), ref.theta) and np.allclose(np.sort(phi), ref.phi)):
                    raise ValueError("Field B's theta/phi grid differs from field A's.")
            if (rot is None) != (ref.rot is None) or (rot is not None and not np.allclose(rot, ref.rot, atol=1e-6)):
                raise ValueError("Field B's grid rotation (GEM xlat1/xlon1/xlat2/xlon2) differs from field A's.")
            theta, phi = ref.theta, ref.phi
        elif theta is None or phi is None:
            nt, nph = yin.shape
            theta = np.linspace(float(self.v_tmin.get()), float(self.v_tmax.get()), nt) * DEG
            phi = np.linspace(float(self.v_pmin.get()), float(self.v_pmax.get()), nph) * DEG
        return YYData(yin, yang, theta, phi, name, rot=rot), axis

    def set_field(self, field, theta, phi, name, rot=None):
        """Install field A: a single 2-D array holding both panels (optionally with its grid)."""
        field = np.asarray(field, float)
        if field.ndim != 2:
            self.error(f"Expected a single 2-D field holding both panels; got shape {field.shape}.")
            return
        self.raw = (field, theta, phi, name, rot)
        state = "disabled" if theta is not None and phi is not None else "normal"
        if state == "disabled":
            self.v_tmin.set(f"{theta[0] / DEG:.4g}"); self.v_tmax.set(f"{theta[-1] / DEG:.4g}")
            self.v_pmin.set(f"{phi[0] / DEG:.4g}"); self.v_pmax.set(f"{phi[-1] / DEG:.4g}")
        for e in self.grid_entries:
            e.configure(state=state)
        self._guess_transpose()
        self.rebuild()

    def set_field_b(self, field, theta, phi, name, rot=None):
        """Install field B (compared against A)."""
        field = np.asarray(field, float)
        if field.ndim != 2:
            self.error(f"Expected a single 2-D field holding both panels; got shape {field.shape}.")
            return
        self.raw_b = (field, theta, phi, name, rot)
        self.rebuild()
        if self.data_b is not None:
            self.status(f"Field B loaded ({name}). Choose Show → '{SHOW_MODES['diff']}' to see A − B.")

    def clear_b(self):
        self.raw_b = None
        self.data_b = None
        self.v_show.set(SHOW_MODES["A"])
        self.rebuild()

    def on_layout_change(self):
        if self.raw is not None:
            self._guess_transpose()
            self.rebuild()

    def _spectra_series(self):
        """(YYData, label) pairs to overlay in the spectrum view, or None."""
        a, b = getattr(self, "data_a", None), getattr(self, "data_b", None)
        if not (self.v_overlay.get() or self.v_overlay_diff.get()) or a is None or b is None:
            return None
        series = []
        if self.v_overlay.get():
            series += [(a, f"A: {a.name}"), (b, f"B: {b.name}")]
        if self.v_overlay_diff.get():
            series.append((a.with_values(a.yin - b.yin, a.yang - b.yang, "A − B"), "A − B"))
        return series

    def on_overlay_change(self):
        if (self.v_overlay.get() or self.v_overlay_diff.get()):
            if self.data_b is None:
                self.status("Load field B to overlay spectra.")
            elif not self.v_view.get().startswith("Power spectrum"):
                self.v_view.set("Power spectrum (spherical harmonics)")   # jump to the spectrum view
        self.redraw()

    def on_show_change(self):
        if self._show_key() == "diff" and not self.v_sym.get():
            self.v_sym.set(True)              # differences read best on a range centred on 0
            self._sym_auto = True
        elif self._show_key() != "diff" and getattr(self, "_sym_auto", False):
            self.v_sym.set(False)             # undo the automatic choice when leaving A − B
            self._sym_auto = False
        self.compose()

    def rebuild(self):
        """Re-split both fields with the current layout/transpose/grid settings."""
        if self.raw is None:
            return
        try:
            self.data_a, axis = self._build(self.raw)
        except Exception as exc:
            self.data_a = self.data = None
            self.error(str(exc))
            return
        self.data_b, self._axis = None, axis
        if self.raw_b is not None:
            try:
                self.data_b, _ = self._build(self.raw_b, ref=self.data_a)
            except Exception as exc:
                self.raw_b = None
                self.error(f"Field B removed: {exc}")
        self.compose()

    def compose(self):
        """Choose what to display (A, B or A − B), update the info box and redraw."""
        a, b = self.data_a, self.data_b
        if a is None:
            return
        mode = self._show_key()
        if mode != "A" and b is None:
            self.v_show.set(SHOW_MODES["A"])
            mode = "A"
            self.status("No field B loaded — showing field A.")
        if mode == "A":
            self.data = a
        elif mode == "B":
            self.data = b
        else:
            self.data = a.with_values(a.yin - b.yin, a.yang - b.yang, f"{a.name} − {b.name}")
        d = self.data
        both = np.concatenate([d.yin.ravel(), d.yang.ravel()])
        split = "Yin above Yang" if self._axis == 0 else "Yin left of Yang"
        lines = [f"A: {a.name}  {self.raw[0].shape} → {split}"]
        if b is not None:
            diff = np.concatenate([(a.yin - b.yin).ravel(), (a.yang - b.yang).ravel()])
            lines.append(f"B: {b.name}")
            lines.append(f"A−B: rms {np.sqrt(np.nanmean(diff ** 2)):.4g}, "
                         f"max |A−B| {np.nanmax(np.abs(diff)):.4g}")
        if d.rot is not None:
            lines.append("maps in true geographic lat/lon (grid rotation from file)")
        lines += [f"panel shape (nθ, nφ) = {d.shape}",
                  f"θ: {d.theta[0] / DEG:.2f}…{d.theta[-1] / DEG:.2f}°  "
                  f"φ: {d.phi[0] / DEG:.2f}…{d.phi[-1] / DEG:.2f}°",
                  f"shown ({SHOW_MODES[mode]}): min {np.nanmin(both):.4g}  max {np.nanmax(both):.4g}"]
        self.lbl_info.configure(text="\n".join(lines))
        self.redraw()

    def load_demo(self):
        a, b = demo_field(variant="A"), demo_field(variant="B")
        self.v_layout.set(LAYOUTS["auto"])
        self.raw_b = None
        self.set_field(np.concatenate([a.yin, a.yang], axis=0), a.theta, a.phi, a.name, a.rot)
        self.set_field_b(np.concatenate([b.yin, b.yang], axis=0), b.theta, b.phi, b.name, b.rot)

    def _ask_field(self, which):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title=f"Yin-Yang field {which} (one 2-D array with both panels)",
            filetypes=[("Arrays", "*.npy *.npz *.txt *.dat *.csv"), ("All", "*.*")])
        if not path:
            return None
        try:
            return load_field(path) + (os.path.basename(path),)
        except Exception as exc:
            self.error(str(exc))
            return None

    def load_file(self):
        got = self._ask_field("A")
        if got:
            self.set_field(*got)

    def load_file_b(self):
        if self.raw is None:
            self.error("Load field A first.")
            return
        got = self._ask_field("B")
        if got:
            self.set_field_b(*got)

    # ---- GEM model output
    def _gem_initialdir(self):
        d = GEM_DEFAULT_DIR
        while d and not os.path.isdir(d):
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return d if os.path.isdir(d) else os.path.expanduser("~")

    def open_gem_dir(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="GEM output folder (e.g. …/laststep_…/001-001)",
                                       initialdir=self._gem_initialdir(), mustexist=True)
        if path:
            self.open_gem(path)

    def open_gem_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="GEM output file (RPN standard file)",
                                          initialdir=self._gem_initialdir())
        if path:
            self.open_gem(path)

    def open_gem(self, path, var=None, load=None):
        """Index GEM output (file or folder); fill the selectors; optionally load as 'A'/'B'.
        Returns True on success."""
        try:
            self.status(f"Reading GEM directory of {path} …")
            self.root.update_idletasks()
            ds = GemDataset(path)
        except Exception as exc:
            self.error(f"{type(exc).__name__}: {exc}")
            return False
        self.gem = ds
        nfiles = len(ds.files)
        note = f"{os.path.basename(os.path.normpath(path)) or path}: {len(ds.records)} Yin-Yang fields, " \
               f"{len(ds.variables())} variables, {nfiles} file{'s' if nfiles != 1 else ''}"
        if ds.problems:
            note += f"\n({len(ds.problems)} file(s) skipped: unreadable)"
        self.lbl_gem.configure(text=note)
        vars_ = ds.variables()
        if var is None or var not in vars_:
            var = next((v for v in GEM_PREFERRED_VARS if v in vars_), vars_[0])
        cb, v = self.cb_gem["var"]
        cb.configure(values=vars_)
        v.set(var)
        self._fill_gem_levels()
        if load is None:
            load = "A" if (self.v_gem_auto.get() or self.raw is None) else None
        if load:
            self.load_gem_selected(load)
        else:
            self.status(note)
        return True

    def _fill_gem_levels(self, keep=None):
        var = self.cb_gem["var"][1].get()
        ips = self.gem.levels(var)
        labels = [self.gem.level_label(ip) for ip in ips]
        cb, v = self.cb_gem["lev"]
        cb.configure(values=labels)
        self._gem_levels = ips
        if keep in ips:
            v.set(labels[ips.index(keep)])
        else:
            v.set(labels[0] if labels else "")
        self._fill_gem_times()

    def _fill_gem_times(self, keep=None):
        var = self.cb_gem["var"][1].get()
        ip1 = self._gem_current_ip1()
        times = self.gem.times(var, ip1)
        labels = [self.gem.time_label(*t) for t in times]
        cb, v = self.cb_gem["time"]
        cb.configure(values=labels)
        self._gem_times = times
        if keep in times:
            v.set(labels[times.index(keep)])
        else:
            v.set(labels[0] if labels else "")

    def _gem_current_ip1(self):
        cb, v = self.cb_gem["lev"]
        labels = list(cb.cget("values"))
        return self._gem_levels[labels.index(v.get())] if v.get() in labels else None

    def _gem_current_time(self):
        cb, v = self.cb_gem["time"]
        labels = list(cb.cget("values"))
        return self._gem_times[labels.index(v.get())] if v.get() in labels else None

    def on_gem_select(self, which):
        if self.gem is None:
            return
        if which == "var":
            self._fill_gem_levels(keep=self._gem_current_ip1())
        elif which == "lev":
            self._fill_gem_times(keep=self._gem_current_time())
        if self.v_gem_auto.get():
            self.load_gem_selected("A")

    def load_gem_selected(self, target="A"):
        if self.gem is None:
            self.error("Open a GEM output folder first.")
            return
        var = self.cb_gem["var"][1].get()
        try:
            rec = self.gem.find(var, self._gem_current_ip1(), self._gem_current_time())
            self.status(f"Decoding {var} …")
            self.root.update_idletasks()
            field, theta, phi, rot, name = self.gem.load(rec)
        except Exception as exc:
            self.error(f"{type(exc).__name__}: {exc}")
            return
        self.v_layout.set(LAYOUTS["rows"])          # GEM: Yin rows first, then Yang
        if target == "B":
            if self.raw is None:
                self.error("Load field A first.")
                return
            self.set_field_b(field, theta, phi, name, rot)
        else:
            self.set_field(field, theta, phi, name, rot)
            self.v_transpose.set(False)

    # ---- drawing / output
    def _is_sphere(self):
        return self.v_view.get().startswith("Sphere")

    def redraw(self, fast=False):
        """Redraw the figure. fast=True (used while dragging the globe) uses a
        coarser merged grid, no line labels, and renders synchronously."""
        if self.data is None:
            return
        try:
            opts = self.options()
            if fast:
                opts.update(nlat=min(opts["nlat"], 91), nlon=min(opts["nlon"], 181), clabels=False)
            else:
                self.status("Drawing…")
                self.root.update_idletasks()
            result = draw(self.fig, self.data, self.v_view.get(), **opts)
            self.sphere_axes = list(result) if self._is_sphere() else []
            if fast:
                self.canvas.draw()
            else:
                self.canvas.draw_idle()
                hint = "  |  drag a globe with the mouse to rotate it" if self._is_sphere() else ""
                self.status(f"{self.v_view.get()} — {self.data.name}{hint}")
        except Exception as exc:
            self._drag = None
            self.fig.clf()
            self.canvas.draw_idle()
            self.error(f"{type(exc).__name__}: {exc}")

    # ---- mouse rotation of the globe (sphere view)
    def _on_press(self, ev):
        if (ev.button != 1 or not self._is_sphere() or self.toolbar.mode
                or ev.inaxes is None or ev.inaxes not in self.sphere_axes):
            return
        idx = self.sphere_axes.index(ev.inaxes)
        width = ev.inaxes.get_window_extent().width
        try:
            lon0, lat0 = float(self.v_lon0.get() or 0), float(self.v_lat0.get() or 0)
        except ValueError:
            lon0, lat0 = 0.0, 0.0
        # Grab-the-surface feel: dragging across the globe's full width turns it by 180°.
        # The far-side globe (idx 1) has its latitude mirrored, hence the sign flip.
        self._drag = dict(x=ev.x, y=ev.y, lon=lon0, lat=lat0,
                          sgn=1.0 if idx == 0 else -1.0, k=180.0 / max(width, 1.0))
        self.canvas.get_tk_widget().configure(cursor="fleur")

    def _on_motion(self, ev):
        d = self._drag
        if not d or ev.x is None:
            return
        lon = d["lon"] - (ev.x - d["x"]) * d["k"]
        lat = d["lat"] - d["sgn"] * (ev.y - d["y"]) * d["k"]
        lon = (lon + 180.0) % 360.0 - 180.0
        lat = float(np.clip(lat, -90.0, 90.0))
        self.v_lon0.set(f"{lon:.1f}")
        self.v_lat0.set(f"{lat:.1f}")
        if self._drag_job is None:          # coalesce motion events into one redraw
            self._drag_job = self.root.after(10, self._drag_redraw)

    def _drag_redraw(self):
        self._drag_job = None
        if self._drag:
            self.redraw(fast=True)

    def _on_release(self, ev):
        if not self._drag:
            return
        self._drag = None
        if self._drag_job is not None:
            self.root.after_cancel(self._drag_job)
            self._drag_job = None
        self.canvas.get_tk_widget().configure(cursor="")
        self.redraw()                        # full-resolution redraw at the final position

    def save_fig(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                       ("SVG", "*.svg"), ("All", "*.*")])
        if path:
            self.fig.savefig(path, dpi=200)
            self.status(f"Saved {path}")

    def export_global(self):
        from tkinter import filedialog
        if self.data is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy", "*.npz")])
        if path:
            o = self.options()
            lon, lat, G = to_global(self.data, o["nlat"], o["nlon"])
            np.savez(path, lon=lon, lat=lat, field=G)
            self.status(f"Exported merged grid {G.shape} to {path}")

    def export_spectrum(self):
        from tkinter import filedialog
        if self.data is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv"), ("Text", "*.txt")])
        if not path:
            return
        try:
            opts = self.options()
            series = opts["spectra"] or [(self.data, self.data.name)]
            lmax = auto_lmax(series[0][0]) if opts["lmax"] is None else opts["lmax"]
            cols, names, info = [], ["l"], []
            for k, (d, label) in enumerate(series):
                l, S, mean_sq = power_spectrum(d, lmax)
                if not cols:
                    cols.append(l)
                short = label.split(":")[0].replace("−", "minus").replace(" ", "_")
                key = "".join(ch if ch.isalnum() else "_" for ch in short).strip("_") or f"f{k}"
                cols += [S, S / (2 * l + 1)]
                names += [f"S_{key}_per_degree", f"S_{key}_per_coefficient"]
                info.append(f"{label}: sum S = {S.sum():.8g}, <f^2> = {mean_sq:.8g}")
            np.savetxt(path, np.column_stack(cols), delimiter=",",
                       header=("Yin-Yang SH power spectra; S(l)=(1/4pi)sum_m|f_lm|^2\n"
                               + "\n".join(info) + "\n" + ",".join(names)),
                       fmt=["%d"] + ["%.10e"] * (len(cols) - 1))
            self.status(f"Exported {len(series)} spectrum/spectra (lmax={lmax}) to {path}")
        except Exception as exc:
            self.error(str(exc))


def show(field=None, theta=None, phi=None, layout="auto", name="field", degrees=None,
         field_b=None, name_b="field B", mode="A", var=None):
    """Launch the GUI, optionally with a field already loaded.

    field may also be a path to GEM output (folder or RPN standard file);
    var then selects the variable shown first (e.g. "TT").

    field  : one 2-D array holding both panels, Yin above Yang (layout="rows")
             or Yin left of Yang (layout="cols"); "auto" detects the split.
    theta, phi : 1-D panel coordinates (radians, or degrees if degrees=True or
             values exceed 2*pi). If omitted, the GUI's extent fields define the grid.
    field_b : optional second field, same layout and grid, for comparison.
    mode   : initial display, "A", "B" or "diff" (A - B).
    """
    import tkinter as tk
    root = tk.Tk()
    app = YinYangApp(root)
    if isinstance(field, (str, os.PathLike)):
        app.open_gem(os.fspath(field), var=var, load="A")
        field = None
    if field is not None:
        if degrees:
            theta = None if theta is None else np.asarray(theta, float).ravel() * DEG
            phi = None if phi is None else np.asarray(phi, float).ravel() * DEG
        else:
            theta, phi = _to_radians(theta), _to_radians(phi)
        app.v_layout.set(LAYOUTS.get(layout, LAYOUTS["auto"]))
        app.set_field(field, theta, phi, name)
        if field_b is not None:
            app.set_field_b(field_b, None, None, name_b)
            app.v_show.set(SHOW_MODES.get(mode, SHOW_MODES["A"]))
            app.on_show_change()
    root.mainloop()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Contour plots of a field on a Yin-Yang grid. Input is either a GEM output "
                    "folder/RPN standard file, or a file holding one 2-D array with both panels.")
    p.add_argument("file", nargs="?",
                   help="GEM output folder or RPN file, or .npy/.npz/.txt/.dat/.csv 2-D field (A). "
                        f"Default: {GEM_DEFAULT_DIR} if it exists, else the demo fields")
    p.add_argument("file_b", nargs="?", help="optional second field/GEM output (B) to compare with A")
    p.add_argument("--var", help="GEM variable to show first (e.g. TT); same level/time is used for B")
    p.add_argument("--diff", action="store_true", help="start by showing the difference A - B")
    p.add_argument("--layout", choices=list(LAYOUTS), default="auto",
                   help="how the panels are stacked: rows = Yin above Yang, "
                        "cols = Yin left of Yang (default: auto-detect)")
    a = p.parse_args(argv)

    import tkinter as tk
    root = tk.Tk()
    app = YinYangApp(root)
    app.v_layout.set(LAYOUTS[a.layout])
    def is_gem(path):
        return os.path.isdir(path) or is_rpn_file(path)

    try:
        src = a.file or (GEM_DEFAULT_DIR if os.path.isdir(GEM_DEFAULT_DIR) else None)
        if src and is_gem(src):
            if app.open_gem(src, var=a.var, load="A") and a.file_b:
                ip1, t = app._gem_current_ip1(), app._gem_current_time()
                var = app.cb_gem["var"][1].get()
                dsb = GemDataset(a.file_b) if is_gem(a.file_b) else None
                if dsb is None:
                    raise RPNError("The second input must also be GEM output when the first one is.")
                try:
                    rec = dsb.find(var, ip1, t)
                except RPNError:
                    rec = dsb.find(var, ip1)        # different run date: same variable and level
                fb, thb, phb, rotb, nameb = dsb.load(rec)
                app.set_field_b(fb, thb, phb, f"{os.path.basename(os.path.normpath(a.file_b))}: {nameb}", rotb)
        elif src:
            field, th, ph = load_field(src)
            app.set_field(field, th, ph, os.path.basename(src))
            if a.file_b:
                fb, thb, phb = load_field(a.file_b)
                app.set_field_b(fb, thb, phb, os.path.basename(a.file_b))
        else:
            app.load_demo()
        if a.diff and app.data_b is not None:
            app.v_show.set(SHOW_MODES["diff"])
            app.on_show_change()
    except Exception as exc:
        app.error(str(exc))
    root.mainloop()


if __name__ == "__main__":
    main()
