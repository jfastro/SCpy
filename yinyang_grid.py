"""
Yin-Yang grid: latitude/longitude coordinates on the Yin and Yang panels.

The Yin panel is the low-latitude part of a standard lat-lon grid:
    colatitude  theta in [pi/4 - d, 3pi/4 + d]
    longitude   phi   in [-3pi/4 - d, 3pi/4 + d]
where d is a small overlap margin.

The Yang panel uses the same (theta, phi) computational grid in its own
rotated frame. Yang and Yin Cartesian coordinates are related by
    (x_yin, y_yin, z_yin) = (-x_yang, z_yang, y_yang)
(Kageyama & Sato, 2004). We use that rotation to get the geographic
(Yin-frame) latitude and longitude of every Yang grid point.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
n_theta = 61            # points in colatitude
n_phi = 181             # points in longitude
overlap = np.deg2rad(2.0)   # overlap margin d

theta_1d = np.linspace(np.pi / 4 - overlap, 3 * np.pi / 4 + overlap, n_theta)
phi_1d = np.linspace(-3 * np.pi / 4 - overlap, 3 * np.pi / 4 + overlap, n_phi)

# 2D computational grid (identical for both panels)
phi, theta = np.meshgrid(phi_1d, theta_1d)          # shape (n_theta, n_phi)

# ---------------------------------------------------------------------------
# Yin panel: computational coords are the geographic coords
# ---------------------------------------------------------------------------
yin_lat = 90.0 - np.rad2deg(theta)                  # latitude  [deg]
yin_lon = np.rad2deg(phi)                           # longitude [deg]

# ---------------------------------------------------------------------------
# Yang panel: rotate its local frame into the Yin (geographic) frame
# ---------------------------------------------------------------------------
x_loc = np.sin(theta) * np.cos(phi)
y_loc = np.sin(theta) * np.sin(phi)
z_loc = np.cos(theta)

x_geo = -x_loc
y_geo = z_loc
z_geo = y_loc

yang_lat = np.rad2deg(np.arcsin(np.clip(z_geo, -1.0, 1.0)))   # latitude  [deg]
yang_lon = np.rad2deg(np.arctan2(y_geo, x_geo))               # longitude [deg], (-180, 180]

# ---------------------------------------------------------------------------
# Plot with contourf on each panel's computational grid
# ---------------------------------------------------------------------------
comp_lon = np.rad2deg(phi)            # panel-local longitude (x axis)
comp_lat = 90.0 - np.rad2deg(theta)   # panel-local latitude  (y axis)

fields = [
    (yin_lat,  "Yin panel: latitude",   "RdBu_r", np.linspace(-90, 90, 37)),
    (yin_lon,  "Yin panel: longitude",  "twilight", np.linspace(-180, 180, 37)),
    (yang_lat, "Yang panel: latitude",  "RdBu_r", np.linspace(-90, 90, 37)),
    (yang_lon, "Yang panel: longitude", "twilight", np.linspace(-180, 180, 37)),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)

for ax, (field, title, cmap, levels) in zip(axes.flat, fields):
    cf = ax.contourf(comp_lon, comp_lat, field, levels=levels, cmap=cmap)
    ax.contour(comp_lon, comp_lat, field, levels=levels[::3],
               colors="k", linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel("panel-local longitude φ (deg)")
    ax.set_ylabel("panel-local latitude (deg)")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, label="degrees", shrink=0.85)

fig.suptitle("Geographic latitude and longitude on the Yin-Yang grid panels")
fig.savefig("yinyang_latlon.png", dpi=150)
plt.show()
