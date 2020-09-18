"""
Dipole source solution
======================

A simple example showing how to use pfsspy to compute the solution to a dipole
source field.
"""

###############################################################################
# First, import required modules
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import sunpy.map
import pfsspy
import pfsspy.coords as coords
from pfsspy import analytic, tracing

from helpers import pffspy_output

###############################################################################
# Compare the the pfsspy solution to the analytic solutions. Cuts are taken
# on the source surface at a constant phi value to do a 1D comparison.
l = 1
m = 0
nphi = 360
ns = 180
nr = 50
rss = 2


def theta_fline(r, theta0, r0):
    z = r / const.R_sun / rss
    z0 = r0 / const.R_sun / rss
    return np.arccos(np.cos(theta0) *
                     np.sqrt((z0**3 + 2) /
                             (z**3 + 2) *
                             (z / z0))
                     )


pfsspy_out = pffspy_output(nphi, ns, nr, rss, l, m)

phi = 180 * u.deg
theta0 = np.linspace(31, 33, 10) * u.deg
r0 = 1 * const.R_sun
seeds = SkyCoord(radius=r0, lat=theta0, lon=phi, frame=pfsspy_out.coordinate_frame)

tracer = tracing.FortranTracer(step_size=0.005)
flines = tracer.trace(seeds, pfsspy_out)

thetas = [theta_fline(fline.coords.radius, theta, r0) for
          fline, theta in zip(flines, theta0)]

fig, axs = plt.subplots(nrows=2, sharex=True)
for fline, theta in zip(flines, thetas):
    r = (fline.coords.radius).to(const.R_sun)
    theta_analytic = theta.to(u.deg) * np.sign(fline.coords.lat)

    ax = axs[0]
    ax.plot(r, fline.coords.lat.to(u.deg))
    ax.plot(r, theta_analytic, color='k')

    ax = axs[1]
    ax.plot(r, fline.coords.lat.to(u.deg) - theta_analytic)

plt.show()
