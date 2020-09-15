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
from pfsspy import analytic

from helpers import brss_pfsspy, brss_analytic

###############################################################################
# Compare the the pfsspy solution to the analytic solutions. Cuts are taken
# on the source surface at a constant phi value to do a 1D comparison.
l = 1
m = 0
nphi = 360
ns = 180
rss = 2.5

fig, axs = plt.subplots(nrows=2)

br_actual = brss_analytic(nphi, ns, rss, l, m)
axs[0].plot(br_actual[:, 180], label='analytic')

for nr in [2, 5, 10]:
    print(nr)
    br_pfsspy = brss_pfsspy(nphi, ns, nr, rss, l, m)

    axs[0].plot(br_pfsspy[:, 180], label=f'nr = {nr}')
    axs[1].scatter(nr, np.nanmean(br_pfsspy[:, 180] / br_actual[:, 180]))

axs[0].legend()
plt.show()
