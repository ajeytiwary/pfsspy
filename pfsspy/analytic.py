"""
Analytic inputs and solutions to the PFSS equations.
"""
from functools import partial

import numpy as np
import scipy.special
import sympy


def _Ynm(l, m, theta, phi):
    return scipy.special.sph_harm(m, l, phi, theta)


def _cot(theta):
    return 1 / np.tan(theta)


_extras = {'Ynm': _Ynm, 'cot': _cot}


def spherical_harmonic_sympy(l, m):
    """
    Return a complex spherical harmonic with numbers ``l, m``.
    """
    L, M = sympy.symbols('l, m')
    theta, phi = sympy.symbols('theta, phi')
    harm = sympy.Ynm(L, M, theta, phi)
    harm = harm.subs([(L, l), (M, m)])
    return harm, theta, phi


def real_spherical_harmonic_sympy(l, m):
    """
    Return a real spherical harmonic.
    """
    sph, theta, phi = spherical_harmonic_sympy(l, m)
    if m == 0:
        return sph, theta, phi
    elif m < 0:
        # Multiply by i to get imaginary part later
        return sympy.sqrt(2) * (-1)**m * 1j * sph, theta, phi
    elif m > 0:
        return sympy.sqrt(2) * (-1)**m * sph, theta, phi


def _c(l, zss):
    """
    """
    def cl(z):
        return (z**(-l - 2) *
                (l + 1 + l * (z / zss)**(2 * l + 1)) /
                (l + 1 + l * zss**(-2 * l - 1)))

    return cl


def _d(l, zss):
    """
    """
    def dl(z):
        return (z**(-l - 2) *
                (1 - (z / zss)**(2 * l + 1)) /
                (l + 1 + l * zss**(-2 * l - 1)))

    return dl


def Br(l, m, zss):
    """
    Returns
    -------
    function :
        ``Br(r, theta, phi)``, which takes coordiantes and returns the radial
        magnetic field component.
    """
    sph, t, p = real_spherical_harmonic_sympy(l, m)
    sph = sympy.lambdify((t, p), sph, _extras)

    def f(r, theta, phi):
        return _c(l, zss)(r) * np.real(sph(theta, phi))

    return f


def Btheta(l, m, zss):
    """
    Returns
    -------
    function :
        ``Btheta(r, theta, phi)``, which takes coordiantes and returns the
        radial magnetic field component.
    """
    sph, t, p = real_spherical_harmonic_sympy(l, m)
    sph = sympy.diff(sph, t)
    sph = sympy.lambdify((t, p), sph, [_extras, 'numpy'])

    def f(r, theta, phi):
        return _d(l, zss)(r) * np.real(sph(theta, phi))

    return f


def Bphi(l, m, zss):
    sph, t, p = real_spherical_harmonic_sympy(l, m)
    sphi = symp.diff(sph, p)
    sph = sympy.lambdify((t, p), sph, [_extras, 'numpy'])

    def f(r, theta, phi):
        return _d(l, zss)(r) * np.real(sph(theta, phi)) / np.sin(theta)

    return f
