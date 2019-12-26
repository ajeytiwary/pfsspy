import astropy.constants as const
import astropy.coordinates as coord
from astropy.tests.helper import quantity_allclose

import matplotlib
import numpy as np
import pfsspy
import sunpy.map

import pfsspy.coords
from pfsspy import tracing

from .example_maps import dipole_map, zero_map
matplotlib.use('Agg')


def test_expansion_factor(dipole_map):
    inp, out = dipole_map

    tracer = tracing.PythonTracer()
    field_line = tracer.trace(
        np.array(pfsspy.coords.strum2cart(0.01, 0.9, 0)), out)[0]
    assert field_line.expansion_factor > 1

    field_line = tracer.trace(
        np.array(pfsspy.coords.strum2cart(0.01, -0.9, 0)), out)[0]
    assert field_line.expansion_factor > 1

    # This is a closed field line
    eq_field_line = tracer.trace(
        np.array([0, 1, 0.1]), out)[0]
    assert np.isnan(eq_field_line.expansion_factor)

    # Check that a field line near the equator has a bigger expansion
    # factor than one near the pole
    pil_field_line = tracer.trace(
        np.array(pfsspy.coords.strum2cart(np.log(2.5 - 0.01), 0.1, 0)), out)[0]
    assert pil_field_line.expansion_factor > field_line.expansion_factor


def test_field_line_polarity(dipole_map):
    input, out = dipole_map

    tracer = tracing.PythonTracer()
    field_line = tracer.trace(np.array([0, 0, 1.01]), out)
    assert field_line[0].polarity == 1

    field_line = tracer.trace(np.array([0, 0, -1.01]), out)
    assert field_line[0].polarity == -1

    # This is a closed field line
    eq_field_line = tracer.trace(
        np.array([0, 1, 0.1]), out)[0]
    assert eq_field_line.polarity == 0


def test_footpoints(dipole_map):
    input, out = dipole_map

    tracer = tracing.PythonTracer(atol=1e-8, rtol=1e-8)

    def check_radius(coord, r):
        coord.representation_type = 'spherical'
        assert quantity_allclose(coord.radius, r)

    def check_open_fline(fline):
        check_radius(fline.solar_footpoint, const.R_sun)
        check_radius(fline.source_surface_footpoint, 2.5 * const.R_sun)

    field_line = tracer.trace(np.array([0, 0, 1.01]), out)[0]
    check_open_fline(field_line)

    field_line = tracer.trace(np.array([0, 0, -1.01]), out)[0]
    check_open_fline(field_line)

    field_line = tracer.trace(np.array([0, 1, 0.1]), out)[0]
    check_radius(field_line.solar_footpoint, const.R_sun)
    check_radius(field_line.source_surface_footpoint, const.R_sun)


def test_shape(zero_map):
    # Test output map shapes
    input, out = zero_map
    nr = input.grid.nr
    nphi = input.grid.nphi
    ns = input.grid.ns

    out = pfsspy.pfss(input)
    alr, als, alp = out.al
    for comp in (alr, als, alp):
        assert np.all(comp == 0)

    assert alr.shape == (nphi + 1, ns + 1, nr)
    assert als.shape == (nphi + 1, ns, nr + 1)
    assert alp.shape == (nphi, ns + 1, nr + 1)

    br, bs, bp = out.bc
    for comp in (br, bs, bp):
        assert np.all(comp == 0)

    assert br.shape == (nphi + 2, ns + 2, nr + 1)
    assert bs.shape == (nphi + 2, ns + 1, nr + 2)
    assert bp.shape == (nphi + 1, ns + 2, nr + 2)

    bg = out.bg
    assert np.all(bg == 0)
    assert bg.shape == (nphi + 1, ns + 1, nr + 1, 3)


def test_sunpy_map_input(zero_map):
    zero_in, _ = zero_map
    # Check that loading an input map works
    header = {'cunit1': 'degree', 'cunit2': 'degree'}
    map = sunpy.map.Map((zero_in.br, header))
    input = pfsspy.Input(map, zero_in.grid.nr, zero_in.grid.rss)
    assert (input.br == zero_in.br).all()


def test_input_output(dipole_map):
    # Smoke test of saving/loading files
    _, out = dipole_map
    out.save('test.npz')
    new_out = pfsspy.load_output('test.npz')
    assert (new_out.al[0] == out.al[0]).all()


def test_plot_input(dipole_map):
    # Smoke test of input plotting
    inp, out = dipole_map
    inp.plot_input()


def test_plot_source_surface(dipole_map):
    # Smoke test of source surface plotting
    inp, out = dipole_map
    out.plot_source_surface()


def test_plot_pil(dipole_map):
    # Smoke test of PIL plotting
    inp, out = dipole_map
    out.plot_pil()
