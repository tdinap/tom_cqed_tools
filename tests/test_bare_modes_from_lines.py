"""Round-trip checks for interactive_tools/bare_modes_from_lines_marimo.py:
choose bare frequencies, compute the lines they produce with the forward model,
invert, and require the original bare frequencies back.

Run with:  uv run pytest tests/test_bare_modes_from_lines.py -v
"""

import pytest
import scqubits as scq

from interactive_tools.bare_modes_from_lines_marimo import (
    bare_samples,
    g_from_chi,
    g_from_chi_with_error,
    two_level,
    invert_joint,
    invert_known_g,
    parse_peaks,
    spectrum,
    transmon_for,
    weight,
)

ALPHA = -0.120
W_T = 4.2253


def lines(s):
    return [s["E"][s["m1"][0]], s["E"][s["m1"][1]]]


def test_transmon_for_hits_target():
    EJ, EC = transmon_for(W_T, ALPHA)
    t = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=30, truncated_dim=3)
    assert t.E01() == pytest.approx(W_T, abs=1e-9)
    assert t.anharmonicity() == pytest.approx(ALPHA, abs=1e-9)


@pytest.mark.parametrize("g", [0.005, 0.018, 0.040])
@pytest.mark.parametrize("delta", [-0.030, -0.005, 0.005, 0.030])
def test_known_g_round_trip(g, delta):
    s = invert_known_g(lines(spectrum(W_T, W_T + delta, g, ALPHA)), g, ALPHA, alice_below=delta < 0)
    assert s["w_t"] == pytest.approx(W_T, abs=1e-9)
    assert s["w_a"] == pytest.approx(W_T + delta, abs=1e-9)


def test_known_g_other_ordering_is_the_mirror_image():
    """The two main lines alone can't tell which mode is on top: the other
    ordering also fits, with the bare frequencies swapped."""
    s = invert_known_g(lines(spectrum(W_T, W_T - 0.02, 0.018, ALPHA)), 0.018, ALPHA, alice_below=False)
    assert s["w_t"] == pytest.approx(W_T - 0.02, abs=2e-4)
    assert s["w_a"] == pytest.approx(W_T, abs=2e-4)


def test_joint_round_trip():
    g, wa, wb = 0.0177, 4.2068, 4.3178
    sa, sb = invert_joint(lines(spectrum(W_T, wa, g, ALPHA)), lines(spectrum(W_T, wb, g, ALPHA)), ALPHA)
    assert sa["w_t"] == pytest.approx(W_T, abs=1e-9)
    assert sa["g_eff"] == pytest.approx(g, abs=1e-9)
    assert sa["w_a"] == pytest.approx(wa, abs=1e-9)
    assert sb["w_a"] == pytest.approx(wb, abs=1e-9)


@pytest.mark.parametrize("wa,g", [(4.2614, 0.0197), (4.30, 0.035), (4.05, 0.020)])
def test_g_from_chi_round_trip(wa, g):
    s = spectrum(W_T, wa, g, ALPHA)
    E = s["E"]
    tl = max(s["m1"], key=lambda i: weight(s, i, 1, 0))
    cl = next(i for i in s["m1"] if i != tl)
    e1 = max(s["m2"], key=lambda j: weight(s, j, 1, 1))
    r = g_from_chi(E[tl], E[cl], E[e1] - E[cl] - E[tl], ALPHA)
    assert r["g_eff"] == pytest.approx(g, abs=1e-9)
    assert r["w_t"] == pytest.approx(W_T, abs=1e-9)
    assert r["w_a"] == pytest.approx(wa, abs=1e-9)


def test_g_from_chi_reproduces_cd16():
    """CD16 PNRQS: qubit 4.21158 GHz, Alice (Buffer 1) 4.26915 GHz, chi = -10.0 MHz."""
    assert g_from_chi(4.21158, 4.26915, -0.0100012, ALPHA)["g_eff"] == pytest.approx(0.01972, abs=1e-5)


def test_zero_errors_give_the_exact_answer():
    """With every input error at zero, each Monte-Carlo sample is the exact inversion."""
    for below in (True, False):
        nominal = (invert_known_g([4.196, 4.236], 0.0197, ALPHA, below),
                   invert_known_g([4.222, 4.321], 0.0197, ALPHA, False))
        s, kept = bare_samples([4.196, 4.236], [4.222, 4.321], nominal, 0.0197, 0.0, 0.0, below, n=5)
        assert kept == 1.0
        assert s["wt_a"] == pytest.approx(nominal[0]["w_t"], abs=1e-12)
        assert s["wa_b"] == pytest.approx(nominal[1]["w_a"], abs=1e-12)


@pytest.mark.parametrize("lines,g", [([4.1955, 4.2365], 0.0197), ([4.197, 4.236], 0.0190), ([4.195, 4.237], 0.0200)])
def test_sampling_shortcut_matches_exact_inversion(lines, g):
    """bare_samples uses the two-level relations plus the exact correction frozen at the
    nominal point. Across MHz-sized perturbations it must match the exact inversion to
    far better than the error bars (here 10 kHz)."""
    nominal = (invert_known_g([4.196, 4.236], 0.0197, ALPHA, True), None)
    t0, a0 = two_level(4.196, 4.236, 0.0197, True)
    t, a = two_level(*lines, g, True)
    exact = invert_known_g(lines, g, ALPHA, True)
    assert t + nominal[0]["w_t"] - t0 == pytest.approx(exact["w_t"], abs=1e-5)
    assert a + nominal[0]["w_a"] - a0 == pytest.approx(exact["w_a"], abs=1e-5)


def test_chi_error_matches_brute_force():
    """Central differences in g_from_chi_with_error agree with resampling chi directly."""
    _, s, parts = g_from_chi_with_error(4.21158, 4.26915, -0.0100012, 0.0001, ALPHA, 0.0)
    g = [g_from_chi(4.21158, 4.26915, c, ALPHA)["g_eff"] for c in (-0.0100012 + d * 0.0001 for d in (-1, 1))]
    assert s == pytest.approx(abs(g[1] - g[0]) / 2, rel=1e-6)
    assert parts["α"] == 0.0


def test_lines_closer_than_2g_are_rejected():
    assert invert_known_g([4.196, 4.236], 0.025, ALPHA, alice_below=True) is None


def test_inconsistent_cooldowns_are_rejected():
    assert invert_joint([4.200, 4.205], [4.100, 4.400], ALPHA) is None


def test_parse_peaks_accepts_any_separator_and_mhz():
    assert parse_peaks("4.127, 4.203;4.257\n4093") == pytest.approx([4.127, 4.203, 4.257, 4.093])
