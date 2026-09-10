"""Verification for the strong-coupling transmon-Alice model.

Two independent kinds of check:

1. Convergence -- vary the basis truncations and confirm the reported physics
   does not move. Catches an under-converged basis.
2. Analytic limits -- compare against results we know in closed form (vacuum
   Rabi splitting at resonance, the Koch dispersive-chi formula at large
   detuning). Catches a *wrong Hamiltonian*, which a convergence sweep never
   would.

Run with:  uv run pytest tests/test_strong_coupling_convergence.py -v
"""

import numpy as np
import pytest
import scqubits as scq

EJ, EC = 20.55, 0.114
G = 0.025


def solve(Eosc=4.327, g=G, ncut=30, nt=6, nc=6):
    """Diagonalize the transmon-cavity system; label states by excitation manifold."""
    tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=ncut, truncated_dim=nt)
    cav = scq.Oscillator(E_osc=Eosc, truncated_dim=nc)
    h = scq.HilbertSpace([tmon, cav])
    apad = cav.annihilation_operator() + cav.creation_operator()
    h.add_interaction(g_strength=g, op1=tmon.n_operator, op2=(apad, cav))

    evals, evecs = h.eigensys(evals_count=h.dimension)
    evals = np.array(evals)
    V = np.array([evecs[i].full().flatten() for i in range(len(evals))])

    N_t = np.kron(np.diag(np.arange(nt)), np.eye(nc))
    N_c = np.kron(np.eye(nt), np.diag(np.arange(nc)))
    exp = lambda op: np.real(np.einsum("ij,jk,ik->i", V.conj(), op, V))
    nt_e, nc_e = exp(N_t), exp(N_c)
    N = nt_e + nc_e

    def manifold(n):
        idx = np.where(np.abs(N - n) < 0.15)[0]
        return idx[np.argsort(evals[idx])]

    m1 = manifold(1)
    lp, up = int(m1[0]), int(m1[1])
    tl, cl = (lp, up) if nt_e[lp] > nt_e[up] else (up, lp)

    m2 = manifold(2)
    f = int(m2[np.argmax(nt_e[m2])])
    c2 = int(m2[np.argmax(nc_e[m2])])
    mixed = [int(i) for i in m2 if i not in (f, c2)]

    n01 = abs(tmon.matrixelement_table("n_operator")[0, 1])
    return dict(
        w_t=evals[tl] - evals[0],
        w_c=evals[cl] - evals[0],
        alpha=(evals[f] - 2 * evals[tl] + evals[0]),
        K=(evals[c2] - 2 * evals[cl] + evals[0]),
        chi=(evals[mixed[0]] - evals[tl] - evals[cl] + evals[0]) if mixed else np.nan,
        splitting=evals[up] - evals[lp],
        p_tmon=nt_e[tl],
        N_err=np.abs(N - np.round(N))[:12].max(),
        n01=n01,
        g_eff=g * n01,
        bare_w_t=tmon.E01(),
        bare_alpha=tmon.anharmonicity(),
    )


REF = solve()


@pytest.mark.parametrize(
    "ncut,nt,nc",
    [(10, 6, 6), (20, 6, 6), (40, 6, 6), (30, 8, 8), (30, 10, 10), (40, 14, 14)],
)
@pytest.mark.parametrize("key", ["w_t", "w_c", "alpha", "K", "chi"])
def test_converged_in_truncation(ncut, nt, nc, key):
    """Physics must not move as the basis grows. Tolerance 1 kHz = 1e-6 GHz."""
    assert abs(solve(ncut=ncut, nt=nt, nc=nc)[key] - REF[key]) < 1e-6


# Best basis we can afford, at the very top of the g slider. Everything smaller
# is compared against this.
REF_MAX_G = solve(g=0.2, ncut=40, nt=14, nc=14)


@pytest.mark.parametrize("nt,nc", [(6, 6), (8, 8), (10, 10), (12, 12)])
@pytest.mark.parametrize("key", ["w_t", "w_c", "alpha", "K", "chi"])
def test_converged_at_max_coupling(nt, nc, key):
    """Convergence is not a single fact -- it depends on how hard you drive the
    system. The margin shrinks by ~1000x from the default coupling to the top of
    the g slider (0.8 Hz -> 941 Hz of truncation error in alpha), so verifying it
    only at g=25 MHz proves nothing about g=200 MHz. Tolerance 2 kHz.
    """
    assert abs(solve(g=0.2, nt=nt, nc=nc)[key] - REF_MAX_G[key]) < 2e-6


def test_excitation_number_is_conserved():
    """Manifold labeling is only valid while N is a good quantum number."""
    assert REF["N_err"] < 1e-3


def test_not_ultrastrong():
    """Guards the RWA-adjacent assumptions; USC begins around g/w ~ 0.1."""
    assert REF["g_eff"] / REF["w_c"] < 0.01


def test_vacuum_rabi_splitting_equals_2g_eff():
    """At resonance the polariton splitting must be exactly 2*g*n01."""
    r = solve(Eosc=solve()["bare_w_t"])
    assert abs(r["splitting"] - 2 * r["g_eff"]) < 5e-4  # 0.5 MHz


def test_modes_are_balanced_at_resonance():
    """On resonance each polariton is half transmon -- this is why the
    'transmon-like'/'Alice-like' labels must be suppressed there."""
    r = solve(Eosc=solve()["bare_w_t"])
    assert abs(r["p_tmon"] - 0.5) < 0.02


def chi_perturbative(Eosc, g=G, nlev=6):
    """Cross-Kerr from second-order perturbation theory in H_int = g*n*(a+adag).

        chi = g^2 (S_1 - S_0),   S_j = sum_k |n_kj|^2 [1/(w_c - d_kj) - 1/(w_c + d_kj)]

    with d_kj = E_k - E_j. This is the energy combination
    E(e,1) - E(e,0) - E(g,1) + E(g,0), which equals 2*chi in the usual
    chi*sigma_z*adag*a convention.

    Do NOT compare against the bare two-level Koch result
    g^2*alpha/(Delta*(Delta+alpha)) (Koch et al., PRA 76, 042319 (2007)):
      * its Delta is w_q - w_c, the opposite sign from this module's Delta, so
        the denominator becomes Delta*(Delta - alpha) here;
      * it is missing the factor 2 above;
      * it truncates at the 0-1-2 levels, so it omits the n_23, n_34, ...
        contributions that the exact diagonalization includes. That omission
        grows with detuning, so no constant prefactor can reconcile the two.
    The full sum below has none of those limitations.
    """
    t = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=30, truncated_dim=nlev)
    E = t.eigenvals(evals_count=nlev)
    # evals_count is required: this defaults to 6 levels regardless of
    # truncated_dim, so without it the sum below IndexErrors for nlev > 6.
    n = t.matrixelement_table("n_operator", evals_count=nlev)

    def S(j):
        return sum(
            abs(n[k, j]) ** 2
            * (1.0 / (Eosc - (E[k] - E[j])) - 1.0 / (Eosc + (E[k] - E[j])))
            for k in range(nlev)
            if k != j
        )

    return g**2 * (S(1) - S(0))


@pytest.mark.parametrize("delta", [0.8, 1.2, 2.0, 3.0, 4.0])
def test_chi_matches_second_order_perturbation_theory(delta):
    """Exact diagonalization must reproduce 2nd-order PT in the dispersive
    regime. This tests the interaction Hamiltonian itself -- a convergence
    sweep would never catch a wrong H_int, since a wrong Hamiltonian converges
    perfectly well to the wrong answer."""
    Eosc = REF["bare_w_t"] + delta
    assert solve(Eosc=Eosc)["chi"] == pytest.approx(chi_perturbative(Eosc), rel=0.01)


@pytest.mark.parametrize("g,tol", [(0.002, 1e-4), (0.005, 5e-4), (0.010, 1e-3)])
def test_chi_approaches_perturbation_theory_as_coupling_vanishes(g, tol):
    """The residual is a 4th-order effect, so it must shrink like g^2. Observed:
    ratio 0.9856 at g=50 MHz, 0.9963 at 25, 0.9994 at 10, 1.0000 at 2."""
    Eosc = REF["bare_w_t"] + 1.2
    ratio = solve(Eosc=Eosc, g=g)["chi"] / chi_perturbative(Eosc, g=g)
    assert abs(ratio - 1.0) < tol
