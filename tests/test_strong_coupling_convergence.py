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


@pytest.mark.xfail(
    reason="UNRESOLVED -- do not trust chi quantitatively until this is closed. "
    "The numerical cross-Kerr does not approach the leading-order Koch result; "
    "the ratio chi_num/chi_koch GROWS with detuning (1.73 at 0.8 GHz, 2.06 at "
    "1.2, 2.52 at 2.0, 3.32 at 4.0) instead of settling on a constant. A pure "
    "convention mismatch would give a fixed factor (2 is expected, since the "
    "energy combination E(e1)-E(e0)-E(g1)+E(g0) equals 2*chi in the chi*sigma_z*adag_a "
    "convention). A drifting ratio means something else: most likely the Koch "
    "formula omits the cavity's coupling to the higher transmon transitions "
    "(n_12, n_23, ...), which this exact diagonalization includes -- or the N=2 "
    "'mixed' state selection drifts at large detuning. Resolve by (a) summing "
    "the full second-order perturbation series over transmon levels, and "
    "(b) printing the bare-state content of the N=2 manifold at each detuning.",
    strict=True,
)
def test_chi_matches_koch_dispersive_formula():
    """Far off resonance, chi -> g_eff^2 * alpha / (Delta * (Delta + alpha)),
    Koch et al. PRA 76, 042319 (2007). Agreement here would mean the interaction
    Hamiltonian itself is right, not merely converged."""
    r0 = solve()
    Eosc = r0["bare_w_t"] + 1.2  # 1.2 GHz detuned: firmly dispersive
    r = solve(Eosc=Eosc)
    delta = Eosc - r["bare_w_t"]
    predicted = r["g_eff"] ** 2 * r["bare_alpha"] / (delta * (delta + r["bare_alpha"]))
    assert r["chi"] == pytest.approx(predicted, rel=0.15)
