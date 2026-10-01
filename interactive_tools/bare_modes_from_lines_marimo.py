import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    import re
    import warnings

    import marimo as mo
    import numpy as np
    import scqubits as scq
    from scipy.optimize import fsolve

    NL, NCUT = 6, 30
    GLOSS = {(2, 0): "transmon doubly excited", (1, 1): "one in each", (0, 2): "two photons in Alice"}


@app.cell
def _():
    mo.md(r"""
    # Where were the bare modes?

    Give it the two lines you measured in each cooldown. It works out where the bare
    transmon and bare Alice actually were, how strongly they're mixed, and every line a
    two-tone sweep should show. Paste your measured two-tone peaks and it tells you which
    picture they support.

    States are written **|transmon, Alice⟩**: |10⟩ = transmon excited, |01⟩ = one photon in Alice.
    """)
    return


@app.cell
def _(chi_inputs):
    chi_inputs
    return


@app.cell
def _(chi_view):
    chi_view
    return


@app.cell
def _(inputs):
    inputs
    return


@app.cell
def _(peakfit_view):
    peakfit_view
    return


@app.cell
def _(result_view):
    result_view
    return


@app.cell
def _():
    mo.md(r"""
    ---
    ## Implementation

    Everything below is machinery. The physics is the same exact diagonalization as
    `strong_coupling_scq_marimo.py`, run backwards: the two lowest excited states of the
    coupled system are matched to the measured lines, and the bare frequencies that
    produce them are solved for.

    **g known:** each cooldown is solved on its own. The two measured lines fix
    how far apart the bare modes were, but not which one was on top. Both orderings are
    kept, and the two-tone lines tell them apart.

    **g unknown:** the two cooldowns are solved together, assuming the bare transmon
    and the coupling didn't change between them. That fixes everything, including which
    mode was on top.
    """)
    return


@app.cell
def _():
    a1 = mo.ui.number(3.0, 7.0, step=0.0001, value=4.196, label="line 1", debounce=True)
    a2 = mo.ui.number(3.0, 7.0, step=0.0001, value=4.236, label="line 2", debounce=True)
    b1 = mo.ui.number(3.0, 7.0, step=0.0001, value=4.222, label="line 1", debounce=True)
    b2 = mo.ui.number(3.0, 7.0, step=0.0001, value=4.321, label="line 2", debounce=True)
    alpha_ui = mo.ui.number(-400.0, -20.0, step=0.1, value=-106.0, debounce=True,
                            label="transmon anharmonicity α (MHz) — default from Andre's pre-etch fit")
    alpha_err_ui = mo.ui.number(0.0, 100.0, step=0.5, value=5.0, debounce=True, label="±")
    g_ui = mo.ui.number(0.0, 200.0, step=0.01, value=20.08, debounce=True,
                        label="coupling g_eff (MHz) — 0 if unknown")
    g_err_ui = mo.ui.number(0.0, 50.0, step=0.01, value=0.17, debounce=True, label="±")
    line_err_ui = mo.ui.number(0.0, 20.0, step=0.05, value=0.5, debounce=True,
                               label="± on each line (MHz) — your fit error, or half the last digit you wrote down")
    q_chi = mo.ui.number(3.0, 7.0, step=0.00001, value=4.21158, label="qubit line", debounce=True)
    a_chi = mo.ui.number(3.0, 7.0, step=0.00001, value=4.26915, label="Alice line", debounce=True)
    chi_ui = mo.ui.number(-200.0, 200.0, step=0.01, value=-10.0, debounce=True,
                          label="χ (MHz, negative if the qubit moves down with a photon)")
    chi_err_ui = mo.ui.number(0.0, 20.0, step=0.01, value=0.1, debounce=True, label="±")
    chi_inputs = mo.vstack([
        mo.md("**0 · Optional: get g_eff from a cooldown where the modes were well apart** "
              "(qubit line and Alice line in GHz, χ from PNRQS; uses α from step 2)"),
        mo.hstack([q_chi, a_chi, chi_ui, chi_err_ui], justify="start", gap=1.5),
    ])
    peaks_ui = mo.ui.text_area(placeholder="e.g. 4.127, 4.203, 4.257", rows=2, full_width=True,
                               label="two-tone peaks from the no-rod cooldown (GHz), any separator")

    inputs = mo.vstack([
        mo.md("**1 · The two main lines you measured (GHz, any order)**"),
        mo.hstack([mo.md("No-rod cooldown"), a1, a2], widths=[1, 2, 2], gap=1.5),
        mo.hstack([mo.md("Rod cooldown"), b1, b2], widths=[1, 2, 2], gap=1.5),
        line_err_ui,
        mo.md("**2 · What you know**"),
        mo.hstack([alpha_ui, alpha_err_ui, g_ui, g_err_ui], justify="start", gap=1.5),
        mo.md("**3 · Optional: check against two-tone data**"),
        peaks_ui,
    ])
    return (a1, a2, a_chi, alpha_err_ui, alpha_ui, b1, b2, chi_err_ui, chi_inputs, chi_ui,
            g_err_ui, g_ui, inputs, line_err_ui, peaks_ui, q_chi)


@app.cell
def _(a_chi, alpha_err_ui, alpha_ui, chi_err_ui, chi_ui, q_chi):
    sc, sg_chi, parts = g_from_chi_with_error(q_chi.value, a_chi.value, chi_ui.value / 1e3,
                                              chi_err_ui.value / 1e3, alpha_ui.value / 1e3,
                                              alpha_err_ui.value / 1e3)
    if sc is None:
        chi_view = mo.callout(mo.md(
            "**No coupling reproduces these three numbers.** Check the sign of χ: it's negative "
            "if the qubit line moves down when Alice holds a photon."), kind="danger")
    else:
        _d = (sc["w_a"] - sc["w_t"]) * 1e3
        _tl = max(sc["m1"], key=lambda i: weight(sc, i, 1, 0))
        _g, _s = sc["g_eff"] * 1e3, sg_chi * 1e3
        chi_view = mo.callout(mo.md(
            f"**g_eff = {_g:.2f} ± {_s:.2f} MHz** ("
            + ", ".join(f"±{v * 1e3:.2f} from {k}" for k, v in parts.items()) + ") · "
            f"bare transmon {sc['w_t']:.4f} GHz, bare Alice {sc['w_a']:.4f} GHz "
            f"(Alice {abs(_d):.1f} MHz {'above' if _d > 0 else 'below'}) · qubit line is "
            f"{100 * weight(sc, _tl, 1, 0):.0f}% transmon.\n\n"
            f"To use it, put **{_g:.2f}** and **± {_s:.2f}** in step 2."), kind="success")
    return (chi_view,)


@app.cell
def _(a1, a2, alpha_err_ui, alpha_ui, line_err_ui, peaks_ui):
    # g for the no-rod cooldown from its own two-tone peaks: needs no g from any other
    # cooldown. Independent of step 2, so it works even when the g there is wrong.
    _peaks = parse_peaks(peaks_ui.value)
    _lines = sorted([a1.value, a2.value])
    _extra = [p for p in _peaks if min(abs(p - x) for x in _lines) > 3 * line_err_ui.value / 1e3]
    if len(_extra) < 2:
        peakfit_view = mo.md(
            "*Paste at least two two-tone peaks besides the main lines and the tool will also fit "
            "g_eff for the no-rod cooldown from them.*" if _peaks else "")
    else:
        _fit = fit_g_from_peaks(_lines, _peaks, alpha_ui.value / 1e3, alpha_err_ui.value / 1e3,
                                line_err_ui.value / 1e3)
        _res = {n: best_g(*v) for n, v in _fit.items()}
        _best = min(_res, key=lambda n: _res[n][3])
        _other = next(n for n in _res if n != _best)
        _g0, _glo, _ghi, _c2 = _res[_best]
        _limit = (_lines[1] - _lines[0]) / 2
        _msg = (f"**From your two-tone peaks alone: g_eff = {_g0 * 1e3:.1f} MHz "
                f"(68%: {_glo * 1e3:.1f} to {_ghi * 1e3:.1f}), {_best.lower()}.**")
        if _ghi > 0.99 * _limit:
            _msg += f" The range runs up to the resonance limit ({_limit * 1e3:.1f} MHz)."
        _dc2 = _res[_other][3] - _c2
        _msg += (f" The other ordering fits clearly worse (Δχ² = {_dc2:.0f})." if _dc2 > 4 else
                 f" The other ordering fits almost as well (Δχ² = {_dc2:.1f}), so which mode was on top stays open.")
        _bad = _c2 / max(len(_peaks) - 2, 1) > 4
        if _bad:
            _msg += (" ⚠️ Even the best fit misses your peaks by more than your error bars allow — "
                     "check α and whether every peak belongs to this cooldown.")
        _msg += f"\n\nTo use it, put **{_g0 * 1e3:.1f}** in the g_eff box in step 2."
        peakfit_view = mo.callout(mo.md(_msg), kind="warn" if _bad else "success")
    return (peakfit_view,)


@app.function
def solve(r, x0):
    """fsolve, silenced: success is judged by the caller from the residual, because
    fsolve's own status flag reports false failures when it lands on the answer to
    machine precision."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fsolve(r, x0, xtol=1e-13)


@app.function
def transmon_for(w_t, alpha):
    """(EJ, EC) giving a bare transmon with E01 = w_t and anharmonicity alpha (GHz)."""
    def r(x):
        e = scq.Transmon(EJ=x[0], EC=x[1], ng=0.0, ncut=NCUT, truncated_dim=3).eigenvals(evals_count=3)
        return np.array([e[1] - e[0] - w_t, e[2] - 2 * e[1] + e[0] - alpha])

    # Newton with the Jacobian of E01 ~ sqrt(8 EJ EC) - EC, alpha ~ -EC: a few diagonalizations
    # instead of fsolve's dozens. Falls back to fsolve if EJ/EC is too small for that to hold.
    ec = -alpha
    x = np.array([(w_t + ec) ** 2 / (8 * ec), ec])
    try:
        for _ in range(30):
            res = r(x)
            if abs(res).max() < 1e-12:
                return x
            s = np.sqrt(8 * x[0] * x[1])
            x = x - np.linalg.solve([[4 * x[1] / s, 4 * x[0] / s - 1], [0.0, -1.0]], res)
            if not (np.all(np.isfinite(x)) and np.all(x > 0)):
                break
    except Exception:
        pass
    return solve(r, [(w_t + ec) ** 2 / (8 * ec), ec])


@app.function
def spectrum(w_t, w_a, g_eff, alpha):
    """Exact spectrum of transmon + Alice, energies relative to ground (GHz).
    g_eff is the physical coupling, g * |<0|n|1>|."""
    EJ, EC = transmon_for(w_t, alpha)
    t = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=NCUT, truncated_dim=NL)
    c = scq.Oscillator(E_osc=w_a, truncated_dim=NL)
    nb = t.matrixelement_table("n_operator", evals_count=NL)
    h = scq.HilbertSpace([t, c])
    ap = c.annihilation_operator() + c.creation_operator()
    h.add_interaction(g_strength=g_eff / abs(nb[0, 1]), op1=t.n_operator, op2=(ap, c))
    ev, vec = h.eigensys(evals_count=h.dimension)
    E = np.array(ev) - ev[0]
    V = np.array([v.full().flatten() for v in vec])
    nt = np.real(np.einsum("ij,jk,ik->i", V.conj(), np.kron(np.diag(np.arange(NL)), np.eye(NL)), V))
    nc = np.real(np.einsum("ij,jk,ik->i", V.conj(), np.kron(np.eye(NL), np.diag(np.arange(NL))), V))
    N = nt + nc

    def manifold(k):
        i = np.where(np.abs(N - k) < 0.15)[0]
        return [int(j) for j in i[np.argsort(E[i])]]

    return dict(E=E, V=V, n=V @ np.kron(nb, np.eye(NL)) @ V.conj().T,
                m1=manifold(1), m2=manifold(2), w_t=float(w_t), w_a=float(w_a),
                g_eff=float(g_eff), alpha=alpha)


@app.function
def weight(s, i, tmon, alice):
    """Probability that dressed state i is the bare state |tmon, alice>."""
    return float(abs(s["V"][i, tmon * NL + alice]) ** 2)


@app.function
def invert_known_g(lines, g_eff, alpha, alice_below):
    """Bare frequencies reproducing two measured lines at a given g_eff, with Alice
    below or above the transmon. None if impossible (lines closer than 2*g_eff)."""
    lo, up = sorted(lines)
    if up - lo < 2 * g_eff:
        return None
    delta = np.sqrt((up - lo) ** 2 - 4 * g_eff ** 2) * (-1 if alice_below else 1)

    def r(x):
        s = spectrum(x[0], x[1], g_eff, alpha)
        return [s["E"][s["m1"][0]] - lo, s["E"][s["m1"][1]] - up]

    mid = (lo + up) / 2
    x = solve(r, [mid - delta / 2, mid + delta / 2])
    if max(abs(np.array(r(x)))) > 1e-7 or (x[1] < x[0]) != alice_below:
        return None
    return spectrum(x[0], x[1], g_eff, alpha)


@app.function
def invert_joint(lines_a, lines_b, alpha):
    """Unknown g: same bare transmon and same g_eff in both cooldowns, only Alice moves.
    Returns (spectrum_a, spectrum_b), or None if no such solution exists."""
    (la, ua), (lb, ub) = sorted(lines_a), sorted(lines_b)
    if abs((lb + ub) - (la + ua)) < 1e-6:
        return None
    wt, wa0, wb0, g0 = two_level_joint(la, ua, lb, ub)   # starting guess
    if not np.isfinite(g0):
        return None

    def r(x):
        sa, sb = spectrum(x[0], x[1], x[3], alpha), spectrum(x[0], x[2], x[3], alpha)
        return [sa["E"][sa["m1"][0]] - la, sa["E"][sa["m1"][1]] - ua,
                sb["E"][sb["m1"][0]] - lb, sb["E"][sb["m1"][1]] - ub]

    x = solve(r, [wt, wa0, wb0, g0])
    if max(abs(np.array(r(x)))) > 1e-7 or x[3] <= 0:
        return None
    return spectrum(x[0], x[1], x[3], alpha), spectrum(x[0], x[2], x[3], alpha)


@app.function
def g_from_chi(f_qubit, f_alice, chi, alpha):
    """Bare transmon, bare Alice and g_eff from one cooldown: the qubit line, the Alice
    line, and chi = how far the qubit line moves per Alice photon (PNRQS). None if no
    coupling reproduces all three."""
    d = f_qubit - f_alice
    g2 = abs(chi * d * (d + alpha) / (2 * alpha))   # dispersive formula, as a starting guess

    def r(x):
        s = spectrum(x[0], x[1], x[2], alpha)
        E = s["E"]
        tl = max(s["m1"], key=lambda i: weight(s, i, 1, 0))
        cl = next(i for i in s["m1"] if i != tl)
        e1 = max(s["m2"], key=lambda j: weight(s, j, 1, 1))
        return [E[tl] - f_qubit, E[cl] - f_alice, E[e1] - E[cl] - E[tl] - chi]

    x = solve(r, [f_qubit, f_alice, np.sqrt(g2)])
    if max(abs(np.array(r(x)))) > 1e-7 or x[2] <= 0:
        return None
    return spectrum(x[0], x[1], x[2], alpha)


@app.function
def g_from_chi_with_error(f_qubit, f_alice, chi, s_chi, alpha, s_alpha):
    """g_from_chi plus its 1-sigma error, split by source. Central differences on the
    exact model: fine here because the modes are well apart, so g varies smoothly."""
    s0 = g_from_chi(f_qubit, f_alice, chi, alpha)
    if s0 is None:
        return None, None, {}

    def g(c, a):
        s = g_from_chi(f_qubit, f_alice, c, a)
        return np.nan if s is None else s["g_eff"]

    parts = {"χ": abs(g(chi + s_chi, alpha) - g(chi - s_chi, alpha)) / 2 if s_chi else 0.0,
             "α": abs(g(chi, alpha + s_alpha) - g(chi, alpha - s_alpha)) / 2 if s_alpha else 0.0}
    return s0, float(np.sqrt(sum(v ** 2 for v in parts.values()))), parts


@app.function
def two_level(lo, up, g, below):
    """Bare (transmon, Alice) from two lines at coupling g in the two-level picture:
    lo + up = w_t + w_a and (up - lo)^2 = Delta^2 + 4 g^2. Vectorised; NaN where the
    lines are closer than 2g."""
    d2 = (up - lo) ** 2 - 4 * g ** 2
    d = np.sqrt(np.where(d2 >= 0, d2, np.nan)) * (-1 if below else 1)
    return (lo + up) / 2 - d / 2, (lo + up) / 2 + d / 2


@app.function
def two_level_joint(la, ua, lb, ub):
    """Shared transmon and g from two cooldowns in the two-level picture (vectorised).
    Returns (w_t, w_a in cooldown a, w_a in cooldown b, g); g is NaN if no solution."""
    D = (lb + ub) - (la + ua)
    P = ((ub - lb) ** 2 - (ua - la) ** 2) / D
    da, db = (P - D) / 2, (P + D) / 2
    g2 = (ua - la) ** 2 - da ** 2
    wt = (la + ua - da) / 2
    return wt, wt + da, wt + db, np.sqrt(np.where(g2 > 0, g2, np.nan)) / 2


@app.function
def bare_samples(lines_a, lines_b, nominal, g=None, s_g=0.0, s_line=0.0, below=True, n=4000, seed=0):
    """Monte-Carlo samples of the bare frequencies (GHz), with Gaussian errors s_line on
    each measured line and s_g on g (g=None: the joint fit, which infers g).

    Each sample uses the two-level relations plus the exact-model correction taken at
    the nominal point; that correction is constant to a few kHz over MHz-sized error
    bars (tests check this). Samples whose lines sit closer than 2g are physically
    impossible and are dropped. Returns (samples, fraction kept)."""
    rng = np.random.default_rng(seed)
    L = np.array([*sorted(lines_a), *sorted(lines_b)]) + rng.normal(0.0, s_line, (n, 4))
    la, ua = np.minimum(L[:, 0], L[:, 1]), np.maximum(L[:, 0], L[:, 1])
    lb, ub = np.minimum(L[:, 2], L[:, 3]), np.maximum(L[:, 2], L[:, 3])
    sa, sb = nominal
    if g is None:
        t0, a0, b0, g0 = two_level_joint(*sorted(lines_a), *sorted(lines_b))
        wt, wa, wb, gs = two_level_joint(la, ua, lb, ub)
        out = dict(wt_a=wt + sa["w_t"] - t0, wa_a=wa + sa["w_a"] - a0, g=gs + sa["g_eff"] - g0,
                   wt_b=wt + sb["w_t"] - t0, wa_b=wb + sb["w_a"] - b0)
    else:
        # The exact correction is taken at the nominal solution's own g, which is g itself
        # except right at the resonance limit, where the nominal is solved just inside it.
        gs = rng.normal(g, s_g, n)
        t0, a0 = two_level(*sorted(lines_a), sa["g_eff"], below)
        wt, wa = two_level(la, ua, gs, below)
        out = dict(wt_a=wt + sa["w_t"] - t0, wa_a=wa + sa["w_a"] - a0, g=gs)
        if sb is not None:
            tb0, ab0 = two_level(*sorted(lines_b), sb["g_eff"], False)
            tb, ab = two_level(lb, ub, gs, False)
            out |= dict(wt_b=tb + sb["w_t"] - tb0, wa_b=ab + sb["w_a"] - ab0)
    keep = np.all([np.isfinite(v) for v in out.values()], axis=0)
    return {k: v[keep] for k, v in out.items()}, float(keep.mean())


@app.function
def line_errors(samples, alpha, s_alpha, k=150, seed=1):
    """1-sigma spread (GHz) of every predicted no-rod line, keyed like predicted_lines,
    from k exact spectra drawn from the bare-frequency samples and from alpha +- s_alpha."""
    if len(samples["g"]) == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples["g"]), size=min(k, len(samples["g"])), replace=False)
    f = {}
    for i, a in zip(idx, rng.normal(alpha, s_alpha, len(idx))):
        for x in predicted_lines(spectrum(samples["wt_a"][i], samples["wa_a"][i], samples["g"][i], a)):
            f.setdefault(x["key"], []).append(x["f"])
    return {key: float(np.std(v)) for key, v in f.items()}


@app.function
def fit_g_from_peaks(lines, peaks, alpha, s_alpha, s_peak, n=80):
    """Scan g_eff for one cooldown, both orderings, scoring each by how well the
    predicted lines match the measured two-tone peaks:
        chi2(g) = sum over peaks of ((peak - nearest predicted line) / sigma)^2,
    sigma combining the peak's error with that line's shift for alpha +- s_alpha.
    Bare frequencies come from the two-level relation, exact here to < 50 kHz, far
    below MHz-level peak errors. Returns {ordering: (g grid, chi2)}."""
    lo, up = sorted(lines)
    gs = np.linspace(0.001, (up - lo) / 2 * 0.998, n)
    out = {}
    for name, below in (("Alice below the transmon", True), ("Alice above the transmon", False)):
        c2 = []
        for g in gs:
            wt, wa = two_level(lo, up, g, below)
            f0 = predicted_lines(spectrum(wt, wa, g, alpha))
            if s_alpha:
                fp = {x["key"]: x["f"] for x in predicted_lines(spectrum(wt, wa, g, alpha + s_alpha))}
                fm = {x["key"]: x["f"] for x in predicted_lines(spectrum(wt, wa, g, alpha - s_alpha))}
            total = 0.0
            for p in peaks:
                x = min(f0, key=lambda x: abs(x["f"] - p))
                s_a = abs(fp[x["key"]] - fm[x["key"]]) / 2 if s_alpha else 0.0
                total += ((p - x["f"]) / max(np.hypot(s_a, s_peak), 1e-6)) ** 2
            c2.append(total)
        out[name] = (gs, np.array(c2))
    return out


@app.function
def best_g(gs, c2):
    """Best g on the grid, its 68% range (chi2 within 1 of the minimum), and the minimum."""
    i = int(np.argmin(c2))
    ok = gs[c2 <= c2[i] + 1]
    return gs[i], ok.min(), ok.max(), float(c2[i])


@app.function
def err(center, samples, digits=1):
    """68% error on `center` from samples (same units): '± e', '+hi / −lo' when lopsided,
    or the 68% range itself when the centre falls outside it."""
    if len(samples) == 0:
        return "(no valid samples)"
    lo, hi = np.percentile(samples, [15.87, 84.13])
    dlo, dhi = center - lo, hi - center
    if dlo < 0 or dhi < 0:
        return f"(68%: {lo:.{digits}f} to {hi:.{digits}f})"
    if abs(dhi - dlo) <= 0.25 * max(dhi, dlo, 1e-12):
        return f"± {(dhi + dlo) / 2:.{digits}f}"
    return f"+{dhi:.{digits}f} / −{dlo:.{digits}f}"


@app.function
def predicted_lines(s):
    """Every line a qubit-drive two-tone sweep can show, with a plain description."""
    E, n = s["E"], s["n"]

    def mode(i):
        return f"{E[i]:.4f} mode ({100 * weight(s, i, 1, 0):.0f}% transmon)"

    def ends_in(j):
        parts = sorted(((weight(s, j, *k), k) for k in GLOSS), reverse=True)
        top = parts[0][1]
        mix = " + ".join(f"{100 * w:.0f}% |{a}{b}⟩" for w, (a, b) in parts if w > 0.03)
        return f"{mix}  — mostly {GLOSS[top]}"

    # key = which line this is by energy rank, so the same line can be followed across
    # the Monte-Carlo samples even when its frequency moves.
    out = [dict(f=E[i], what=f"main line → {mode(i)}", short=mode(i), group="ground", key=("g", a),
                title="Plain two-tone (no π pulse)", strength=abs(n[i, 0])) for a, i in enumerate(s["m1"])]
    for a, i in enumerate(s["m1"]):
        for b, j in enumerate(s["m2"]):
            out.append(dict(f=E[j] - E[i], what=f"from the {mode(i)} → {ends_in(j)}", short=ends_in(j),
                            group=i, key=("x", a, b), title=f"After a π pulse on the {E[i]:.4f} line",
                            strength=abs(n[j, i])))
    for b, j in enumerate(s["m2"]):
        out.append(dict(f=E[j] / 2, what=f"two-photon from ground → {ends_in(j)} (high power only)",
                        short=ends_in(j), group="2ph", key=("2", b),
                        title="Two-photon lines (high drive power only)", strength=None))
    for g in {x["group"] for x in out if x["strength"] is not None}:
        top = max(x["strength"] for x in out if x["group"] == g)
        for x in out:
            if x["group"] == g:
                x["strength"] /= top
    return sorted(out, key=lambda x: x["f"])


@app.function
def strength_word(v):
    if v is None:
        return "—"
    return f"{'strong' if v >= 0.5 else 'medium' if v >= 0.2 else 'weak' if v >= 0.05 else 'very weak'} ({v:.2f})"


@app.function
def esc(text):
    """Kets contain '|', which would split a markdown table cell."""
    return text.replace("|", "\\|")


@app.function
def parse_peaks(text):
    """Floats from free text. Values above 100 are taken to be MHz."""
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return [v / 1e3 if v > 100 else v for v in vals]


@app.function
def match(peaks, lines, line_sig, s_peak):
    """Nearest predicted line to each peak. Returns rows (peak, line, sigma), the rms
    offset in MHz, and the rms offset in units of sigma, where sigma combines the
    prediction's spread with the peak's own error."""
    rows = []
    for p in peaks:
        x = min(lines, key=lambda x: abs(x["f"] - p))
        rows.append((p, x, float(np.hypot(line_sig.get(x["key"], 0.0), s_peak))))
    if not rows:
        return rows, None, None
    rms = float(np.sqrt(np.mean([(p - x["f"]) ** 2 for p, x, _ in rows]))) * 1e3
    z = float(np.sqrt(np.mean([((p - x["f"]) / s) ** 2 if s > 0 else np.inf for p, x, s in rows])))
    return rows, rms, z


@app.cell
def _(a1, a2, alpha_ui, b1, b2, g_err_ui, g_ui, line_err_ui):
    alpha = alpha_ui.value / 1e3
    g_known = g_ui.value / 1e3
    lines_a, lines_b = sorted([a1.value, a2.value]), sorted([b1.value, b2.value])
    at_limit = False

    if g_known > 0:
        scenarios = {}
        for name, below in (("Alice below the transmon", True), ("Alice above the transmon", False)):
            sa = invert_known_g(lines_a, g_known, alpha, alice_below=below)
            if sa is not None:
                scenarios[name] = (sa, invert_known_g(lines_b, g_known, alpha, alice_below=False))
        # Lines a hair closer than 2g, but within 2 sigma of it: that's "on resonance",
        # not "impossible". Solve just inside the limit (bare modes 2 MHz apart) as the
        # nominal point and let the Monte Carlo say how close they really were.
        split = lines_a[1] - lines_a[0]
        slack = 2 * np.hypot(2 * g_err_ui.value, np.sqrt(2) * line_err_ui.value) / 1e3
        if not scenarios and 2 * g_known - split <= slack:
            at_limit = True
            sa = invert_known_g(lines_a, np.sqrt(split ** 2 - 0.002 ** 2) / 2, alpha, alice_below=True)
            if sa is not None:
                scenarios["on resonance (within your error bars)"] = (
                    sa, invert_known_g(lines_b, g_known, alpha, alice_below=False))
    else:
        joint = invert_joint(lines_a, lines_b, alpha)
        scenarios = {} if joint is None else {"same transmon & coupling in both cooldowns": joint}
    return alpha, at_limit, g_known, lines_a, lines_b, scenarios


@app.function
def values(sa, sb):
    """The nominal bare frequencies under the same keys bare_samples uses."""
    v = dict(wt_a=sa["w_t"], wa_a=sa["w_a"], g=sa["g_eff"])
    if sb is not None:
        v |= dict(wt_b=sb["w_t"], wa_b=sb["w_a"])
    return v


@app.cell
def _(alpha, alpha_err_ui, g_err_ui, g_known, line_err_ui, lines_a, lines_b, scenarios):
    # Error propagation. Kept apart from the view cell so that pasting two-tone peaks
    # doesn't rerun it.
    s_line, s_g, s_alpha = line_err_ui.value / 1e3, g_err_ui.value / 1e3, alpha_err_ui.value / 1e3
    mc = {}
    for _name, _nominal in scenarios.items():
        _below = _name.startswith("Alice below")
        _g = g_known if g_known > 0 else None
        _smp, _kept = bare_samples(lines_a, lines_b, _nominal, _g, s_g, s_line, _below)
        _budget = {}
        if _g is not None:
            for _src, _sg, _sl in (("g_eff", s_g, 0.0), ("the line centres", 0.0, s_line)):
                _sub, _ = bare_samples(lines_a, lines_b, _nominal, _g, _sg, _sl, _below)
                _budget[_src] = float(np.std(_sub["wa_a"] - _sub["wt_a"])) if len(_sub["g"]) else np.nan
        mc[_name] = dict(s=_smp, v=values(*_nominal), kept=_kept, budget=_budget,
                         lines=line_errors(_smp, alpha, s_alpha))
    return mc, s_alpha, s_g, s_line


@app.cell
def _(alpha, at_limit, g_known, lines_a, mc, peaks_ui, s_alpha, s_g, s_line, scenarios):
    peaks = parse_peaks(peaks_ui.value)
    split_a = (lines_a[1] - lines_a[0]) * 1e3
    names = list(scenarios)
    views = []

    def q(n, fn, ghz=False):
        """Nominal value of fn(values) with its 68% error. MHz, or GHz with the error in MHz."""
        v, s = mc[n]["v"], mc[n]["s"]
        try:
            c = fn(v)
        except KeyError:
            return "—"
        if ghz:
            return f"{c:.4f} ({err(c * 1e3, fn(s) * 1e3)} MHz)"
        return f"{c * 1e3:+.1f} {err(c * 1e3, fn(s) * 1e3)}"

    delta_a = lambda d: d["wa_a"] - d["wt_a"]  # noqa: E731
    rod = lambda d: d["wa_b"] - d["wa_a"]  # noqa: E731
    moved = lambda d: d["wt_b"] - d["wt_a"]  # noqa: E731

    # ── Headline ─────────────────────────────────────────────────────────────
    if not scenarios and g_known > 0:
        headline = mo.callout(mo.md(
            f"**These can't both be real modes with that coupling.** The no-rod lines are "
            f"{split_a:.1f} MHz apart, but with g_eff = {g_known * 1e3:.1f} MHz two coupled modes "
            f"can never be closer than 2g_eff = {2 * g_known * 1e3:.1f} MHz. Either g_eff is smaller "
            f"than {split_a / 2:.1f} MHz, or one of those lines isn't what you think it is."), kind="danger")
    elif not scenarios:
        headline = mo.callout(mo.md(
            "**These four lines can't come from one transmon frequency and one coupling.** "
            "Most likely the transmon moved between cooldowns. Enter g_eff to solve each "
            "cooldown on its own."), kind="danger")
    else:
        fits = ({n: match(peaks, predicted_lines(scenarios[n][0]), mc[n]["lines"], s_line) for n in names}
                if peaks else {})
        z = {n: fits[n][2] for n in fits}

        def said(n):
            sa, sb = scenarios[n]
            lo, hi = sa["m1"]
            d = delta_a(mc[n]["v"]) * 1e3

            def who(i):
                p = weight(sa, i, 1, 0)
                return f"**{100 * max(p, 1 - p):.0f}% {'transmon' if p >= 0.5 else 'Alice'}**"
            txt = (f"Alice was **{abs(d):.1f} MHz {'below' if d < 0 else 'above'}** the transmon in the "
                   f"no-rod cooldown ({err(abs(d), abs(delta_a(mc[n]['s'])) * 1e3).strip('()')} MHz). The {sa['E'][lo]:.4f} line "
                   f"is {who(lo)}, the {sa['E'][hi]:.4f} line is {who(hi)}.")
            if sb is not None:
                txt += f" The rod moved Alice by {q(n, rod)} MHz."
            return txt

        def fit_txt(n):
            return f"{fits[n][1]:.1f} MHz rms = {z[n]:.1f}σ"

        if at_limit:
            n = names[0]
            d = np.abs(delta_a(mc[n]["s"])) * 1e3
            msg = (f"**On resonance.** With g_eff = {g_known * 1e3:.1f} ± {s_g * 1e3:.2f} MHz, the no-rod lines "
                   f"({split_a:.1f} MHz apart) sit at the closest two coupled modes can ever be "
                   f"(2g_eff = {2 * g_known * 1e3:.1f} MHz). At the nominal values they're a hair past it, "
                   f"but inside your error bars. So bare transmon and bare Alice coincided to within "
                   f"**{np.percentile(d, 84.13):.1f} MHz** (68%), and each line is close to 50/50 transmon/Alice. "
                   "Which one was on top can't be read off these two lines; the two-tone lines can tell.")
            if scenarios[n][1] is not None:
                msg += f" The rod moved Alice by {q(n, rod)} MHz."
            kind = "warn"
        elif len(names) == 1:
            n = names[0]
            msg = (f"**g_eff = {q(n, lambda d: d['g']).lstrip('+')} MHz** (inferred, assuming the transmon "
                   f"and coupling didn't change between cooldowns). {said(n)}")
            if fits:
                msg += f"\n\nYour two-tone peaks sit {fit_txt(n)} from the predicted lines"
                msg += (". That's more than your error bars allow — check α first."
                        if z[n] > 3 else ", consistent with your error bars.")
            kind = "success"
        elif len(names) == 2 and fits and max(z.values()) > 2 and max(z.values()) > 2 * min(z.values()):
            best = min(z, key=z.get)
            msg = f"**Your two-tone data picks: {best}.** {said(best)}\n\n" + " · ".join(
                f"{n}: {fit_txt(n)}" for n in names)
            kind = "success"
        elif len(names) == 2:
            msg = ("**Two pictures fit the main lines equally well.** The two-tone lines tell them "
                   "apart — paste your peaks in step 3.\n\n" + "\n\n".join(f"- *{n}:* {said(n)}" for n in names))
            if fits:
                msg += ("\n\nWithin your error bars your peaks don't clearly prefer either: "
                        + " · ".join(f"{n}: {fit_txt(n)}" for n in names))
            kind = "info"
        else:
            msg = f"Only one ordering is possible with this g_eff. {said(names[0])}"
            kind = "info"
        headline = mo.callout(mo.md(msg), kind=kind)
    views.append(headline)

    # ── Bare frequencies ─────────────────────────────────────────────────────
    if scenarios:
        rows = [
            ("bare transmon, no rod (GHz)", lambda n: q(n, lambda d: d["wt_a"], ghz=True)),
            ("bare Alice, no rod (GHz)", lambda n: q(n, lambda d: d["wa_a"], ghz=True)),
            ("Alice − transmon, no rod (MHz)", lambda n: q(n, delta_a)),
            ("g_eff (MHz)", lambda n: q(n, lambda d: d["g"]).lstrip("+")),
            ("bare transmon, rod (GHz)", lambda n: q(n, lambda d: d["wt_b"], ghz=True)),
            ("bare Alice, rod (GHz)", lambda n: q(n, lambda d: d["wa_b"], ghz=True)),
            ("transmon moved between cooldowns (MHz)", lambda n: q(n, moved)),
        ]
        md = "### Bare frequencies\n\n| | " + " | ".join(names) + " |\n|:--|" + "--:|" * len(names) + "\n"
        md += "\n".join(f"| {label} | " + " | ".join(f(n) for n in names) + " |" for label, f in rows)
        md += "\n\n*Errors are 68% intervals, from the uncertainties you entered.*"
        if len(names) == 2:
            md += (" *If you believe the transmon frequency held steady between cooldowns, "
                   "the column where it barely moved is the more plausible one.*")
        for n in names:
            if mc[n]["kept"] < 0.95:
                md += (f"\n\n⚠️ *{n}:* {100 * (1 - mc[n]['kept']):.0f}% of the trial inputs within your error "
                       "bars put the no-rod lines closer than 2g_eff, which is impossible, so they were "
                       "dropped. That's expected for a cooldown sitting right on resonance, but it means "
                       "the no-rod numbers lean on your error bars.")
        b = mc[names[0]]["budget"]
        if b:
            md += (f"\n\n**Biggest source of error in the no-rod numbers: {max(b, key=b.get)}** ("
                   + ", ".join(f"±{v * 1e3:.1f} MHz from {k}" for k, v in b.items())
                   + "). Tightening that one helps most.")
        views.append(mo.md(md))

        # ── Rod-cooldown α check ────────────────────────────────────────────
        sb = next((scenarios[n][1] for n in names if scenarios[n][1] is not None), None)
        if sb is not None:
            def ef_at(a):
                s = spectrum(sb["w_t"], sb["w_a"], sb["g_eff"], a)
                rank = max(range(2), key=lambda k: weight(s, s["m1"][k], 1, 0))   # the transmon-like line
                return max((x for x in predicted_lines(s) if x["key"][:2] == ("x", rank)),
                           key=lambda x: x["strength"])["f"]
            ef = ef_at(alpha)
            ef_err = abs(ef_at(alpha + s_alpha) - ef_at(alpha - s_alpha)) / 2 * 1e3 if s_alpha else 0.0
            views.append(mo.md(
                f"**Check α:** in the rod cooldown the ef line should be at **{ef:.4f} GHz** "
                f"(± {ef_err:.1f} MHz from α). If you measured it somewhere else, change α until this "
                "matches — every two-tone prediction below depends on it."))

        # ── Predicted two-tone lines, no-rod cooldown ───────────────────────
        tabs = {}
        for n in names:
            lines = predicted_lines(scenarios[n][0])
            sig = mc[n]["lines"]
            md = ""
            for grp in ["ground", *scenarios[n][0]["m1"], "2ph"]:
                sub = [x for x in lines if x["group"] == grp]
                md += (f"**{sub[0]['title']}**\n\n| GHz | ± MHz | ends in | strength |\n|--:|--:|:--|:--|\n"
                       + "\n".join(f"| {x['f']:.4f} | {sig.get(x['key'], np.nan) * 1e3:.1f} | {esc(x['short'])} "
                                   f"| {strength_word(x['strength'])} |" for x in sub) + "\n\n")
            md += "*Strength is for a qubit drive, relative to the other lines in the same table.*"
            if peaks:
                rows_, rms_, z_ = match(peaks, lines, sig, s_line)
                md += ("\n\n**Your peaks**\n\n| measured | nearest predicted | off by (MHz) | ± (MHz) | which line |"
                       "\n|--:|--:|--:|--:|:--|\n")
                md += "\n".join(f"| {p:.4f} | {x['f']:.4f} | {(p - x['f']) * 1e3:+.1f} | {s * 1e3:.1f} | "
                                f"{esc(x['what'])} |" for p, x, s in rows_)
                md += f"\n\n**rms: {rms_:.1f} MHz = {z_:.1f}σ**"
            tabs[n] = mo.md(md)
        views.append(mo.md("### Predicted two-tone lines, no-rod cooldown"))
        views.append(mo.ui.tabs(tabs) if len(tabs) > 1 else next(iter(tabs.values())))

    result_view = mo.vstack(views, gap=1)
    return (result_view,)


if __name__ == "__main__":
    app.run()
