import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Transmon–Alice Strong Coupling Analysis

    Exact diagonalization of a transmon capacitively coupled to a cavity ("Alice").
    Valid from the dispersive limit all the way through resonance — no rotating-wave
    or dispersive approximation is made anywhere.
    """
    )
    return


@app.cell
def _(mo):
    # Collapsing needs mo.accordion; markdown-level syntax for it does not
    # survive marimo's prose editor.
    mo.accordion(
        {
            "📖 How this model works (click to expand)": mo.md(
                r"""
    ### The Hamiltonian

    $$H = H_{\rm transmon} + \omega_c a^\dagger a + g\,\hat n\,(a + a^\dagger)$$

    diagonalized exactly by `scqubits`. Both the transmon and the cavity are kept
    to `Levels per mode` states each, and the transmon itself is solved in a charge
    basis of size $2\,\mathrm{ncut}+1$.

    ### ⚠️ The slider `g` is not the coupling rate

    `scqubits` adds $g\,\hat n(a+a^\dagger)$, and the transmon charge matrix element
    $|\langle 0|\hat n|1\rangle| \approx 1.5$ — not 1. The physical vacuum-Rabi
    coupling is therefore

    $$g_{\rm eff} = g\,n_{01}, \qquad n_{01} \approx \left(\frac{E_J}{32E_C}\right)^{1/4}$$

    and the vacuum Rabi splitting on resonance is $2g_{\rm eff}$. Because $n_{01}$
    depends on $E_J$ and $E_C$, **moving the EJ or EC slider changes the coupling**
    even when the `g` slider has not moved. Quote $g_{\rm eff}$, never the slider.

    ### How states are labeled

    The obvious approach — "which eigenstate has the largest
    $|\langle i|\hat n|g\rangle|$?" — **fails at resonance**. There both polaritons
    are 50/50 mixtures, so the matrix elements are equal to within numerical noise
    and the search for the transmon and the search for the cavity can return the
    *same* eigenstate. `scqubits`' own `generate_lookup()` fails there for the same
    reason. This is not a numerical accuracy problem: at $\Delta = 0$ neither
    eigenstate *is* "the transmon", so the question has no answer.

    Instead we label by total excitation number
    $N = \langle n_{\rm transmon}\rangle + \langle a^\dagger a\rangle$, then order
    by energy inside each manifold. $N$ is conserved to $\sim 10^{-4}$ at the
    default coupling, which makes this unambiguous and continuous through
    resonance. "Transmon-like" and "Alice-like" are then applied on top, and
    *suppressed* whenever the two modes are within 40–60 % hybridized.

    ### Cross-Kerr

    The reported $\chi_2 = E_{e1}-E_{e0}-E_{g1}+E_{g0}$ equals $2\chi$ in the usual
    $\chi\,\sigma_z a^\dagger a$ convention — both are shown. It is only a
    *dispersive* cross-Kerr when $|\Delta| \gg g_{\rm eff}$; below that the energy
    combination is still exact but its interpretation is not, and the notebook says
    so. Verified against second-order perturbation theory in
    `tests/test_strong_coupling_convergence.py`.

    ### What is measured input, not model output

    The storage-mode rates $\kappa_1$, $\kappa_\varphi$ and the beamsplitter
    duration describe a second mode that is **not** in this Hilbert space. They are
    yours to set from measurement; nothing here derives them.
    """
            )
        }
    )
    return


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(
    K_cav,
    T1_d_c,
    T1_d_t,
    T2_d_c,
    T2_d_t,
    alpha_t,
    bare_w_cav,
    bs_fid,
    chi_txt,
    detuning,
    g_eff,
    g_slider,
    lbl_c,
    lbl_t,
    mo,
    n01,
    p_t_cav,
    p_t_tmon,
    t_bs,
    w_01,
    warn_banner,
    w_ge,
):
    mo.md(
        f"""
    ## Summary
    {warn_banner}
    | mode | ω (GHz) | α or K (MHz) | transmon | T₁ (µs) | T₂ (µs) |
    |:--|--:|--:|--:|--:|--:|
    | **{lbl_t}** | {w_ge:.6f} | {alpha_t*1e3:+.3f} | {p_t_tmon*100:.1f} % | {T1_d_t:.1f} | {T2_d_t:.1f} |
    | **{lbl_c}** | {w_01:.6f} | {K_cav*1e3:+.3f} | {p_t_cav*100:.1f} % | {T1_d_c:.1f} | {T2_d_c:.1f} |

    | coupling | | detuning | | cross-Kerr | beamsplitter |
    |:--|--:|:--|--:|:--|:--|
    | **g_eff** | **{g_eff*1e3:.3f} MHz** | Δ = ω_c − ω_t | {detuning*1e3:+.3f} MHz | {chi_txt} | F({t_bs*1e6:.2f} µs) = **{bs_fid:.4f}** |
    | 2g_eff (vac. Rabi) | {2*g_eff*1e3:.3f} MHz | Δ / g_eff | {detuning/g_eff:+.2f} | | |
    | slider g × n₀₁ = {g_slider.value*1e3:.1f} × {n01:.4f} | | g_eff / ω_c | {g_eff/bare_w_cav:.5f} | | |
    """
    )
    return


@app.cell
def _(
    K_cav,
    N_err,
    alpha_t,
    bare_alpha,
    bare_w_cav,
    bare_w_tmon,
    lbl_c,
    lbl_t,
    mo,
    p_t_cav,
    p_t_cav_old,
    p_t_tmon,
    p_t_tmon_old,
    w_01,
    w_ge,
):
    mo.accordion(
        {
            "🔬 Detail — bare values, participation, diagnostics": mo.md(
                f"""
    **Dressed ({lbl_t}):** ω_ge = {w_ge:.6f} GHz, α = {alpha_t*1e3:.3f} MHz,
    transmon content {p_t_tmon:.4f} *(old matrix-element estimate: {p_t_tmon_old:.4f})*

    **Dressed ({lbl_c}):** ω_01 = {w_01:.6f} GHz, K = {K_cav*1e3:.3f} MHz,
    transmon content {p_t_cav:.6f} *(old matrix-element estimate: {p_t_cav_old:.6f})*

    **Bare:** transmon ω = {bare_w_tmon:.6f} GHz, α = {bare_alpha*1e3:.3f} MHz ·
    cavity ω = {bare_w_cav:.6f} GHz, K = 0 (harmonic)

    **Diagnostic:** excitation number conserved to {N_err:.2e}
    *(validates the manifold labeling; warns above 1e-2)*
    """
            )
        }
    )
    return


@app.cell
def _(mo, rabi_table):
    mo.accordion({"📐 Rabi rates & matrix elements": rabi_table})
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ---
    ## Implementation

    Everything below is machinery. The cells are ordered for reading, not for
    execution — marimo runs them in dependency order regardless of position.
    """
    )
    return


@app.cell
def _(mo):
    EJ_slider = mo.ui.slider(10.0, 30.0, step=0.1, value=20.55, label="EJ (GHz)", show_value=True, include_input=True)
    EC_slider = mo.ui.slider(0.05, 0.5, step=0.01, value=0.114, label="EC (GHz)", show_value=True, include_input=True)
    E_osc_slider = mo.ui.slider(3.5, 5.5, step=0.001, value=4.327, label="Cavity E_osc (GHz)", show_value=True, include_input=True)
    g_slider = mo.ui.slider(0.001, 0.2, step=0.001, value=0.025, label="Coupling g (GHz)", show_value=True, include_input=True)

    T1_tmon_ui = mo.ui.number(1.0, 500.0, value=60.0, label="Transmon T1 (µs)", step=1.0)
    T2_tmon_ui = mo.ui.number(1.0, 500.0, value=30.0, label="Transmon T2 (µs)", step=1.0)
    T1_cav_ui = mo.ui.number(1.0, 2000.0, value=143.0, label="Cavity T1 (µs)", step=1.0)
    T2_cav_ui = mo.ui.number(1.0, 2000.0, value=286.0, label="Cavity T2 (µs)", step=1.0)

    kappa_1_s_ui = mo.ui.number(0.0, 1e6, value=2e3, label="Storage κ₁ (s⁻¹, measured)", step=100.0)
    kappa_2_s_ui = mo.ui.number(0.0, 1e6, value=10e3, label="Storage κ_φ (s⁻¹, measured)", step=100.0)
    t_bs_ui = mo.ui.number(0.01, 100.0, value=2.0, label="Beamsplitter duration (µs)", step=0.1)

    trunc_ui = mo.ui.number(4, 20, value=8, label="Levels per mode", step=1)
    ncut_ui = mo.ui.number(10, 60, value=30, label="Transmon ncut", step=5)

    # Laid out here so the cell that displays the controls stays one line, keeping
    # the sliders adjacent to the summary output.
    controls = mo.hstack(
        [
            mo.vstack([mo.md("**System**"), EJ_slider, EC_slider, E_osc_slider, g_slider]),
            mo.vstack([mo.md("**Bare coherence**"), T1_tmon_ui, T2_tmon_ui, T1_cav_ui, T2_cav_ui]),
            mo.vstack([mo.md("**Beamsplitter** *(measured)*"), kappa_1_s_ui, kappa_2_s_ui, t_bs_ui,
                       mo.md("**Numerics**"), trunc_ui, ncut_ui]),
        ],
        widths="equal",
        gap=2,
    )
    return (
        EC_slider,
        EJ_slider,
        E_osc_slider,
        T1_cav_ui,
        T1_tmon_ui,
        T2_cav_ui,
        T2_tmon_ui,
        controls,
        g_slider,
        kappa_1_s_ui,
        kappa_2_s_ui,
        ncut_ui,
        t_bs_ui,
        trunc_ui,
    )


@app.cell
def _(EC_slider, EJ_slider, E_osc_slider, g_slider, ncut_ui, np, scq, trunc_ui):
    # Physics only. Deliberately independent of the coherence and beamsplitter
    # inputs so that editing a T1 does not re-run the diagonalization.
    scq.settings.OVERLAP_THRESHOLD = 0.25

    n_lev = int(trunc_ui.value)
    tmon = scq.Transmon(
        EJ=EJ_slider.value, EC=EC_slider.value, ng=0.0,
        ncut=int(ncut_ui.value), truncated_dim=n_lev,
    )
    cavity = scq.Oscillator(E_osc=E_osc_slider.value, truncated_dim=n_lev)

    hilbert = scq.HilbertSpace([tmon, cavity])
    a_plus_adag = cavity.annihilation_operator() + cavity.creation_operator()
    hilbert.add_interaction(
        g_strength=g_slider.value, op1=tmon.n_operator, op2=(a_plus_adag, cavity)
    )

    evals, evecs = hilbert.eigensys(evals_count=hilbert.dimension)
    evec_matrix = np.array([evecs[i].full().flatten() for i in range(len(evals))])

    # Both operators lifted into the product space and rotated with our own evecs,
    # rather than via hilbert.op_in_dressed_eigenbasis() (which reads scqubits'
    # private _data["evecs"]).
    n_t, n_c = tmon.truncated_dim, cavity.truncated_dim
    # evals_count is REQUIRED: matrixelement_table defaults to 6 levels regardless
    # of truncated_dim, silently returning a 6x6 table that breaks the kron below.
    n_bare_eig = tmon.matrixelement_table("n_operator", evals_count=n_lev)
    x_cav_bare = a_plus_adag

    n_matrix = evec_matrix @ np.kron(n_bare_eig, np.eye(n_c)) @ evec_matrix.conj().T
    x_matrix = evec_matrix @ np.kron(np.eye(n_t), x_cav_bare) @ evec_matrix.conj().T

    # ── State identification by excitation-number manifold (see notes at top) ──
    g_idx = 0
    N_t_op = np.kron(np.diag(np.arange(n_t)), np.eye(n_c))
    N_c_op = np.kron(np.eye(n_t), np.diag(np.arange(n_c)))

    def _expect(op):
        return np.real(np.einsum("ij,jk,ik->i", evec_matrix.conj(), op, evec_matrix))

    nt_exp, nc_exp = _expect(N_t_op), _expect(N_c_op)
    N_exp = nt_exp + nc_exp
    N_err = float(np.abs(N_exp - np.round(N_exp))[:12].max())

    def _manifold(n):
        idx = np.where(np.abs(N_exp - n) < 0.15)[0]
        return idx[np.argsort(evals[idx])]

    m1 = _manifold(1)
    lp_idx, up_idx = int(m1[0]), int(m1[1])   # lower / upper polariton

    t_frac = nt_exp / np.where(N_exp > 1e-9, N_exp, 1.0)
    if t_frac[lp_idx] >= t_frac[up_idx]:
        e_idx, c1_idx = lp_idx, up_idx
    else:
        e_idx, c1_idx = up_idx, lp_idx
    # Inside this window neither mode deserves a character label at all.
    hybridized = bool(0.4 <= t_frac[e_idx] <= 0.6)

    m2 = _manifold(2)
    f_idx = int(m2[np.argmax(nt_exp[m2])])
    c2_idx = int(m2[np.argmax(nc_exp[m2])])
    _mixed = [int(i) for i in m2 if i not in (f_idx, c2_idx)]
    c1e1_idx = _mixed[0] if _mixed else -1

    # ── Spectrum ─────────────────────────────────────────────────────────────
    w_ge = float(evals[e_idx] - evals[g_idx])
    alpha_t = float(evals[f_idx] - evals[e_idx] - w_ge)
    w_01 = float(evals[c1_idx] - evals[g_idx])
    K_cav = float(evals[c2_idx] - evals[c1_idx] - w_01)
    chi = (
        float(evals[c1e1_idx] - evals[e_idx] - evals[c1_idx] + evals[g_idx])
        if c1e1_idx >= 0 else float("nan")
    )
    splitting = float(evals[up_idx] - evals[lp_idx])

    n01 = float(np.abs(tmon.matrixelement_table("n_operator")[0, 1]))
    g_eff = float(g_slider.value) * n01
    detuning = float(cavity.E_osc - tmon.E01())
    dispersive = bool(abs(detuning) > 5.0 * g_eff)

    bare_w_tmon = float(tmon.E01())
    bare_alpha = float(tmon.anharmonicity())
    bare_w_cav = float(cavity.E_osc)

    # Transmon participation, read straight off the eigenvector. The *_old values
    # use a matrix-element-ratio definition instead, shown alongside for comparison.
    n_bare = tmon.matrixelement_table("n_operator")
    p_t_tmon = float(nt_exp[e_idx])
    p_t_cav = float(nt_exp[c1_idx])
    p_t_tmon_old = float(np.abs(n_matrix[e_idx, g_idx]) ** 2 / np.abs(n_bare[0, 1]) ** 2)
    p_t_cav_old = float(np.abs(n_matrix[c1_idx, g_idx]) ** 2 / np.abs(n_bare[0, 1]) ** 2)

    # ── Rabi rate ratios ─────────────────────────────────────────────────────
    me_ge = float(np.abs(n_matrix[e_idx, g_idx]))
    me_ef = float(np.abs(n_matrix[f_idx, e_idx]))
    me_ge_bare = float(np.abs(n_bare[0, 1]))
    me_ef_bare = float(np.abs(n_bare[1, 2]))
    me_c01 = float(np.abs(x_matrix[c1_idx, g_idx]))
    me_c12 = float(np.abs(x_matrix[c2_idx, c1_idx]))
    me_c01_bare = float(np.abs(x_cav_bare[1, 0]))
    me_c12_bare = float(np.abs(x_cav_bare[2, 1]))
    rabi_ratio_tmon_dressed = float(me_ef / me_ge)
    rabi_ratio_tmon_bare = float(me_ef_bare / me_ge_bare)
    rabi_ratio_cav_dressed = float(me_c12 / me_c01)
    rabi_ratio_cav_bare = float(me_c12_bare / me_c01_bare)

    lbl_t = "hybridized A" if hybridized else "transmon-like"
    lbl_c = "hybridized B" if hybridized else "Alice-like"

    # Built here so the summary cell stays a bare mo.md() call, which marimo's
    # editor renders as prose rather than as code.
    warn_banner = ""
    if N_err > 1e-2:
        warn_banner += (
            f"\n> ⚠️ **Approaching ultrastrong coupling** — N conserved only to {N_err:.1e}, "
            f"g_eff/ω_c = {g_eff/bare_w_cav:.3f}. Eigenvalues stay exact, but the "
            "manifold labeling is losing its justification.\n"
        )
    if hybridized:
        warn_banner += (
            f"\n> ⚠️ **Modes are ~50/50 hybridized** (Δ = {detuning*1e3:+.2f} MHz, "
            f"g_eff = {g_eff*1e3:.2f} MHz). Neither mode is 'the transmon' or 'Alice' — "
            f"they are the lower/upper polariton, split by {splitting*1e3:.2f} MHz. "
            "Rows are ordered by energy, not character.\n"
        )
    chi_txt = (
        f"**χ₂ = {chi*1e3:.3f} MHz**  (σ_z convention: χ = {chi*1e3/2:.3f} MHz)"
        if dispersive
        else f"level shift {chi*1e3:.3f} MHz — ⚠️ *not a dispersive cross-Kerr*, |Δ|/g_eff = {abs(detuning)/g_eff:.2f}"
    )
    return (
        K_cav,
        N_err,
        alpha_t,
        bare_alpha,
        bare_w_cav,
        bare_w_tmon,
        chi_txt,
        detuning,
        g_eff,
        lbl_c,
        lbl_t,
        me_c01,
        me_c01_bare,
        me_c12,
        me_c12_bare,
        me_ef,
        me_ef_bare,
        me_ge,
        me_ge_bare,
        n01,
        p_t_cav,
        p_t_cav_old,
        p_t_tmon,
        p_t_tmon_old,
        rabi_ratio_cav_bare,
        rabi_ratio_cav_dressed,
        rabi_ratio_tmon_bare,
        rabi_ratio_tmon_dressed,
        w_01,
        warn_banner,
        w_ge,
    )


@app.cell
def _(
    T1_cav_ui,
    T1_tmon_ui,
    T2_cav_ui,
    T2_tmon_ui,
    kappa_1_s_ui,
    kappa_2_s_ui,
    np,
    p_t_cav,
    p_t_tmon,
    t_bs_ui,
):
    # Coherence only. Depends on the physics cell's participation ratios, so
    # editing a T1/T2/kappa re-runs just this and the summary, not the eigensolve.
    #
    # A dressed mode inherits loss from both bare modes in proportion to its
    # participation. Combining 1/T2 linearly is exact, not an approximation:
    # 1/T2 = 1/(2*T1) + 1/Tphi is linear in Gamma_1 and Gamma_phi, so weighting
    # 1/T2 gives the same answer as weighting the two rates separately.
    def _mix(p, bare_tmon_us, bare_cav_us):
        a, b = bare_tmon_us * 1e-6, bare_cav_us * 1e-6
        return float(1.0 / (p / a + (1 - p) / b) * 1e6)

    T1_d_t = _mix(p_t_tmon, T1_tmon_ui.value, T1_cav_ui.value)
    T1_d_c = _mix(p_t_cav, T1_tmon_ui.value, T1_cav_ui.value)
    T2_d_t = _mix(p_t_tmon, T2_tmon_ui.value, T2_cav_ui.value)
    T2_d_c = _mix(p_t_cav, T2_tmon_ui.value, T2_cav_ui.value)

    # Beamsplitter fidelity. The _s rates are the measured storage mode, which is
    # not in this Hilbert space; the _a rates come from the Alice-like mode above.
    kappa_1_a = 1 / (T1_d_c * 1e-6)
    kappa_2_a = 1 / (T2_d_c * 1e-6) - 1 / (2 * T1_d_c * 1e-6)
    k1_avg = 0.5 * (kappa_1_a + float(kappa_1_s_ui.value))
    kphi_avg = 0.5 * (kappa_2_a + float(kappa_2_s_ui.value))
    t_bs = float(t_bs_ui.value) * 1e-6
    bs_fid = float(0.5 * (np.exp(-k1_avg * t_bs) + np.exp(-(k1_avg + kphi_avg) * t_bs)))
    return T1_d_c, T1_d_t, T2_d_c, T2_d_t, bs_fid, t_bs


@app.cell
def _(
    me_c01,
    me_c01_bare,
    me_c12,
    me_c12_bare,
    me_ef,
    me_ef_bare,
    me_ge,
    me_ge_bare,
    mo,
    np,
    pd,
    rabi_ratio_cav_bare,
    rabi_ratio_cav_dressed,
    rabi_ratio_tmon_bare,
    rabi_ratio_tmon_dressed,
):
    rabi_table = mo.vstack([
        mo.md("*Identifies true anharmonicity independently of state labeling.*"),
        mo.ui.table(
            pd.DataFrame({
                "Quantity": [
                    "--- TRANSMON (drive via n̂) ---",
                    "|⟨e|n̂|g⟩| dressed", "|⟨f|n̂|e⟩| dressed", "Ω_ef / Ω_ge  dressed",
                    "|⟨e|n̂|g⟩| bare", "|⟨f|n̂|e⟩| bare", "Ω_ef / Ω_ge  bare",
                    "--- CAVITY (drive via x̂ = a+a†) ---",
                    "|⟨c₁|x̂|g⟩|  dressed", "|⟨c₂|x̂|c₁⟩| dressed", "Ω_12 / Ω_01  dressed",
                    "|⟨c₁|x̂|g⟩|  bare", "|⟨c₂|x̂|c₁⟩| bare", "Ω_12 / Ω_01  bare",
                ],
                "Value": [
                    "", f"{me_ge:.6f}", f"{me_ef:.6f}", f"{rabi_ratio_tmon_dressed:.6f}",
                    f"{me_ge_bare:.6f}", f"{me_ef_bare:.6f}", f"{rabi_ratio_tmon_bare:.6f}",
                    "", f"{me_c01:.6f}", f"{me_c12:.6f}", f"{rabi_ratio_cav_dressed:.6f}",
                    f"{me_c01_bare:.6f}", f"{me_c12_bare:.6f}", f"{rabi_ratio_cav_bare:.6f}",
                ],
                "Note": [
                    "", "", "", "coupling shifts from √2",
                    "", "", f"bare limit = √2 ≈ {np.sqrt(2):.4f}",
                    "", "", "", "coupling shifts from √2",
                    "", "", f"harmonic limit = √2 ≈ {np.sqrt(2):.4f}",
                ],
            }),
            selection=None,
        ),
    ])
    return (rabi_table,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import scqubits as scq

    return mo, np, pd, scq


if __name__ == "__main__":
    app.run()
