import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # Transmon-Alice Strong Coupling Analysis
    Explore the highly hybridized regime of a transmon strongly coupled to a cavity.
    Adjust the sliders to see the impact on effective parameters, spectrum, and dressed coherence.
    """)
    return


@app.cell
def _(mo):
    EJ_slider = mo.ui.slider(10.0, 30.0, step=0.1, value=20.55, label="EJ (GHz)", show_value=True)
    EC_slider = mo.ui.slider(0.05, 0.5, step=0.01, value=0.114, label="EC (GHz)", show_value=True)
    E_osc_slider = mo.ui.slider(3.5, 5.5, step=0.001, value=4.327, label="Cavity E_osc (GHz)", show_value=True)
    g_slider = mo.ui.slider(0.001, 0.2, step=0.001, value=0.025, label="Coupling g (GHz)", show_value=True)
    T1_tmon_ui = mo.ui.number(1.0, 500.0, value=60.0, label="Transmon T1 (µs)", step=1.0)
    T2_tmon_ui = mo.ui.number(1.0, 500.0, value=30.0, label="Transmon T2 (µs)", step=1.0)
    T1_cav_ui = mo.ui.number(1.0, 2000.0, value=143.0, label="Cavity T1 (µs)", step=1.0)
    T2_cav_ui = mo.ui.number(1.0, 2000.0, value=286.0, label="Cavity T2 (µs)", step=1.0)

    # Beamsplitter parameters. kappa_1_s / kappa_2_s are MEASURED rates for the
    # second (storage) mode, which is not part of this Hilbert space -- they are
    # inputs, not results of this model. They used to be bare literals buried in
    # the fidelity calculation; exposed here so it is obvious they are ours to set.
    kappa_1_s_ui = mo.ui.number(0.0, 1e6, value=2e3, label="Storage κ₁ (s⁻¹, measured)", step=100.0)
    kappa_2_s_ui = mo.ui.number(0.0, 1e6, value=10e3, label="Storage κ_φ (s⁻¹, measured)", step=100.0)
    t_bs_ui = mo.ui.number(0.01, 100.0, value=2.0, label="Beamsplitter duration (µs)", step=0.1)
    return (
        EC_slider,
        EJ_slider,
        E_osc_slider,
        T1_cav_ui,
        T1_tmon_ui,
        T2_cav_ui,
        T2_tmon_ui,
        g_slider,
        kappa_1_s_ui,
        kappa_2_s_ui,
        t_bs_ui,
    )


@app.cell
def _(
    EC_slider,
    EJ_slider,
    E_osc_slider,
    T1_cav_ui,
    T1_tmon_ui,
    T2_cav_ui,
    T2_tmon_ui,
    g_slider,
    kappa_1_s_ui,
    kappa_2_s_ui,
    mo,
    np,
    pd,
    scq,
    t_bs_ui,
):
    # ── scqubits setup ───────────────────────────────────────────────────────
    scq.settings.OVERLAP_THRESHOLD = 0.25

    tmon = scq.Transmon(EJ=EJ_slider.value, EC=EC_slider.value, ng=0.0, ncut=30, truncated_dim=6)
    cavity = scq.Oscillator(E_osc=E_osc_slider.value, truncated_dim=6)

    hilbert = scq.HilbertSpace([tmon, cavity])
    a_plus_adag = cavity.annihilation_operator() + cavity.creation_operator()

    hilbert.add_interaction(
        g_strength=g_slider.value,
        op1=tmon.n_operator,
        op2=(a_plus_adag, cavity),
    )

    dim = hilbert.dimension
    evals, evecs = hilbert.eigensys(evals_count=dim)
    hilbert.generate_lookup()

    evec_matrix = np.array([evecs[i].full().flatten() for i in range(len(evals))])

    # ── Matrix Elements ──────────────────────────────────────────────────────
    n_dressed = hilbert.op_in_dressed_eigenbasis(tmon.n_operator)
    n_matrix = n_dressed.full()

    n_t = tmon.truncated_dim
    x_cav_bare = cavity.annihilation_operator() + cavity.creation_operator()
    x_full = np.kron(np.eye(n_t), x_cav_bare)
    x_matrix = evec_matrix @ x_full @ evec_matrix.conj().T

    # ── State Identification by excitation-number manifold ───────────────────
    # An earlier version picked states by argmax of |<i|n|g>| / |<i|x|g>|. That
    # fails exactly at resonance: both polaritons are 50/50 mixtures, so both
    # matrix elements are equal to within numerical noise and the two searches
    # can return the SAME eigenstate (verified: at Delta=0 it returned index 1
    # for both the transmon-like and the cavity-like mode). scqubits'
    # generate_lookup() fails there for the same reason.
    #
    # Instead label by total excitation number N = <n_transmon> + <a-dag a>,
    # then order by energy inside each manifold. N is conserved to ~8e-5 here
    # (g_eff/w_c ~ 0.009, far below the ultrastrong threshold of ~0.1), so this
    # is unambiguous and continuous through resonance.
    g_idx = 0
    n_c = cavity.truncated_dim
    N_t_op = np.kron(np.diag(np.arange(n_t)), np.eye(n_c))
    N_c_op = np.kron(np.eye(n_t), np.diag(np.arange(n_c)))

    def _expect(op):
        return np.real(np.einsum("ij,jk,ik->i", evec_matrix.conj(), op, evec_matrix))

    nt_exp = _expect(N_t_op)      # transmon excitations per dressed state
    nc_exp = _expect(N_c_op)      # photons per dressed state
    N_exp = nt_exp + nc_exp
    N_err = float(np.abs(N_exp - np.round(N_exp))[:12].max())

    def _manifold(n):
        idx = np.where(np.abs(N_exp - n) < 0.15)[0]
        return idx[np.argsort(evals[idx])]

    m1 = _manifold(1)
    lp_idx, up_idx = int(m1[0]), int(m1[1])   # lower / upper polariton

    # Character labels: keep "transmon-like" / "Alice-like" wherever they mean
    # something, i.e. wherever one mode is clearly more transmon than the other.
    t_frac = nt_exp / np.where(N_exp > 1e-9, N_exp, 1.0)
    if t_frac[lp_idx] >= t_frac[up_idx]:
        e_idx, c1_idx = lp_idx, up_idx
    else:
        e_idx, c1_idx = up_idx, lp_idx
    # Guard: inside this window the two modes are ~half-and-half and neither
    # deserves a character label at all.
    hybridized = bool(0.4 <= t_frac[e_idx] <= 0.6)

    # Two-excitation manifold: doubly-transmon, doubly-photonic, and the mixed
    # one-and-one state whose energy defines the cross-Kerr.
    m2 = _manifold(2)
    f_idx = int(m2[np.argmax(nt_exp[m2])])
    c2_idx = int(m2[np.argmax(nc_exp[m2])])
    _mixed = [int(i) for i in m2 if i not in (f_idx, c2_idx)]
    c1e1_idx = _mixed[0] if _mixed else -1

    # ── Spectrum Extraction ──────────────────────────────────────────────────
    w_ge = float(evals[e_idx] - evals[g_idx])
    w_ef = float(evals[f_idx] - evals[e_idx])
    alpha_t = float(w_ef - w_ge)

    w_01 = float(evals[c1_idx] - evals[g_idx])
    w_12 = float(evals[c2_idx] - evals[c1_idx])
    K_cav = float(w_12 - w_01)

    chi = (
        float(evals[c1e1_idx] - evals[e_idx] - evals[c1_idx] + evals[g_idx])
        if c1e1_idx >= 0
        else float("nan")
    )

    # ── Coupling: the slider is NOT the coupling rate ────────────────────────
    # scqubits adds H_int = g * n * (a + a-dag). The charge matrix element
    # <0|n|1> is ~1.5 for a transmon, not 1, so the physical vacuum-Rabi
    # coupling is g_eff = g * n01 -- and because n01 ~ (EJ/8EC)^(1/4)/sqrt(2),
    # moving the EJ or EC slider changes g_eff even with g held fixed.
    n01 = float(np.abs(tmon.matrixelement_table("n_operator")[0, 1]))
    g_eff = float(g_slider.value) * n01
    detuning = float(cavity.E_osc - tmon.E01())
    # chi is only a dispersive cross-Kerr when the modes are well separated.
    dispersive = bool(abs(detuning) > 5.0 * g_eff)

    bare_w_tmon = float(tmon.E01())
    bare_alpha = float(tmon.anharmonicity())
    bare_w_cav = float(cavity.E_osc)

    # ── Participation ratios & coherence ─────────────────────────────────────
    n_bare = tmon.matrixelement_table("n_operator")

    # Transmon participation = the transmon-excitation content of each dressed
    # mode, read straight off the eigenvector. The old definition below took a
    # ratio of charge matrix elements instead; it happens to agree to ~0.005
    # absolute, but it is not the same quantity and it is evaluated on whichever
    # state argmax picked. Both are shown so the difference stays visible.
    p_t_tmon = float(nt_exp[e_idx])
    p_t_cav = float(nt_exp[c1_idx])

    me_ge_dressed_sq = float(np.abs(n_matrix[e_idx, g_idx]) ** 2)
    me_c01_dressed_sq = float(np.abs(n_matrix[c1_idx, g_idx]) ** 2)
    me_ge_bare_sq = float(np.abs(n_bare[0, 1]) ** 2)

    p_t_tmon_old = float(me_ge_dressed_sq / me_ge_bare_sq)
    p_t_cav_old = float(me_c01_dressed_sq / me_ge_bare_sq)

    T1_t_bare = float(T1_tmon_ui.value) * 1e-6
    T2_t_bare = float(T2_tmon_ui.value) * 1e-6
    T1_c_bare = float(T1_cav_ui.value) * 1e-6
    T2_c_bare = float(T2_cav_ui.value) * 1e-6

    T1_d_t = float(1.0 / (p_t_tmon / T1_t_bare + (1 - p_t_tmon) / T1_c_bare) * 1e6)
    T1_d_c = float(1.0 / (p_t_cav / T1_t_bare + (1 - p_t_cav) / T1_c_bare) * 1e6)
    T2_d_t = float(1.0 / (p_t_tmon / T2_t_bare + (1 - p_t_tmon) / T2_c_bare) * 1e6)
    T2_d_c = float(1.0 / (p_t_cav / T2_t_bare + (1 - p_t_cav) / T2_c_bare) * 1e6)

    # ── BS fidelity ──────────────────────────────────────────────────────────
    kappa_1_a = 1 / (T1_d_c * 1e-6)
    kappa_1_s = float(kappa_1_s_ui.value)
    kappa_2_a = 1 / (T2_d_c * 1e-6) - 1 / (2 * T1_d_c * 1e-6)
    kappa_2_s = float(kappa_2_s_ui.value)
    k1_avg = 0.5 * (kappa_1_a + kappa_1_s)
    kphi_avg = 0.5 * (kappa_2_a + kappa_2_s)
    t_bs = float(t_bs_ui.value) * 1e-6
    bs_fid = float(0.5 * (np.exp(-k1_avg * t_bs) + np.exp(-(k1_avg + kphi_avg) * t_bs)))

    # ── Rabi rate ratios ─────────────────────────────────────────────────────
    me_ge = float(np.abs(n_matrix[e_idx, g_idx]))
    me_ef = float(np.abs(n_matrix[f_idx, e_idx]))
    me_ge_bare = float(np.abs(n_bare[0, 1]))
    me_ef_bare = float(np.abs(n_bare[1, 2]))

    rabi_ratio_tmon_dressed = float(me_ef / me_ge)
    rabi_ratio_tmon_bare = float(me_ef_bare / me_ge_bare)

    me_c01 = float(np.abs(x_matrix[c1_idx, g_idx]))
    me_c12 = float(np.abs(x_matrix[c2_idx, c1_idx]))
    me_c01_bare = float(np.abs(x_cav_bare[1, 0]))
    me_c12_bare = float(np.abs(x_cav_bare[2, 1]))

    rabi_ratio_cav_dressed = float(me_c12 / me_c01)
    rabi_ratio_cav_bare = float(me_c12_bare / me_c01_bare)

    # ── Display ──────────────────────────────────────────────────────────────
    lbl_t = "Hybridized (mode A)" if hybridized else "Transmon-like mode"
    lbl_c = "Hybridized (mode B)" if hybridized else "Alice-like mode"
    hyb_note = (
        rf"""
    > ⚠️ **Modes are ~50/50 hybridized** ($\Delta$ = {detuning*1e3:+.2f} MHz,
    > $g_{{eff}}$ = {g_eff*1e3:.2f} MHz). Neither eigenstate is "the transmon" or
    > "Alice" here -- they are the lower and upper polariton, split by
    > {(evals[up_idx]-evals[lp_idx])*1e3:.2f} MHz. Labels below are ordered by
    > energy, not character.
    """
        if hybridized
        else ""
    )
    chi_line = (
        rf"- $\chi$ = {chi:.6f} GHz  ({chi*1e3:.3f} MHz)"
        if dispersive
        else rf"""- Level shift = {chi*1e3:.3f} MHz
    > ⚠️ **Not a dispersive cross-Kerr.** $|\Delta|/g_{{eff}}$ =
    > {abs(detuning)/g_eff:.2f}, below the $\gtrsim 5$ needed for $\chi$ to mean
    > anything. The energy combination is still exact; its *interpretation* is not."""
    )

    mo.vstack([
        EJ_slider, EC_slider, E_osc_slider, g_slider,
        mo.md("**Bare Coherence**"),
        T1_tmon_ui, T2_tmon_ui, T1_cav_ui, T2_cav_ui,
        mo.md("**Beamsplitter (storage-mode rates are measured inputs, not model outputs)**"),
        kappa_1_s_ui, kappa_2_s_ui, t_bs_ui,

        mo.md("---"),

        mo.md(rf"""
    ### Coupling

    - Slider $g$ = {g_slider.value*1e3:.3f} MHz  *(prefactor of $g\,\hat n(\hat a + \hat a^\dagger)$, **not** the coupling rate)*
    - $|\langle 0|\hat n|1\rangle|$ = {n01:.4f}
    - **$g_{{eff}} = g\,n_{{01}}$ = {g_eff*1e3:.3f} MHz**  → vacuum Rabi splitting $2g_{{eff}}$ = {2*g_eff*1e3:.3f} MHz
    - Detuning $\Delta = \omega_c - \omega_t$ = {detuning*1e3:+.3f} MHz  ($\Delta/g_{{eff}}$ = {detuning/g_eff:+.2f})
    - $g_{{eff}}/\omega_c$ = {g_eff/bare_w_cav:.5f} *(ultrastrong would need $\gtrsim 0.1$)*
    - Excitation number conserved to {N_err:.1e} *(validates manifold labeling)*

    ⓘ $n_{{01}}$ depends on EJ and EC, so **moving the EJ/EC sliders changes $g_{{eff}}$**
    even with $g$ fixed.
    {hyb_note}
    ### Dressed Parameters (scqubits exact diag)

    **{lbl_t}:**
    - $\omega_{{ge}}$ = {w_ge:.6f} GHz
    - $\alpha$ = {alpha_t:.6f} GHz  ({alpha_t*1e3:.3f} MHz)
    - Transmon content: {p_t_tmon:.4f}  *(old matrix-element estimate: {p_t_tmon_old:.4f})*

    **{lbl_c}:**
    - $\omega_{{01}}$ = {w_01:.6f} GHz
    - $K$ (Self-Kerr) = {K_cav:.6f} GHz  ({K_cav*1e3:.3f} MHz)
    - Transmon content: {p_t_cav:.6f}  *(old matrix-element estimate: {p_t_cav_old:.6f})*

    **Cross-Kerr:**
    {chi_line}

    **Bare values for comparison:**
    - Bare $\omega_{{transmon}}$ = {bare_w_tmon:.6f} GHz, $\alpha$ = {bare_alpha:.6f} GHz
    - Bare $\omega_{{cavity}}$ = {bare_w_cav:.6f} GHz, K = 0 (harmonic)

    ### Dressed Coherence Estimates
    - **{lbl_t}:** $T_1$ = {T1_d_t:.1f} µs, $T_2$ = {T2_d_t:.1f} µs
    - **{lbl_c}:** $T_1$ = {T1_d_c:.1f} µs, $T_2$ = {T2_d_c:.1f} µs
    - **Estimated BS Fidelity ({t_bs*1e6:.2f} µs):** {bs_fid:.4f}
        """),

        mo.md("---"),

        mo.md("### Rabi Rate & Matrix Element Analysis"),
        mo.md("*Crucial for identifying true anharmonicity independent of state labeling.*"),
        mo.ui.table(pd.DataFrame({
            'Quantity': [
                '--- TRANSMON (drive via n̂) ---',
                '|⟨e|n̂|g⟩| dressed',
                '|⟨f|n̂|e⟩| dressed',
                'Ω_ef / Ω_ge  dressed',
                '|⟨e|n̂|g⟩| bare',
                '|⟨f|n̂|e⟩| bare',
                'Ω_ef / Ω_ge  bare',
                '--- CAVITY (drive via x̂ = a+a†) ---',
                '|⟨c₁|x̂|g⟩|  dressed',
                '|⟨c₂|x̂|c₁⟩| dressed',
                'Ω_12 / Ω_01  dressed',
                '|⟨c₁|x̂|g⟩|  bare',
                '|⟨c₂|x̂|c₁⟩| bare',
                'Ω_12 / Ω_01  bare',
            ],
            'Value': [
                '',
                f'{me_ge:.6f}',
                f'{me_ef:.6f}',
                f'{rabi_ratio_tmon_dressed:.6f}',
                f'{me_ge_bare:.6f}',
                f'{me_ef_bare:.6f}',
                f'{rabi_ratio_tmon_bare:.6f}',
                '',
                f'{me_c01:.6f}',
                f'{me_c12:.6f}',
                f'{rabi_ratio_cav_dressed:.6f}',
                f'{me_c01_bare:.6f}',
                f'{me_c12_bare:.6f}',
                f'{rabi_ratio_cav_bare:.6f}',
            ],
            'Note': [
                '',
                '', '', 'coupling shifts from √2',
                '', '', f'bare limit = √2 ≈ {np.sqrt(2):.4f}',
                '',
                '', '', 'coupling shifts from √2',
                '', '', f'harmonic limit = √2 ≈ {np.sqrt(2):.4f}',
            ],
        })),
    ])
    return


@app.cell
def _():
    import marimo as mo
    import scqubits as scq
    import numpy as np
    import pandas as pd

    return mo, np, pd, scq


if __name__ == "__main__":
    app.run()
