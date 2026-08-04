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
    return (
        EC_slider,
        EJ_slider,
        E_osc_slider,
        T1_cav_ui,
        T1_tmon_ui,
        T2_cav_ui,
        T2_tmon_ui,
        g_slider,
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
    mo,
    np,
    pd,
    scq,
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

    # ── Robust State Identification (Bypassing scqubits heuristic) ───────────
    # The ground state is always index 0
    g_idx = 0

    # Transmon-like mode (driven via n)
    e_idx = int(np.argmax(np.abs(n_matrix[:, g_idx])))
    n_from_e = np.abs(n_matrix[:, e_idx]).copy()
    n_from_e[g_idx] = 0
    f_idx = int(np.argmax(n_from_e))

    # Cavity-like mode (driven via x)
    c1_idx = int(np.argmax(np.abs(x_matrix[:, g_idx])))
    x_from_c1 = np.abs(x_matrix[:, c1_idx]).copy()
    x_from_c1[g_idx] = 0
    c2_idx = int(np.argmax(x_from_c1))

    # Mixed state (driven via n from c1, or x from e)
    n_from_c1 = np.abs(n_matrix[:, c1_idx]).copy()
    n_from_c1[g_idx] = 0
    c1e1_idx = int(np.argmax(n_from_c1))

    # ── Spectrum Extraction ──────────────────────────────────────────────────
    w_ge = float(evals[e_idx] - evals[g_idx])
    w_ef = float(evals[f_idx] - evals[e_idx])
    alpha_t = float(w_ef - w_ge)

    w_01 = float(evals[c1_idx] - evals[g_idx])
    w_12 = float(evals[c2_idx] - evals[c1_idx])
    K_cav = float(w_12 - w_01)

    chi = float(evals[c1e1_idx] - evals[e_idx] - evals[c1_idx] + evals[g_idx])

    bare_w_tmon = float(tmon.E01())
    bare_alpha = float(tmon.anharmonicity())
    bare_w_cav = float(cavity.E_osc)

    # ── Participation ratios & coherence ─────────────────────────────────────
    n_bare = tmon.matrixelement_table("n_operator")

    me_ge_dressed_sq = float(np.abs(n_matrix[e_idx, g_idx]) ** 2)
    me_c01_dressed_sq = float(np.abs(n_matrix[c1_idx, g_idx]) ** 2)
    me_ge_bare_sq = float(np.abs(n_bare[0, 1]) ** 2)

    p_t_tmon = float(me_ge_dressed_sq / me_ge_bare_sq)
    p_t_cav = float(me_c01_dressed_sq / me_ge_bare_sq)

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
    kappa_1_s = 2e3
    kappa_2_a = 1 / (T2_d_c * 1e-6) - 1 / (2 * T1_d_c * 1e-6)
    kappa_2_s = 10e3
    k1_avg = 0.5 * (kappa_1_a + kappa_1_s)
    kphi_avg = 0.5 * (kappa_2_a + kappa_2_s)
    t_bs = 2e-6
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
    mo.vstack([
        EJ_slider, EC_slider, E_osc_slider, g_slider,
        mo.md("**Bare Coherence**"),
        T1_tmon_ui, T2_tmon_ui, T1_cav_ui, T2_cav_ui,

        mo.md("---"),

        mo.md(rf"""
    ### Dressed Parameters (scqubits exact diag)

    **Transmon-like mode:**
    - $\omega_{{ge}}$ = {w_ge:.6f} GHz
    - $\alpha$ = {alpha_t:.6f} GHz  ({alpha_t*1e3:.3f} MHz)
    - Participation (transmon): {p_t_tmon:.4f}

    **Cavity-like mode:**
    - $\omega_{{01}}$ = {w_01:.6f} GHz
    - $K$ (Self-Kerr) = {K_cav:.6f} GHz  ({K_cav*1e3:.3f} MHz)
    - Participation (transmon): {p_t_cav:.6f}

    **Cross-Kerr:**
    - $\chi$ = {chi:.6f} GHz  ({chi*1e3:.3f} MHz)

    **Bare values for comparison:**
    - Bare $\omega_{{transmon}}$ = {bare_w_tmon:.6f} GHz, $\alpha$ = {bare_alpha:.6f} GHz
    - Bare $\omega_{{cavity}}$ = {bare_w_cav:.6f} GHz, K = 0 (harmonic)

    ### Dressed Coherence Estimates
    - **Transmon-like:** $T_1$ = {T1_d_t:.1f} µs, $T_2$ = {T2_d_t:.1f} µs
    - **Cavity-like:** $T_1$ = {T1_d_c:.1f} µs, $T_2$ = {T2_d_c:.1f} µs
    - **Estimated BS Fidelity (2µs):** {bs_fid:.4f}
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
