import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # Time-Dependent Detuning Toy (2-Level System)

    This toy simulates a purely coherent 2-level system (Qubit + 1 Storage Mode) under a sideband drive with an exponentially sweeping detuning, representing the flux ramp.

    Play with the sliders to see if a purely coherent 2-level model can produce the signature features of the data (large initial drop + low-contrast steady state tail oscillating around ~0.3-0.5).
    """)
    return


@app.cell
def _(mo):
    g_slider = mo.ui.slider(start=0.01, stop=2.0, step=0.01, value=0.25, label="$g$ (MHz)", include_input=True)
    d0_slider = mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=-4.0, label="$\\Delta_0$ Initial Detuning (MHz)", include_input=True)
    dstat_slider = mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=0.0, label="$\\Delta_{stat}$ Final Detuning (MHz)", include_input=True)
    tau_slider = mo.ui.slider(start=0.01, stop=1.0, step=0.01, value=0.15, label="$\\tau$ Settling Time ($\\mu$s)", include_input=True)
    return d0_slider, dstat_slider, g_slider, tau_slider


@app.cell
def _(d0_slider, dstat_slider, g_slider, mo, np, plt, qutip, tau_slider):
    g = g_slider.value
    d0 = d0_slider.value
    dstat = dstat_slider.value
    tau = tau_slider.value
    dshift = d0 - dstat

    t_arr = np.linspace(0, 5.0, 500)

    # 2-Level Hamiltonian
    N = 2
    a = qutip.destroy(N)
    I = qutip.qeye(N)

    bob = qutip.tensor(a, I)
    si = qutip.tensor(I, a)
    n_bob = bob.dag() * bob
    n_si = si.dag() * si

    psi0 = qutip.tensor(qutip.basis(N, 1), qutip.basis(N, 0))

    H_bs = 2 * np.pi * g * (si.dag() * bob + si * bob.dag())

    def H_det_coeff(t, args):
        return 2 * np.pi * (dstat + dshift * np.exp(-t / tau))

    H = [H_bs, [n_si, H_det_coeff]]

    result = qutip.sesolve(H, psi0, t_arr, e_ops=[n_bob])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_arr, result.expect[0], 'b-', lw=2, label="Qubit $P_e$")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time ($\\mu$s)")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Coherent 2-Level Detuning Sweep")

    mo.vstack([
        g_slider,
        d0_slider,
        dstat_slider,
        tau_slider,
        fig
    ])
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import qutip

    return mo, np, plt, qutip


if __name__ == "__main__":
    app.run()
