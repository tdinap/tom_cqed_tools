import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Quantum Coherence & Beam Splitter Fidelity Analyzer
    This workspace maps the operational fidelity of a beam splitter operation under localized environmental coupling. Use the sidebar controls to explore the cross-sections of your parameter space dynamically.
    """)
    return


@app.cell
def _(mo):
    # Unified slider widgets with attached, editable text boxes
    k1_slider = mo.ui.slider(
        start=0.0, stop=50.0, step=1.0, value=5.0, 
        include_input=True
    )
    kphi_slider = mo.ui.slider(
        start=0.0, stop=100.0, step=1.0, value=30.0, 
        include_input=True
    )
    tbs_slider = mo.ui.slider(
        start=0.05, stop=5.0, step=0.05, value=1.0, 
        include_input=True
    )
    return k1_slider, kphi_slider, tbs_slider


@app.cell
def _(np):
    def calc_fidelity(k1_khz, kphi_khz, tbs_us):
        # Conversion scaling: kHz * us = 1e3 * 1e-6 = 1e-3 standard dimensionless units
        product_factor = 1e-3
        term1 = np.exp(-k1_khz * tbs_us * product_factor)
        term2 = np.exp(-(k1_khz + kphi_khz) * tbs_us * product_factor)
        return 0.5 * (term1 + term2)

    return (calc_fidelity,)


@app.cell
def _(calc_fidelity, go, k1_slider, kphi_slider, mo, np, tbs_slider):
    # 1. Capture slider configurations
    k1_val = k1_slider.value
    kphi_val = kphi_slider.value
    tbs_val = tbs_slider.value

    current_fid = calc_fidelity(k1_val, kphi_val, tbs_val)

    # 2. Setup resolution arrays for 2D meshes & 1D line arrays
    k1_arr = np.linspace(0, 50, 100)
    kphi_arr = np.linspace(0, 100, 100)
    tbs_arr = np.linspace(0.01, 5, 100)

    # 3. Dedicated configuration dictionary for 2D contours (Zoomed to top 10%)
    contour_opts = dict(
        start=0.90,
        end=1.0,
        size=0.01,          # Fine-grained 1% steps
        showlines=True,
        showlabels=True,    # Inlines numeric labels directly on the lines
        coloring='heatmap',
        labelfont=dict(size=11, color='white')
    )
    line_style = dict(color='rgba(255, 255, 255, 0.3)', width=1)

    # --- 2D Plot 1: Decay vs Dephasing ---
    Z1 = np.array([[calc_fidelity(x, y, tbs_val) for x in k1_arr] for y in kphi_arr])
    fig1 = go.Figure(data=[
        go.Contour(x=k1_arr, y=kphi_arr, z=Z1, colorscale='Plasma', zmin=0.90, zmax=1.0, contours=contour_opts, line=line_style, hovertemplate='κ₁: %{x} kHz<br>κ_φ: %{y} kHz<br>Fidelity: %{z:.4f}<extra></extra>'),
        go.Scatter(x=[k1_val], y=[kphi_val], mode='markers', marker=dict(symbol='star', size=14, color='cyan', line=dict(color='black', width=1.5)), showlegend=False, hoverinfo='skip')
    ])
    fig1.update_layout(title=f'Fixed: t_BS = {tbs_val:.2f} µs', xaxis_title='κ₁ (kHz)', yaxis_title='κ_φ (kHz)', width=380, height=360, margin=dict(l=50, r=10, t=50, b=50))

    # --- 2D Plot 2: Gate Time vs Decay ---
    Z2 = np.array([[calc_fidelity(y, kphi_val, x) for x in tbs_arr] for y in k1_arr])
    fig2 = go.Figure(data=[
        go.Contour(x=tbs_arr, y=k1_arr, z=Z2, colorscale='Plasma', zmin=0.90, zmax=1.0, contours=contour_opts, line=line_style, hovertemplate='t_BS: %{x} µs<br>κ₁: %{y} kHz<br>Fidelity: %{z:.4f}<extra></extra>'),
        go.Scatter(x=[tbs_val], y=[k1_val], mode='markers', marker=dict(symbol='star', size=14, color='cyan', line=dict(color='black', width=1.5)), showlegend=False, hoverinfo='skip')
    ])
    fig2.update_layout(title=f'Fixed: κ_φ = {kphi_val:.1f} kHz', xaxis_title='t_BS (µs)', yaxis_title='κ₁ (kHz)', width=380, height=360, margin=dict(l=50, r=10, t=50, b=50))

    # --- 2D Plot 3: Gate Time vs Dephasing ---
    Z3 = np.array([[calc_fidelity(k1_val, y, x) for x in tbs_arr] for y in kphi_arr])
    fig3 = go.Figure(data=[
        go.Contour(x=tbs_arr, y=kphi_arr, z=Z3, colorscale='Plasma', zmin=0.90, zmax=1.0, contours=contour_opts, line=line_style, hovertemplate='t_BS: %{x} µs<br>κ_φ: %{y} kHz<br>Fidelity: %{z:.4f}<extra></extra>'),
        go.Scatter(x=[tbs_val], y=[kphi_val], mode='markers', marker=dict(symbol='star', size=14, color='cyan', line=dict(color='black', width=1.5)), showlegend=False, hoverinfo='skip')
    ])
    fig3.update_layout(title=f'Fixed: κ₁ = {k1_val:.1f} kHz', xaxis_title='t_BS (µs)', yaxis_title='κ_φ (kHz)', width=380, height=360, margin=dict(l=50, r=10, t=50, b=50))

    # --- 1D Cross-Sections Generation ---
    f_vs_k1 = [calc_fidelity(x, kphi_val, tbs_val) for x in k1_arr]
    f_vs_kphi = [calc_fidelity(k1_val, y, tbs_val) for y in kphi_arr]
    f_vs_tbs = [calc_fidelity(k1_val, kphi_val, t) for t in tbs_arr]

    # Shared style for the indicator trace
    marker_line_style = dict(color='red', width=1.5, dash='dash')

    # 1D Trace 1: vs Decay
    fig1_1d = go.Figure()
    fig1_1d.add_trace(go.Scatter(x=k1_arr, y=f_vs_k1, mode='lines', line=dict(color='#636EFA', width=3), name='Fidelity Profile'))
    fig1_1d.add_trace(go.Scatter(x=[k1_val, k1_val], y=[0.88, 1.01], mode='lines', line=marker_line_style, name='Current Value'))
    fig1_1d.update_layout(title="Fidelity vs Decay (κ₁)", xaxis_title="κ₁ (kHz)", yaxis_title="Fidelity", yaxis=dict(range=[0.88, 1.01]), width=380, height=280, showlegend=True, template="plotly_white", margin=dict(l=50, r=20, t=50, b=50), legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"))

    # 1D Trace 2: vs Dephasing
    fig2_1d = go.Figure()
    fig2_1d.add_trace(go.Scatter(x=kphi_arr, y=f_vs_kphi, mode='lines', line=dict(color='#EF553B', width=3), name='Fidelity Profile'))
    fig2_1d.add_trace(go.Scatter(x=[kphi_val, kphi_val], y=[0.88, 1.01], mode='lines', line=marker_line_style, name='Current Value'))
    fig2_1d.update_layout(title="Fidelity vs Dephasing (κ_φ)", xaxis_title="κ_φ (kHz)", yaxis_title="Fidelity", yaxis=dict(range=[0.88, 1.01]), width=380, height=280, showlegend=True, template="plotly_white", margin=dict(l=50, r=20, t=50, b=50), legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"))

    # 1D Trace 3: vs Gate Time
    fig3_1d = go.Figure()
    fig3_1d.add_trace(go.Scatter(x=tbs_arr, y=f_vs_tbs, mode='lines', line=dict(color='#00CC96', width=3), name='Fidelity Profile'))
    fig3_1d.add_trace(go.Scatter(x=[tbs_val, tbs_val], y=[0.88, 1.01], mode='lines', line=marker_line_style, name='Current Value'))
    fig3_1d.update_layout(title="Fidelity vs Gate Time (t_BS)", xaxis_title="t_BS (µs)", yaxis_title="Fidelity", yaxis=dict(range=[0.88, 1.01]), width=380, height=280, showlegend=True, template="plotly_white", margin=dict(l=50, r=20, t=50, b=50), legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.7)"))

    # 4. Consolidate Dashboard Panel layouts using explicit LaTeX mapping
    control_panel = mo.vstack([
        mo.md(f"### Operational Fidelity:\n# **{current_fid:.4f}**"),
        mo.md("---"),
        mo.md(r"**Decay Rate ($\kappa_1$):**"),
        k1_slider,
        mo.md(""),
        mo.md(r"**Dephasing Rate ($\kappa_\phi$):**"),
        kphi_slider,
        mo.md(""),
        mo.md(r"**Gate Pulse Time ($t_{\text{BS}}$):**"),
        tbs_slider,
    ], align="stretch").style({"min-width": "340px", "padding-right": "20px"})

    dashboard_view = mo.hstack([
        control_panel,
        mo.vstack([
            mo.md("### 2D Parameter Landscapes"),
            mo.hstack([fig1, fig2, fig3], justify="start"),
            mo.md("---"),
            mo.md("### 1D Cross-Section Profiles"),
            mo.hstack([fig1_1d, fig2_1d, fig3_1d], justify="start")
        ])
    ], align="start", justify="start")
    return (dashboard_view,)


@app.cell
def _(dashboard_view):
    # Render unified interactive dashboard interface
    dashboard_view
    return


if __name__ == "__main__":
    app.run()
