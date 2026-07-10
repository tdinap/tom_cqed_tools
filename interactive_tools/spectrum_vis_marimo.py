import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def imports():
    """
    CELL 1: IMPORTS
    All external libraries are loaded here. In marimo, imports are available globally, 
    but variables created in other cells are strictly scoped.
    """
    import marimo as mo
    import plotly.graph_objects as go
    import numpy as np
    import importlib.util
    import os
    from itertools import combinations

    return combinations, go, importlib, mo, np, os


@app.cell
def dashboard_ui(mo):
    """
    CELL 2: THE DASHBOARD UI
    This cell defines the interactive elements. Because marimo is reactive, 
    changing any of these toggles will instantly push the new values to the 
    downstream physics and plotting cells.
    """
    mo.md("## MMC2 Spectral Map Dashboard")

    # 1. Calculation Method Dropdown
    calc_method_ui = mo.ui.dropdown(
        options=['average', 'alice', 'bob'],
        value='average',
        label='Storage Mode Source Calculation:'
    )

    # 2. Parasitic Toggle
    show_parasitics_ui = mo.ui.checkbox(
        value=True, 
        label='Calculate and show parasitic cross-terms (T-Cav, St-St, Buff-Buff)'
    )

    # 3. File Configurations (Allows easy swapping without touching code)
    params_file_ui = mo.ui.text(value='qubit_params_440_mPhi0.py', label='Params File:')
    bbq_file_ui = mo.ui.text(value='bbq_snailmon_spectrum.npz', label='BBQ Data (Optional):')

    # Render the UI
    dashboard = mo.vstack([
        mo.hstack([calc_method_ui, show_parasitics_ui], justify="start", gap=2),
        mo.hstack([params_file_ui, bbq_file_ui], justify="start", gap=2)
    ])

    return (
        bbq_file_ui,
        calc_method_ui,
        dashboard,
        params_file_ui,
        show_parasitics_ui,
    )


@app.cell
def physics_engine(
    bbq_file_ui,
    calc_method_ui,
    combinations,
    importlib,
    np,
    os,
    params_file_ui,
    show_parasitics_ui,
):
    """
    CELL 3: PHYSICS LOGIC & MODE GENERATOR
    This takes the inputs from the UI and calculates the frequencies. 
    It is fully isolated from the plotting code, making it easy to audit.
    """
    def load_params(filepath):
        if not os.path.exists(filepath):
            return None
        spec = importlib.util.spec_from_file_location("qubit_params", filepath)
        qp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qp_module)
        return qp_module.qubit_parameters['q0']

    def generate_modes(params_file, calc_method, show_parasitics, bbq_file):
        qp = load_params(params_file)
        if not qp:
            return [] # Return empty if file not found

        modes = []
        # External parasitic list from your CSVs
        parasitic_buffers = [5.657452, 6.485098, 7.595492] if show_parasitics else []

        def add_mode(name, freq, m_type):
            modes.append({'name': name, 'freq': freq, 'type': m_type})

        # --- Base Frequencies (Converted to GHz) ---
        f_alice = qp['cav_alice_freq'] / 1e9
        f_bob = qp['cav_bob_freq'] / 1e9
        f_transmon = qp['qb_freq'] / 1e9

        add_mode('Transmon GE', f_transmon, 'Transmon')
        add_mode('Transmon EF', qp['qb_ef_freq'] / 1e9, 'Transmon')
        add_mode('Readout', qp['ro_freq'] / 1e9, 'Readout')
        add_mode('Alice Buffer', f_alice, 'Buffer')
        add_mode('Bob Buffer', f_bob, 'Buffer')

        # --- Sidebands ---
        if 'sb_alice_freqs' in qp:
            for i, f in enumerate(qp['sb_alice_freqs']):
                add_mode(f'Alice Sideband {i}', f / 1e9, 'Sideband')
        if 'sb_bob_freqs' in qp:
            for i, f in enumerate(qp['sb_bob_freqs']):
                add_mode(f'Bob Sideband {i}', f / 1e9, 'Sideband')

        # --- Storage and Beamsplitters ---
        bs_alice = np.array(qp['bs_alice_freqs']) / 1e9
        bs_bob = np.array(qp['bs_bob_freqs']) / 1e9

        # Index 0 is SNAIL (subtracted). Others are Storage (added).
        signs = np.ones(len(bs_alice))
        signs[0] = -1.0 

        modes_from_alice = f_alice + (bs_alice * signs)
        modes_from_bob = f_bob + (bs_bob * signs)

        if calc_method == 'alice':
            final_storage = modes_from_alice
        elif calc_method == 'bob':
            final_storage = modes_from_bob
        else:
            min_len = min(len(modes_from_alice), len(modes_from_bob))
            final_storage = (modes_from_alice[:min_len] + modes_from_bob[:min_len]) / 2.0

        for i in range(1, len(bs_alice)):
            add_mode(f'Alice-Storage BS {i}', bs_alice[i], 'A-St BS')
        for i in range(1, len(bs_bob)):
            add_mode(f'Bob-Storage BS {i}', bs_bob[i], 'B-St BS')

        for i, f in enumerate(final_storage):
            if i > 0:
                add_mode(f'Storage Mode {i}', f, 'Storage')

        # --- BBQ SNAIL Flux Data & BS ---
        if os.path.exists(bbq_file):
            bbq_data = np.load(bbq_file)
            snail_01 = bbq_data['evals_vs_flux'][0]
            snail_02 = bbq_data['evals_vs_flux'][1]
            flux_axis = bbq_data['bbq_fluxes']

            add_mode('SNAIL mode 1', snail_01, 'SNAIL')
            add_mode('SNAIL mode 2', snail_02, 'SNAIL')
            add_mode('Alice-SNAIL BS', np.abs(snail_01 - f_alice), 'SNAIL BS')
            add_mode('Bob-SNAIL BS', np.abs(snail_01 - f_bob), 'SNAIL BS')
        else:
            flux_axis = None
            add_mode('SNAIL mode 1 (Scalar)', final_storage[0], 'SNAIL')
            add_mode('Alice-SNAIL BS (Scalar)', bs_alice[0], 'A-St BS')
            add_mode('Bob-SNAIL BS (Scalar)', bs_bob[0], 'B-St BS')

        # --- Parasitic Beamsplitters ---
        if show_parasitics:
            add_mode('Alice-Bob BS', np.abs(f_alice - f_bob), 'Alice-Bob BS')
            for i, f in enumerate(final_storage[1:]):
                add_mode(f'T-Storage {i+1} BS', np.abs(f - f_transmon), 'Parasitic BS, T-Cav')
            add_mode('T-Alice BS', np.abs(f_transmon - f_alice), 'Parasitic BS, T-Cav')
            add_mode('T-Bob BS', np.abs(f_transmon - f_bob), 'Parasitic BS, T-Cav')

            for i, (a, b) in enumerate(combinations(final_storage[1:], 2)):
                add_mode(f'St-St BS {i+1}', np.abs(b - a), 'Parasitic BS, St-St')

            for i, f in enumerate(parasitic_buffers):
                add_mode(f'Ext Buffer {i+1}', f, 'Buffer')
                add_mode(f'Ext A-Buff BS {i+1}', np.abs(f - f_alice), 'Parasitic BS, Buff-Buff')
                add_mode(f'Ext B-Buff BS {i+1}', np.abs(f - f_bob), 'Parasitic BS, Buff-Buff')
                for j, (a, b) in enumerate(combinations(parasitic_buffers, 2)):
                    add_mode(f'Ext Buff-Buff BS {j+1}', np.abs(b - a), 'Parasitic BS, Buff-Buff')

        return modes, flux_axis

    # Execute the generation using the UI values
    calculated_modes, shared_flux_axis = generate_modes(
        params_file=params_file_ui.value,
        calc_method=calc_method_ui.value,
        show_parasitics=show_parasitics_ui.value,
        bbq_file=bbq_file_ui.value
    )
    return calculated_modes, shared_flux_axis


@app.cell
def visualizer(calculated_modes, go, np, shared_flux_axis):
    """
    CELL 4: PLOTLY VISUALIZATION
    Takes the pure data generated by the physics cell and renders it.
    It automatically handles the difference between scalar frequencies and 
    flux-dependent NumPy arrays.
    """
    if not calculated_modes:
        fig_out = "File not found or missing data. Please check file paths in the UI."
    else:
        fig = go.Figure()

        # Design system matching your notebook
        TYPE_COLORS = {
            'SNAIL': '#ff3300', 'Transmon': '#c0392b', 'Readout': '#8e44ad',
            'Buffer': '#292bb9', 'Storage': '#16a085', 'Sideband': '#e6a522',
            'A-St BS': '#d35400', 'B-St BS': '#e74c3c', 'SNAIL BS': '#f39c12',
            'Alice-Bob BS': '#2c3e50', 'Parasitic BS, T-Cav': '#95a5a6',
            'Parasitic BS, Buff-Buff': '#dfe6e9', 'Parasitic BS, St-St': '#dfe6e9'
        }

        legend_groups_added = set()

        # Determine plot bounds
        if shared_flux_axis is not None:
            flux_min, flux_max = shared_flux_axis.min(), shared_flux_axis.max()
        else:
            flux_min, flux_max = -0.6, 0.6 # Fallback if no BBQ data

        for mode in calculated_modes:
            m_type = mode['type']
            color = TYPE_COLORS.get(m_type, '#7f8c8d') # Gray fallback

            first_in_group = m_type not in legend_groups_added
            if first_in_group:
                legend_groups_added.add(m_type)

            # Route for Flux-Dependent Arrays vs Scalar Lines
            if isinstance(mode['freq'], np.ndarray) and shared_flux_axis is not None:
                fig.add_trace(go.Scatter(
                    x=shared_flux_axis,
                    y=mode['freq'],
                    mode='lines',
                    line=dict(color=color, width=2.5, dash='dash'),
                    hoverinfo='x+y+name',
                    name=mode['name'],
                    legendgroup=m_type,
                    showlegend=first_in_group
                ))
            else:
                # Handle scalar frequencies
                freq_val = mode['freq'][0] if isinstance(mode['freq'], np.ndarray) else mode['freq']
                fig.add_trace(go.Scatter(
                    x=[flux_min, flux_max],
                    y=[freq_val, freq_val],
                    mode='lines+markers',
                    marker=dict(symbol='square', size=6, opacity=0.8),
                    line=dict(color=color, width=1.5, dash='15,6'),
                    hoverinfo='text',
                    text=f"{mode['name']}<br>Frequency: {freq_val:.3f} GHz",
                    name=mode['name'],
                    legendgroup=m_type,
                    showlegend=first_in_group
                ))

        fig.update_layout(
            height=750,
            title='MMC2 Dynamic Spectral Map',
            xaxis_title='Flux (Φ₀)',
            yaxis_title='Frequency (GHz)',
            template='plotly_white',
            legend=dict(
                title="Click to Toggle Layers",
                x=-0.2, y=0.98,
                bgcolor='rgba(245,245,245,0.96)',
                bordercolor='rgba(0,0,0,0.08)',
                borderwidth=1
            ),
            margin=dict(l=20, r=120)
        )
        fig_out = fig

    return (fig_out,)


@app.cell
def render_output(dashboard, fig_out, mo):
    """
    CELL 5: RENDER
    This tells marimo to draw the UI dashboard at the top, and the Plotly figure below it.
    """
    mo.vstack([dashboard, mo.md("---"), fig_out])
    return


if __name__ == "__main__":
    app.run()
