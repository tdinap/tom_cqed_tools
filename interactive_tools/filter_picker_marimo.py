import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def imports():
    """CELL 1: IMPORTS + the physics module next door."""
    import sys
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go

    try:
        notebook_dir = Path(mo.notebook_dir())
    except Exception:
        notebook_dir = Path(__file__).parent
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    import sideband_bands as sb

    return Path, go, mo, notebook_dir, sb


@app.cell
def intro(mo):
    mo.md(
        """
        <style>
        /* marimo's default heading face is Lora; this notebook uses the UI sans.
           Scoped to this notebook, so the others keep whatever they have.
           marimo's own .markdown rule outranks a bare h1, hence !important. */
        .markdown h1, .markdown h2, .markdown h3, .markdown h4,
        h1, h2, h3, h4 {
            font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif !important;
        }
        </style>

        # Drive-line filter picker

        A drive at $f_d$ dresses every mode, so line noise at $|f_m - n f_d|$ is
        converted straight into that mode.  Those frequencies, and the bare mode
        frequencies, are what the filter has to **reject** while still **passing**
        $f_d$.

        Edit the two boxes, pick which drives you actually run, and the bottom
        table is the spec to shop with.
        """
    )
    return


@app.cell
def source_ui(mo, notebook_dir):
    """CELL 2: where the defaults come from."""
    default_cfg = (notebook_dir.parent / "examples" / "supporting files"
                   / "qubit_params_0p358Phi0.py")
    params_file_ui = mo.ui.text(
        value=str(default_cfg), label="Params file (optional):", full_width=True
    )
    snail_ui = mo.ui.text(value="SNAIL, 3.5", label="Coupler mode(s), comma-separated:")
    mo.vstack([params_file_ui, snail_ui])
    return params_file_ui, snail_ui


@app.cell
def load_defaults(Path, params_file_ui, sb, snail_ui):
    """CELL 3: turn the params file into starting text for the two boxes."""
    load_error = ""
    modes_text = "\n".join(
        f"{n}, {f / sb.GHZ:.6f}"
        for n, f in sb.parse_list(snail_ui.value, prefix="coupler")
    )
    drives_text = ""
    path = Path(params_file_ui.value.strip('"'))
    if path.is_file():
        try:
            _p = sb.load_params(path)
            modes_text += "\n" + "\n".join(
                f"{m.name}, {m.freq / sb.GHZ:.6f}" for m in sb.modes_from_params(_p)
            )
            drives_text = "\n".join(
                f"{d.name}, {d.freq / sb.GHZ:.6f}" for d in sb.drives_from_params(_p)
            )
        except Exception as exc:
            load_error = f"could not read {path.name}: {type(exc).__name__}: {exc}"
    elif params_file_ui.value.strip():
        load_error = f"no such file: {path}"
    return drives_text, load_error, modes_text


@app.cell
def editors(drives_text, load_error, mo, modes_text):
    """CELL 4: the editable inputs.  Re-created when the params file changes."""
    modes_box = mo.ui.text_area(
        value=modes_text, label="Modes to protect (name, GHz)", rows=9, full_width=True
    )
    drives_box = mo.ui.text_area(
        value=drives_text, label="Drive frequencies (name, GHz)", rows=9, full_width=True
    )
    mo.vstack(
        ([mo.md(f"**{load_error}**")] if load_error else [])
        + [mo.hstack([modes_box, drives_box], widths="equal", gap=1)]
    )
    return drives_box, modes_box


@app.cell
def knobs(mo):
    """CELL 5: the knobs that change the answer."""
    order_ui = mo.ui.slider(
        start=1, stop=3, step=1, value=1, label="Include sidebands up to $n$:",
        include_input=True,
    )
    guard_mode_ui = mo.ui.radio(
        options=["absolute", "% of centre"], value="absolute", inline=True,
        label="Guard band:",
    )
    guard_abs_ui = mo.ui.slider(
        start=10, stop=1000, step=10, value=200, label="guard (MHz)", include_input=True
    )
    guard_pct_ui = mo.ui.slider(
        start=1, stop=40, step=1, value=10, label="guard (% of centre)", include_input=True
    )
    margin_ui = mo.ui.slider(
        start=0, stop=200, step=5, value=20, label="Margin around each drive (MHz):",
        include_input=True,
    )
    mo.vstack([
        mo.hstack([order_ui, margin_ui], justify="start", gap=2),
        mo.hstack([guard_mode_ui, guard_abs_ui, guard_pct_ui], justify="start", gap=2),
    ])
    return guard_abs_ui, guard_mode_ui, guard_pct_ui, margin_ui, order_ui


@app.cell
def parse_inputs(drives_box, mo, modes_box, sb):
    """CELL 6: text -> objects, and the drive selector."""
    parse_error = ""
    try:
        all_modes = [sb.Mode(n, f) for n, f in sb.parse_list(modes_box.value, "mode")]
        all_drives = [sb.Drive(n, f) for n, f in sb.parse_list(drives_box.value, "drive")]
    except ValueError as exc:
        all_modes, all_drives, parse_error = [], [], f"could not parse: {exc}"

    drive_labels = {f"{d.name}  ({d.freq / sb.GHZ:.4f} GHz)": d for d in all_drives}
    pick_ui = mo.ui.multiselect(
        options=list(drive_labels), value=list(drive_labels),
        label="Drives in play:", full_width=True,
    )
    mo.vstack(([mo.md(f"**{parse_error}**")] if parse_error else []) + [pick_ui])
    return all_modes, drive_labels, parse_error, pick_ui


@app.cell
def compute(
    all_modes, drive_labels, guard_abs_ui, guard_mode_ui, guard_pct_ui,
    margin_ui, order_ui, pick_ui, sb,
):
    """CELL 7: the whole calculation."""
    drives = [drive_labels[k] for k in pick_ui.value]
    if guard_mode_ui.value == "absolute":
        guard_fn = lambda centre: guard_abs_ui.value * sb.MHZ  # noqa: E731
    else:
        guard_fn = lambda centre: centre * guard_pct_ui.value / 100.0  # noqa: E731

    analysed = sb.analyse(all_modes, drives, max_n=int(order_ui.value))
    bands, impossible = sb.plan_bands(
        analysed, guard_fn=guard_fn, margin=margin_ui.value * sb.MHZ
    )
    band_of = {d.name: i for i, b in enumerate(bands) for d in b.drives}
    return analysed, band_of, bands, guard_fn, impossible


@app.cell
def figure(analysed, band_of, bands, go, guard_fn, sb):
    """CELL 8: one row per drive; shaded columns are the filters to buy."""
    DRIVE, SPUR, BAND = "#2a78d6", "#e34948", "#2a78d6"
    INK, MUTED = "#0b0b0b", "#898781"

    rows = sorted(analysed, key=lambda a: a.drive.freq)
    # Anything above the lowest mode is already rejected by whatever rejects that
    # mode, so it only squashes the axis.  Clip, and say so in the subtitle.
    x_hi = max(
        max((a.drive.freq for a in rows), default=3 * sb.GHZ) * 1.12,
        min((s.freq for a in rows for s in a.spurs if s.n == 0),
            default=3.5 * sb.GHZ) * 1.06,
    ) / sb.GHZ
    ylab = {
        a.drive.name: (f"{a.drive.name}" if a.drive.name in band_of
                       else f"✗ {a.drive.name}")
        for a in rows
    }
    fig = go.Figure()

    for _i, _band in enumerate(bands):
        _guard = guard_fn(_band.centre)
        _lo_allowed, _hi_allowed = _band.allowed(_guard)
        _hi_allowed = min(_hi_allowed, _band.pass_max + 4 * _guard)
        fig.add_vrect(
            x0=_lo_allowed / sb.GHZ, x1=_hi_allowed / sb.GHZ,
            fillcolor=BAND, opacity=0.06, line_width=0, layer="below",
        )
        fig.add_vrect(
            x0=_band.pass_min / sb.GHZ, x1=_band.pass_max / sb.GHZ,
            fillcolor=BAND, opacity=0.16, line_width=1, line_dash="dot",
            line_color=BAND, layer="below",
            annotation_text=f"F{_i + 1}", annotation_position="top left",
            annotation_font_size=11, annotation_font_color=INK,
        )

    for _a in rows:
        _name = ylab[_a.drive.name]
        fig.add_trace(go.Scatter(
            x=[s.freq / sb.GHZ for s in _a.spurs], y=[_name] * len(_a.spurs),
            mode="markers", marker=dict(symbol="x", size=8, color=SPUR),
            name="must reject", legendgroup="spur",
            showlegend=_a is rows[0],
            text=[s.label for s in _a.spurs],
            hovertemplate="%{text}<br>%{x:.4f} GHz<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[_a.drive.freq / sb.GHZ], y=[_name], mode="markers",
            marker=dict(symbol="triangle-up", size=13, color=DRIVE,
                        line=dict(width=2, color="#fcfcfb")),
            name="drive", legendgroup="drive", showlegend=_a is rows[0],
            hovertemplate=f"{_a.drive.name}<br>%{{x:.4f}} GHz<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="square", size=12, color=BAND, opacity=0.3),
        name="filter passband",
    ))
    fig.update_layout(
        height=max(320, 42 * len(rows) + 150),
        template="plotly_white",
        title=dict(
            text="Drive tones, the noise frequencies each one opens up, "
                 "and the filters that cover them<br>"
                 f"<sup>Modes above {x_hi:.1f} GHz are off-scale: anything that "
                 "rejects the nearest one rejects those too.</sup>",
        ),
        xaxis_title="Frequency (GHz)",
        yaxis_title=None,
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=20, t=90, b=50),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="#e1e0d9", zeroline=False, range=[0, x_hi])
    fig.update_yaxes(gridcolor="#e1e0d9", tickfont=dict(color=MUTED))
    fig
    return (fig,)


@app.cell
def spec_table(bands, guard_fn, impossible, mo, sb):
    """CELL 9: the thing you actually shop with."""
    def lo_edge(value, empty="DC"):
        return empty if value <= 0 else f"{value / sb.GHZ:.3f}"

    def hi_edge(value, empty="none"):
        return empty if value == float("inf") else f"{value / sb.GHZ:.3f}"

    lines = ["| # | shape | must pass (GHz) | may extend to (GHz) | reject by (GHz) "
             "| guard lo / hi (MHz) | drives |",
             "|---|---|---|---|---|---|---|"]
    for _i, _b in enumerate(bands):
        _guard = guard_fn(_b.centre)
        _lo_allowed, _hi_allowed = _b.allowed(_guard)
        lines.append(
            f"| F{_i + 1} | {_b.shape} "
            f"| {_b.pass_min / sb.GHZ:.3f} – {_b.pass_max / sb.GHZ:.3f} "
            f"| {lo_edge(_lo_allowed)} – {hi_edge(_hi_allowed)} "
            f"| {lo_edge(_b.stop_lo, '—')} / {hi_edge(_b.stop_hi, '—')} "
            f"| {_b.guard_lo / sb.MHZ:.0f} / "
            f"{'∞' if _b.guard_hi == float('inf') else f'{_b.guard_hi / sb.MHZ:.0f}'} "
            f"| {', '.join(d.name for d in _b.drives)} |"
        )

    bad = ["", f"### No filter at this guard band ({len(impossible)})", ""]
    if impossible:
        bad += ["| drive | GHz | blocked below by | blocked above by |", "|---|---|---|---|"]
        for _a in impossible:
            _lo = f"{_a.lo_spur.label} @ {_a.stop_lo / sb.GHZ:.4f}" if _a.lo_spur else "-"
            _hi = f"{_a.hi_spur.label} @ {_a.stop_hi / sb.GHZ:.4f}" if _a.hi_spur else "-"
            bad.append(f"| {_a.drive.name} | {_a.drive.freq / sb.GHZ:.4f} | {_lo} | {_hi} |")
    else:
        bad.append("Every selected drive is covered.")

    mo.md("\n".join([f"### Filters to buy ({len(bands)})", ""] + lines + bad))
    return


if __name__ == "__main__":
    app.run()
