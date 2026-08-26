# tom_cqed_tools

A collection of tools for circuit QED analysis, understanding, experiment support, etc. Initially developed for work with bosonic superconducting cavity devices in the Chakram Lab at Rutgers University.

## Installation

Because the package is properly configured in `pyproject.toml`, all dependencies (including `marimo`) are installed automatically.

```bash
pip install -e .
```

## Usage

You can run some of the interactive tools using Marimo. For example, to launch the spectrum visualization:

```bash
marimo edit interactive_tools/spectrum_vis_marimo.py
```

### `analyze_*` filename suffixes

`analyze_rabi`, `analyze_spectroscopy`, `analyze_flattop_rabi`, and
`analyze_flattop_spectroscopy` all take a `suffix` argument that controls
which file each panel loads:

- A plain string (`suffix="bs_a3_rabi"`) is used as-is for every panel —
  this is the original behavior and needs no changes to existing notebooks.
- A template containing `{mode}` (`suffix="bs_a{mode}_rabi"`) is filled in
  per panel from the corresponding entry in `modes`, so one call can sweep
  several files instead of calling the function once per mode.

See the docstring on `analyze_rabi` for details.

## Linting

This repo uses [ruff](https://docs.astral.sh/ruff/) configured to catch only
genuine bugs (undefined names, unused/duplicate imports, unused locals) —
no style or formatting rules. Run it with:

```bash
uvx ruff check src/tom_cqed_tools/analysis_utils.py
```

`uvx` runs ruff without installing anything. Worth running after any large
edit (by hand or by an AI assistant) before committing.
