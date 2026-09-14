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

`analyze_rabi`, `analyze_spectroscopy`, `analyze_t1`, `analyze_ramsey`,
`analyze_flattop_rabi`, and `analyze_flattop_spectroscopy` all take a
`suffix` argument that controls which file each panel loads:

- A plain string (`suffix="bs_a3_rabi"`) is used as-is for every panel —
  this is the original behavior and needs no changes to existing notebooks.
- A template is filled in per panel, so one call can sweep several files
  instead of calling the function once per mode. Available fields:

  | field | value |
  |---|---|
  | `{mode}` | that panel's entry from `modes` |
  | `{filenum}` | that panel's entry from `filenums` |
  | `{a_or_b}` | first letter of `alice_or_bob` (`"a"` / `"b"`) |
  | `{alice_or_bob}` | the full `alice_or_bob` string |

  e.g. `suffix="bs_{a_or_b}{mode}_spectroscopy"`.

All six accept the same fields, so one template can be shared across them.
A template referring to anything else raises a `ValueError` naming the bad
field and listing the valid ones. See `expand_suffix()` for details.

## Linting

This repo uses [ruff](https://docs.astral.sh/ruff/) configured to catch only
genuine bugs (undefined names, unused/duplicate imports, unused locals) —
no style or formatting rules. Run it with:

```bash
uvx ruff check src/tom_cqed_tools/analysis_utils.py
```

`uvx` runs ruff without installing anything. Worth running after any large
edit (by hand or by an AI assistant) before committing.
