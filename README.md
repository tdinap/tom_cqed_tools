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
