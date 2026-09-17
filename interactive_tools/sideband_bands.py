"""Band-pass filter specs for parametrically driven couplers.

A drive at f_d on a mixing element dresses every mode m, so line noise at
|f_m - n*f_d| gets converted into a real excitation of m.  Those frequencies,
plus the bare mode frequencies, are what a drive-line filter has to reject
while still passing f_d.

This module answers one question: what passband do I have to buy?

For a group of drives sharing one filter it reports

    pass_min .. pass_max    the filter MUST pass this (the drives, plus margin)
    stop_lo, stop_hi        the nearest frequency that MUST be rejected, below
                            and above; the gap between a pass edge and its stop
                            edge is the rolloff you have to pay for

`guard_fn` sets how much room the filter needs between passing and rejecting --
a real part does not reject a spur sitting 20 MHz outside its corner.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path

GHZ = 1e9
MHZ = 1e6


@dataclass(frozen=True)
class Mode:
    name: str
    freq: float


@dataclass(frozen=True)
class Drive:
    name: str
    freq: float


@dataclass(frozen=True)
class Spur:
    """A line frequency that must be rejected while this drive is on."""
    freq: float
    mode: str
    n: int          # 0 = the bare mode frequency, 1 = f_m - f_d, ...

    @property
    def label(self) -> str:
        if self.n == 0:
            return self.mode
        if self.n == 1:
            return f"{self.mode} - f_d"
        return f"{self.mode} - {self.n}*f_d"


def spurs_for(modes: list[Mode], drive: Drive, *, max_n: int = 1,
              f_min: float = 10 * MHZ) -> list[Spur]:
    """Every frequency the filter must reject while `drive` is on."""
    out = []
    for mode in modes:
        for n in range(0, max_n + 1):
            freq = abs(mode.freq - n * drive.freq)
            if freq >= f_min:
                out.append(Spur(freq, mode.name, n))
    return sorted(out, key=lambda s: s.freq)


@dataclass
class DriveSpurs:
    drive: Drive
    spurs: list[Spur]

    @property
    def below(self) -> list[Spur]:
        return [s for s in self.spurs if s.freq < self.drive.freq]

    @property
    def above(self) -> list[Spur]:
        return [s for s in self.spurs if s.freq > self.drive.freq]

    @property
    def stop_lo(self) -> float:
        return max((s.freq for s in self.below), default=0.0)

    @property
    def stop_hi(self) -> float:
        return min((s.freq for s in self.above), default=math.inf)

    @property
    def lo_spur(self) -> Spur | None:
        return max(self.below, key=lambda s: s.freq) if self.below else None

    @property
    def hi_spur(self) -> Spur | None:
        return min(self.above, key=lambda s: s.freq) if self.above else None


def analyse(modes: list[Mode], drives: list[Drive], *, max_n: int = 1) -> list[DriveSpurs]:
    return [DriveSpurs(d, spurs_for(modes, d, max_n=max_n)) for d in drives]


@dataclass
class FilterBand:
    """One filter to buy, and the drives it serves."""
    drives: list[Drive]
    pass_min: float
    pass_max: float
    stop_lo: float
    stop_hi: float
    lo_spur: Spur | None = None
    hi_spur: Spur | None = None

    @property
    def guard_lo(self) -> float:
        return self.pass_min - self.stop_lo

    @property
    def guard_hi(self) -> float:
        return self.stop_hi - self.pass_max

    @property
    def centre(self) -> float:
        return math.sqrt(self.pass_min * self.pass_max)

    @property
    def frac_bw(self) -> float:
        return (self.pass_max - self.pass_min) / self.centre

    @property
    def shape(self) -> str:
        """Nothing to reject below means a low-pass will do, and it is cheaper."""
        if self.stop_lo <= 0:
            return "lowpass" if math.isfinite(self.stop_hi) else "none needed"
        return "highpass" if not math.isfinite(self.stop_hi) else "bandpass"

    def allowed(self, guard: float) -> tuple[float, float]:
        """How wide the passband is allowed to be, given the guard band."""
        lo = self.stop_lo + guard if self.stop_lo > 0 else 0.0
        hi = self.stop_hi - guard if math.isfinite(self.stop_hi) else math.inf
        return lo, hi

    def spec(self) -> str:
        hi = "none" if not math.isfinite(self.stop_hi) else f"{self.stop_hi / GHZ:.3f}"
        return (f"{self.shape}: pass {self.pass_min / GHZ:.3f}-{self.pass_max / GHZ:.3f} GHz, "
                f"reject by {self.stop_lo / GHZ:.3f} and {hi} GHz")


def _group_band(items: list[DriveSpurs], margin: float) -> FilterBand:
    lo_item = max(items, key=lambda a: a.stop_lo)
    hi_item = min(items, key=lambda a: a.stop_hi)
    return FilterBand(
        drives=[a.drive for a in items],
        pass_min=min(a.drive.freq for a in items) - margin,
        pass_max=max(a.drive.freq for a in items) + margin,
        stop_lo=lo_item.stop_lo,
        stop_hi=hi_item.stop_hi,
        lo_spur=lo_item.lo_spur,
        hi_spur=hi_item.hi_spur,
    )


def _fits(band: FilterBand, guard_fn) -> bool:
    guard = guard_fn(band.centre)
    lo_ok = band.pass_min - band.stop_lo >= guard
    hi_ok = (not math.isfinite(band.stop_hi)) or (band.stop_hi - band.pass_max >= guard)
    return lo_ok and hi_ok and band.pass_min > 0


def plan_bands(analysed: list[DriveSpurs], *, guard_fn=lambda centre: 200 * MHZ,
               margin: float = 20 * MHZ) -> tuple[list[FilterBand], list[DriveSpurs]]:
    """Fewest filters that cover every drive.

    Exact over contiguous groupings, which is what you want on a shopping list:
    one part per stretch of the band.  Drives that no filter can serve at this
    guard band come back separately.
    """
    items = sorted(analysed, key=lambda a: a.drive.freq)
    ok = [a for a in items if _fits(_group_band([a], margin), guard_fn)]
    ok_ids = {id(a) for a in ok}
    impossible = [a for a in items if id(a) not in ok_ids]

    n = len(ok)
    best: list[float] = [0.0] * (n + 1)
    choice: list[int] = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        best[i] = math.inf
        for j in range(i, n):
            if not _fits(_group_band(ok[i:j + 1], margin), guard_fn):
                break
            if 1 + best[j + 1] < best[i]:
                best[i], choice[i] = 1 + best[j + 1], j
    bands, i = [], 0
    while i < n:
        j = choice[i]
        bands.append(_group_band(ok[i:j + 1], margin))
        i = j + 1
    return bands, impossible


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def parse_freq(text) -> float:
    """'3.5', '3.5G', '2.9e9', '1750M' -> Hz.  Bare numbers below 100 are GHz."""
    if isinstance(text, (int, float)):
        value = float(text)
        return value * GHZ if 0 < value < 100 else value
    s = str(text).strip().replace("Hz", "").replace("hz", "")
    mult = 1.0
    if s and s[-1] in "GgMmKk":
        mult = {"g": GHZ, "m": MHZ, "k": 1e3}[s[-1].lower()]
        s = s[:-1]
    value = float(s)
    return value * (GHZ if mult == 1.0 and 0 < value < 100 else mult)


def parse_list(text: str, prefix: str = "f") -> list[tuple[str, float]]:
    """One entry per line: 'name, freq' or just 'freq'.  Blank/# lines ignored."""
    out = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        name, _, rest = line.rpartition(",")
        name = name.strip() or f"{prefix}{len(out)}"
        out.append((name, parse_freq(rest.strip())))
    return out


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}


def _literal(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp):
        value = _literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_literal(node.left), _literal(node.right))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"unsupported expression on line {getattr(node, 'lineno', '?')}")


def load_params(path, func_name: str = "single_qubit_parameters") -> dict:
    """The params dict, read as a literal.  Nothing is imported or executed."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value is not None:
                    return _literal(stmt.value)
    raise ValueError(f"no literal {func_name}() found in {path}")


def drives_from_params(params: dict, keys=("bs_alice", "bs_bob"),
                       drop_placeholders: bool = True) -> list[Drive]:
    """Drive frequencies from the params arrays.

    The config marks unmeasured entries with 1e9 in both the frequency and the
    fwhm; those are dropped by default.
    """
    out = []
    for key in keys:
        freqs = params.get(f"{key}_freqs", [])
        fwhms = params.get(f"{key}_fwhms", [None] * len(freqs))
        for i, freq in enumerate(freqs):
            fwhm = fwhms[i] if i < len(fwhms) else None
            if drop_placeholders and (freq == 1e9 or (fwhm is not None and fwhm >= 1e9)):
                continue
            out.append(Drive(f"{key}[{i}]", float(freq)))
    return out


def modes_from_params(params: dict, extra: list[Mode] | None = None) -> list[Mode]:
    """Modes worth protecting that the params file already knows about."""
    named = [("transmon_ge", "qb_freq"), ("transmon_ef", "qb_ef_freq"),
             ("buf_alice", "cav_alice_freq"), ("buf_bob", "cav_bob_freq"),
             ("readout", "ro_freq")]
    out = [Mode(name, params[key]) for name, key in named if params.get(key)]
    return out + list(extra or [])
