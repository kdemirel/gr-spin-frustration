"""Shared conventions of the PRL data figures: fonts, colours, sizes, data access.

Every figure script imports this module, parses the common command line
(``--results-root``, ``--out``, ``--entry-map``) and reads only the CSV/JSON
outputs of ``grspin analyze`` / ``grspin compare`` below the results root:

    <root>/<ENTRY>/analysis/*.csv, summary.json        per-run analyses
    <root>/_comparisons/<CMP>/*.csv                     comparisons

``--entry-map KEY=ID`` replaces the directory name of an entry (or of a
comparison) by another one, e.g. ``--entry-map M=S3_N5000_periodic`` while the
production run of ``M`` is not finished.  Missing files raise
:class:`FigureDataError` with the path that was looked for.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ----------------------------------------------------------------------------
# Geometry and typography (PRL)
# ----------------------------------------------------------------------------
SINGLE = 3.375  # column width in inches
DOUBLE = 7.0  # text width in inches
FONTSIZE = 8.0
SMALL = 7.0  # legends, tick labels of secondary axes
DPI = 200

# Okabe-Ito palette (as in fig1_schematic.py) plus greys
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#8A8A8A"
LIGHT = "#C8C8C8"
BLACK = "black"

XLABEL = r"$r/\bar{r}_{\mathrm{nn}}$"
R_NN = 0.55396  # mean nearest-neighbour distance of the Poisson process at rho = 1

LW = 1.0  # curves
MS = 3.0  # marker size (points)
ELW = 0.6  # error-bar line width

DASHED = (0, (3.0, 1.6))
DOTTED = (0, (1.0, 1.4))
DASHDOT = (0, (3.0, 1.2, 1.0, 1.2))


class FigureDataError(RuntimeError):
    """A required analysis output is missing or malformed."""


def setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FONTSIZE,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "legend.fontsize": SMALL,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "lines.linewidth": LW,
            "lines.markersize": MS,
            "lines.markeredgewidth": 0.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.columnspacing": 0.8,
            "axes.unicode_minus": True,
        }
    )


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def script_dir() -> Path:
    return Path(sys.modules["__main__"].__file__).resolve().parent  # type: ignore[arg-type]


def default_results_root() -> Path:
    return (script_dir() / "../../results").resolve()


def default_out_dir() -> Path:
    return (script_dir() / "../../../spin_spin_paper_for_PRL/figures").resolve()


def make_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="root with <ENTRY>/analysis and _comparisons/<CMP> (default: <repo>/results)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: ../../../spin_spin_paper_for_PRL/figures relative to the script)",
    )
    p.add_argument(
        "--entry-map",
        nargs="*",
        default=[],
        metavar="KEY=ID",
        help="replace entry/comparison KEY by directory ID (e.g. M=S3_N5000_periodic)",
    )
    p.add_argument("--dpi", type=int, default=DPI, help="PNG resolution (default: 200)")
    return p


def parse_entry_map(items) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--entry-map expects KEY=ID, got {item!r}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


class Root:
    """Resolves entry ids (through the entry map) to analysis files below the results root."""

    def __init__(self, results_root: Path | None, entry_map: dict[str, str] | None = None):
        self.path = Path(results_root).expanduser().resolve() if results_root else default_results_root()
        self.map = dict(entry_map or {})
        if not self.path.is_dir():
            raise FigureDataError(f"results root {self.path} does not exist")

    @classmethod
    def from_args(cls, args) -> Root:
        return cls(args.results_root, parse_entry_map(args.entry_map))

    def resolve(self, key: str) -> str:
        return self.map.get(key, key)

    def label_suffix(self, key: str) -> str:
        """' [stand-in: ID]' when the entry is mapped, else ''.  Used in the console report."""
        return f" [stand-in: {self.map[key]}]" if key in self.map else ""

    # per-run analyses ----------------------------------------------------
    def analysis_dir(self, entry: str) -> Path:
        return self.path / self.resolve(entry) / "analysis"

    def analysis_file(self, entry: str, name: str, required: bool = True) -> Path | None:
        f = self.analysis_dir(entry) / name
        if not f.is_file():
            if required:
                raise FigureDataError(
                    f"entry {entry!r} (directory {self.resolve(entry)!r}): missing {f}\n"
                    f"  run: grspin analyze <results>/{self.resolve(entry)} --out {self.analysis_dir(entry)}"
                )
            return None
        return f

    def has_entry(self, entry: str, name: str = "ensemble_table.csv") -> bool:
        return (self.analysis_dir(entry) / name).is_file()

    def table(self, entry: str) -> pd.DataFrame:
        """``ensemble_table.csv`` of an entry with the derived columns of B."""
        df = pd.read_csv(self.analysis_file(entry, "ensemble_table.csv"))
        for col in ("x_center", "x_mean", "mean_C_final", "sem_C_final"):
            if col not in df.columns:
                raise FigureDataError(f"{entry}: ensemble_table.csv has no column {col!r}")
        return add_B(df)

    def summary(self, entry: str) -> dict:
        return json.loads(self.analysis_file(entry, "summary.json").read_text())

    def landmarks(self, entry: str) -> dict[str, tuple[float, float]]:
        """{name: (value, err)} from summary.json (NaN where undefined)."""
        lm = self.summary(entry).get("landmarks", {}) or {}
        out = {}
        for k, v in lm.items():
            if isinstance(v, dict):
                out[k] = (_nan(v.get("value")), _nan(v.get("err")))
        return out

    def csv(self, entry: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.analysis_file(entry, name))

    # comparisons -----------------------------------------------------------
    def comparison_dir(self, cmp: str) -> Path:
        return self.path / "_comparisons" / self.resolve(cmp)

    def comparison_file(self, cmp: str, name: str, required: bool = True) -> Path | None:
        f = self.comparison_dir(cmp) / name
        if not f.is_file():
            if required:
                raise FigureDataError(
                    f"comparison {cmp!r}: missing {f}\n  run: grspin plan analyze {cmp}  (or grspin compare ...)"
                )
            return None
        return f

    def paired_file(self, cmp: str, obs: str, label: str, reference: str, required: bool = True) -> Path | None:
        """``paired_<obs>_<label>_minus_<reference>.csv`` (or ``unpaired_*``) of a comparison."""
        d = self.comparison_dir(cmp)
        stem = f"{obs}_{safe_name(label)}_minus_{safe_name(reference)}.csv"
        for prefix in ("paired_", "unpaired_"):
            f = d / (prefix + stem)
            if f.is_file():
                return f
        if required:
            raise FigureDataError(
                f"comparison {cmp!r}: missing {d / ('paired_' + stem)}\n"
                f"  available: {sorted(p.name for p in d.glob(f'*{obs}_*minus*.csv')) if d.is_dir() else 'no directory'}"
            )
        return None


def safe_name(label: str) -> str:
    """File-name version of a run label (same rule as ``grspin.report.safe_name``)."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def add_B(df: pd.DataFrame) -> pd.DataFrame:
    """B = C - 3P with SEM in quadrature (mean_B_* columns of newer analyses are kept if present)."""
    for state in ("final", "initial"):
        if f"mean_B_{state}" not in df.columns and f"mean_P_{state}" in df.columns:
            df[f"mean_B_{state}"] = df[f"mean_C_{state}"] - 3.0 * df[f"mean_P_{state}"]
            df[f"sem_B_{state}"] = np.hypot(df[f"sem_C_{state}"], 3.0 * df[f"sem_P_{state}"])
    return df


def series(df: pd.DataFrame, obs: str, state: str = "final", x: str = "x_center"):
    """(x, mean, sem) arrays of an observable, rows with a defined mean only."""
    m, s = f"mean_{obs}_{state}", f"sem_{obs}_{state}"
    if m not in df.columns:
        raise FigureDataError(f"ensemble table has no column {m!r}")
    ok = np.isfinite(df[m].to_numpy(float))
    sem = df[s].to_numpy(float) if s in df.columns else np.full(len(df), np.nan)
    return df[x].to_numpy(float)[ok], df[m].to_numpy(float)[ok], sem[ok]


def dimer_fraction(x) -> np.ndarray:
    """f_m(r) = exp(-9 pi rho r^3 / 4) with rho = 1 and r = x r_nn."""
    x = np.asarray(x, float)
    return np.exp(-9.0 * math.pi * (x * R_NN) ** 3 / 4.0)


def isolated_sphere(x) -> np.ndarray:
    """exp(-4 pi rho r^3 / 3): the one-sided (isolated-sphere) estimate."""
    x = np.asarray(x, float)
    return np.exp(-4.0 * math.pi * (x * R_NN) ** 3 / 3.0)


def interp_at(x, y, x0: float) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.interp(x0, x[ok], y[ok]))


def fmt(v: float, err: float | None = None, digits: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return "undefined"
    if err is None or not np.isfinite(err) or err <= 0:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} ± {err:.{digits}f}"


def report(title: str, items: dict | None = None, **kw) -> None:
    """Print numbers read off the data so that the captions can be checked."""
    print(f"[{title}]")
    allitems = dict(items or {})
    allitems.update(kw)
    for k, v in allitems.items():
        print(f"  {k}: {v}")


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def curve(ax, x, y, sem=None, color=BLACK, marker="o", ls="-", lw=LW, ms=MS, label=None, zorder=3, **kw):
    """Line with markers; error bars are added at save time only where larger than the marker.

    Pass ``marker=None`` for a plain line.  Error bars with ``sem`` NaN are skipped.
    """
    (line,) = ax.plot(x, y, color=color, marker=marker or "None", ls=ls, lw=lw, ms=ms, label=label, zorder=zorder, **kw)
    if sem is not None:
        pending = ax.__dict__.setdefault("_pending_errorbars", [])
        pending.append((np.asarray(x, float), np.asarray(y, float), np.asarray(sem, float), color, ms if marker else 0.0, zorder))
    return line


def _flush_errorbars(fig) -> None:
    """Draw the deferred error bars where the half-length exceeds the marker radius."""
    fig.canvas.draw()  # final layout and limits
    for ax in fig.axes:
        pending = ax.__dict__.pop("_pending_errorbars", None)
        if not pending:
            continue
        bbox = ax.get_window_extent()
        y0, y1 = ax.get_ylim()
        if ax.get_yscale() == "log":
            pts_per_decade = bbox.height / max(np.log10(y1 / y0), 1e-9)
        else:
            pts_per_unit = bbox.height / max(abs(y1 - y0), 1e-12) * 72.0 / fig.dpi
        for x, y, sem, color, ms, zorder in pending:
            ok = np.isfinite(sem) & np.isfinite(y) & (sem > 0)
            if ax.get_yscale() == "log":
                # marker radius in decades
                radius_dec = (ms / 2.0) / (pts_per_decade * 72.0 / fig.dpi) if ms else 0.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    ok &= np.log10((y + sem) / np.maximum(y - sem, 1e-300)) / 2.0 > radius_dec
            else:
                radius = (ms / 2.0) / pts_per_unit if ms else 0.0  # marker radius in data units
                ok &= sem > radius
            if ok.any():
                ax.errorbar(
                    x[ok], y[ok], yerr=sem[ok], fmt="none", ecolor=color, elinewidth=ELW, capsize=0, zorder=zorder - 0.5
                )


def zero_line(ax, y=0.0, **kw):
    kw.setdefault("color", GREY)
    kw.setdefault("lw", 0.5)
    kw.setdefault("zorder", 1)
    return ax.axhline(y, **kw)


def panel_label(ax, text: str, x: float = 0.0, y: float = 1.0, inside: bool = False, **kw):
    """Bold panel label.  Default: just above the top-left corner of the axes (outside the frame);
    ``inside=True`` puts it inside at the top left."""
    if inside:
        kw.setdefault("ha", "left")
        kw.setdefault("va", "top")
        return ax.text(0.03, 0.96, text, transform=ax.transAxes, fontweight="bold", fontsize=FONTSIZE, zorder=10, **kw)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "bottom")
    return ax.annotate(
        text,
        (x, y),
        xycoords="axes fraction",
        xytext=(0, 1.5),
        textcoords="offset points",
        fontweight="bold",
        fontsize=FONTSIZE,
        zorder=10,
        annotation_clip=False,
        **kw,
    )


def legend(ax, **kw):
    kw.setdefault("fontsize", SMALL)
    kw.setdefault("frameon", False)
    return ax.legend(**kw)


def xaxis_r(ax, xmax: float = 4.0, label: bool = True):
    ax.set_xlim(0, xmax)
    if label:
        ax.set_xlabel(XLABEL)


def finish(fig, outdir: Path | None, stem: str, dpi: int = DPI) -> tuple[Path, Path]:
    """Flush the deferred error bars and write ``<stem>.pdf`` and ``<stem>.png``."""
    out = Path(outdir).expanduser().resolve() if outdir else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    _flush_errorbars(fig)
    pdf, png = out / f"{stem}.pdf", out / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")
    return pdf, png


def run_main(build, description: str, stem: str):
    """Common ``main`` of the figure scripts: parse, build (returns a figure), save."""
    setup_rc()
    args = make_parser(description).parse_args()
    try:
        root = Root.from_args(args)
        fig = build(root, args)
    except FigureDataError as exc:
        print(f"{stem}: cannot draw the figure\n{exc}", file=sys.stderr)
        sys.exit(2)
    finish(fig, args.out, stem, dpi=args.dpi)
