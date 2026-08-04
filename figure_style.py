"""Shared figure style for the Scientific Data submission.

Nature Portfolio sizes figures at their final printed width, so both figures are
built at 183 mm (two columns) and kept under the 170 mm height limit, which
leaves room for the caption below.

Type sizes follow the Nature Portfolio range of 5-7 pt for ordinary text, with
8 pt bold for panel letters. Scientific Data additionally asks for one typeface
and one set of sizes across all figures, which is why this module exists rather
than each script setting its own.

Text is kept as text: pdf.fonttype 42 embeds TrueType outlines that remain
editable and searchable, as required, instead of converting glyphs to paths.
"""

import matplotlib as mpl

MM = 1.0 / 25.4
WIDTH_1COL = 89 * MM          # inches
WIDTH_2COL = 183 * MM
HEIGHT_MAX = 170 * MM

FS_AXIS_LABEL = 7
FS_TICK = 6
FS_LEGEND = 6
FS_ANNOT = 6.5
FS_PANEL_LETTER = 8


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_ANNOT,
        "axes.labelsize": FS_AXIS_LABEL,
        "axes.titlesize": FS_ANNOT,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_AXIS_LABEL,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def panel_letter(fig, x, y, letter):
    """Lower-case, bold, upright panel letter at figure coordinates."""
    fig.text(x, y, letter, fontsize=FS_PANEL_LETTER, fontweight="bold",
             va="top", ha="left")


def save(fig, path):
    """Write the vector formats Illustrator can edit, plus a raster preview.

    SVG is written with svg.fonttype "none", so glyphs stay as <text> elements
    that Illustrator can re-edit with the live font. The PDF embeds TrueType
    outlines (pdf.fonttype 42) rather than converting text to paths, so its text
    also remains selectable. The PNG is a separate 600 dpi preview.
    """
    for suffix, kw in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(path.with_suffix(suffix), bbox_inches="tight", **kw)
    return [path.with_suffix(s) for s in (".svg", ".pdf", ".png")]


def parse_paths(description):
    """Resolve the BIDS root and the output directory from the command line.

    The scripts are released as a standalone repository, so the data are not
    necessarily beside them. --bids_root defaults to the parent of the directory
    holding the script, which is correct when the code sits in the dataset's own
    code/ directory, and can be pointed anywhere otherwise.
    """
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--bids_root", type=Path, default=None,
                   help="root of the BIDS dataset (default: parent of this script's directory)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: <bids_root>/derivatives/technical_validation)")
    a = p.parse_args()

    root = (a.bids_root or Path(__import__("sys").argv[0]).resolve().parents[1]).resolve()
    if not (root / "dataset_description.json").exists():
        raise SystemExit(f"{root} does not look like a BIDS dataset "
                         "(no dataset_description.json); pass --bids_root")
    out = (a.out or root / "derivatives" / "technical_validation").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return root, out
