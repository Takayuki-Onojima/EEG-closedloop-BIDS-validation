"""Figure 2 - phase-targeting accuracy of the closed-loop stimulation.

Anchored on the photosensor (B), the measured luminance change on the display,
which is the dataset's primary external reference for actual visual onset. The
comparison against the two upstream triggers (Speedgoat A, stimulation PC) is
reported numerically in Table 5 and is deliberately not repeated here.

Panels
  a  circular histograms of the phase 200 ms before visual onset, one per
     target phase, with the target and the realised circular mean drawn on top
  b  phase-locking factor around visual onset, one curve per target phase

Reads the cache written by code/compare_trigger_references.py; run that first if
the .npz is missing. Sizing and type follow code/figure_style.py.
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_style as st

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}
REFERENCE_MS = -200.0
ANCHOR = "B"
HIST_COLOUR = "#9EBCDA"
MEAN_COLOUR = "#E45756"


def label(cond):
    return f"{math.degrees(TARGET[cond]):.0f}°"


def circmean(a):
    return math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))


def rlen(a):
    return math.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def load(cache: Path):
    if not cache.exists():
        raise SystemExit(f"missing {cache}\nrun code/compare_trigger_references.py first")
    z = np.load(cache, allow_pickle=True)
    return z["phases"].item(), z["offsets"]


def plot(path: Path, phases, offsets):
    fig = plt.figure(figsize=(st.WIDTH_2COL, 136 * st.MM))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.90],
                          hspace=0.80, wspace=0.55,
                          top=0.930, bottom=0.075, left=0.075, right=0.985)

    # ---- a: realised phase, one panel per target -------------------------
    for k, cond in enumerate(range(1, 7)):
        ax = fig.add_subplot(gs[k // 3, (k % 3)*2:(k % 3)*2 + 2], projection="polar")
        a = np.asarray(phases[ANCHOR][cond][REFERENCE_MS], float)
        target = TARGET[cond]

        counts, edges = np.histogram(a, bins=36, range=(-math.pi, math.pi))
        ax.bar(edges[:-1] + np.diff(edges)/2, counts, width=np.diff(edges),
               color=HIST_COLOUR, alpha=0.85, edgecolor="none", zorder=1)
        top = counts.max() if counts.size and counts.max() > 0 else 1

        m = circmean(a)
        ax.plot([target, target], [0, top], "--", color="black", lw=1.0, zorder=5)
        ax.plot([m, m], [0, top], "-", color=MEAN_COLOUR, lw=1.2, zorder=6)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)          # counter-clockwise
        ax.set_yticklabels([])
        ax.tick_params(pad=0.5)
        ax.grid(lw=0.4, color="#CCCCCC")
        ax.spines["polar"].set_linewidth(0.6)
        ax.set_title(f"target {label(cond)}\n"
                     f"n = {a.size}, R = {rlen(a):.2f}, "
                     f"bias = {math.degrees(wrap(m - target)):+.1f}°",
                     fontsize=st.FS_ANNOT, pad=6)

    # ---- b: phase-locking factor ------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    for cond in range(1, 7):
        r = [rlen(np.asarray(phases[ANCHOR][cond][o], float))
             if len(phases[ANCHOR][cond][o]) else np.nan for o in offsets]
        ax.plot(offsets, r, lw=0.9, label=f"target {label(cond)}")
    ax.axvline(REFERENCE_MS, color="black", ls="--", lw=0.9)
    ax.axvline(0, color="grey", ls="--", lw=0.9)
    ax.set_xlabel("time relative to photosensor-confirmed visual onset (ms)")
    ax.set_ylabel("phase-locking factor")
    ax.set_xlim(offsets[0], offsets[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=st.FS_LEGEND, loc="upper right",
              bbox_to_anchor=(0.998, 0.995), columnspacing=1.0,
              handlelength=1.3, labelspacing=0.35, borderaxespad=0.2)
    ax.text(REFERENCE_MS, 1.02, "prestimulus target", ha="center",
            fontsize=st.FS_TICK)
    ax.text(0, 1.02, "visual onset", ha="center", fontsize=st.FS_TICK, color="grey")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # the legend belongs to panel a; place it in the clear band between the
    # bottom of the polar block (whose 270 deg labels hang below the axes) and
    # the top of panel b
    polar_bottom = min(fig.axes[i].get_position().y0 for i in range(3, 6))
    panel_b_top = ax.get_position().y1
    handles = [plt.Line2D([], [], color="black", ls="--", lw=1.0, label="target phase"),
               plt.Line2D([], [], color=MEAN_COLOUR, lw=1.2,
                          label="realised circular mean")]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, polar_bottom - 0.052),
               ncol=2, fontsize=st.FS_LEGEND)

    st.panel_letter(fig, 0.010, 0.995, "a")
    st.panel_letter(fig, 0.010, panel_b_top + 0.045, "b")

    for f in st.save(fig, path):
        print(f"wrote {f}")
    plt.close(fig)


def write_summary(path: Path, phases):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_deg", "n", "circular_mean_deg", "resultant_length", "bias_deg"])
        for c in range(1, 7):
            a = np.asarray(phases[ANCHOR][c][REFERENCE_MS], float)
            m = circmean(a)
            w.writerow([f"{math.degrees(TARGET[c]):.1f}", a.size, f"{math.degrees(m):.2f}",
                        f"{rlen(a):.4f}", f"{math.degrees(wrap(m - TARGET[c])):+.2f}"])


def main():
    st.apply()
    root, out = st.parse_paths("Figure 2 - phase-targeting accuracy of the closed-loop stimulation")
    phases, offsets = load(out / "trigger_reference_comparison.npz")

    plot(out / "figure2_phase_targeting_accuracy.png", phases, offsets)
    write_summary(out / "figure2_phase_targeting_summary.csv", phases)

    print(f"anchor: photosensor (B), {REFERENCE_MS:.0f} ms")
    for c in range(1, 7):
        a = np.asarray(phases[ANCHOR][c][REFERENCE_MS], float)
        print(f"  target {label(c):>5s}: n={a.size:5d}  R={rlen(a):.3f}  "
              f"bias={math.degrees(wrap(circmean(a) - TARGET[c])):+6.2f}°")


if __name__ == "__main__":
    main()
