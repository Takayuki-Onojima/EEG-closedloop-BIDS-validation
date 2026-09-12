"""Figure 2 - phase-targeting accuracy of the closed-loop stimulation.

Anchored on the photodiode onset, the measured luminance change on the display,
which is the dataset's primary external reference for actual visual onset. It is
taken from the analog PhotoSensor channel rather than from the StimTrak marker
B, because B fires at a threshold that was set by hand once per session and so
carries a per-session offset. Panel a additionally overlays the
stimulation-PC anchor; B and the Speedgoat trigger A are reported numerically in
Table 5 and are deliberately not repeated here.

Panels
  a  circular histograms of the phase 200 ms before visual onset, one per
     target phase, with the target drawn on top and the realized circular mean
     shown for two anchors at once. The online delay compensation was applied
     to the presentation command, so the stimulation-PC anchor shows what the
     controller achieved and the photosensor anchor what the eye received; the
     gap between the two is the display latency, and nothing else.
  b  scalp topography of the same phase locking, at seven latencies around
     visual onset, showing that it is not confined to the control channel
  c  phase-locking factor over time on the control channel, one curve per
     target phase

Reads two caches: `trigger_reference_comparison.npz` from
code/compare_trigger_references.py for panels a and c, and
`plf_topography_cache.npz` from code/compute_plf_topography.py for panel b. Run
both first if either is missing. Sizing and type follow code/figure_style.py.
"""

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

import compute_plf_topography as topo
import figure_style as st

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}
REFERENCE_MS = -200.0

# Panel a overlays two anchors. The online delay compensation was applied to the
# presentation command, so `stim` shows what the controller achieved; the display
# then needs time of its own to change luminance, so `photo` shows what the eye
# actually received. `photo` is the analog photodiode rise rather than the
# StimTrak marker B, whose offset moves with a threshold set by hand once per
# session; B is reported in Table 5 instead. Panels b and c use `photo`.
ANCHOR = "photo"
OVERLAY = (("stim", "#4C78A8", "stimulation-PC trigger"),
           ("photo", "#E45756", "photodiode onset"))

TOPO_CMAP = "RdBu_r"          # red high, blue low
TOPO_FS = 500                 # rate the cached analytic signals were computed at


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


def panel_a(fig, gs, phases):
    """Realized phase at the prestimulus reference, one polar panel per target.

    Both anchors are drawn on the same axes. They are the same trials read at
    two instants either side of the display latency, so the two distributions
    differ by a rigid rotation and nothing else; overlaying them shows that
    latency directly, as the gap between the two mean vectors.
    """
    axes = []
    for k, cond in enumerate(range(1, 7)):
        ax = fig.add_subplot(gs[k // 3, (k % 3)*2:(k % 3)*2 + 2], projection="polar")
        axes.append(ax)
        target = TARGET[cond]

        hists = []
        for ref, color, _ in OVERLAY:
            a = np.asarray(phases[ref][cond][REFERENCE_MS], float)
            counts, edges = np.histogram(a, bins=36, range=(-math.pi, math.pi))
            hists.append((edges[:-1] + np.diff(edges) / 2, np.diff(edges),
                          counts, color, circmean(a)))

        top = max(h[2].max() for h in hists) or 1
        for centers, widths, counts, color, _ in hists:
            ax.bar(centers, counts, width=widths, color=color, alpha=0.45,
                   edgecolor="none", zorder=1)

        # the stim mean lands on the target by design, so one of the two has to be
        # drawn over the other. The target goes on top: it is dashed, so the mean
        # underneath still shows through the gaps, whereas the reverse would hide
        # the target completely.
        for _, _, _, color, m in hists:
            ax.plot([m, m], [0, top], "-", color=color, lw=1.2, zorder=5)
        ax.plot([target, target], [0, top], "--", color="black", lw=1.0, zorder=7)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)          # counter-clockwise
        ax.set_ylim(0, top)
        ax.set_yticklabels([])
        ax.tick_params(pad=0.5)
        ax.grid(lw=0.4, color="#CCCCCC")
        ax.spines["polar"].set_linewidth(0.6)
        # Only the target is named on the panel. The bias for both anchors is a
        # number, not a pattern, so it belongs in Table 5 and in
        # figure2_phase_targeting_summary.csv rather than on the figure; what the
        # figure has to show is the gap between the two mean vectors, which it
        # shows without being annotated.
        ax.set_title(f"target {label(cond)}", fontsize=st.FS_ANNOT, pad=6)
    return axes


def panel_b(fig, gs, channels, plf):
    """Scalp topography of the same locking, one map per latency."""
    info = mne.create_info(channels, TOPO_FS, ch_types="eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="warn")

    lat = topo.LATENCIES_MS
    vmax = float(max(plf[o].max() for o in lat))
    axes, im = [], None
    for k, o in enumerate(lat):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        im, _ = mne.viz.plot_topomap(plf[o], info, axes=ax, show=False,
                                     cmap=TOPO_CMAP, vlim=(0, vmax),
                                     contours=4, sensors=True, outlines="head")
        weight = "bold" if o == REFERENCE_MS else "normal"
        ax.set_title(f"{o:+d} ms", fontsize=st.FS_ANNOT, fontweight=weight, pad=3)
    axes[0].set_ylabel("relative to visual onset", fontsize=st.FS_TICK, labelpad=1)

    # the colorbar column spans the full row height, but the topomaps are square
    # and so occupy only part of it; match the colorbar to their drawn extent
    pos = axes[0].get_position()
    cax = fig.add_axes([gs[0, -1].get_position(fig).x0, pos.y0, 0.008, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("phase-locking factor", fontsize=st.FS_ANNOT, labelpad=2)
    cb.ax.tick_params(labelsize=st.FS_TICK, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    return axes


def panel_c(fig, gs, phases, offsets):
    """Phase-locking factor over time on the control channel."""
    ax = fig.add_subplot(gs[0, 0])
    for cond in range(1, 7):
        r = [rlen(np.asarray(phases[ANCHOR][cond][o], float))
             if len(phases[ANCHOR][cond][o]) else np.nan for o in offsets]
        ax.plot(offsets, r, lw=0.9, label=f"target {label(cond)}")
    ax.axvline(REFERENCE_MS, color="black", ls="--", lw=0.9)
    ax.axvline(0, color="gray", ls="--", lw=0.9)
    ax.set_xlabel("time relative to the photodiode onset (ms)")
    ax.set_ylabel("phase-locking factor")
    ax.set_xlim(offsets[0], offsets[-1])
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, fontsize=st.FS_LEGEND, loc="upper right",
              bbox_to_anchor=(0.998, 0.995), columnspacing=1.0,
              handlelength=1.3, labelspacing=0.35, borderaxespad=0.2)
    ax.text(REFERENCE_MS, 1.02, "prestimulus target", ha="center", fontsize=st.FS_TICK)
    ax.text(0, 1.02, "visual onset", ha="center", fontsize=st.FS_TICK, color="gray")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def plot(path: Path, phases, offsets, channels, plf):
    fig = plt.figure(figsize=(st.WIDTH_2COL, 166 * st.MM))
    # four blocks: the polar array, a spacer that holds panel a's key, the
    # topography row, the time course. The topography row gets only as much
    # height as a square map needs at this width, since plot_topomap keeps its
    # aspect and any surplus becomes margin.
    outer = fig.add_gridspec(4, 1, height_ratios=[2.30, 0.16, 0.72, 1.05],
                             hspace=0.30, top=0.955, bottom=0.055,
                             left=0.075, right=0.985)

    axes_a = panel_a(fig, outer[0].subgridspec(2, 6, hspace=0.80, wspace=0.55), phases)
    axes_b = panel_b(fig, outer[2].subgridspec(1, 8, width_ratios=[1]*7 + [0.06],
                                               wspace=0.15), channels, plf)
    ax_c = panel_c(fig, outer[3].subgridspec(1, 1), phases, offsets)

    # the target/mean key belongs to panel a; center it in the spacer, which is
    # clear of both the 270 deg labels hanging below the polar block and the
    # latency titles above the topographies
    spacer = outer[1].get_position(fig)
    handles = [plt.Line2D([], [], color="black", ls="--", lw=1.0, label="target phase")]
    handles += [plt.Line2D([], [], color=c, lw=1.2, label=f"realized, anchored on {name}")
                for _, c, name in OVERLAY]
    fig.legend(handles=handles, loc="center",
               bbox_to_anchor=(0.5, spacer.y0 + spacer.height / 2),
               ncol=3, fontsize=st.FS_LEGEND, frameon=False, columnspacing=1.8)

    st.panel_letter(fig, 0.010, 0.995, "a")
    st.panel_letter(fig, 0.010, max(ax.get_position().y1 for ax in axes_b) + 0.030, "b")
    st.panel_letter(fig, 0.010, ax_c.get_position().y1 + 0.038, "c")

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
    channels, plf, n_subj, n_trials = topo.load(out)

    plot(out / "figure2_phase_targeting_accuracy.png", phases, offsets, channels, plf)
    write_summary(out / "figure2_phase_targeting_summary.csv", phases)

    print(f"anchor: {ANCHOR}, {REFERENCE_MS:.0f} ms")
    for c in range(1, 7):
        a = np.asarray(phases[ANCHOR][c][REFERENCE_MS], float)
        print(f"  target {label(c):>5s}: n={a.size:5d}  R={rlen(a):.3f}  "
              f"bias={math.degrees(wrap(circmean(a) - TARGET[c])):+6.2f}°")
    print(f"\ntopography: {n_subj} participants, {n_trials} stimuli")
    for o in topo.LATENCIES_MS:
        i = int(np.argmax(plf[o]))
        print(f"  {o:+5d} ms  max PLF {plf[o][i]:.3f} at {channels[i]}")


if __name__ == "__main__":
    main()
