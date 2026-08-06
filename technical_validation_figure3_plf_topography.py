"""Figure 3 - scalp topography of phase locking to the intended target phase.

Figure 2 shows how well the realised phase matched its target on the one channel
each participant's controller was driven from. This figure asks where else on the
scalp that locking is present, and when.

For every phase-dependent stimulus the phase of each scalp electrode is measured
at a series of latencies around the photodiode-confirmed visual onset, and the
target phase assigned to that trial is subtracted. Locking is then summarised per
electrode as the resultant length of those differences,

    PLF = | mean over trials of exp(i * (phase - target)) |

so a value near 1 means the electrode carried the intended phase on every trial
and a value near 0 means it carried no consistent relation to it. Subtracting the
target is what makes the six conditions poolable: without it they would cancel,
since they are spread evenly around the cycle by design.

PLF is computed within each participant and then averaged across participants, so
that participants with slightly more trials do not dominate.

Preprocessing matches code/compare_trigger_references.py: linked-earlobe
reference, 500 Hz, 6-8 Hz zero-phase band-pass, Hilbert transform.
"""

import csv
import math
from pathlib import Path

import figure_style as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

TARGET_FS = 500
BAND = (6.0, 8.0)
TRANS_BW = 1.0
RECORDING_REFERENCE = "A1"
RIGHT_EARLOBE = "X2"
NON_SCALP = {"X2", "HEOG", "VEOG", "PhotoSensor"}

# latencies shown, relative to photodiode-confirmed visual onset
LATENCIES_MS = (-300, -250, -200, -150, -100, -50, 0)
REFERENCE_MS = -200.0

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analytic_all_channels(vhdr: Path):
    """Analytic signal for every scalp electrode, preprocessed as in Figure 2."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    scalp = [c for c in raw.ch_names if c not in NON_SCALP]
    raw.pick(scalp + [RIGHT_EARLOBE]).load_data(verbose="ERROR")
    raw = mne.add_reference_channels(raw, RECORDING_REFERENCE)
    raw.set_eeg_reference([RECORDING_REFERENCE, RIGHT_EARLOBE], verbose="ERROR")
    raw.pick(scalp)

    raw.resample(TARGET_FS, verbose="ERROR")
    raw.filter(BAND[0], BAND[1], method="fir", fir_design="firwin", phase="zero",
               l_trans_bandwidth=TRANS_BW, h_trans_bandwidth=TRANS_BW, verbose="ERROR")
    raw.apply_hilbert(verbose="ERROR")
    return raw.get_data(), scalp


def collect(root: Path):
    chmap = {r["participant_id"]: r["phase_estimation_channel"]
             for r in read_tsv(root / "participants.tsv")}

    per_subject = {}          # sub -> {latency: complex sum}, and a trial count
    channels = None

    for sub in sorted({p.name for p in root.glob("sub-*") if p.is_dir()}):
        if chmap.get(sub, "n/a") == "n/a":
            continue
        acc = {o: None for o in LATENCIES_MS}
        n_trials = 0

        for ev_path in sorted(root.glob(f"{sub}/eeg/*_task-phasedep_run-*_events.tsv")):
            vhdr = ev_path.with_name(ev_path.name[: -len("_events.tsv")] + "_eeg.vhdr")
            if not vhdr.exists():
                continue
            z, names = analytic_all_channels(vhdr)
            if channels is None:
                channels = names
            elif names != channels:
                raise SystemExit(f"{vhdr.name}: channel order differs from earlier runs")
            for o in LATENCIES_MS:
                if acc[o] is None:
                    acc[o] = np.zeros(len(names), complex)

            rows = read_tsv(ev_path)
            for i, r in enumerate(rows):
                if r["trial_type"] != "visual_stimulus":
                    continue
                cond = r.get("oscillation_phase")
                if cond not in set("123456"):
                    continue
                t_stim = float(r["onset"])
                t_b = None
                for r2 in rows[i+1:i+6]:
                    if r2["trial_type"] == "photosensor":
                        d = float(r2["onset"]) - t_stim
                        if 0 <= d <= 0.050:
                            t_b = float(r2["onset"])
                        break
                if t_b is None:
                    continue

                ok = True
                sample = {}
                for o in LATENCIES_MS:
                    x = (t_b + o / 1000.0) * TARGET_FS
                    j = int(np.floor(x))
                    if j < 0 or j + 1 >= z.shape[1]:
                        ok = False
                        break
                    f = x - j
                    sample[o] = (1 - f) * z[:, j] + f * z[:, j + 1]
                if not ok:
                    continue

                target = TARGET[int(cond)]
                for o in LATENCIES_MS:
                    acc[o] += np.exp(1j * (np.angle(sample[o]) - target))
                n_trials += 1
            print(f"  {vhdr.stem}", flush=True)

        if n_trials:
            per_subject[sub] = ({o: acc[o] / n_trials for o in LATENCIES_MS}, n_trials)

    return channels, per_subject


def plot(path: Path, channels, plf, n_subj, n_trials):
    info = mne.create_info(channels, TARGET_FS, ch_types="eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="warn")

    # each topomap is square, so the height follows from the width available to
    # one column; anything taller only adds white space
    fig = plt.figure(figsize=(st.WIDTH_2COL, 36 * st.MM))
    gs = fig.add_gridspec(1, len(LATENCIES_MS) + 1,
                          width_ratios=[1] * len(LATENCIES_MS) + [0.06],
                          wspace=0.15, top=0.86, bottom=0.02, left=0.01, right=0.95)

    vmax = float(max(plf[o].max() for o in LATENCIES_MS))
    im = None
    axes = []
    for k, o in enumerate(LATENCIES_MS):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)
        im, _ = mne.viz.plot_topomap(plf[o], info, axes=ax, show=False,
                                     cmap="viridis", vlim=(0, vmax),
                                     contours=4, sensors=True, outlines="head")
        weight = "bold" if o == REFERENCE_MS else "normal"
        ax.set_title(f"{o:+d} ms", fontsize=st.FS_ANNOT, fontweight=weight, pad=3)

    # the colourbar column spans the full figure height, but the topomaps are
    # square and so occupy only part of it; match the colourbar to their extent
    pos = axes[0].get_position()
    cax = fig.add_axes([gs[0, -1].get_position(fig).x0, pos.y0, 0.008, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("phase-locking factor", fontsize=st.FS_ANNOT, labelpad=2)
    cb.ax.tick_params(labelsize=st.FS_TICK, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)

    for f in st.save(fig, path):
        print(f"wrote {f}")
    plt.close(fig)


def main():
    st.apply()
    root, out = st.parse_paths("Figure 3 - scalp topography of phase locking")
    cache = out / "plf_topography_cache.npz"

    if cache.exists():
        print(f"loading {cache}")
        z = np.load(cache, allow_pickle=True)
        channels = list(z["channels"])
        per_subject = z["per_subject"].item()
    else:
        channels, per_subject = collect(root)
        np.savez_compressed(cache, channels=np.array(channels),
                            per_subject=np.array(per_subject, dtype=object))

    # average the per-participant resultant vectors, then take their length
    plf = {o: np.abs(np.mean([np.abs(v[0][o]) for v in per_subject.values()], axis=0))
           for o in LATENCIES_MS}
    n_trials = sum(v[1] for v in per_subject.values())

    with (out / "figure3_plf_topography.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel"] + [f"plf_{o}ms" for o in LATENCIES_MS])
        for i, ch in enumerate(channels):
            w.writerow([ch] + [f"{plf[o][i]:.4f}" for o in LATENCIES_MS])

    plot(out / "figure3_plf_topography", channels, plf, len(per_subject), n_trials)

    print(f"\nparticipants {len(per_subject)}, stimuli {n_trials}")
    for o in LATENCIES_MS:
        i = int(np.argmax(plf[o]))
        print(f"  {o:+5d} ms  max PLF {plf[o][i]:.3f} at {channels[i]:5s}  "
              f"mean {plf[o].mean():.3f}")


if __name__ == "__main__":
    main()
