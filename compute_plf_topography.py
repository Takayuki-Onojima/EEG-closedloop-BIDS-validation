"""Scalp topography of phase locking to the intended target phase.

Panel a of Figure 2 shows how well the realized phase matched its target on the
one channel each participant's controller was driven from. On that channel a high
resultant length is partly what the loop was built to produce, so this script
asks the separate question of where else on the scalp the intended phase was
present, and when. The answer is drawn as panel b of Figure 2.

Latencies are relative to the photodiode onset from code/photodiode_onset.py,
the same anchor the rest of Figure 2 uses.

For every phase-dependent stimulus the phase of each scalp electrode is measured
at a series of latencies around the photodiode-confirmed visual onset. Locking is
summarized per electrode as the resultant length within each target condition,
averaged over the six conditions,

    PLF = mean over conditions of | mean over trials of exp(i * phase) |

The conditions are kept separate rather than pooled because they are spread
evenly around the cycle by design and would otherwise cancel. Pooling them after
subtracting each trial's own target is the alternative; it measures the same
quantity but confounds the within-condition spread with any disagreement between
the conditions' biases, and it needs the subtraction to be explained. Keeping
them separate measures only the spread, which is what panel b asks about; the
biases are the subject of Table 5.

Within a condition the trials of all participants are pooled before the resultant
length is taken. A resultant length is biased upwards at small N, by roughly
sqrt(rho^2 + (1 - rho^2)/N), and averaging per-participant values would remove
only the variance of that bias, not the bias itself. Pooling keeps N as large as
the data allow, which matters most where locking is weak. Trial counts are near
equal between participants, so pooling gives none of them undue weight. It also
makes this panel and panel c the same measurement, differing only in which
channel each reads: panel c follows the control channel, which is not the same
electrode in every participant, whereas the maps here read one fixed electrode
across all of them, so the two do not coincide at any single site.

A value near 1 means the electrode held a fixed relation to the intended phase on
every trial, not that the relation was zero: a constant offset leaves the
resultant length unchanged. A value near 0 means no consistent relation.

Preprocessing matches code/compare_trigger_references.py: linked-earlobe
reference, 500 Hz, 6-8 Hz zero-phase band-pass, Hilbert transform. Applying it to
all 63 scalp electrodes rather than one makes this the slowest script in the
directory, so the result is cached and Figure 2 reads the cache.
"""

import csv
import math
from pathlib import Path

import figure_style as st
import photodiode_onset as pdo

import mne
import numpy as np

TARGET_FS = 500
BAND = (6.0, 8.0)
TRANS_BW = 1.0
RECORDING_REFERENCE = "A1"
RIGHT_EARLOBE = "X2"
NON_SCALP = {"X2", "HEOG", "VEOG", "PhotoSensor"}

# latencies shown, relative to the photodiode onset (the analog rise, not the
# StimTrak marker B, whose offset moves with a hand-set threshold)
LATENCIES_MS = (-300, -250, -200, -150, -100, -50, 0)

# The target phases themselves are not needed here: the resultant length within a
# condition does not depend on where that condition's target sits, because
# subtracting a constant only rotates the trials and leaves the modulus alone.
CACHE_NAME = "plf_topography_cache.npz"


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analytic_all_channels(vhdr: Path):
    """Analytic signal for every scalp electrode, preprocessed as in Figure 2a."""
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


def collect(root: Path, out: Path):
    chmap = {r["participant_id"]: r["phase_estimation_channel"]
             for r in read_tsv(root / "participants.tsv")}
    analog = {stem: dict(zip(np.round(stim, 6), an))
              for stem, (_, stim, _, an) in pdo.load(out).items()}

    per_subject = {}          # sub -> ({latency: complex mean}, trial count)
    channels = None

    for sub in sorted({p.name for p in root.glob("sub-*") if p.is_dir()}):
        if chmap.get(sub, "n/a") == "n/a":
            continue
        acc = {}          # (latency, condition) -> complex sum over that subject's trials
        n_cond = {}       # condition -> trial count
        n_trials = 0

        for ev_path in sorted(root.glob(f"{sub}/eeg/*_task-phasedep_run-*_events.tsv")):
            vhdr = ev_path.with_name(ev_path.name[: -len("_events.tsv")] + "_eeg.vhdr")
            if not vhdr.exists():
                continue
            z, names = analytic_all_channels(vhdr)
            an = analog.get(ev_path.name[: -len("_events.tsv")], {})
            if channels is None:
                channels = names
            elif names != channels:
                raise SystemExit(f"{vhdr.name}: channel order differs from earlier runs")
            rows = read_tsv(ev_path)
            for i, r in enumerate(rows):
                if r["trial_type"] != "visual_stimulus":
                    continue
                cond = r.get("oscillation_phase")
                if cond not in set("123456"):
                    continue
                t_stim = float(r["onset"])
                t_b = an.get(round(t_stim, 6))
                if t_b is None or math.isnan(t_b):
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

                c = int(cond)
                for o in LATENCIES_MS:
                    key = (o, c)
                    if key not in acc:
                        acc[key] = np.zeros(len(names), complex)
                    acc[key] += np.exp(1j * np.angle(sample[o]))
                n_cond[c] = n_cond.get(c, 0) + 1
                n_trials += 1
            print(f"  {vhdr.stem}", flush=True)

        if n_trials:
            # the complex sums and the trial counts, not a resultant length: the
            # sums can be added across participants, a resultant length cannot
            per_subject[sub] = (acc, dict(n_cond))

    return channels, per_subject


def load(out: Path):
    """Channels, the PLF per latency, the participant count and the trial count.

    The trials of every participant are pooled within each condition before the
    resultant length is taken, and the six conditions are then averaged. Pooling
    first is what keeps N large: a resultant length is biased upwards by roughly
    sqrt(rho^2 + (1 - rho^2)/N), and averaging per-participant values would not
    remove that bias, only its variance. Trial counts are near equal between
    participants, so pooling gives no participant undue weight.
    """
    cache = out / CACHE_NAME
    if not cache.exists():
        raise SystemExit(f"missing {cache}\nrun code/compute_plf_topography.py first")
    z = np.load(cache, allow_pickle=True)
    channels = list(z["channels"])
    per_subject = z["per_subject"].item()

    conds = sorted({c for _, n in per_subject.values() for c in n})
    total = {}
    n_cond = {c: 0 for c in conds}
    for sums, counts in per_subject.values():
        for key, v in sums.items():
            total[key] = total.get(key, 0) + v
        for c, m in counts.items():
            n_cond[c] += m
    plf = {o: np.mean([np.abs(total[(o, c)] / n_cond[c]) for c in conds], axis=0)
           for o in LATENCIES_MS}
    return channels, plf, len(per_subject), sum(n_cond.values())


def main():
    root, out = st.parse_paths("Phase-locking topography feeding panel b of Figure 2")
    cache = out / CACHE_NAME

    if not cache.exists():
        channels, per_subject = collect(root, out)
        np.savez_compressed(cache, channels=np.array(channels),
                            per_subject=np.array(per_subject, dtype=object))
        print(f"wrote {cache}")

    channels, plf, n_subj, n_trials = load(out)

    csv_path = out / "plf_topography.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel"] + [f"plf_{o}ms" for o in LATENCIES_MS])
        for i, ch in enumerate(channels):
            w.writerow([ch] + [f"{plf[o][i]:.4f}" for o in LATENCIES_MS])
    print(f"wrote {csv_path}")

    print(f"\nparticipants {n_subj}, stimuli {n_trials}")
    for o in LATENCIES_MS:
        i = int(np.argmax(plf[o]))
        print(f"  {o:+5d} ms  max PLF {plf[o][i]:.3f} at {channels[i]:5s}  "
              f"mean {plf[o].mean():.3f}")


if __name__ == "__main__":
    main()
