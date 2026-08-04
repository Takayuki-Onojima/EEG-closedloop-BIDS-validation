"""Compare phase-targeting accuracy against each of the three available triggers.

Three timestamps exist per stimulus:
  A    - realtime_trigger, emitted by the Speedgoat when the target phase was detected
  stim - the stimulation PC's own trigger, i.e. the presentation command
  B    - photosensor, the actual luminance change on the display

The intended -200 ms prestimulus reference can be anchored to any of them, and
the choice shifts the realised phase by the inter-trigger latency (median
A->stim 1.0 ms, stim->B 6.2 ms). This script measures the resulting phase
distribution for all three so the correct anchor can be chosen from the data.

Phase is taken from the analytic signal at 500 Hz but interpolated linearly in
the complex plane, because index rounding at 500 Hz alone would introduce up to
+-2.7 deg - the same order as the bias being measured.
"""

import csv
import math
import os
from pathlib import Path

import figure_style as st

import numpy as np
from scipy.signal import decimate, filtfilt, firwin, hilbert

NCH, FS = 67, 5000
DS = 10
FS_DS = FS // DS
FIR_ORDER = 128
BAND = (6.0, 8.0)
REFERENCE_MS = -200.0
OFFSETS_MS = np.arange(-400, 101, 5)
REFS = ("A", "stim", "B")

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analytic_at(z, t_s):
    """Linear interpolation of the complex analytic signal at time t (seconds)."""
    x = t_s * FS_DS
    i = int(np.floor(x))
    if i < 0 or i + 1 >= len(z):
        return None
    f = x - i
    return (1.0 - f) * z[i] + f * z[i + 1]


def collect(root: Path):
    chmap = {r["participant_id"]: r["phase_estimation_channel"]
             for r in read_tsv(root / "participants.tsv")}
    taps = firwin(FIR_ORDER + 1, BAND, pass_zero=False, fs=FS_DS)

    # phases[ref][cond][offset] -> list
    phases = {ref: {c: {o: [] for o in OFFSETS_MS} for c in range(1, 7)} for ref in REFS}
    counts = {ref: 0 for ref in REFS}

    for ev_path in sorted(root.glob("sub-*/eeg/*_task-phasedep_run-*_events.tsv")):
        sub = ev_path.name.split("_")[0]
        ch = chmap.get(sub)
        if not ch or ch == "n/a":
            continue
        stem = ev_path.name[: -len("_events.tsv")]
        eeg_path = ev_path.with_name(stem + "_eeg.eeg")
        if not eeg_path.exists():
            continue

        names = [r["name"] for r in read_tsv(ev_path.with_name(stem + "_channels.tsv"))]
        n = os.path.getsize(eeg_path) // (NCH * 4)
        mm = np.memmap(eeg_path, dtype="<f4", mode="r", shape=(n, NCH))
        sig = (np.asarray(mm[:, names.index(ch)], np.float64)
               - np.asarray(mm[:, names.index("X2")], np.float64) / 2.0)
        del mm
        z = hilbert(filtfilt(taps, 1.0, decimate(sig, DS, ftype="fir", zero_phase=True)))

        rows = read_tsv(ev_path)
        last_a = None
        for i, r in enumerate(rows):
            tt = r["trial_type"]
            if tt == "realtime_trigger":
                last_a = float(r["onset"])
                continue
            if tt != "visual_stimulus" or r.get("oscillation_phase") not in set("123456"):
                continue
            cond = int(r["oscillation_phase"])
            t_stim = float(r["onset"])

            t_b = None
            for r2 in rows[i+1:i+6]:
                if r2["trial_type"] == "photosensor":
                    d = float(r2["onset"]) - t_stim
                    if 0 <= d <= 0.050:
                        t_b = float(r2["onset"])
                    break

            for ref, t0 in (("A", last_a), ("stim", t_stim), ("B", t_b)):
                if t0 is None:
                    continue
                counts[ref] += 1
                for o in OFFSETS_MS:
                    v = analytic_at(z, t0 + o / 1000.0)
                    if v is not None:
                        phases[ref][cond][o].append(math.atan2(v.imag, v.real))
            last_a = None
        print(f"  {stem}  ch={ch}", flush=True)

    return phases, counts


def circmean(a):
    return math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))


def rlen(a):
    return math.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def main():
    root, out = st.parse_paths("Realised phase measured against each of the three available triggers")
    cache = out / "trigger_reference_comparison.npz"

    if cache.exists():
        print(f"loading {cache}")
        z = np.load(cache, allow_pickle=True)
        phases = z["phases"].item()
        counts = z["counts"].item()
    else:
        phases, counts = collect(root)
        np.savez_compressed(cache, phases=np.array(phases, dtype=object),
                            counts=np.array(counts, dtype=object),
                            offsets=OFFSETS_MS)

    rows = []
    print(f"\n{'ref':>5s} {'cond':>5s} {'target':>8s} {'actual':>8s} {'bias':>8s} {'R':>7s} {'n':>7s}")
    for ref in REFS:
        biases = []
        for c in range(1, 7):
            a = np.asarray(phases[ref][c][REFERENCE_MS], float)
            if a.size == 0:
                continue
            cm = circmean(a)
            b = math.degrees(wrap(cm - TARGET[c]))
            biases.append(b)
            print(f"{ref:>5s} {c:5d} {math.degrees(TARGET[c]):8.1f} "
                  f"{math.degrees(cm):8.1f} {b:+8.2f} {rlen(a):7.3f} {a.size:7d}")
            rows.append((ref, c, math.degrees(TARGET[c]), math.degrees(cm), b, rlen(a), a.size))
        print(f"{ref:>5s}  mean bias {np.mean(biases):+6.2f} deg   SD {np.std(biases):.2f}\n")

    # zero-bias crossing per reference
    print("zero-bias crossing (ms relative to each trigger):")
    for ref in REFS:
        prof = []
        for o in OFFSETS_MS:
            bs = [math.degrees(wrap(circmean(np.asarray(phases[ref][c][o], float)) - TARGET[c]))
                  for c in range(1, 7) if len(phases[ref][c][o])]
            if bs:
                prof.append((o, float(np.mean(bs))))
        for (o1, b1), (o2, b2) in zip(prof, prof[1:]):
            if -190 >= o1 >= -215 and (b1 <= 0 <= b2 or b2 <= 0 <= b1):
                print(f"  {ref:>5s}: {o1 + (o2-o1)*(0-b1)/(b2-b1):7.1f} ms")
                break

    with (out / "trigger_reference_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference_trigger", "condition", "target_deg", "actual_deg",
                    "bias_deg", "resultant_length", "n"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                        f"{r[5]:.4f}", r[6]])
    print(f"\nwrote {out/'trigger_reference_comparison.csv'}")


if __name__ == "__main__":
    main()
