"""Compare phase-targeting accuracy against each of the four available anchors.

Four timestamps exist per stimulus:
  A     - realtime_trigger, emitted by the Speedgoat when the target phase was detected
  stim  - the stimulation PC's own trigger, i.e. the presentation command
  B     - the StimTrak marker, emitted when the photodiode signal crossed threshold
  photo - the same rise measured from the analogue PhotoSensor channel itself, at
          half the pulse amplitude, by code/photodiode_onset.py

The intended -200 ms prestimulus reference can be anchored to any of them, and
the choice shifts the realised phase by the inter-trigger latency. This script
measures the resulting phase distribution for all four so the anchor can be
chosen from the data.

B and photo track the same physical flash and carry the same trial-to-trial
jitter: B minus photo has a within-participant SD of only 0.06 ms, so the marker
adds no noise of its own. But B fires wherever a hand-set threshold happened to
sit on the rise, and participant means of B span 3.00 ms where those of photo
span 0.47 ms. photo is therefore the anchor used for absolute phase, and B is
retained here so that a reader who prefers the marker can read off the
correction.

Preprocessing reproduces the online control pipeline and is carried out with
MNE-Python, so that channel names, units and the reference scheme are handled by
the same library a reader would use to reopen the data:

  1. read the BrainVision recording, which gives channel names and units from
     the header rather than from any assumption about the file layout
  2. restore the recording reference as a zero-valued channel (A1, the left
     earlobe) and re-reference to the average of A1 and X2, the recorded right
     earlobe; this is the linked-earlobe montage the online system used
  3. resample to 500 Hz, the rate the online system worked at
  4. band-pass 6-8 Hz, zero-phase, with 1 Hz transition bands
  5. Hilbert transform, then read the phase at a chosen offset from each trigger

The band-pass deserves a note. Online, the controller had to run causally and
used a 128th-order FIR at 500 Hz, which is only 258 ms long and therefore passes
a good deal more than 6-8 Hz. Offline there is no such constraint, so the filter
here is designed properly and measures the phase of the 6-8 Hz component itself.
The two definitions do not agree: against the controller's own broadband
definition the six conditions reach a resultant length of about 0.93, against
the narrow-band definition about 0.69. Both are correct measurements of
different quantities, and the narrow-band one is used here because it is what a
reader recomputing the phase from the released data will obtain.

Phase is interpolated linearly in the complex plane, because index rounding at
500 Hz alone would introduce up to +-2.7 deg, the same order as the bias being
measured.
"""

import csv
import math
from pathlib import Path

import figure_style as st
import photodiode_onset as pd

import mne
import numpy as np

TARGET_FS = 500                 # rate the online system worked at
BAND = (6.0, 8.0)
# 1 Hz transition bands either side; MNE then picks the filter length needed to
# achieve them, which is far longer than the 128th-order filter the online system
# had to make do with in real time. See the note in the module docstring.
TRANS_BW = 1.0
REFERENCE_MS = -200.0
OFFSETS_MS = np.arange(-400, 101, 5)
REFS = ("A", "stim", "B", "photo")
RECORDING_REFERENCE = "A1"      # left earlobe, not stored as a channel
RIGHT_EARLOBE = "X2"

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analytic_signal(vhdr: Path, channel: str):
    """Preprocessed analytic signal for one channel, following the online pipeline."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
    missing = [c for c in (channel, RIGHT_EARLOBE) if c not in raw.ch_names]
    if missing:
        raise SystemExit(f"{vhdr.name}: missing channel(s) {missing}")

    raw.pick([channel, RIGHT_EARLOBE]).load_data(verbose="ERROR")
    raw = mne.add_reference_channels(raw, RECORDING_REFERENCE)
    raw.set_eeg_reference([RECORDING_REFERENCE, RIGHT_EARLOBE], verbose="ERROR")

    raw.resample(TARGET_FS, verbose="ERROR")
    raw.filter(BAND[0], BAND[1], picks=[channel], method="fir", fir_design="firwin",
               phase="zero", l_trans_bandwidth=TRANS_BW, h_trans_bandwidth=TRANS_BW,
               verbose="ERROR")
    raw.apply_hilbert(picks=[channel], verbose="ERROR")
    return raw.get_data(picks=[channel])[0]


def analytic_at(z, t_s):
    """Linear interpolation of the complex analytic signal at time t (seconds)."""
    x = t_s * TARGET_FS
    i = int(np.floor(x))
    if i < 0 or i + 1 >= len(z):
        return None
    f = x - i
    return (1.0 - f) * z[i] + f * z[i + 1]


def collect(root: Path, out: Path):
    chmap = {r["participant_id"]: r["phase_estimation_channel"]
             for r in read_tsv(root / "participants.tsv")}
    # analogue onsets, keyed per run by the stimulation-PC trigger time they
    # belong to, so they can be looked up while walking the event file
    analog = {stem: dict(zip(np.round(stim, 6), an))
              for stem, (_, stim, _, an) in pd.load(out).items()}

    # phases[ref][cond][offset] -> list
    phases = {ref: {c: {o: [] for o in OFFSETS_MS} for c in range(1, 7)} for ref in REFS}
    counts = {ref: 0 for ref in REFS}

    for ev_path in sorted(root.glob("sub-*/eeg/*_task-phasedep_run-*_events.tsv")):
        sub = ev_path.name.split("_")[0]
        ch = chmap.get(sub)
        if not ch or ch == "n/a":
            continue
        stem = ev_path.name[: -len("_events.tsv")]
        vhdr = ev_path.with_name(stem + "_eeg.vhdr")
        if not vhdr.exists():
            continue

        z = analytic_signal(vhdr, ch)
        an = analog.get(stem, {})

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

            t_photo = an.get(round(t_stim, 6))
            if t_photo is not None and math.isnan(t_photo):
                t_photo = None

            for ref, t0 in (("A", last_a), ("stim", t_stim), ("B", t_b),
                            ("photo", t_photo)):
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
        phases, counts = collect(root, out)
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
