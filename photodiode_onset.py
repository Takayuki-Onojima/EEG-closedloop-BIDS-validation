"""Visual onset measured from the photodiode signal itself, rather than from `B`.

The dataset carries two timing references derived from the same photodiode: the
analog `PhotoSensor` channel, and marker `B`, which the in-line StimTrak emits
when that signal crosses a threshold. `B` is present on every stimulus and its
trial-to-trial jitter is the display's own, but the threshold had to be set by
hand for each session, and a threshold sets *when during the rise* the trigger
fires. Any change to it shifts every `B` in that session by a constant.

Measuring the same rise from the analog channel with one criterion for all
sessions removes that free parameter. The white patch is drawn for a single
frame, so the photodiode sees a brief flash that saturates the amplifier; the
onset is therefore taken as the crossing of half the pulse amplitude, linearly
interpolated between samples at the native 5 kHz, which is stable against the
exact saturation level.

Neither reference is the retinal onset: the photodiode watches the upper-left
corner of the screen while the stimuli sit near the middle, and an LCD refreshes
top to bottom, so both carry a constant positional offset of up to one frame.
That offset is common to every trial, participant and condition.

Writes one cache holding, per phase-dependent visual stimulus, the stimulation-PC
trigger time, the `B` time and the analog onset time. `compare_trigger_
references.py` and Table 4 both read it.
"""

import csv
from pathlib import Path

import figure_style as st

import mne
import numpy as np

PHOTO_CHANNEL = "PhotoSensor"
SEARCH_S = 0.030          # window after the stimulation-PC trigger to look in
MIN_PULSE_UV = 20000.0    # a real flash saturates far above this
RISE_FRACTION = 0.5

CACHE_NAME = "photodiode_onset_cache.npz"


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analog_onset(photo, fs, t_stim):
    """Half-amplitude crossing of the photodiode pulse, in seconds, or None."""
    j = int(round(t_stim * fs))
    w = photo[j: j + int(SEARCH_S * fs)]
    if w.size < 10:
        return None
    peak = w.max()
    if peak < MIN_PULSE_UV:
        return None
    k = int(np.argmax(w > RISE_FRACTION * peak))
    if k == 0:
        return None
    y0, y1 = w[k - 1], w[k]
    frac = (RISE_FRACTION * peak - y0) / (y1 - y0)
    return t_stim + (k - 1 + frac) / fs


def collect(root: Path):
    rec = {}      # run stem -> (subject, stim[], b[], analog[])
    for ev_path in sorted(root.glob("sub-*/eeg/*_task-phasedep_run-*_events.tsv")):
        stem = ev_path.name[: -len("_events.tsv")]
        vhdr = ev_path.with_name(stem + "_eeg.vhdr")
        if not vhdr.exists():
            continue
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        if PHOTO_CHANNEL not in raw.ch_names:
            raise SystemExit(f"{vhdr.name}: no {PHOTO_CHANNEL} channel")
        fs = raw.info["sfreq"]
        raw.pick([PHOTO_CHANNEL]).load_data(verbose="ERROR")
        photo = raw.get_data()[0] * 1e6          # volts -> microvolts

        rows = read_tsv(ev_path)
        stim, bt, an = [], [], []
        for i, r in enumerate(rows):
            if r["trial_type"] != "visual_stimulus":
                continue
            t = float(r["onset"])
            b = np.nan
            for r2 in rows[i+1:i+6]:
                if r2["trial_type"] == "photosensor":
                    d = float(r2["onset"]) - t
                    if 0 <= d <= 0.050:
                        b = float(r2["onset"])
                    break
            a = analog_onset(photo, fs, t)
            stim.append(t)
            bt.append(b)
            an.append(np.nan if a is None else a)

        rec[stem] = (stem.split("_")[0], np.array(stim), np.array(bt), np.array(an))
        n_bad = int(np.isnan(rec[stem][3]).sum())
        print(f"  {stem}  {len(stim)} stimuli"
              + (f"  ({n_bad} without a detectable pulse)" if n_bad else ""), flush=True)
    return rec


def load(out: Path):
    """{run stem: (subject, stim times, B times, analog onset times)} in seconds."""
    cache = out / CACHE_NAME
    if not cache.exists():
        raise SystemExit(f"missing {cache}\nrun code/photodiode_onset.py first")
    return np.load(cache, allow_pickle=True)["runs"].item()


def main():
    root, out = st.parse_paths("Visual onset measured from the photodiode signal")
    cache = out / CACHE_NAME

    if not cache.exists():
        rec = collect(root)
        np.savez_compressed(cache, runs=np.array(rec, dtype=object))
        print(f"wrote {cache}")
    rec = load(out)

    # per participant: the analog onset, B, and the constant between them
    by_sub = {}
    for sub, stim, bt, an in rec.values():
        d = by_sub.setdefault(sub, [[], []])
        d[0].extend((an - stim) * 1000)
        d[1].extend((bt - stim) * 1000)

    print(f"\n{'sub':<8}{'n':>6}{'analog (ms)':>16}{'B (ms)':>16}{'B - analog':>13}")
    rows = []
    for sub in sorted(by_sub):
        a = np.asarray(by_sub[sub][0], float)
        b = np.asarray(by_sub[sub][1], float)
        ok = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[ok], b[ok]
        rows.append((sub, a.size, np.median(a), a.std(), np.median(b), b.std(),
                     np.median(b - a), (b - a).std()))
        print(f"{sub:<8}{a.size:6d}{np.median(a):9.2f} +/-{a.std():4.2f}"
              f"{np.median(b):9.2f} +/-{b.std():4.2f}"
              f"{np.median(b - a):+9.2f} +/-{(b - a).std():4.2f}")

    csv_path = out / "photodiode_onset_per_participant.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "n", "analog_median_ms", "analog_sd_ms",
                    "B_median_ms", "B_sd_ms", "B_minus_analog_median_ms",
                    "B_minus_analog_sd_ms"])
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{v:.3f}" for v in r[2:]])
    print(f"\nwrote {csv_path}")

    am = np.array([r[2] for r in rows])
    bm = np.array([r[4] for r in rows])
    dm = np.array([r[6] for r in rows])
    for name, v in (("analog", am), ("B", bm), ("B - analog", dm)):
        print(f"between-participant {name:>10s}: {v.min():+6.2f} to {v.max():+6.2f} ms, "
              f"spread {v.max()-v.min():4.2f} ms = {(v.max()-v.min())/1000*7*360:4.1f} deg at 7 Hz")


if __name__ == "__main__":
    main()
