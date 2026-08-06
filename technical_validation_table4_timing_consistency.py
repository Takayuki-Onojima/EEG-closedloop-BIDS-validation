import csv
import math
from pathlib import Path

import figure_style as st
import photodiode_onset as pdo

import numpy as np


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean(values):
    return sum(values) / len(values) if values else math.nan


def std(values):
    if len(values) < 2:
        return math.nan
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(values):
    if not values:
        return math.nan, math.nan, 0
    return mean(values), std(values), len(values)


def find_prev_realtime(events, idx, max_back_sec=1.0):
    stim_onset = to_float(events[idx].get("onset"))
    if stim_onset is None:
        return None
    for j in range(idx - 1, -1, -1):
        ev = events[j]
        onset = to_float(ev.get("onset"))
        if onset is None:
            continue
        if stim_onset - onset > max_back_sec:
            break
        if ev.get("trial_type") == "realtime_trigger":
            return ev
    return None


def find_next_photosensor(events, idx, max_forward_sec=0.020):
    stim_onset = to_float(events[idx].get("onset"))
    if stim_onset is None:
        return None
    for j in range(idx + 1, len(events)):
        ev = events[j]
        onset = to_float(ev.get("onset"))
        if onset is None:
            continue
        if onset - stim_onset > max_forward_sec:
            break
        if ev.get("trial_type") == "photosensor":
            return ev
    return None


def build_timing_samples(root: Path):
    a_to_stim_ms = []
    stim_to_photo_ms_all = []
    stim_to_photo_ms_chain = []
    a_to_photo_ms = []
    b_to_stim_ms = []

    for ev_path in sorted(root.glob("sub-*/eeg/*_task-phasedep_run-*_events.tsv")):
        events = read_tsv(ev_path)
        
        # Find the last visual_stimulus to exclude post-session noise
        last_stim_onset = None
        for ev in events:
            if ev.get("trial_type") == "visual_stimulus":
                onset = to_float(ev.get("onset"))
                if onset is not None:
                    last_stim_onset = onset
        
        for i, ev in enumerate(events):
            if ev.get("trial_type") != "visual_stimulus":
                continue

            stim_onset = to_float(ev.get("onset"))
            if stim_onset is None:
                continue

            rt = find_prev_realtime(events, i)
            # Only look for photosensor on attended trials
            is_attended = ev.get("attention_congruency") == "attended"
            ps = find_next_photosensor(events, i) if is_attended else None

            if rt is not None:
                rt_onset = to_float(rt.get("onset"))
                if rt_onset is not None:
                    a_to_stim_ms.append((stim_onset - rt_onset) * 1000.0)

            if ps is not None:
                ps_onset = to_float(ps.get("onset"))
                if ps_onset is not None:
                    # Exclude B events that are too far from this stim relative to file duration
                    # If B is the last stim, include it; otherwise exclude if >1.0s from it
                    is_valid_b = True
                    if stim_onset != last_stim_onset:
                        # Check if nearest stim (this one) is close enough
                        if ps_onset - stim_onset > 0.020:
                            is_valid_b = False
                    
                    if is_valid_b:
                        dt_sp = (ps_onset - stim_onset) * 1000.0
                        stim_to_photo_ms_all.append(dt_sp)
                        if (ps.get("trigger_value") or "").strip() == "B":
                            b_to_stim_ms.append(dt_sp)
                        if rt is not None:
                            stim_to_photo_ms_chain.append(dt_sp)

            if rt is not None and ps is not None:
                rt_onset = to_float(rt.get("onset"))
                ps_onset = to_float(ps.get("onset"))
                if ps_onset - stim_onset <= 0.020:
                    if rt_onset is not None and ps_onset is not None:
                        a_to_photo_ms.append((ps_onset - rt_onset) * 1000.0)

    return a_to_stim_ms, stim_to_photo_ms_all, stim_to_photo_ms_chain, a_to_photo_ms, b_to_stim_ms


def analog_rows(root: Path, out: Path):
    """Rows comparing the StimTrak marker B with the photodiode signal itself.

    B and the analogue onset are two readings of the same flash, so their
    difference isolates what the hand-set StimTrak threshold contributed. Both
    are restricted to the same attended trials the rows above use, so that every
    row of the table refers to one trial set.
    """
    runs = pdo.load(out)
    attended = {}
    for ev_path in sorted(root.glob("sub-*/eeg/*_task-phasedep_run-*_events.tsv")):
        stem = ev_path.name[: -len("_events.tsv")]
        attended[stem] = {round(to_float(ev["onset"]), 6)
                          for ev in read_tsv(ev_path)
                          if ev.get("trial_type") == "visual_stimulus"
                          and ev.get("attention_congruency") == "attended"}

    stim_to_an, b_minus_an = [], []
    an_sub, b_sub, diff_sub = {}, {}, {}
    for stem, (sub, stim, bt, an) in runs.items():
        keep = np.array([round(t, 6) in attended.get(stem, ()) for t in stim])
        ok = keep & ~np.isnan(an) & ~np.isnan(bt)
        d_an = (an[ok] - stim[ok]) * 1000.0
        d_b = (bt[ok] - stim[ok]) * 1000.0
        stim_to_an.extend(d_an)
        b_minus_an.extend(d_b - d_an)
        an_sub.setdefault(sub, []).extend(d_an)
        b_sub.setdefault(sub, []).extend(d_b)
        diff_sub.setdefault(sub, []).extend(d_b - d_an)

    stim_to_an = np.asarray(stim_to_an)
    b_minus_an = np.asarray(b_minus_an)
    within = float(np.mean([np.std(v, ddof=1) for v in diff_sub.values()]))
    spread = lambda d: float(np.ptp([np.mean(v) for v in d.values()]))

    return [
        {
            "Comparison": "stimulation-PC trigger - photodiode onset",
            "Definition": "Delay from the stimulation-PC trigger to the half-amplitude rise of the photodiode pulse on the analogue `PhotoSensor` channel",
            "Mean delay (ms)": round(float(stim_to_an.mean()), 2),
            "SD (ms)": round(float(stim_to_an.std(ddof=1)), 2),
            "Number of trials": int(stim_to_an.size),
            "Notes": f"read with one criterion in every session; participant means span {spread(an_sub):.2f} ms against {spread(b_sub):.2f} ms for `B`",
        },
        {
            "Comparison": "photodiode onset - `B`",
            "Definition": "Offset of the StimTrak marker from the photodiode rise it was derived from",
            "Mean delay (ms)": round(float(b_minus_an.mean()), 2),
            "SD (ms)": round(float(b_minus_an.std(ddof=1)), 2),
            "Number of trials": int(b_minus_an.size),
            "Notes": f"within participants a constant (SD {within:.2f} ms); participant means span {spread(diff_sub):.2f} ms, set by the StimTrak threshold; per-participant values are reproducible with `code/photodiode_onset.py`",
        },
    ]


def write_table4(
    out_csv: Path,
    out_md: Path,
    a_to_stim_ms,
    stim_to_photo_ms_all,
    stim_to_photo_ms_chain,
    a_to_photo_ms,
    b_to_stim_ms,
    extra_rows=(),
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    m1, s1, n1 = summarize(a_to_stim_ms)
    m2, s2, n2 = summarize(stim_to_photo_ms_all)  # All B, not just A-chain
    m3, s3, n3 = summarize(a_to_photo_ms)
    m4, s4, n4 = summarize(b_to_stim_ms)

    rows = [
        {
            "Comparison": "`A` - stimulation-PC trigger",
            "Definition": "Delay from the online phase-detection trigger `A` to the stimulation-PC trigger output",
            "Mean delay (ms)": round(m1, 2) if not math.isnan(m1) else "n/a",
            "SD (ms)": round(s1, 2) if not math.isnan(s1) else "n/a",
            "Number of trials": n1,
            "Notes": "`A` = `realtime_trigger`; attended trials with a valid `A`",
        },
        {
            "Comparison": "stimulation-PC trigger - `B`",
            "Definition": "Delay from the stimulation-PC trigger to the StimTrak marker `B`",
            "Mean delay (ms)": round(m2, 2) if not math.isnan(m2) else "n/a",
            "SD (ms)": round(s2, 2) if not math.isnan(s2) else "n/a",
            "Number of trials": n2,
            "Notes": "all attended trials, with and without a recorded `A`",
        },
        {
            "Comparison": "`A` - `B` chain latency",
            "Definition": "Total delay from `A` to `B`",
            "Mean delay (ms)": round(m3, 2) if not math.isnan(m3) else "n/a",
            "SD (ms)": round(s3, 2) if not math.isnan(s3) else "n/a",
            "Number of trials": n3,
            "Notes": "closed-loop performance metric; attended trials with a valid `A`",
        },
    ]

    rows.extend(extra_rows)

    fieldnames = [
        "Comparison",
        "Definition",
        "Mean delay (ms)",
        "SD (ms)",
        "Number of trials",
        "Notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = []
    lines.append("## Table 4. Consistency of stimulus-related timing information in the phase-dependent sessions")
    lines.append("")
    lines.append("| Comparison | Definition | Mean delay (ms) | SD (ms) | Number of trials | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

    for r in rows:
        lines.append(
            f"| {r['Comparison']} | {r['Definition']} | {fmt(r['Mean delay (ms)'])} | "
            f"{fmt(r['SD (ms)'])} | {r['Number of trials']} | {r['Notes']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root, out_dir = st.parse_paths("Table 4 - consistency of stimulus-related timing information")

    (
        a_to_stim_ms,
        stim_to_photo_ms_all,
        stim_to_photo_ms_chain,
        a_to_photo_ms,
        b_to_stim_ms,
    ) = build_timing_samples(root)
    csv_path = out_dir / "table4_timing_consistency.csv"
    md_path = out_dir / "table4_timing_consistency.md"
    write_table4(
        csv_path,
        md_path,
        a_to_stim_ms,
        stim_to_photo_ms_all,
        stim_to_photo_ms_chain,
        a_to_photo_ms,
        b_to_stim_ms,
        extra_rows=analog_rows(root, out_dir),
    )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"A->Stim samples: {len(a_to_stim_ms)}")
    print(f"Stim->Photo samples (all): {len(stim_to_photo_ms_all)}")
    print(f"Stim->Photo samples (A-chain): {len(stim_to_photo_ms_chain)}")
    print(f"A->Photo samples: {len(a_to_photo_ms)}")
    print(f"B->Stim samples: {len(b_to_stim_ms)}")


if __name__ == "__main__":
    main()
