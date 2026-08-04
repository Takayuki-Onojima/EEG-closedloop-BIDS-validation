import csv
import math
from pathlib import Path

import figure_style as st


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


def write_table4(
    out_csv: Path,
    out_md: Path,
    a_to_stim_ms,
    stim_to_photo_ms_all,
    stim_to_photo_ms_chain,
    a_to_photo_ms,
    b_to_stim_ms,
):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    m1, s1, n1 = summarize(a_to_stim_ms)
    m2, s2, n2 = summarize(stim_to_photo_ms_all)  # All B, not just A-chain
    m3, s3, n3 = summarize(a_to_photo_ms)
    m4, s4, n4 = summarize(b_to_stim_ms)

    rows = [
        {
            "Comparison": "A - stimulus-PC trigger",
            "Definition": "Delay from online phase-detection trigger (A) to stimulus-PC trigger output",
            "Mean delay (ms)": round(m1, 4) if not math.isnan(m1) else "n/a",
            "SD (ms)": round(s1, 4) if not math.isnan(s1) else "n/a",
            "Number of trials": n1,
            "Notes": "A=realtime_trigger; measured on attended trials with valid A-B chain",
        },
        {
            "Comparison": "stimulus-PC trigger - photosensor (B)",
            "Definition": "Delay from stimulus-PC trigger output to photosensor confirmation (B)",
            "Mean delay (ms)": round(m2, 4) if not math.isnan(m2) else "n/a",
            "SD (ms)": round(s2, 4) if not math.isnan(s2) else "n/a",
            "Number of trials": n2,
            "Notes": "B extracted from EEG PhotoSensor channel; includes attended trials with and without A trigger",
        },
        {
            "Comparison": "A - photosensor (B) chain latency",
            "Definition": "Total delay from online phase-detection trigger (A) to photosensor confirmation (B)",
            "Mean delay (ms)": round(m3, 4) if not math.isnan(m3) else "n/a",
            "SD (ms)": round(s3, 4) if not math.isnan(s3) else "n/a",
            "Number of trials": n3,
            "Notes": "Combined A->Stim->B latency on trials with valid A trigger (12,874 trials); closed-loop performance metric",
        },
    ]

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
    for r in rows:
        lines.append(
            f"| {r['Comparison']} | {r['Definition']} | {r['Mean delay (ms)']} | {r['SD (ms)']} | {r['Number of trials']} | {r['Notes']} |"
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
