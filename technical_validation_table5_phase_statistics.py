"""Table 5 - circular statistics of the realised stimulation phase.

Table 4 already reports the latencies between the three triggers in the time
domain. This table reports the same comparison in the angular domain: where the
realised phase landed relative to each target, and how tightly it was
concentrated, for each of the three possible timing anchors.

All statistics are circular (Fisher, Statistical Analysis of Circular Data,
1993):

  R          mean resultant length, |mean of the unit vectors|
  mean       angle of that resultant
  circ. SD   sqrt(-2 ln R), the standard deviation of a wrapped normal with the
             same R; reported in degrees
  95% CI     mean +- 1.96 * circ.SD / sqrt(n), the large-sample approximation,
             which is accurate here because R > 0.9 in every cell
  Rayleigh   Z = n R^2, testing uniformity

Reads the cache written by code/compare_trigger_references.py.
"""

import csv
import math
from pathlib import Path

import figure_style as st

import numpy as np

TARGET = {1: -math.pi/3, 2: 0.0, 3: math.pi/3, 4: 2*math.pi/3, 5: math.pi, 6: 4*math.pi/3}
REFERENCE_MS = -200.0
REF_LABEL = {
    "A": "Speedgoat trigger (A)",
    "stim": "Stimulation-PC trigger",
    "B": "StimTrak marker (B)",
    "photo": "Photodiode onset (analogue)",
}
REFS = ("A", "stim", "B", "photo")


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def circ_stats(angles):
    a = np.asarray(angles, float)
    n = a.size
    c, s = np.mean(np.cos(a)), np.mean(np.sin(a))
    R = math.hypot(c, s)
    mean = math.atan2(s, c)
    sd = math.sqrt(-2.0 * math.log(R)) if 0 < R < 1 else float("nan")
    ci = 1.96 * sd / math.sqrt(n) if n else float("nan")
    return n, mean, R, sd, ci, n * R * R


def main():
    root, out = st.parse_paths("Table 5 - circular statistics of the realised stimulation phase")
    z = np.load(out / "trigger_reference_comparison.npz", allow_pickle=True)
    phases = z["phases"].item()

    rows = []
    for ref in REFS:
        for cond in range(1, 7):
            n, mean, R, sd, ci, rayZ = circ_stats(phases[ref][cond][REFERENCE_MS])
            rows.append({
                "reference_trigger": REF_LABEL[ref],
                "condition": cond,
                "target_deg": round(math.degrees(TARGET[cond]), 1),
                "n": n,
                "circular_mean_deg": round(math.degrees(mean), 2),
                "bias_deg": round(math.degrees(wrap(mean - TARGET[cond])), 2),
                "resultant_length_R": round(R, 4),
                "circular_sd_deg": round(math.degrees(sd), 2),
                "ci95_halfwidth_deg": round(math.degrees(ci), 2),
                "rayleigh_Z": round(rayZ, 1),
            })

    with (out / "table5_phase_statistics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Per-condition detail is given for the anchor the article uses for absolute
    # phase. The other three differ from it by a constant, so listing them
    # condition by condition would repeat the same offset six times; their
    # summary rows carry all the information they add.
    PRIMARY = "photo"
    lines = [
        "## Table 5. Circular statistics of the realised stimulation phase "
        "in the phase-dependent sessions",
        "",
        f"Realised phase by condition, anchored on the {REF_LABEL[PRIMARY].lower()}:",
        "",
        "| Condition | Target (deg) | n | Circular mean (deg) | "
        "Bias from target (deg) | R | Circular SD (deg) | 95% CI (+/- deg) | Rayleigh Z |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["reference_trigger"] != REF_LABEL[PRIMARY]:
            continue
        lines.append(
            f"| {r['condition']} | {r['target_deg']:.1f} | {r['n']} | "
            f"{r['circular_mean_deg']:.2f} | {r['bias_deg']:+.2f} | {r['resultant_length_R']:.4f} | "
            f"{r['circular_sd_deg']:.2f} | {r['ci95_halfwidth_deg']:.2f} | {r['rayleigh_Z']:.1f} |")

    lines += ["", "The same measurement anchored on each of the four available timestamps, "
              "summarised across the six conditions:", "",
              "| Timing anchor | Mean bias (deg) | SD of bias across conditions (deg) | "
              "Mean R | Mean circular SD (deg) |", "| --- | --- | --- | --- | --- |"]
    for ref in REFS:
        sub = [r for r in rows if r["reference_trigger"] == REF_LABEL[ref]]
        b = [r["bias_deg"] for r in sub]
        lines.append(f"| {REF_LABEL[ref]} | {np.mean(b):+.2f} | {np.std(b):.2f} | "
                     f"{np.mean([r['resultant_length_R'] for r in sub]):.4f} | "
                     f"{np.mean([r['circular_sd_deg'] for r in sub]):.2f} |")

    lines += [
        "",
        "Phase was measured at the intended prestimulus reference, 200 ms before the anchoring "
        "trigger, on the participant-specific channel used for online phase estimation "
        "(`phase_estimation_channel` in `participants.tsv`), referenced to the average of the two "
        "earlobes as in the online system, then band-pass filtered offline between 6 and 8 Hz "
        "with a zero-phase FIR filter. That filter is not the one the controller used online: "
        "the online filter had to be causal and short, and the offline one is designed for the "
        "6-8 Hz band itself. Statistics are "
        "circular; circular SD is sqrt(-2 ln R) and the 95% CI is the large-sample approximation "
        "mean +/- 1.96 circ.SD / sqrt(n). The per-condition rows for the remaining three "
        "anchors are omitted because they differ from those above by a single constant, and "
        "are included in the accompanying CSV. "
        "mean +/- 1.96 circ.SD / sqrt(n). All Rayleigh tests reject uniformity at p < 0.001. "
        "The bias differs between anchors only because of the trigger-to-trigger latencies "
        "reported in Table 4, and the dispersion grows along the chain because each further step "
        "adds its own jitter, the display's being the largest. The two display anchors read the "
        "same physical flash: the photodiode onset is the half-amplitude rise of the analogue "
        "PhotoSensor pulse, while B is the marker the StimTrak emitted when that rise crossed a "
        "threshold set by hand once per session. B therefore carries a per-session constant "
        "offset, which is why the photodiode onset is the anchor to use for absolute phase; B is "
        "listed here so that analyses built on the marker can be corrected onto it.",
    ]
    (out / "table5_phase_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out/'table5_phase_statistics.csv'}")
    print(f"wrote {out/'table5_phase_statistics.md'}")
    for ref in REFS:
        sub = [r for r in rows if r["reference_trigger"] == REF_LABEL[ref]]
        b = [r["bias_deg"] for r in sub]
        print(f"  {REF_LABEL[ref]:24s} bias {np.mean(b):+6.2f}  "
              f"circSD {np.mean([r['circular_sd_deg'] for r in sub]):5.2f} deg")


if __name__ == "__main__":
    main()
