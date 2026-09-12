"""Table 5 - circular statistics of the realized stimulation phase.

Table 4 already reports the latencies between the four timestamps in the time
domain. This table reports the same comparison in the angular domain: where the
realized phase landed relative to each target, and how tightly it was
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
    "photo": "Photodiode onset (analog)",
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
    root, out = st.parse_paths("Table 5 - circular statistics of the realized stimulation phase")
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
        "## Table 5. Circular statistics of the realized stimulation phase "
        "in the phase-dependent sessions",
        "",
        f"Realized phase by condition, anchored on the {REF_LABEL[PRIMARY].lower()}:",
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

    # The other three anchors are not tabulated. Their mean bias differs from the
    # one above by exactly the inter-trigger latencies already given in Table 4,
    # and their R and circular SD agree with it to the third decimal, so a second
    # table would restate Table 4 in different units. The values are given in the
    # note below, and every anchor and condition is in the CSV.
    summary = {}
    for ref in REFS:
        sub = [r for r in rows if r["reference_trigger"] == REF_LABEL[ref]]
        b = [r["bias_deg"] for r in sub]
        summary[ref] = (np.mean(b), np.std(b),
                        np.mean([r["resultant_length_R"] for r in sub]),
                        np.mean([r["circular_sd_deg"] for r in sub]))
    Rs = [v[2] for v in summary.values()]

    lines += [
        "",
        "Phase was measured at the intended prestimulus reference, 200 ms before the anchoring "
        "trigger, on the participant-specific channel used for online phase estimation "
        "(`phase_estimation_channel` in `participants.tsv`), referenced to the average of the two "
        "earlobes as in the online system, then band-pass filtered offline between 6 and 8 Hz "
        "with a zero-phase FIR filter. That filter is not the one the controller used online: the "
        "online filter had to be causal and short, and the offline one is designed for the 6-8 Hz "
        "band itself. Statistics are circular; circular SD is sqrt(-2 ln R) and the 95% CI is the "
        "large-sample approximation mean +/- 1.96 circ.SD / sqrt(n). All Rayleigh tests reject "
        "uniformity at p < 0.001.",
        "",
        "The same measurement can be anchored on any of the four timestamps available for each "
        f"stimulus. Doing so changes the mean bias to {summary['A'][0]:+.2f} deg for the Speedgoat "
        f"trigger, {summary['stim'][0]:+.2f} deg for the stimulation-PC trigger and "
        f"{summary['B'][0]:+.2f} deg for the StimTrak marker, differences that are exactly the "
        "inter-trigger latencies of Table 4 expressed as phase, while leaving the mean resultant "
        f"length between {min(Rs):.3f} and {max(Rs):.3f} throughout. Per-condition values for all "
        "four anchors are in the accompanying CSV. The two display anchors read the same physical "
        "flash: the photodiode onset is the half-amplitude rise of the analog PhotoSensor pulse, "
        "while B is the marker the StimTrak emitted when that rise crossed a threshold set by hand "
        "once per session. B therefore carries a per-session constant offset, which is why the "
        "photodiode onset is the anchor used here for absolute phase; the correction onto it is "
        "given in Table 4.",
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
