import csv
from pathlib import Path

import figure_style as st


NOMINAL_TRIALS = {
    "baseline": 140,
    "phasedep": 140,
}


def detect_task(file_name: str) -> str:
    if "_task-baseline_" in file_name:
        return "baseline"
    if "_task-phasedep_" in file_name:
        return "phasedep"
    return "other"


def extract_run(file_name: str) -> str:
    if "_run-" not in file_name:
        return "n/a"
    return file_name.split("_run-")[1].split("_")[0]


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def summarize_trial_completeness(root: Path):
    rows_out = []
    for ev_path in sorted(root.glob("sub-*/eeg/*_events.tsv")):
        task = detect_task(ev_path.name)
        if task not in {"baseline", "phasedep"}:
            continue

        events = read_tsv(ev_path)
        recorded_trials = sum(1 for r in events if r.get("trial_type") == "visual_stimulus")
        nominal = NOMINAL_TRIALS[task]
        delta = recorded_trials - nominal

        if delta == 0:
            notes = "complete"
        elif delta < 0:
            notes = f"missing {-delta} trials"
        else:
            notes = f"{delta} extra trials"

        rows_out.append(
            {
                "Participant": ev_path.parent.parent.name,
                "Task": task,
                "Session / run": f"run-{extract_run(ev_path.name)}",
                "Nominal trials": nominal,
                "Recorded trials": recorded_trials,
                "Notes": notes,
            }
        )

    return rows_out


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Participant",
        "Task",
        "Session / run",
        "Nominal trials",
        "Recorded trials",
        "Notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("## Table 3. Trial completeness across participants and sessions")
    lines.append("")
    lines.append("| Participant | Task | Session / run | Nominal trials | Recorded trials | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| {Participant} | {Task} | {Session / run} | {Nominal trials} | {Recorded trials} | {Notes} |".format(
                **r
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root, out_dir = st.parse_paths("Table 3 - trial completeness across participants")
    rows = summarize_trial_completeness(root)

    csv_path = out_dir / "table3_trial_completeness.csv"
    md_path = out_dir / "table3_trial_completeness.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    n_incomplete = sum(1 for r in rows if r["Notes"] != "complete")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Rows: {len(rows)}")
    print(f"Incomplete rows: {n_incomplete}")


if __name__ == "__main__":
    main()
