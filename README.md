# Code release for the manuscript: "An EEG dataset acquired during closed-loop phase-dependent visual stimulation"

Scripts that regenerate the technical-validation outputs reported in the data
descriptor: Tables 3, 4 and 5, and Figure 2. They read a released copy of the
BIDS dataset and write to `derivatives/technical_validation/` inside it. None of
them modify the released data.

This directory is self-contained: it is both the `code/` directory of the BIDS
dataset and the whole of the repository at
https://github.com/Takayuki-Onojima/EEG-closedloop-BIDS-validation.

## Scripts overview

### 1. Phase extraction

`compare_trigger_references.py` — measures the realised stimulation phase against
each of the three timestamps available for every stimulus:

- `A`, the trigger the Speedgoat emitted when it detected the target phase
- the stimulation PC's own trigger, that is, the presentation command
- `B`, the photosensor, which marks the actual luminance change on the display

Because the three differ only by the inter-trigger latency, the choice of anchor
shifts the realised phase; this script quantifies that. It writes a cache that
Table 5 and Figure 2 both consume, so it has to run first.

### 2. Tables

- `technical_validation_table3_trial_completeness.py` — trial counts per participant
- `technical_validation_table4_timing_consistency.py` — latencies between the three triggers
- `technical_validation_table5_phase_statistics.py` — circular statistics of the realised phase

### 3. Figure

- `technical_validation_figure2_phase_targeting.py` — phase-targeting accuracy, written as SVG, PDF and PNG
- `figure_style.py` — shared figure sizing, type and path handling; imported by the others, not run on its own

## Usage examples

Every script takes `--bids_root`, which defaults to the parent of the directory
holding the script. That default is correct when this directory sits inside the
dataset as `code/`; pass the option explicitly when the code lives elsewhere.

```
# from inside the dataset
python code/compare_trigger_references.py
python code/technical_validation_table5_phase_statistics.py

# with the code checked out separately
python compare_trigger_references.py --bids_root /path/to/BIDS_EEG_Closed-loop_Visual_Stimu_Exp
python technical_validation_figure2_phase_targeting.py --bids_root /path/to/dataset --out /tmp/figures
```

Run order: `compare_trigger_references.py` first, then anything else. Tables 3
and 4 are independent of it and of each other.

```
python code/compare_trigger_references.py                       # about 30 min
python code/technical_validation_table3_trial_completeness.py
python code/technical_validation_table4_timing_consistency.py
python code/technical_validation_table5_phase_statistics.py
python code/technical_validation_figure2_phase_targeting.py
```

## Implementation details

### Phase extraction (`compare_trigger_references.py`)

Reproduces the online control pipeline so that the measured phase is comparable
with what the closed-loop system was targeting:

- takes the participant-specific channel named in the `phase_estimation_channel`
  column of `participants.tsv`
- re-references it to the average of the two earlobes, `signal − X2/2`, since
  `X2` is the right earlobe recorded against the left-earlobe reference
- decimates from 5000 to 500 Hz, the rate the online system worked at
- band-pass filters between 6 and 8 Hz with a 128th-order FIR filter applied
  forwards and backwards, giving zero phase distortion
- converts to an analytic signal by the Hilbert transform and interpolates it
  linearly in the complex plane, so the sampling instant is not tied to the 2 ms
  grid; index rounding alone would introduce up to ±2.7°, the same order as the
  bias being measured
- reads the phase 200 ms before the anchoring trigger, the prestimulus reference
  the closed-loop system targeted

Reading and filtering all 109 phase-dependent runs takes roughly half an hour, so
the extracted phases are cached in
`derivatives/technical_validation/trigger_reference_comparison.npz`. Delete that
file to force a full recomputation; with the cache present, Table 5 and Figure 2
rerun in seconds.

### Circular statistics (`technical_validation_table5_phase_statistics.py`)

Follows Fisher, *Statistical Analysis of Circular Data* (1993): mean resultant
length `R`; circular standard deviation `sqrt(-2 ln R)`; a large-sample 95 %
confidence interval `mean ± 1.96 · circSD / sqrt(n)`, accurate here because `R`
exceeds 0.9 in every cell; and the Rayleigh statistic `Z = n R²`.

### Figures (`figure_style.py`)

Sets one typeface and one set of type sizes for all figures, at the widths and
height limit of the target journal. Text is exported as text rather than
outlines, so the SVG and PDF remain editable.

## Requirements

Python 3.10 or later. Tables 3 and 4 use only the standard library; the phase
analysis and Figure 2 additionally need the packages below, pinned to the
versions the released outputs were produced with.

```
pip install -r requirements.txt
```

| Package | Tested version |
| --- | --- |
| `numpy` | 2.0.1 |
| `scipy` | 1.15.3 |
| `matplotlib` | 3.10.6 |

Figure 2 is typeset in Arial; if it is unavailable, matplotlib falls back to
Helvetica and then to DejaVu Sans and the figure still renders.

### Setting up an environment

With `venv` and `pip`:

```
python -m venv .venv
source .venv/bin/activate          # Linux, macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

With conda:

```
conda create -n closedloop python=3.10 numpy scipy matplotlib
conda activate closedloop
```

## License

The code in this directory is released under the MIT License; see `LICENSE`.
The dataset it analyses is distributed separately and under its own terms.
