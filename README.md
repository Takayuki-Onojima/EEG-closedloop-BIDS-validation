# Code release for the manuscript: "An EEG dataset acquired during closed-loop phase-dependent visual stimulation"

Scripts that regenerate the technical-validation outputs reported in the data
descriptor: Tables 3, 4 and 5, and Figure 2. They read a released copy of
the BIDS dataset and write to `derivatives/technical_validation/` inside it. None
of them modify the released data.

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

`technical_validation_figure2_phase_targeting.py` draws all three panels of
Figure 2 and writes them as SVG, PDF and PNG:

- **a** phase-targeting accuracy on the control channel, from the cache written by `compare_trigger_references.py`
- **b** scalp topography of the same phase locking at seven latencies around visual onset, from the cache written by `compute_plf_topography.py`
- **c** the phase-locking factor over time on the control channel

`compute_plf_topography.py` is the slow half of panel b and is kept separate so
it can be run once; `figure_style.py` holds the shared figure sizing, type and
path handling and is imported rather than run.

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

Run order: the two extraction scripts first, then anything that reads their
caches. Figure 2 needs both. Tables 3 and 4 are independent of everything else.

```
python code/compare_trigger_references.py                       # about 10 min
python code/compute_plf_topography.py                           # about 30 min
python code/technical_validation_table3_trial_completeness.py
python code/technical_validation_table4_timing_consistency.py
python code/technical_validation_table5_phase_statistics.py
python code/technical_validation_figure2_phase_targeting.py
```

## Implementation details

### Phase extraction (`compare_trigger_references.py`)

Reproduces the online control pipeline with MNE-Python, so that the measured
phase is comparable with what the closed-loop system was targeting and so that
channel names, units and the reference scheme are taken from the BrainVision
header rather than assumed:

- takes the participant-specific channel named in the `phase_estimation_channel`
  column of `participants.tsv`
- restores the recording reference, the left earlobe, as a zero-valued channel
  and re-references to the average of that channel and `X2`, the recorded right
  earlobe, giving the linked-earlobe montage the online system used
- resamples from 5000 to 500 Hz, the rate the online system worked at
- band-pass filters between 6 and 8 Hz with a zero-phase FIR filter (`firwin`,
  1 Hz transition bands either side)
- converts to an analytic signal by the Hilbert transform and interpolates it
  linearly in the complex plane, so the sampling instant is not tied to the 2 ms
  grid; index rounding alone would introduce up to ±2.7°, the same order as the
  bias being measured
- reads the phase 200 ms before the anchoring trigger, the prestimulus reference
  the closed-loop system targeted

The band-pass deserves a note. Online, the controller had to run causally and
used a 128th-order FIR at 500 Hz, which is only 258 ms long and passes 4.4 to
9.6 Hz at half power rather than the nominal 6 to 8 Hz. Offline there is no such
constraint, and the filter used here is 3.3 s long and passes 5.6 to 8.4 Hz. The
two therefore define "phase" differently: against the controller's own broadband
definition the six conditions reach a resultant length of about 0.93, against the
narrow-band definition about 0.69. Both are correct measurements of different
quantities; the narrow-band one is used here because it is what a reader
recomputing the phase of the 6-8 Hz component from the released data will obtain.

Reading and filtering all 109 phase-dependent runs takes roughly ten minutes, so
the extracted phases are cached in
`derivatives/technical_validation/trigger_reference_comparison.npz`. Delete that
file to force a full recomputation; with the cache present, Table 5 and Figure 2
rerun in seconds.

### Topography (`compute_plf_topography.py`)

Same preprocessing, but applied to all 63 scalp electrodes rather than one, and
therefore about three times slower. For each stimulus the target phase assigned
to that trial is subtracted from the measured phase, which is what makes the six
conditions poolable — without it they would cancel, since they are spread evenly
around the cycle by design. Locking is summarised per electrode as the resultant
length of those differences, computed within each participant and then averaged
across participants so that participants with more trials do not dominate.
Electrode positions come from MNE's `standard_1005` montage. Results are cached
in `plf_topography_cache.npz` and also written as `plf_topography.csv`; with the
cache present, Figure 2 redraws in seconds.

### Circular statistics (`technical_validation_table5_phase_statistics.py`)

Follows Fisher, *Statistical Analysis of Circular Data* (1993): mean resultant
length `R`; circular standard deviation `sqrt(-2 ln R)`; a large-sample 95 %
confidence interval `mean ± 1.96 · circSD / sqrt(n)`, which is adequate here
because every cell pools more than two thousand stimuli; and the Rayleigh
statistic `Z = n R²`.

### Figures (`figure_style.py`)

Sets one typeface and one set of type sizes for all figures, at the widths and
height limit of the target journal. Text is exported as text rather than
outlines, so the SVG and PDF remain editable.

## Requirements

Python 3.10 or later. Tables 3 and 4 use only the standard library; the phase
analysis and Figure 2 additionally need the packages below.

```
pip install -r requirements.txt
```

| Package | Tested version |
| --- | --- |
| `numpy` | 2.0.1 |
| `scipy` | 1.15.3 |
| `matplotlib` | 3.10.6 |
| `mne` | 1.12.1 |

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
pip install mne
```

## License

The code in this directory is released under the MIT License; see `LICENSE`.
Copyright (c) 2025 Takayuki Onojima, Keiichi Kitajo.

The dataset these scripts analyse is distributed separately, under the Creative
Commons Attribution-NonCommercial 4.0 International License.
