# Honey classification with residual GAT and LightGBM

Code for “Cross-acquisition hyperspectral modelling for honey adulteration using wavelength-graph representation and reduced-waveband analysis”, submitted to *Chemometrics and Intelligent Laboratory Systems*.

## Installation

Use Python 3.10 with PyTorch installed, then run:

```bash
pip install -r requirements.txt
```

## Run

Open the project in PyCharm and run `run.py`. Set `EXPERIMENT` to:

| Value | Experiment |
| --- | --- |
| `main` | Residual GAT + LightGBM |
| `analysis` | Ablation, k sensitivity, wavelength attribution and waveband reduction |
| `baselines` | GraphSAGE, GCN, GIN and standard GAT, each followed by LightGBM |
| `graph_structure` | Self-loops only, adjacent edges with self-loops, and the full graph |

Adjust settings in `experiments/<entry>/config.py`. Neural encoders use seeds 2026–2030, up to 50 epochs, validation after every epoch and early stopping with patience 5. Results are saved to `results_*` under the selected entry.

All graph models use three node features: SG-smoothed SNV values, first derivatives (×10) and second derivatives (×100). The graph-structure ablation uses the same encoder and LightGBM pipeline as the main model.

## Reusing main-model results

To reuse the completed five-seed main-model results, run `run_revision.py` and select the results folder containing `tasks/`. It runs feature ablation, k sensitivity and waveband reduction, then exports Acq5/Acq6 attribution and class-wise results.

Results are collected in `results_revision/results_for_revision.zip`.

## Data and licence

The CSV is included in `data/`. Acquisitions 1–4 are used for training, Acq5 for validation and band selection, and Acq6 for testing.

Code: [MIT](LICENSE). Dataset: [CC BY 4.0; source and attribution](data/README.md).
