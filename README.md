# Honey classification with residual GAT and LightGBM
This repository contains the code for our manuscript “Cross-acquisition hyperspectral modelling for honey adulteration using wavelength-graph representation and reduced-waveband analysis”, submitted to Chemometrics and Intelligent Laboratory Systems.

Python implementation of residual GAT + LightGBM for hyperspectral honey-adulteration classification. The repository includes feature ablation, correlation-neighbour sensitivity, wavelength attribution, reduced-waveband experiments, spectral LightGBM, and four GNN comparators.

## Setup

Use Python 3.10 and an environment with PyTorch installed. A tested CPU setup can be installed with:

```bash
python -m pip install torch==2.4.1
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build for your GPU using the [official installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`. The locally tested dependency versions are listed in `requirements-tested.txt`.

## Run

Open the repository in PyCharm, select the Python environment, and run `run.py`. Its `EXPERIMENT` setting selects one of the following entry points. The CSV is included in `data/`, and paths are resolved relative to the code.

| Entry point | Experiments | Encoder configuration |
| --- | --- | --- |
| `main` | Full-spectrum residual GAT + LightGBM; configurable ablation, k sensitivity, and waveband experiments | Maximum 50 epochs; patience 5; downstream validation every epoch |
| `analysis` | Feature ablation, k sensitivity, 16/32/48-waveband comparisons, visible/NIR comparisons, and spectral LightGBM | Analysis configuration: 5 epochs; cosine schedule 50; checkpoint evaluation at epoch 1, every 5 epochs, and the final epoch |
| `baselines` | GraphSAGE, GCN, GIN, and standard GAT | Recorded comparator configuration: 5 epochs; original per-model settings |

The main model and analysis experiments have separate training configurations. Each entry point records its own configuration and outputs. Change settings in the corresponding `experiments/<entry>/config.py`. In the main-model launcher, set `EXPERIMENTS = ["all"]` to run all its experiment groups.

Command-line examples, from the repository root:

```bash
python run.py --check
python run.py --experiment main
python run.py --experiment analysis
python run.py --experiment baselines
```

The code is organised as follows:

```text
run.py                  # Select and run an experiment
experiments/
  main/                 # Residual GAT + LightGBM
  analysis/             # Ablation, k sensitivity, attribution, and wavebands
  baselines/            # GraphSAGE, GCN, GIN, and standard GAT
data/                   # Dataset and attribution
```

Each experiment directory contains `config.py` for run settings, `model.py` for the shared model definitions, and `pipeline.py` for the experiment workflow. Baseline architectures are in `experiments/baselines/models.py`.

Results are written under each entry point's `results_*` directory. Outputs include per-run and summary CSVs, checkpoints, validation records, attribution rankings, selected wavebands, and test predictions. Existing completed tasks are reused only when their recorded configuration matches.

## Data and evaluation

The unmodified dataset is distributed with attribution under CC BY 4.0. Its source, authors, licence, and checksum are given in [data/README.md](data/README.md).

The experiments use five concentration classes: 0%, 5%, 10%, 25%, and 50%. The fixed cohort filter excludes `Brand=C4`, `Class=ManukaUMF10`, `Concentration_Class=50`. After filtering, Acquisitions 1–4 contain 5600 training spectra, Acquisition 5 contains 1400 validation spectra, and Acquisition 6 contains 1400 test spectra, each with 128 wavelengths.

Correlation graphs are estimated from Acquisitions 1–4. Checkpoint selection, waveband selection, LightGBM stopping, and threshold calibration use Acquisition 5. The encoder is then fixed, and the final LightGBM is refitted on Acquisitions 1–5 before test evaluation. GNN comparators retain their three node features and embedding-only LightGBM inputs; the residual model uses four node features and combines its embedding with six raw-spectrum statistics.

The final main-model LightGBM uses a maximum of 3000 boosting rounds, a learning rate of 0.01, 31 leaves, and early stopping with a patience of 100 rounds on Acquisition 5 multiclass log loss. The 500-round LightGBM probe is used for encoder checkpoint selection. Comparator-specific parameters are defined in `experiments/baselines/baseline_config.py`.

Accuracy, balanced accuracy, macro-F1, and class-0 recall are saved. Repeated-run summaries use the sample standard deviation (`ddof=1`). Defaults use seeds 42, 123, 2024, 3407, and 9999.

## Checks

```bash
python check_installation.py
python check_installation.py --tests
```

The first command checks dependencies, CSV loading, and all three entry points. `--tests` also runs the existing protocol and regression tests with mock or small synthetic fixtures.

`reference_model.py` and the text files in `architecture_sources/` are reference snapshots for these regression tests. Run experiments through the documented entry points.

## Licence

Project code is provided under the [MIT licence](LICENSE). The bundled third-party dataset retains its [CC BY 4.0 licence and attribution](data/README.md). Dependencies retain their own licences.
