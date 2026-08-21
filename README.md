# ResilientVisionAutonomous

ResilientVisionAutonomous is a research project for studying the resilience of
autonomous-driving vision systems to rolling-shutter laser attacks, and for
detecting those attacks. It combines two complementary pipelines that were
developed separately and are now integrated in this repository:

1. **Attack & robustness pipeline** (original project): a verified
   rolling-shutter laser attack engine and dataset generator that produce
   attacked image datasets and measure the impact on a YOLO object detector.
2. **CNN attack-detection pipeline** (imported capstone project): a physics
   -based rolling-shutter laser simulator that generates labelled clean /
   attacked training data, and a binary ResNet18 CNN classifier that flags a
   frame as *clean* or *attacked* from the image alone.

A **safety mechanism / trust engine** (imported) then provides a blueprint for
reacting to detector alerts: temporal validation, Bayesian sensor-trust
weighting, a fail-safe state machine, and a tamper-evident audit log. It is kept
deliberately separate from model training and evaluation.

> This repository is research software. It is not a vehicle-control system and
> must not be used to make safety-critical driving decisions.

## Research objective

CMOS rolling-shutter sensors expose image rows at slightly different times. A
time-modulated laser aimed at such a sensor produces structured horizontal
stripes and localised overexposure instead of a uniformly bright frame. These
artifacts can corrupt downstream computer-vision models. The project studies:

- how to generate realistic rolling-shutter laser artifacts in a controlled,
  reproducible way (attack generation),
- how such artifacts affect an object detector (YOLO robustness experiments),
- whether the artifacts can be detected at the frame level by a lightweight CNN
  (binary attack detection),
- how a vehicle could react safely once an attack is suspected (trust engine).

## Repository layout

```text
ResilientVisionAutonomous/
├── attack_engine/          # Verified laser-pattern attack primitives (additive RGB overlay,
│                           #   rolling-shutter artifact patterns, target selection)
├── dataset_generator/      # Dataset discovery, attack orchestration, output writing,
│                           #   CSV provenance metadata
├── dataset_loader/         # KITTI loading, validation, split management
├── config/                 # generator.yaml + paths.py (attack/dataset pipeline)
├── models/                 # YOLO wrapper/trainer/predictor/evaluator/metrics/visualizer (Ultralytics)
├── evaluation/             # Attack-vs-clean detection evaluation for YOLO outputs
├── scripts/                # Experiment runner, YOLO train/eval, pilots, demos
│
├── configs/config.yaml     # CNN pipeline configuration (simulator + training)
├── src/
│   ├── dataset/            # rolling_shutter_simulator_v7.py, dataset_builder.py,
│   │                       #   dataloader.py, kaggle_downloader.py, audit scripts
│   ├── models/             # cnn.py (ResNet18 binary head), ensemble.py (specialists)
│   ├── training/           # train.py (single + ensemble loops), losses.py
│   ├── evaluation/         # evaluate.py (F1/AUC/confusion/per-variation report)
│   └── utils/              # logger.py, extract_nuscenes.py
├── safety_mechanism/       # Trust engine, temporal validator, failsafe state machine,
│                           #   hash-chain audit log, ROS 2 demo nodes, Streamlit dashboard
├── tests/                  # Unit/integration tests + test_pipeline.py sanity checks
├── docs/                   # Design documentation
├── run.py                  # Entry point for the CNN detection pipeline
├── audit_pixels.py         # Pixel-level stripe-visibility audit of a generated dataset
├── exploration.ipynb       # Notebook for inspecting a generated CNN dataset
└── requirements.txt
```

`datasets/`, `data/`, `outputs/`, `results/` and `checkpoints/` hold local data
and generated artifacts; they are gitignored and must be created locally.

## The two rolling-shutter attack implementations

Both are intentional and serve different consumers; neither replaces the other.

| | `attack_engine` (+ `dataset_generator`) | `src/dataset/rolling_shutter_simulator_v7.py` |
|---|---|---|
| Origin | Original ResilientVisionAutonomous | Imported capstone pipeline |
| Approach | Laser-pattern assets projected onto images: additive saturating RGB overlays plus full-frame rolling-shutter artifact patterns (optical integration, CMOS response, bloom, smear) | Physics-based simulation: row-exposure timing × laser PWM modulation, Gaussian elliptical beam, divergence, auto-exposure, sensor noise/defects, domain randomization |
| Output | Attacked datasets under `outputs/` with `metadata.csv` provenance (YOLO-format labels via `scripts/prepare_yolo_dataset.py`) | Labelled PNG dataset under `data/final_dataset/` with `labels.csv` |
| Consumer | YOLO robustness experiments (`models/`, `scripts/experiment_runner.py`) | CNN binary detector (`src/`) |

The simulator file also exists verbatim inside `safety_mechanism/` because that
package is intentionally self-contained (its demos import it flatly). The two
copies are currently identical; treat `src/dataset/` as the canonical copy for
the CNN pipeline.

## Pipeline 1 — CNN attack detection (`run.py`)

### Data flow

```text
clean driving images (data/clean_base/)
        │  rolling_shutter_simulator_v7  (per-variation laser parameters)
        ▼
dataset_builder  ──►  data/final_dataset/<variation>/*.png + labels.csv
        │                (path, label, split, variation, laser params, ...)
        ▼
dataloader (LaserAttackDataset, ImageNet-normalized 224×224 tensors)
        ▼
train (single CNN or per-variation specialist ensemble)  ──►  checkpoints/*.pth
        ▼
evaluate (F1, AUC-ROC, accuracy, precision/recall, confusion matrix,
          per-variation breakdown, single-vs-ensemble table)
```

Binary labels are derived in the dataloader: rows whose `variation` is `clean`
get label 0; every other variation gets label 1. The descriptive variation name
is preserved for per-variation analysis.

### Model

`src/models/cnn.py` implements `LaserAttackCNN`: a torchvision backbone
(ResNet18 by default; ResNet34/EfficientNet-B0 selectable) with the original
classifier replaced by a dropout → Linear(512→256) → ReLU → Linear(256→2) head.
Output is 2-class logits: **0 = clean, 1 = attacked**.

`src/models/ensemble.py` optionally trains one specialist per attack variation
(each specialist sees only clean + its own variation) and averages their
softmax probabilities at inference. Whether the ensemble outperforms the single
CNN on your data is an empirical question — run `--mode compare` after training
both and trust the resulting table, not prior expectations. No benchmark
numbers are bundled with this repository.

### Training setup

`src/training/train.py` provides early stopping (on validation F1), best-checkpoint
selection, cosine/step/plateau schedulers, optional mixed precision on CUDA, and
JSON training history under `paths.results`. `losses.py` offers weighted
cross-entropy or focal loss.

### Dataset requirements (CNN pipeline)

- Place clean background images (e.g. KITTI left-color camera frames) in
  `data/clean_base/`. Any `.png/.jpg/.jpeg/.bmp` files work; they are resized
  to the configured camera resolution (224×224).
- Variation profiles (frequency bands, exposure times, coverage, angles, counts)
  live in `configs/config.yaml`. The shipped default requests ~97,500 images;
  reduce the `count:` values for pilot runs.
- Optional helpers: `python run.py --mode download` (Kaggle KITTI; needs
  `pip install kaggle` and a Kaggle API token) and `python run.py --mode
  extract_nuscenes --nuscenes_root ...` (needs `nuscenes-devkit`).

### Commands (run from the repository root)

```bash
pip install -r requirements.txt

# sanity checks before generating anything
python run.py --mode test            # or: python tests/test_pipeline.py

# generate the labelled dataset (deterministic for a fixed seed)
python run.py --mode generate --seed 42

# train
python run.py --mode train --model single     # one CNN, all variations
python run.py --mode train --model ensemble   # one specialist per variation

# evaluate saved checkpoints
python run.py --mode evaluate --model single  --split test
python run.py --mode evaluate --model ensemble --split test

# single vs ensemble comparison table (requires both trained)
python run.py --mode compare

# audit stripe visibility directly in generated pixels
python audit_pixels.py --dataset data/final_dataset --samples 15
```

## Pipeline 2 — Attack generation and YOLO robustness (original project)

This pipeline measures how laser attacks affect an Ultralytics YOLOv8 detector.

### Data flow

```text
datasets/KITTI/...  +  datasets/patterns/*.png  +  config/generator.yaml
        │
        ▼
dataset_loader (KITTI parsing, validation, splits)
        ▼
attack_engine (laser patterns, projection, rolling-shutter artifacts)
        ▼
dataset_generator (attacked copies + metadata.csv provenance)
        ▼
scripts/prepare_yolo_dataset.py ──► outputs/yolo_dataset (YOLO format)
        ▼
models/ (Ultralytics wrapper: train / predict / validate)  ──►  evaluation/
        ▼
scripts/experiment_runner.py ──► outputs/experiments/<stage>/<run>/ (JSON reports)
```

### Commands

```bash
# end-to-end experiment stages: baseline, random_attack, target_attack, defense
python scripts/experiment_runner.py --stages baseline random_attack \
    --device cuda --epochs 10 --batch-size 16

# individual steps (each script has --help)
python scripts/run_dataset_generator_demo.py      # small attack-generation demo
python scripts/prepare_yolo_dataset.py            # build the YOLO dataset
python scripts/train_yolo.py                      # train the YOLO baseline
python scripts/evaluate_yolo.py                   # evaluate on clean/attacked sets
python scripts/rolling_shutter_artifact_pilot.py  # full-frame artifact pilot renders
```

Dataset placement follows `config/generator.yaml` and `config/paths.py`
(KITTI at `datasets/KITTI/training/image_2`, pattern assets in
`datasets/patterns`). Source datasets are never modified; all results go to
`outputs/`.

## Safety mechanism / trust-engine direction (`safety_mechanism/`)

The imported safety package sketches how a vehicle should react when the CNN
detector raises an alert. It is standalone and deliberately **not** wired into
training or evaluation:

1. `temporal_validator.py` — confirms an attack only if a majority of recent
   frames agree (single-frame false positives are ignored).
2. `trust_engine.py` — Bayesian update of per-sensor trust weights; camera
   trust drops on confirmed attacks and recovers gradually afterwards.
3. `failsafe_state_machine.py` — NORMAL → DEGRADED → EMERGENCY_STOP transitions.
4. `hash_chain_logger.py` — SHA-256 hash-chained, tamper-evident event log.
5. ROS 2 demos (`nuscenes_publisher.py`, `stub_cnn_publisher.py`,
   `demo_dashboard2.py`) replay nuScenes scenes and visualize the pipeline; see
   `safety_mechanism/README.md` and `safety_mechanism_blueprint.md`. They need
   a ROS 2 installation and/or Streamlit and are optional.

Integrating this module with a trained detector's live output is future work.

## Tests

```bash
python -m pytest tests/                 # full unit/integration suite
python -m unittest tests.test_dataset_generator   # generator-focused subset
python tests/test_pipeline.py           # CNN-pipeline sanity checks
```

## Reproducibility

- Dataset generation is seeded (`--seed`, default 42); each variation derives a
  deterministic sub-seed, so regenerating one variation does not disturb others.
- Splits are recorded per image in `data/final_dataset/labels.csv`.
- The attack pipeline records provenance (source image, pattern, dimensions,
  failures) in `metadata.csv` and `generation_summary.json`.
- All hyperparameters and paths for both pipelines live in versioned YAML
  (`config/generator.yaml`, `configs/config.yaml`).
- Datasets, checkpoints, and outputs are never committed; record the seed,
  config, and dataset versions you used alongside any reported result.

## Limitations and future work

- The two attack generators are not interchangeable: `attack_engine` composites
  pattern assets onto source images, while the v7 simulator models row-time ×
  PWM physics. Cross-generating the same scene with both is not automated.
- The CNN dataloader consumes only the `labels.csv` format produced by
  `src/dataset/dataset_builder.py`. An adapter to train on `attack_engine` /
  `dataset_generator` outputs would be needed and does not exist yet.
- The safety mechanism is a standalone design + demos; coupling it to trained
  detector inference is not implemented.
- The simulator produces single frames; video-sequence effects (AE lag, motion
  blur, frame-to-frame drift) are out of scope by design.
- Single-CNN vs ensemble superiority must be established per experiment via
  `run.py --mode compare`; this repository ships no pretrained weights or
  published metrics.
- Kaggle download and nuScenes extraction require optional dependencies and
  external credentials/data.

## References

- S. Köhler, G. Lovisotto, S. Birnbach, R. Baker, and I. Martinovic,
  *"They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS
  Image Sensors,"* ACSAC '21, ACM, 2021.
  DOI: https://doi.org/10.1145/3485832.3488016
- Further parameterizations referenced by the imported modules (duty-cycle /
  stripe-width concealment, full-frame injection) are cited in the module
  docstrings of `src/dataset/` and `safety_mechanism/`.
- KITTI and BDD100K remain the property of their respective creators; comply
  with their licenses before downloading or using them.
