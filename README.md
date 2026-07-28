# ResilientVisionAutonomous

ResilientVisionAutonomous is a research-oriented project for studying the
resilience of autonomous-driving vision workflows to rolling-shutter laser
perturbations. Its first release provides a verified RGB attack engine and a
reproducible pipeline for building attacked image datasets from local source
images and laser-pattern assets.

> This repository is research software. It is not a vehicle-control system and
> must not be used to make safety-critical driving decisions.

## Motivation

CMOS rolling-shutter sensors expose image rows at different times. Time-varying
light can therefore create structured image artifacts that may affect vision
systems. This project provides a small, traceable foundation for creating
controlled attacked datasets and evaluating future resilience methods.

## Objectives

- Generate paired clean and laser-corrupted RGB image datasets.
- Preserve per-image provenance, dimensions, chosen pattern, and failures.
- Keep attack generation independent of training and evaluation workflows.
- Support reproducible experiments with local KITTI, BDD100K, or custom image
  directories.
- Provide a maintainable base for future detection and mitigation research.

## System architecture

```text
source RGB images + laser pattern assets + generator.yaml
                         |
                         v
                 dataset_generator
       discover -> load -> resize pattern -> attack -> save -> metadata.csv
                         |
                         v
                   attack_engine
                         |
                         v
          outputs/generated_images (attacked RGB dataset)
                         |
              +----------+----------+
              v                     v
        training (planned)    evaluation (planned)
```

`attack_engine` owns only the verified additive, saturating RGB overlay.
`dataset_generator` owns discovery, pattern preparation, output layout, and
CSV provenance. It never alters the raw source dataset.

## Directory structure

```text
ResilientVisionAutonomous/
├── attack_engine/       # Verified, standalone attack primitive
├── dataset_generator/   # Dataset discovery, generation, output, metadata
├── config/              # Generator configuration templates
├── datasets/            # Local datasets and pattern assets (gitignored)
├── outputs/             # Generated datasets, logs, checkpoints (gitignored)
├── docs/                # Technical design documentation
├── tests/               # Automated unit tests
├── training/            # Planned training workflows
├── evaluation/          # Planned evaluation workflows
├── models/              # Planned model definitions/artifacts
├── notebooks/           # Exploratory analysis
├── scripts/             # Project utilities
├── requirements.txt     # Runtime dependencies
└── LICENSE              # License selection currently pending
```

## Installation

Requirements:

- Python 3.10 or newer
- `pip`
- A local copy of source images and one or more laser-pattern images

Clone and install:

```bash
git clone <your-fork-or-repository-url>
cd ResilientVisionAutonomous
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Verify the checkout:

```bash
python -m unittest tests.test_dataset_generator
```

The project has no packaged dataset. Dataset directories and generated outputs
are deliberately excluded from version control.

## Usage

1. Place input images in a local directory. The default configuration expects
   KITTI camera images at `datasets/KITTI/training/image_2`.
2. Place PNG, JPG, or JPEG laser patterns in `datasets/patterns`.
3. Review [config/generator.yaml](config/generator.yaml). Paths are interpreted
   relative to that YAML file.
4. Generate the attacked dataset:

   ```bash
   python -m dataset_generator.generate_dataset --config config/generator.yaml
   ```

The generator recursively discovers PNG/JPG/JPEG/BMP source images, uses a
round-robin or seeded-random pattern choice, resizes the selected pattern to
the source dimensions, applies the verified attack, and writes the result under
`outputs/generated_images/kitti_laser` by default. It writes one row per result
or failure to `metadata.csv` and continues after isolated image errors.

For source directories other than KITTI, update `input_dataset` in the YAML
file. BDD100K and custom image trees can use the same image-directory workflow;
dataset-specific annotation processing is not yet implemented.

## Current progress

- Complete and verified standalone `attack_engine` package.
- Functional dataset generator with image discovery, pattern preparation,
  attack orchestration, output writing, CSV metadata, logging, and progress.
- Unit coverage for discovery, pattern resizing, attack integration, output
  writing, and metadata creation.
- Dataset-generator design documentation in
  [docs/DatasetGeneratorDesign.md](docs/DatasetGeneratorDesign.md).

## Roadmap

1. Add dataset manifests and annotation-preservation workflows for KITTI,
   BDD100K, and custom datasets.
2. Add integration fixtures, integrity checks, and continuous integration.
3. Add baseline robustness/detection models and training workflows.
4. Add evaluation, reporting, and reproducibility tooling.
5. Publish experiment configurations and results after dataset licensing and
   ethical-use review.

## License

License selection is pending. The current [LICENSE](LICENSE) file is a
placeholder and does not grant permission to reuse the project. Select and add
an appropriate license before accepting external contributions or publishing a
release intended for reuse.

## References and acknowledgements

- A. Jain, L. E. et al., *They See Me Rollin': Inherent Vulnerability of the
  Rolling Shutter in CMOS Image Sensors*, USENIX Security Symposium, 2021.
  The attack operation was derived from this research:
  <https://www.usenix.org/conference/usenixsecurity21/presentation/jain>.
- KITTI and BDD100K remain the property of their respective creators and are
  not distributed by this repository. Users must comply with their applicable
  licenses, terms, and citation requirements before downloading or using them.
- NumPy, Pillow, PyYAML, and tqdm make the current implementation possible.