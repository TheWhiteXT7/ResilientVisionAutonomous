# Project Audit: GitHub Release Readiness

Audit date: 2026-07-28

## Repository health

The repository has a clear separation between attack logic, dataset generation,
documentation, tests, configuration, and future training/evaluation areas.
The attack engine is isolated from file-system and machine-learning framework
dependencies, which makes it easy to test and reuse. The dataset generator is
also modular: loading, pattern preparation, in-memory attack execution, output
writing, and CSV metadata are separate responsibilities.

The repository is suitable for a first public research release after the owner
selects a license. No source datasets, generated outputs, checkpoints, logs, or
local environments are tracked by the release policy in `.gitignore`.

## Verification performed

- Parsed and compiled all Python modules successfully.
- Imported the dataset generator successfully.
- Reviewed public functions for docstrings and type annotations.
- Reviewed imports for the current small dependency set.
- Checked 79-character PEP-8 line length across dataset-generator and test
  modules.
- Ran `python -m unittest tests.test_dataset_generator`: 5 tests passed.
- Verified `requirements.txt` contains only direct runtime requirements:
  NumPy, Pillow, PyYAML, and tqdm.

The verified `attack_engine` was not modified during this audit. Its few legacy
long lines are noted but intentionally left unchanged to preserve the verified
implementation.

## Completed modules

| Area | Status |
| --- | --- |
| `attack_engine` | Complete and verified additive RGB attack primitive. |
| `dataset_generator.image_loader` | Complete discovery and RGB loading. |
| `dataset_generator.pattern_manager` | Complete pattern loading and resizing. |
| `dataset_generator.attack_pipeline` | Complete validated attack-engine bridge. |
| `dataset_generator.output_writer` | Complete PNG/JPEG generated-image writer. |
| `dataset_generator.metadata` | Complete CSV provenance writer. |
| `dataset_generator.generate_dataset` | Complete configurable CLI orchestration. |
| Tests | Component/integration coverage for the generator's implemented path. |

## Remaining work

- Select and replace the placeholder `LICENSE`; this is the primary public
  release blocker.
- Add dataset-manifest and annotation workflows for formal KITTI, BDD100K, and
  custom-dataset support.
- Add end-to-end test fixtures for the command-line workflow and failure cases.
- Add continuous integration, formatting/lint configuration, and package
  metadata before a broader release.
- Implement planned training, evaluation, model, and reporting modules.

## Recommendations before downloading KITTI

1. Select a project license and review KITTI's current terms, citation, and
   redistribution restrictions. Do not commit downloaded data to this repo.
2. Create and activate a clean Python 3.10+ virtual environment; run
   `pip install -r requirements.txt` and the unit test command from the README.
3. Put only KITTI camera images under `datasets/KITTI/training/image_2` and
   retain the original dataset outside generated-output paths.
4. Add one or more controlled PNG/JPG/JPEG pattern assets to
   `datasets/patterns`, then review `config/generator.yaml`.
5. Run a small subset first, inspect generated RGB images and `metadata.csv`,
   then estimate disk usage before producing the full attacked dataset.
6. Record the exact KITTI release, local directory layout, configuration file,
   pattern assets, and random seed in experiment notes.

## Release recommendation

Publish the repository as a code-only research preview after replacing the
license placeholder. Describe dataset-download and annotation support as
planned/partial, not as bundled data support, until the manifest and annotation
work is complete.