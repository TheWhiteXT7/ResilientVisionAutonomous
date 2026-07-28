# Dataset Generator Technical Design

## 1. Purpose and scope

`dataset_generator` will create reproducible, paired autonomous-driving image
datasets containing source (clean) RGB images and one or more attacked variants.
It consumes the verified `attack_engine` package; it does not reimplement attack
math, train models, or interpret detector annotations.

The first implementation targets KITTI, BDD100K, and a documented custom-dataset
layout. It is intentionally image-centric: source labels are preserved as
provenance references and copied only when explicitly requested in configuration.
This keeps generated images traceable without claiming that every perturbation
requires a changed label.

### Design goals

- Preserve a one-to-many relationship between each source image and its attacked
  variants.
- Make every generated artifact traceable to its input image, pattern, attack
  engine version, configuration, and output checksum.
- Provide a deterministic order and deterministic output identities for a fixed
  manifest and configuration.
- Keep data-set discovery, pattern preparation, attack execution, persistence,
  and metadata ownership separate.
- Fail one bad sample without silently corrupting the rest of a batch, while
  recording the failure in the run manifest.

### Non-goals

- Pattern synthesis and rolling-shutter physical simulation.
- Resizing inside `attack_engine`; patterns are prepared before that boundary.
- Dataset download, model training, evaluation, or annotation conversion.
- Mutating source datasets.

## 2. Package structure

```text
dataset_generator/
├── image_loader.py
├── pattern_manager.py
├── attack_pipeline.py
├── output_writer.py
├── metadata.py
├── generate_dataset.py
└── README.md
```

`dataset_generator` is a package-level workflow; the listed modules are the
planned public surface. The implementation may add private helpers (names
prefixed with `_`) and tests, but must not move public responsibilities between
these modules without updating this design.

## 3. Core data contracts

Implementation should use typed dataclasses (or equivalent immutable mappings)
for the following boundaries. Paths are stored in manifests relative to the
declared dataset or run root whenever possible, using POSIX separators for
portability.

| Contract | Required fields | Meaning |
| --- | --- | --- |
| `SourceSample` | `sample_id`, `dataset_name`, `split`, `image_path`, `source_relative_path`, `width`, `height`, `label_paths` | One discoverable source image and optional associated annotation paths. |
| `PatternSpec` | `pattern_id`, `source_path`, `mode`, `parameters` | A named pattern asset and its preparation rules. |
| `PreparedPattern` | `pattern_id`, `rgb_array`, `source_sha256`, `preparation` | RGB `uint8` array compatible with a particular source image, plus provenance. |
| `AttackVariant` | `variant_id`, `pattern_id`, `parameters` | One requested attack instance for a source image. |
| `GeneratedSample` | `generated_id`, `source_sample_id`, `variant_id`, `clean_output_path`, `attacked_output_path`, `record_id` | The successfully written paired result. |
| `FailureRecord` | `source_sample_id`, `variant_id` (if known), `stage`, `error_type`, `message` | A recoverable per-item failure. |

### Image and identifier rules

- All images crossing into `attack_engine.apply_attack(image, pattern)` are
  RGB NumPy arrays with `dtype=uint8` and shape `(H, W, 3)`.
- Input alpha channels are converted to RGB according to loader policy; grayscale
  images are rejected by default rather than silently replicated.
- A generated ID is stable: it is derived from the source sample ID, pattern ID,
  variant parameters, and configuration fingerprint. It must not depend on
  processing time or enumeration order.
- `sample_id` is dataset-qualified and collision-safe. For example,
  `kitti/training/image_2/000123` and `bdd100k/train/abc123`.

## 4. Dataset configuration and output layout

The command-line entry point accepts one JSON or YAML configuration file. The
design intentionally keeps the parser behind `generate_dataset.py` so internal
modules receive validated contracts, not raw configuration dictionaries.

```yaml
run:
  output_root: outputs/generated_images/kitti_laser_v1
  copy_clean_images: true
  on_error: continue            # continue | fail_fast
  overwrite: false
  seed: 20260728
source:
  adapter: kitti                # kitti | bdd100k | custom
  root: datasets/KITTI
  split: training
  custom_manifest: null         # required only for adapter: custom
patterns:
  - id: laser_a
    path: patterns/laser_a.png
    fit: resize                 # exact | resize | tile | crop
    interpolation: nearest
  - id: laser_b
    path: patterns/laser_b.png
    fit: tile
attacks:
  selection: cartesian          # cartesian | explicit
  variants:
    - pattern_id: laser_a
    - pattern_id: laser_b
output:
  image_format: png
  labels: reference_only        # reference_only | copy
```

The implementation validates the configuration before it scans source images.
Unknown adapter, fit, selection, and error-policy values are configuration
errors. Explicit variant selection permits future attack parameters without
changing this schema.

```text
<output_root>/
├── clean/<source_relative_path>.png              # optional, never overwrites source
├── attacked/<variant_id>/<source_relative_path>.png
├── metadata/
│   ├── samples.jsonl                             # one record per written variant
│   ├── failures.jsonl                            # one record per recoverable failure
│   ├── run_manifest.json                         # immutable run-level provenance
│   └── source_manifest.jsonl                     # resolved source inventory
└── labels/                                       # only if output.labels: copy
```

Original filename stems and relative hierarchy are retained. If the configured
image format changes a suffix, metadata retains both the source and output
paths. Writes use a temporary sibling file followed by an atomic rename; a
metadata success record is appended only after all artifacts for that generated
sample have been committed.

## 5. Module designs

### `image_loader.py`

**Purpose.** Discover source images through dataset adapters, validate readable
RGB data, and load source samples without changing the source dataset.

**Public API.**

```python
discover_samples(source_config: SourceConfig) -> Iterable[SourceSample]
load_rgb_image(sample: SourceSample) -> np.ndarray
get_dataset_adapter(name: str) -> DatasetAdapter
```

`DatasetAdapter` is a small protocol with `discover(config)` and is the
extension point for new datasets. Built-in adapter names are `kitti`, `bdd100k`,
and `custom`.

**Inputs.** A validated `SourceConfig`; `SourceSample` for image loading;
dataset roots and, for custom data, an explicit manifest. The custom manifest is
JSONL with at least `sample_id`, `image_path`, and optional `split` and
`label_paths`, all paths relative to its declared root.

**Outputs.** A deterministic iterable of `SourceSample` ordered by normalized
relative path, and independently allocated RGB `uint8` arrays.

**Error handling.** Missing root, unknown adapter, malformed custom manifest,
duplicate sample IDs, unreadable image, unsupported image mode, and invalid
path escape are reported as typed `DatasetDiscoveryError` or `ImageLoadError`.
Discovery errors stop the run; image-specific load errors are recorded per
sample and follow `on_error`.

**Interactions.** `generate_dataset.py` calls discovery once and sends each
sample to `attack_pipeline.py`. The pipeline calls `load_rgb_image`; it never
opens paths itself. `metadata.py` receives the resolved source inventory.

#### Built-in adapter behavior

| Adapter | Discovery rule | Labels/provenance |
| --- | --- | --- |
| KITTI | Enumerate the configured split's camera-image directory (default `training/image_2`) recursively or by its documented flat layout. | Pair same-stem annotation files when present (for example `training/label_2`). |
| BDD100K | Enumerate the configured `images/<split>` tree. | Associate the configured BDD100K labels JSON if supplied; image-level label lookup is stored as a reference. |
| Custom | Read the explicit custom JSONL manifest; no directory guessing. | Use optional manifest `label_paths` unchanged after containment validation. |

The exact roots are configuration fields rather than hard-coded assumptions,
allowing official releases and reorganized local copies to coexist.

### `pattern_manager.py`

**Purpose.** Load, validate, prepare, and cache reusable attack pattern assets
so every prepared pattern exactly satisfies `attack_engine`'s image contract.

**Public API.**

```python
load_pattern_specs(config: GeneratorConfig) -> list[PatternSpec]
prepare_pattern(spec: PatternSpec, target_shape: tuple[int, int, int]) -> PreparedPattern
validate_prepared_pattern(pattern: PreparedPattern, target_shape: tuple[int, int, int]) -> None
```

An optional `PatternManager` class owns an in-memory cache keyed by pattern
source checksum plus target dimensions and preparation parameters.

**Inputs.** `PatternSpec`, target `(height, width, 3)`, and RGB PNG/JPEG pattern
assets. Supported fit modes are `exact` (dimensions must match), `resize`,
`tile`, and `crop`. Preparation is explicit in metadata; it does not alter the
immutable source pattern file.

**Outputs.** `PreparedPattern` whose `rgb_array` is a new RGB `uint8` array with
the requested target shape, source SHA-256, selected fit mode, interpolation,
and any crop/tile coordinates.

**Error handling.** Missing/unreadable pattern, unsupported mode, non-RGB
conversion policy violation, impossible crop, and an `exact` size mismatch raise
`PatternPreparationError`. A failure affects every variant depending on that
pattern; under `continue`, the pipeline records a failure for each affected
variant rather than substituting a different pattern.

**Interactions.** `attack_pipeline.py` asks this module for prepared patterns
after a source image is loaded. It does not call `attack_engine` and owns no
output paths. Its preparation record is incorporated by `metadata.py`.

### `attack_pipeline.py`

**Purpose.** Coordinate one source image and one or more attack variants,
calling the verified attack engine and returning in-memory results plus
structured provenance. It is the only dataset-generator module allowed to call
`attack_engine.apply_attack`.

**Public API.**

```python
build_variants(config: GeneratorConfig) -> list[AttackVariant]
process_sample(sample: SourceSample, variants: Sequence[AttackVariant],
               loader: ImageLoader, patterns: PatternManager,
               run_context: RunContext) -> SampleProcessResult
apply_variant(image: np.ndarray, variant: AttackVariant,
              pattern: PreparedPattern) -> np.ndarray
```

**Inputs.** `SourceSample`, resolved variants, a loaded RGB image, compatible
`PreparedPattern`, and immutable run context (configuration fingerprint, engine
version, and seed). Current variants select a pattern; their `parameters`
mapping is reserved for future transforms and attack models.

**Outputs.** `SampleProcessResult`: the clean image (when requested), zero or
more `AttackResult` objects containing attacked RGB arrays and draft metadata,
and zero or more `FailureRecord` objects. Output arrays are not written here.

**Error handling.** It validates image and prepared-pattern shapes before each
call. Contract violations from `attack_engine`, unexpected memory errors, or
failed pattern preparation are converted to contextual failures with sample and
variant IDs. In `fail_fast`, the first failure is re-raised after recording
context; in `continue`, unaffected variants and later samples proceed. No
partially generated output is exposed from this module.

**Interactions.** Receives discovery/load services from `image_loader.py` and
prepared patterns from `pattern_manager.py`; calls `attack_engine.apply_attack`;
passes results to `output_writer.py`; obtains record builders from `metadata.py`.
`generate_dataset.py` owns run-level iteration and policy decisions.

### `output_writer.py`

**Purpose.** Persist paired images and optional copied labels safely, preserving
relative source hierarchy and preventing an incomplete sample from looking
successful.

**Public API.**

```python
plan_output_paths(sample: SourceSample, variant: AttackVariant,
                  output_config: OutputConfig) -> OutputPaths
write_clean_image(image: np.ndarray, paths: OutputPaths) -> WrittenArtifact
write_attacked_image(image: np.ndarray, paths: OutputPaths) -> WrittenArtifact
copy_labels(sample: SourceSample, paths: OutputPaths) -> list[WrittenArtifact]
commit_sample(artifacts: Sequence[WrittenArtifact]) -> None
```

**Inputs.** RGB `uint8` arrays, `SourceSample` label references, output
configuration, and paths planned inside `output_root`.

**Outputs.** Committed output files and `WrittenArtifact` records containing
relative path, byte size, format, and SHA-256. Clean-image writes are deduplicated
per source within a run; attacked writes are one per source/variant.

**Error handling.** The writer rejects path traversal, unsupported format,
non-RGB arrays, collisions when `overwrite=false`, and encoder/I/O failures
with `OutputWriteError`. It cleans up its temporary files on failure and never
touches source files. Existing committed outputs are not deleted automatically.

**Interactions.** Called by `generate_dataset.py` after an in-memory pipeline
result succeeds. It returns checksums and paths to `metadata.py`; it does not
decide whether a failed item should continue.

### `metadata.py`

**Purpose.** Define the provenance schema and write append-only run manifests
that let training and evaluation reproduce or audit each generated image.

**Public API.**

```python
create_run_manifest(config: GeneratorConfig, environment: EnvironmentInfo) -> RunManifest
create_sample_record(sample: SourceSample, variant: AttackVariant,
                     prepared_pattern: PreparedPattern,
                     artifacts: Sequence[WrittenArtifact], run: RunManifest) -> SampleRecord
write_jsonl_record(path: Path, record: Mapping[str, Any]) -> None
record_failure(failure: FailureRecord, run: RunManifest) -> None
finalize_run(run: RunManifest, summary: RunSummary) -> None
```

**Inputs.** Validated configuration, source and pattern provenance, write
artifacts, attack-engine version (or package/module fingerprint), environment
information, and structured failures.

**Outputs.** `run_manifest.json`, `source_manifest.jsonl`, `samples.jsonl`, and
`failures.jsonl`. JSONL records are machine-readable and append-friendly;
`run_manifest.json` is updated atomically at finalization with counts and end
state.

**Error handling.** All records are JSON-serializable before writing. Metadata
write failures are run-fatal because an image without provenance violates the
package contract. Individual corrupt/missing source metadata becomes a
per-sample failure only if it can be represented in `failures.jsonl`.

**Interactions.** All modules supply facts to this module, but only
`generate_dataset.py` starts/finalizes a run. `output_writer.py` supplies final
artifact checksums; `attack_pipeline.py` supplies attack/pattern facts;
`image_loader.py` supplies the source inventory.

Minimum `SampleRecord` shape:

```json
{
  "record_id": "...",
  "generated_id": "...",
  "source": {"sample_id": "...", "dataset": "kitti", "split": "training", "image_path": "...", "sha256": "..."},
  "attack": {"variant_id": "...", "pattern_id": "...", "pattern_source_sha256": "...", "preparation": {"fit": "resize"}, "engine": "attack_engine.apply_attack"},
  "artifacts": {"clean": {"path": "...", "sha256": "..."}, "attacked": {"path": "...", "sha256": "..."}},
  "run": {"id": "...", "config_sha256": "...", "seed": 20260728}
}
```

### `generate_dataset.py`

**Purpose.** Provide the command-line entry point and run orchestration. It
validates configuration, initializes services, applies policy, and emits a
concise end-of-run summary.

**Public API.**

```python
load_config(path: str | Path) -> GeneratorConfig
run_generation(config: GeneratorConfig) -> RunSummary
main(argv: Sequence[str] | None = None) -> int
```

Planned CLI:

```text
python -m dataset_generator.generate_dataset --config config/generate_kitti.yaml
```

Optional operational flags may include `--dry-run`, `--limit`, and `--resume`.
They must not override provenance invisibly: effective values are written to the
run manifest.

**Inputs.** Configuration path and command-line overrides. It accepts neither
raw images nor patterns directly; those are declared in configuration.

**Outputs.** Exit status (`0` only for a fully successful run), output tree,
run manifest, source manifest, sample records, failures, and `RunSummary`.
With `on_error=continue`, a completed run with failed samples uses a distinct
nonzero partial-success exit code to make automation notice it.

**Error handling.** Configuration and initialization failures are fatal before
output processing begins. Per-sample failures follow the configured policy.
The orchestrator catches unexpected exceptions only to finalize the run as
`failed` with context, then re-raises or returns a nonzero exit code.

**Interactions.** This is the composition root: it calls all other package
modules but contains no dataset-specific discovery, attack math, image encoding,
or metadata serialization logic.

### `README.md`

**Purpose.** Give implementers and users a concise package contract and a
working configuration example, linking to this design for normative detail.

**Public API.** Documentation only; it lists the supported command, configuration
fields, output layout, and the dependency on `attack_engine.apply_attack`.

**Inputs/outputs.** Describes accepted configuration and generated artifacts.

**Error handling.** Documents common setup and validation failures and directs
users to `metadata/failures.jsonl`.

**Interactions.** Remains synchronized with all public APIs specified above;
the technical design is authoritative when the two conflict.

## 6. Complete processing pipeline

```text
configuration file
      |
      v
validate configuration + create immutable run manifest
      |
      v
select adapter -> discover, validate, and order SourceSample inventory
      |
      +----------------------------> write source_manifest.jsonl
      |
      v  (for each source sample)
load and validate RGB source image
      |
      +---- load/validation failure -> failures.jsonl -> policy decision
      |
      v
resolve requested attack variants
      |
      v  (for each variant)
load/cache pattern -> prepare to source dimensions -> validate RGB uint8 shape
      |
      +---- preparation failure -> failures.jsonl -> next variant/policy
      |
      v
attack_engine.apply_attack(clean_rgb, prepared_pattern)
      |
      +---- engine/contract failure -> failures.jsonl -> next variant/policy
      |
      v
plan paths -> atomically write clean image once and attacked image per variant
      |
      +---- write failure -> failures.jsonl -> policy decision
      |
      v
checksum artifacts -> append success SampleRecord to samples.jsonl
      |
      v
finalize run_manifest.json with counts, status, and elapsed time
```

The clean copy is written once per source only if `copy_clean_images=true`; every
attacked record still references the original source and optional clean output.
For two patterns applied to three images, the default Cartesian selection creates
three clean images and six attacked images, each with its own sample record.

### Ordering, resume, and repeatability

1. Normalize and sort source relative paths before processing.
2. Canonically serialize the effective configuration and hash it.
3. Derive IDs from normalized identities and the configuration fingerprint.
4. Record checksums for source images, source pattern files, and written
   artifacts.
5. A future `--resume` verifies the existing run's configuration fingerprint and
   checksums, skips only fully committed matching records, and never treats a
   file without a metadata record as complete.

The current additive engine is deterministic. The run seed is still recorded so
future randomized selection, parameter sampling, or pattern generators can be
reproduced without schema changes.

## 7. Dataset support and extensibility

### Multiple attack patterns

Patterns are named assets, while variants are separate requested applications of
those assets. `cartesian` selection applies every named pattern to every source;
`explicit` selection can later target samples, splits, or parameter sets. The
metadata model records both source pattern checksum and derived preparation,
making it possible to distinguish a resized pattern from the original asset.

### Adding a dataset adapter

To support a future dataset, implement `DatasetAdapter.discover(config)` that
emits validated `SourceSample` objects, register a new adapter name, and add
adapter-specific configuration validation. No change to pattern, attack,
writer, or metadata APIs should be needed. The adapter must provide stable
sample IDs, deterministic ordering inputs, and source-relative paths.

### Adding pattern or attack capabilities

- Add a preparation strategy to `pattern_manager.py` and serialize its full
  parameters in `PreparedPattern.preparation`.
- Add validated fields to `AttackVariant.parameters` for transformations or a
  future engine selector.
- Keep the current engine adapter as the default and preserve its strict RGB
  shape contract. A different attack implementation must have an explicit
  engine identifier and version in every sample record.
- New output formats, label-copy rules, and remote/object storage writers can
  be introduced behind `output_writer.py` without changing pipeline semantics.

## 8. Implementation acceptance criteria

The future implementation is complete when it can:

1. Generate paired clean/attacked PNG datasets from representative KITTI,
   BDD100K, and custom-manifest fixtures.
2. Apply more than one pattern per image and produce the expected one-to-many
   file and metadata relationship.
3. Demonstrate byte/shape compatibility with `attack_engine.apply_attack` and
   preserve its additive saturating result.
4. Reject invalid configuration, path escapes, malformed manifests, incorrect
   pattern dimensions under `exact`, and non-RGB inputs with actionable errors.
5. Continue safely after an isolated corrupt image in `continue` mode, writing a
   structured failure record and no false success record.
6. Produce stable IDs, ordering, and provenance records across repeated runs
   with identical inputs and configuration.

