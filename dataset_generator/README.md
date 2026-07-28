# Dataset Generator

`dataset_generator` creates attacked RGB-image datasets using the verified
`attack_engine.apply_attack` operation. It reads source images without changing
them, matches each image to a laser pattern, writes generated images under a
separate output root, and records every outcome in a CSV file.

## Run

Configure `config/generator.yaml`, then run from the repository root:

```powershell
python -m dataset_generator.generate_dataset --config config/generator.yaml
```

The command displays a progress bar and ends with discovered, successful, and
failed image counts. It exits with code `2` when individual images failed after
the run continued; configuration failures exit with code `1`.

## Configuration

| Setting | Meaning |
| --- | --- |
| `input_dataset` | Root directory containing raw input images. |
| `output_dataset` | Separate root receiving attacked images. |
| `metadata_location` | Destination for `metadata.csv`. |
| `pattern_directory` | Directory containing PNG/JPG/JPEG patterns. |
| `output_image_format` | `png`, `jpeg`, or `jpg`. |
| `overwrite` | Whether existing generated images can be replaced. |
| `recursive_scanning` | Include nested source-image directories. |
| `random_pattern_selection` | Select a random rather than round-robin pattern. |
| `random_seed` | Optional seed for repeatable random pattern selection. |

Relative YAML paths are interpreted from the configuration file's directory.
For safety, the output root cannot be the raw input root or a child of it.

## Output and metadata

Generated files preserve their source-relative parent path. With a source image
at `training/image_2/000123.png` and PNG output, the attacked result is written
to `<output_dataset>/training/image_2/000123.png`.

`metadata.csv` has one record for every processing outcome and includes image
identity, original and generated relative paths, chosen pattern, dimensions,
status, error message, and UTC timestamp. Inspect failed rows to identify
corrupt source images, unsupported patterns, or output collisions.

See [DatasetGeneratorDesign.md](../docs/DatasetGeneratorDesign.md) for the
broader technical design.
