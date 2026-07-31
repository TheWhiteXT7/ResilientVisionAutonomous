# Dataset Generator Layer

The `dataset_generator` package orchestrates end-to-end dataset attack generation for the `ResilientVisionAutonomous` project. It bridges the **Dataset Loader Layer** (`dataset_loader`) and the **Attack Engine Layer** (`attack_engine`) to produce fully corrupted benchmark datasets (including attacked images, label copies, calibration copies, and rich JSON metadata) ready for YOLO training and evaluation.

---

## Architecture & Pipeline

```
+--------------------+
|    KittiLoader     |
+---------+----------+
          |
          v
+--------------------+     +--------------------+
|    KittiSample     | --> |   AttackExecutor   |
+--------------------+     +---------+----------+
                                     |
                                     v
                           +--------------------+
                           |   AttackPipeline   |
                           +---------+----------+
                                     |
                                     v
                           +--------------------+
                           |   Attacked Image   |
                           +---------+----------+
                                     |
                                     v
+--------------------+     +--------------------+
|  ProgressTracker   | <-- |   OutputManager    | --> MetadataWriter
+--------------------+     +---------+----------+
                                     |
                                     v
                           +--------------------+
                           | Generated Dataset  |
                           +--------------------+
```

---

## Folder Structure of Generated Dataset

```
outputs/attacked_dataset/
└── training/
    ├── image_2/            # Attacked camera images (.png / .jpg)
    ├── image_2_orig/       # Optional original image copies
    ├── label_2/            # Copied ground-truth bounding box labels (.txt)
    ├── calib/              # Copied camera calibration files (.txt)
    └── metadata/           # Per-sample attack provenance JSON files (.json)
```

---

## Component Responsibilities

1. **`generator_config.py` (`GeneratorConfig`)**:
   - Immutable configuration dataclass managing output paths, overwrite behavior, label/calib copying flags, batch sizes, worker counts, image formats, and logging levels.

2. **`attack_executor.py` (`AttackExecutor`)**:
   - In-memory execution bridge receiving `KittiSample` objects, delegating attack application to `AttackPipeline`, and calculating timing/execution metadata without filesystem I/O.

3. **`metadata_writer.py` (`MetadataWriter`)**:
   - Serializes per-sample laser spot geometry and configuration parameters into clean JSON metadata files and writes dataset generation summary reports.

4. **`output_manager.py` (`OutputManager`)**:
   - Manages output directory structure creation, image writing, label/calibration file copying, and metadata persistence. Enforces non-destructive handling of source datasets.

5. **`progress_tracker.py` (`ProgressTracker`)**:
   - Tracks sample count, percentage completion, processing rates, ETA, and failure logs with console reporting support.

6. **`dataset_generator.py` (`DatasetGenerator`)**:
   - Main public orchestrator API supporting full generation (`generate_dataset`), subset processing (`generate_subset`), single sample rendering (`generate_single`), and incremental resume (`resume_generation`).

---

## Usage Examples

### 1. Generating a Full Attacked Dataset

```python
from dataset_loader import KittiLoader
from attack_engine import AttackConfig, AttackPipeline
from dataset_generator import DatasetGenerator, GeneratorConfig

# 1. Initialize Dataset Loader
loader = KittiLoader(split="train")

# 2. Initialize Attack Engine configuration
attack_config = AttackConfig(
    laser_color=(255, 0, 0),
    intensity=0.9,
    alpha=0.8,
    spot_radius=15.0,
    max_spots=5,
    random_seed=42,
)
pipeline = AttackPipeline(config=attack_config)

# 3. Configure Dataset Generator
config = GeneratorConfig(
    output_directory="outputs/kitti_attacked_train",
    overwrite_existing=False,
    save_metadata=True,
    copy_labels=True,
    copy_calibration=True,
)

# 4. Instantiate and execute DatasetGenerator
generator = DatasetGenerator(loader=loader, pipeline=pipeline, config=config)
report = generator.generate_dataset(pattern_type="random")

print(f"Dataset generation complete. Total samples: {report['processed_samples']}")
```

### 2. Generating a Subset or Resuming Execution

```python
# Generate only the first 50 samples for testing/debugging
generator.generate_subset(count=50, pattern_type="grid")

# Resume incomplete generation run
generator.resume_generation(pattern_type="random")
```

---

## Extension Points

1. **Multi-Dataset Support**:
   - Extend `DatasetGenerator` to accept any loader subclassing `BaseDatasetLoader` (e.g. BDD100K or custom datasets).

2. **Parallel Worker Execution**:
   - Utilize `GeneratorConfig.workers` to distribute `AttackExecutor` processing across multiprocessing pools or worker threads.

3. **Custom Metadata Formats**:
   - Subclass `MetadataWriter` to emit dataset manifests in COCO, YOLO, or CSV formats.
