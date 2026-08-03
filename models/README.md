# YOLO Integration Layer

The `models` module provides a comprehensive, production-ready integration layer for Ultralytics YOLO object detectors within the **ResilientVisionAutonomous** framework. It bridges clean and laser-attacked KITTI datasets with YOLO training, inference, evaluation, and visualization workflows.

---

## Architecture Overview

```
                Dataset (KITTI / Attacked)
                           │
                           ▼
                      YOLOWrapper
                     /     │     \
                    /      │      \
              Trainer  Predictor  Evaluator
                    \      │      /
                     \     │     /
                      Visualizer
```

### Module Design & Principles
- **SOLID Principles**: Clean separation of concerns between config, wrapper, training, prediction, evaluation, metrics, and visualization.
- **Future Compatibility**: Easily adaptable to new YOLO versions (YOLOv8, YOLOv9, YOLOv10, YOLOv11), target-aware laser attack pipelines, defense mechanisms, and multi-detector benchmark studies without breaking public APIs.
- **Zero Global State**: Pure objects, immutable dataclasses, explicit `pathlib.Path` path handling throughout.

---

## Component Breakdown

| Module | Primary Class / Function | Description |
| :--- | :--- | :--- |
| `yolo_config.py` | `YoloConfig` | Immutable dataclass encapsulating all model hyperparameters and training/eval settings. |
| `yolo_wrapper.py` | `YoloWrapper` | Clean abstraction layer wrapping Ultralytics `YOLO` API for model loading, train, predict, val, and export. |
| `trainer.py` | `YoloTrainer` | Orchestrates training runs, dataset preparation, checkpoint management, and training resumption. |
| `predictor.py` | `YoloPredictor` | Executes inference on single images, image directories, or dataset loaders and returns structured `DetectionResult` objects. |
| `evaluator.py` | `YoloEvaluator` | Computes mAP50, mAP50-95, Precision, Recall, and F1 metrics for clean and attacked datasets and exports JSON reports. |
| `visualizer.py` | `YoloVisualizer` | Renders bounding boxes, class labels, confidence scores, and produces side-by-side comparison images. |
| `metrics.py` | `compute_detection_metrics` | Computes IoU, confusion matrix, per-class AP, and aggregates metrics across multiple evaluation runs. |
| `utils.py` | `prepare_yolo_dataset` | Handles KITTI-to-YOLO bounding box conversions, default class mappings, and `data.yaml` generation. |

---

## Usage Examples

### 1. Configuration Setup
```python
from models import YoloConfig

config = YoloConfig(
    model_name="yolov8n.pt",
    epochs=10,
    batch_size=16,
    image_size=640,
    confidence_threshold=0.25,
    experiment_name="kitti_clean_run"
)
```

### 2. Training Workflow
```python
from dataset_loader import KittiLoader
from models import YoloWrapper, YoloTrainer, YoloConfig

loader = KittiLoader(split="train")
config = YoloConfig(epochs=20, experiment_name="kitti_baseline")

wrapper = YoloWrapper(config=config)
trainer = YoloTrainer(wrapper=wrapper, config=config)

# Train on KITTI dataset
summary = trainer.train(loader)
best_model_path = trainer.save_best("./saved_models")
```

### 3. Inference on Attacked Dataset
```python
from dataset_generator import DatasetGenerator
from models import YoloWrapper, YoloPredictor

wrapper = YoloWrapper(model_path="./saved_models/best.pt")
predictor = YoloPredictor(wrapper=wrapper)

# Predict on attacked KITTI dataset
attacked_dataset = generator.generate(...)
predictions = predictor.predict_dataset(attacked_dataset)
```

### 4. Evaluation & Comparison Report
```python
from models import YoloEvaluator

evaluator = YoloEvaluator()

clean_report = evaluator.evaluate_dataset(clean_dataset, predictor, dataset_name="Clean KITTI")
attacked_report = evaluator.evaluate_dataset(attacked_dataset, predictor, dataset_name="Attacked KITTI")

# Generate comparative metrics report and save JSON
comparison = evaluator.compare(
    clean_results=clean_report,
    attacked_results=attacked_report,
    output_json_path="./outputs/reports/kitti_laser_attack_eval.json"
)
```

### 5. Visualization & Side-by-Side Analysis
```python
from models import YoloVisualizer

visualizer = YoloVisualizer()

# Annotate single prediction
annotated_img = visualizer.visualize_prediction(
    image="sample_000000.png",
    detection_result=predictions[0],
    output_path="outputs/annotated_sample.png"
)

# Side-by-side comparison between clean and attacked predictions
composite_img = visualizer.create_side_by_side_comparison(
    image1="outputs/clean_000000.png",
    image2="outputs/attacked_000000.png",
    title1="Clean Image Predictions",
    title2="Attacked Image Predictions",
    output_path="outputs/comparison_000000.png"
)
```
