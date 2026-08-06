from pathlib import Path

from PIL import Image

from models.yolo_dataset import YoloDataset


def test_yolo_dataset_loads_validation_split_and_converts_labels(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(image_dir / "sample.png")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("path: .\nval: images/val\nnames: {0: Car}\n", encoding="utf-8")

    dataset = YoloDataset.from_yaml(data_yaml)

    assert len(dataset) == 1
    assert dataset[0].sample_id == "val/sample"
    assert dataset[0].annotations[0].class_name == "Car"
    assert dataset[0].annotations[0].bbox == (40.0, 15.0, 60.0, 35.0)


def test_yolo_dataset_integrates_with_existing_evaluator(tmp_path: Path) -> None:
    from models.evaluator import YoloEvaluator
    from models.predictor import DetectionBox, DetectionResult

    root = tmp_path / "prepared"
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50)).save(image_dir / "sample.png")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("path: .\nval: images/val\nnames: {0: Car}\n", encoding="utf-8")
    dataset = YoloDataset.from_yaml(data_yaml)

    class Predictor:
        def predict_dataset(self, samples):
            return [DetectionResult("val/sample", image_dir / "sample.png", [DetectionBox(0, "Car", 0.9, (40.0, 15.0, 60.0, 35.0))])]

    report = YoloEvaluator().evaluate_dataset(dataset, Predictor(), dataset_name="yolo_val")
    assert report.num_samples == 1
    assert report.metrics.mAP50 == 1.0

