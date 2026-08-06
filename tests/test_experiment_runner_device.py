from argparse import Namespace
from pathlib import Path

from PIL import Image


def test_experiment_runner_device_defaults_to_cpu() -> None:
    from scripts.experiment_runner import _build_parser

    assert _build_parser().parse_args([]).device == "cpu"


def test_experiment_runner_passes_gpu_index_to_trainer(monkeypatch, tmp_path: Path) -> None:
    from scripts import experiment_runner

    dataset_dir = tmp_path / "yolo_dataset"
    image_dir = dataset_dir / "images" / "val"
    label_dir = dataset_dir / "labels" / "val"
    train_images = dataset_dir / "images" / "train"
    train_labels = dataset_dir / "labels" / "train"
    for directory in (image_dir, label_dir, train_images, train_labels):
        directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20)).save(image_dir / "sample.png")
    Image.new("RGB", (20, 20)).save(train_images / "sample.png")
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (train_labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (dataset_dir / "data.yaml").write_text(
        f"path: {dataset_dir}\ntrain: images/train\nval: images/val\nnames: {{0: Car}}\n",
        encoding="utf-8",
    )
    captured = {}

    class Trainer:
        def __init__(self, config):
            captured["device"] = config.device
            self.wrapper = object()

        def train(self, **_kwargs):
            return None

        @property
        def best_weights_path(self):
            return tmp_path / "missing-best.pt"

        @property
        def last_weights_path(self):
            return tmp_path / "missing-last.pt"

    class Predictor:
        def __init__(self, **_kwargs):
            pass

        def predict_dataset(self, _dataset):
            return []

    class Report:
        def to_dict(self):
            return {}

        def save_json(self, _path):
            pass

    class Evaluator:
        def evaluate_dataset(self, **_kwargs):
            return Report()

    monkeypatch.setattr(experiment_runner, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(experiment_runner, "YoloTrainer", Trainer)
    monkeypatch.setattr(experiment_runner, "YoloEvaluator", Evaluator)
    monkeypatch.setattr("models.predictor.YoloPredictor", Predictor)
    args = Namespace(epochs=1, batch_size=1, device="0")

    experiment_runner.run_baseline(tmp_path / "experiment", args)

    assert captured["device"] == "0"
