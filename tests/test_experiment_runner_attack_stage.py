from argparse import Namespace
from pathlib import Path

from PIL import Image


def test_random_attack_prepares_non_empty_yolo_train_and_val_images(monkeypatch, tmp_path: Path) -> None:
    from scripts import experiment_runner

    captured = {}
    source_loader = object()
    real_kitti_loader = experiment_runner.KittiLoader

    def loader_factory(kitti_dir=None, split="train", load_images=False, **kwargs):
        if kitti_dir is None:
            assert split == "trainval"
            return source_loader
        return real_kitti_loader(kitti_dir=kitti_dir, split=split, load_images=load_images, **kwargs)

    class Generator:
        def __init__(self, loader, config):
            captured["loader"] = loader
            self.output_dir = Path(config.output_directory)

        def generate_dataset(self, **_kwargs):
            image_dir = self.output_dir / "training" / "image_2"
            label_dir = self.output_dir / "training" / "label_2"
            split_dir = self.output_dir / "ImageSets"
            for directory in (image_dir, label_dir, split_dir):
                directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20)).save(image_dir / "000001.png")
            Image.new("RGB", (20, 20)).save(image_dir / "000002.png")
            label = "Car 0 0 0 2 2 18 18 1 1 1 0 0 0 0\n"
            (label_dir / "000001.txt").write_text(label, encoding="utf-8")
            (label_dir / "000002.txt").write_text(label, encoding="utf-8")
            (split_dir / "train.txt").write_text("000001\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("000002\n", encoding="utf-8")

    class Trainer:
        def __init__(self, config):
            self.wrapper = object()

        def train(self, **_kwargs):
            return None

        @property
        def best_weights_path(self):
            return tmp_path / "missing.pt"

    class Predictor:
        def __init__(self, **_kwargs):
            pass

    class Report:
        def to_dict(self):
            return {}

        def save_json(self, _path):
            pass

    class Evaluator:
        def evaluate_dataset(self, **_kwargs):
            return Report()

    monkeypatch.setattr(experiment_runner, "KittiLoader", loader_factory)
    monkeypatch.setattr(experiment_runner, "DatasetGenerator", Generator)
    monkeypatch.setattr(experiment_runner, "YoloTrainer", Trainer)
    monkeypatch.setattr(experiment_runner, "YoloEvaluator", Evaluator)
    monkeypatch.setattr("models.predictor.YoloPredictor", Predictor)

    experiment_runner.run_attack_stage(
        "random_attack", tmp_path / "experiment", Namespace(epochs=1, batch_size=1, device="cpu")
    )

    assert captured["loader"] is source_loader
    assert list((tmp_path / "experiment" / "dataset_prepared" / "images" / "train").glob("*.png"))

    assert list((tmp_path / "experiment" / "dataset_prepared" / "images" / "val").glob("*.png"))


def test_target_attack_uses_targeted_pattern(monkeypatch, tmp_path: Path) -> None:
    from scripts import experiment_runner

    captured = {}
    real_kitti_loader = experiment_runner.KittiLoader

    def loader_factory(kitti_dir=None, split="train", load_images=False, **kwargs):
        if kitti_dir is None:
            return object()
        return real_kitti_loader(kitti_dir=kitti_dir, split=split, load_images=load_images, **kwargs)

    class Generator:
        def __init__(self, loader, config):
            self.output_dir = Path(config.output_directory)

        def generate_dataset(self, **kwargs):
            captured["pattern_type"] = kwargs.get("pattern_type")
            image_dir = self.output_dir / "training" / "image_2"
            label_dir = self.output_dir / "training" / "label_2"
            split_dir = self.output_dir / "ImageSets"
            for directory in (image_dir, label_dir, split_dir):
                directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20)).save(image_dir / "000001.png")
            Image.new("RGB", (20, 20)).save(image_dir / "000002.png")
            label = "Car 0 0 0 2 2 18 18 1 1 1 0 0 0 0\n"
            (label_dir / "000001.txt").write_text(label, encoding="utf-8")
            (label_dir / "000002.txt").write_text(label, encoding="utf-8")
            (split_dir / "train.txt").write_text("000001\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("000002\n", encoding="utf-8")

    class Trainer:
        def __init__(self, config):
            self.wrapper = object()

        def train(self, **_kwargs):
            return None

        @property
        def best_weights_path(self):
            return tmp_path / "missing.pt"

    class Predictor:
        def __init__(self, **_kwargs):
            pass

    class Report:
        def to_dict(self):
            return {}

        def save_json(self, _path):
            pass

    class Evaluator:
        def evaluate_dataset(self, **_kwargs):
            return Report()

    monkeypatch.setattr(experiment_runner, "KittiLoader", loader_factory)
    monkeypatch.setattr(experiment_runner, "DatasetGenerator", Generator)
    monkeypatch.setattr(experiment_runner, "YoloTrainer", Trainer)
    monkeypatch.setattr(experiment_runner, "YoloEvaluator", Evaluator)
    monkeypatch.setattr("models.predictor.YoloPredictor", Predictor)

    experiment_runner.main(
        ["--stages", "target_attack", "--project", str(tmp_path), "--name", "run_target", "--epochs", "1"]
    )

    assert captured["pattern_type"] == "targeted"



