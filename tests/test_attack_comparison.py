"""Tests for the evaluation-only attack comparison workflow."""

import json
from pathlib import Path

import pytest
from PIL import Image

from evaluation.attack_evaluator import (
    AttackComparisonConfig,
    AttackComparisonEvaluator,
    compare_to_clean,
    count_validation_samples,
    summarize_target_metadata,
    validate_population_sizes,
)
from models.predictor import DetectionResult
from models.yolo_config import YoloConfig

EXPECTED_METRIC_KEYS = {
    "mAP50",
    "mAP50-95",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
}


def make_yolo_dataset(root: Path, prefix: str, n: int, include_names: bool = True) -> Path:
    """Create a minimal YOLO dataset with ``n`` validation samples.

    Args:
        include_names: If False, write an empty ``names: {}`` block to mimic a
            clean data.yaml whose class mapping was not persisted.

    Returns:
        Path to the generated ``data.yaml``.
    """
    img_val = root / "images" / "val"
    lbl_val = root / "labels" / "val"
    img_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    for i in range(n):
        sample_id = f"{prefix}_{i:04d}"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(img_val / f"{sample_id}.png")
        (lbl_val / f"{sample_id}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    names_block = (
        "names:\n  0: Car\n  1: Pedestrian\n  2: Cyclist\n"
        if include_names
        else "names: {}\n"
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n" + names_block
    )
    return data_yaml


class RecordingWrapper:
    """Fake YoloWrapper that records the weights path and forbids training."""

    instances: list = []

    def __init__(self, config=None, model_path=None):
        self.config = config if config is not None else YoloConfig()
        self.model_path = model_path
        type(self).instances.append(model_path)

    def train(self, *args, **kwargs):
        raise AssertionError("YoloWrapper.train() must never be called during evaluation")


class FakePredictor:
    """Fake YoloPredictor recording every dataset it was asked to predict."""

    def __init__(self, wrapper=None, config=None, class_names=None):
        self.wrapper = wrapper
        self.called_datasets = []

    def predict_dataset(self, dataset, **kwargs):
        self.called_datasets.append([s.sample_id for s in dataset])
        return [
            DetectionResult(sample_id=s.sample_id, image_path=s.image_path, boxes=[])
            for s in dataset
        ]


def _install_fakes(monkeypatch, weights: Path, tmp_path: Path):
    from evaluation import attack_evaluator as ae

    weights.write_bytes(b"fake-weights")
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 3)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 3)

    RecordingWrapper.instances = []
    monkeypatch.setattr(ae, "YoloWrapper", RecordingWrapper)
    monkeypatch.setattr(ae, "YoloPredictor", FakePredictor)

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=clean_yaml,
        random_data_yaml=random_yaml,
        target_data_yaml=target_yaml,
        expected_samples=3,
    )
    return clean_yaml, random_yaml, target_yaml, AttackComparisonEvaluator(config)


def test_count_validation_samples_counts_val_split(tmp_path: Path) -> None:
    data_yaml = make_yolo_dataset(tmp_path / "ds", "c", 5)
    assert count_validation_samples(data_yaml) == 5


def test_count_validation_samples_missing_yaml_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data.yaml not found"):
        count_validation_samples(tmp_path / "missing" / "data.yaml")


def test_validate_population_sizes_rejects_mismatched_populations(tmp_path: Path) -> None:
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 3)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 4)

    yamls = {
        "clean": clean_yaml,
        "random_attack": random_yaml,
        "target_attack": target_yaml,
    }
    with pytest.raises(ValueError, match="target_attack"):
        validate_population_sizes(yamls, expected_samples=3)

    target_yaml_ok = make_yolo_dataset(tmp_path / "target_ok", "t", 3)
    counts = validate_population_sizes(
        {**yamls, "target_attack": target_yaml_ok}, expected_samples=3
    )
    assert counts == {"clean": 3, "random_attack": 3, "target_attack": 3}


def test_compare_to_clean_calculates_absolute_and_relative_changes() -> None:
    clean = {"mAP50": 0.5, "mAP50-95": 0.3, "precision": 0.6, "recall": 0.7, "f1": 0.65}
    attacked = {"mAP50": 0.4, "mAP50-95": 0.24, "precision": 0.55, "recall": 0.6, "f1": 0.57}

    comp = compare_to_clean(clean_metrics=clean, attacked_metrics=attacked)

    assert comp["absolute_mAP50_change"] == pytest.approx(-0.1)
    assert comp["absolute_mAP50-95_change"] == pytest.approx(-0.06)
    assert comp["relative_mAP50_change_pct"] == pytest.approx(-20.0)
    assert comp["relative_mAP50-95_change_pct"] == pytest.approx(-20.0)
    assert comp["deltas"]["mAP50"] == pytest.approx(0.1)
    assert comp["deltas"]["mAP50-95"] == pytest.approx(0.06)
    assert comp["percentage_drops"]["mAP50"] == pytest.approx(20.0)
    assert comp["percentage_drops"]["mAP50-95"] == pytest.approx(20.0)


def test_compare_to_clean_handles_zero_baseline() -> None:
    clean = {"mAP50": 0.0, "mAP50-95": 0.0}
    attacked = {"mAP50": 0.0, "mAP50-95": 0.0}
    comp = compare_to_clean(clean_metrics=clean, attacked_metrics=attacked)
    assert comp["relative_mAP50_change_pct"] == 0.0
    assert comp["relative_mAP50-95_change_pct"] == 0.0


def test_same_weights_used_for_all_three_evaluations(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "baseline_best.pt"
    _, _, _, evaluator = _install_fakes(monkeypatch, weights, tmp_path)

    report = evaluator.evaluate()

    assert RecordingWrapper.instances == [str(weights)]
    assert isinstance(evaluator.predictor, FakePredictor)
    assert isinstance(evaluator.predictor.wrapper, RecordingWrapper)
    assert report["sample_counts"] == {"clean": 3, "random_attack": 3, "target_attack": 3}


def test_all_three_datasets_evaluated(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "baseline_best.pt"
    _, _, _, evaluator = _install_fakes(monkeypatch, weights, tmp_path)

    evaluator.evaluate()

    called = [tuple(ids) for ids in evaluator.predictor.called_datasets]
    assert len(called) == 3
    assert all(len(ids) == 3 for ids in called)
    evaluated_ids = {sid for ids in called for sid in ids}
    assert len(evaluated_ids) == 9
    evaluated_stems = {sid.rsplit("/", 1)[-1] for sid in evaluated_ids}
    assert any(sid.startswith("c_") for sid in evaluated_stems)
    assert any(sid.startswith("r_") for sid in evaluated_stems)
    assert any(sid.startswith("t_") for sid in evaluated_stems)


def test_no_training_invoked(monkeypatch, tmp_path: Path) -> None:
    from evaluation import attack_evaluator as ae

    assert not hasattr(ae, "YoloTrainer")
    assert not hasattr(ae, "prepare_yolo_dataset")
    assert not hasattr(ae, "DatasetGenerator")

    weights = tmp_path / "baseline_best.pt"
    _, _, _, evaluator = _install_fakes(monkeypatch, weights, tmp_path)

    report = evaluator.evaluate()

    assert report["model_weights"].endswith("baseline_best.pt")
    # RecordingWrapper.train() raises AssertionError; reaching here proves it
    # was never invoked during the evaluation-only run.


def test_evaluate_rejects_mismatched_population(monkeypatch, tmp_path: Path) -> None:
    from evaluation import attack_evaluator as ae

    weights = tmp_path / "baseline_best.pt"
    weights.write_bytes(b"fake")
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 4)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 3)

    monkeypatch.setattr(ae, "YoloWrapper", RecordingWrapper)
    monkeypatch.setattr(ae, "YoloPredictor", FakePredictor)

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=clean_yaml,
        random_data_yaml=random_yaml,
        target_data_yaml=target_yaml,
        expected_samples=3,
    )
    evaluator = AttackComparisonEvaluator(config)

    with pytest.raises(ValueError, match="random_attack"):
        evaluator.evaluate()
    assert evaluator.wrapper is None
    assert evaluator.predictor is None


def test_report_json_structure_and_target_metadata_summary(
    monkeypatch, tmp_path: Path
) -> None:
    from evaluation import attack_evaluator as ae

    weights = tmp_path / "baseline_best.pt"
    weights.write_bytes(b"fake")
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 3)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 3)

    meta_dir = tmp_path / "target_metadata"
    meta_dir.mkdir()
    (meta_dir / "t_0000.json").write_text(
        json.dumps({"target_found": True, "preserved": False})
    )
    (meta_dir / "t_0001.json").write_text(
        json.dumps({"target_found": True, "preserved": False})
    )
    (meta_dir / "t_0002.json").write_text(
        json.dumps({"target_found": False, "preserved": True})
    )

    RecordingWrapper.instances = []
    monkeypatch.setattr(ae, "YoloWrapper", RecordingWrapper)
    monkeypatch.setattr(ae, "YoloPredictor", FakePredictor)

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=clean_yaml,
        random_data_yaml=random_yaml,
        target_data_yaml=target_yaml,
        expected_samples=3,
        target_metadata_dir=meta_dir,
    )
    evaluator = AttackComparisonEvaluator(config)
    report = evaluator.evaluate()

    out = tmp_path / "report.json"
    path = evaluator.save_report(report, out)
    assert path == out
    data = json.loads(out.read_text())

    for key in (
        "model_weights",
        "datasets",
        "datasets_evaluated",
        "expected_samples",
        "sample_counts",
        "clean",
        "random_attack",
        "target_attack",
        "comparisons",
        "target_metadata_summary",
    ):
        assert key in data, f"missing report key: {key}"

    assert data["datasets_evaluated"] == ["clean", "random_attack", "target_attack"]
    assert data["sample_counts"] == {"clean": 3, "random_attack": 3, "target_attack": 3}

    for population in ("clean", "random_attack", "target_attack"):
        entry = data[population]
        assert entry["num_samples"] == 3
        assert set(entry["metrics"]) == EXPECTED_METRIC_KEYS

    for name in ("random_attack", "target_attack"):
        comp = data["comparisons"][name]
        for key in (
            "absolute_mAP50_change",
            "absolute_mAP50-95_change",
            "relative_mAP50_change_pct",
            "relative_mAP50-95_change_pct",
            "deltas",
            "percentage_drops",
        ):
            assert key in comp, f"missing comparison key: {key}"

    meta = data["target_metadata_summary"]
    assert meta["attacked"] == 2
    assert meta["preserved"] == 1
    assert meta["samples_with_metadata"] == 3
    assert meta["unknown"] == 0


def test_summarize_target_metadata_missing_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata directory not found"):
        summarize_target_metadata(tmp_path / "nope", ["000000"])


def test_evaluation_does_not_modify_or_regenerate_dataset_files(
    monkeypatch, tmp_path: Path
) -> None:
    weights = tmp_path / "baseline_best.pt"
    weights.write_bytes(b"fake")
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 3)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 3)

    dataset_dirs = (clean_yaml.parent, random_yaml.parent, target_yaml.parent)
    before = {
        p: p.read_bytes() for d in dataset_dirs for p in d.rglob("*") if p.is_file()
    }

    from evaluation import attack_evaluator as ae

    RecordingWrapper.instances = []
    monkeypatch.setattr(ae, "YoloWrapper", RecordingWrapper)
    monkeypatch.setattr(ae, "YoloPredictor", FakePredictor)

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=clean_yaml,
        random_data_yaml=random_yaml,
        target_data_yaml=target_yaml,
        expected_samples=3,
    )
    evaluator = AttackComparisonEvaluator(config)
    evaluator.evaluate()

    after = {
        p: p.read_bytes() for d in dataset_dirs for p in d.rglob("*") if p.is_file()
    }
    assert set(before) == set(after)
    for path in before:
        assert before[path] == after[path]


def test_clean_empty_names_yaml_still_matches_detector_classes(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression: a clean data.yaml with empty ``names`` must still resolve
    labels to canonical KITTI class names so ground truth matches the frozen
    detector output.

    Without the canonical fallback the clean ground-truth classes become
    ``class_0..class_7`` while the detector reports ``Car/Pedestrian/...``, so
    clean mAP50 collapses to 0.
    """
    from evaluation import attack_evaluator as ae
    from models.predictor import DetectionBox, DetectionResult

    weights = tmp_path / "baseline_best.pt"
    weights.write_bytes(b"fake-weights")
    clean_yaml = make_yolo_dataset(tmp_path / "clean", "c", 3, include_names=False)
    random_yaml = make_yolo_dataset(tmp_path / "random", "r", 3)
    target_yaml = make_yolo_dataset(tmp_path / "target", "t", 3)

    # Give every clean GT a distinct box so the greedy matcher can match all 3.
    for i, x_center in enumerate((0.2, 0.5, 0.8)):
        (clean_yaml.parent / "labels" / "val" / f"c_{i:04d}.txt").write_text(
            f"0 {x_center} 0.5 0.2 0.2\n"
        )

    class GtMatchingPredictor:
        """Report every GT box as a perfect detection using detector class names."""

        def __init__(self, wrapper=None, config=None, class_names=None):
            self.wrapper = wrapper

        def predict_dataset(self, dataset, **kwargs):
            results = []
            for sample in dataset:
                boxes = [
                    DetectionBox(
                        class_id=0,
                        class_name="Car",
                        confidence=1.0,
                        bbox=ann.bbox,
                    )
                    for ann in sample.annotations
                ]
                results.append(
                    DetectionResult(
                        sample_id=sample.sample_id,
                        image_path=sample.image_path,
                        boxes=boxes,
                    )
                )
            return results

    RecordingWrapper.instances = []
    monkeypatch.setattr(ae, "YoloWrapper", RecordingWrapper)
    monkeypatch.setattr(ae, "YoloPredictor", GtMatchingPredictor)

    config = AttackComparisonConfig(
        weights=weights,
        clean_data_yaml=clean_yaml,
        random_data_yaml=random_yaml,
        target_data_yaml=target_yaml,
        expected_samples=3,
    )
    report = AttackComparisonEvaluator(config).evaluate()

    clean_metrics = report["clean"]["metrics"]
    assert report["clean"]["num_samples"] == 3
    assert clean_metrics["mAP50"] == pytest.approx(1.0)
    assert clean_metrics["mAP50"] > 0.0
    assert clean_metrics["recall"] == pytest.approx(1.0)
