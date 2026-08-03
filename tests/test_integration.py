"""End-to-end integration test for YOLO integration layer in ResilientVisionAutonomous."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest

from dataset_loader.annotation_parser import Annotation
from dataset_loader.kitti_loader import KittiSample
from models.evaluator import ComparisonReport, EvaluationReport, YoloEvaluator
from models.predictor import DetectionBox, DetectionResult, YoloPredictor
from models.trainer import YoloTrainer
from models.visualizer import YoloVisualizer
from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper


@pytest.fixture
def dummy_kitti_dataset(tmp_path: Path) -> list:
    """Create a dummy list of KITTI samples representing a dataset split."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path1 = img_dir / "000000.png"
    img_path2 = img_dir / "000001.png"

    img = Image.new("RGB", (1242, 375), color="gray")
    img.save(img_path1)
    img.save(img_path2)

    sample1 = KittiSample(
        sample_id="000000",
        image_path=img_path1,
        annotations=[
            Annotation(
                class_name="Car",
                truncated=0.0,
                occluded=0,
                alpha=0.0,
                bbox=(100.0, 100.0, 300.0, 300.0),
                dimensions=(1.5, 1.6, 3.5),
                location=(0.0, 0.0, 10.0),
                rotation_y=0.0,
            )
        ],
        image=img,
    )

    sample2 = KittiSample(
        sample_id="000001",
        image_path=img_path2,
        annotations=[
            Annotation(
                class_name="Pedestrian",
                truncated=0.0,
                occluded=0,
                alpha=0.0,
                bbox=(50.0, 50.0, 150.0, 200.0),
                dimensions=(1.7, 0.6, 0.8),
                location=(0.0, 0.0, 5.0),
                rotation_y=0.0,
            )
        ],
        image=img,
    )

    return [sample1, sample2]


@patch("models.yolo_wrapper.YOLO")
def test_full_yolo_workflow_integration(
    mock_yolo_cls: MagicMock, dummy_kitti_dataset: list, tmp_path: Path
) -> None:
    """Test full integrated pipeline: dataset -> config -> wrapper -> trainer -> predictor -> evaluator -> visualizer."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    # Mock Ultralytics training return
    mock_train_res = MagicMock()
    mock_train_res.results_dict = {"metrics/mAP50(B)": 0.88}
    mock_model.train.return_value = mock_train_res

    # Mock Ultralytics prediction return
    mock_pred_res1 = MagicMock()
    mock_pred_res1.orig_shape = (375, 1242)
    mock_pred_res1.names = {0: "Car", 1: "Pedestrian"}
    box1 = MagicMock()
    box1.xyxy = np.array([[100.0, 100.0, 300.0, 300.0]])
    box1.conf = np.array([0.92])
    box1.cls = np.array([0])
    mock_pred_res1.boxes = box1

    mock_pred_res2 = MagicMock()
    mock_pred_res2.orig_shape = (375, 1242)
    mock_pred_res2.names = {0: "Car", 1: "Pedestrian"}
    box2 = MagicMock()
    box2.xyxy = np.array([[50.0, 50.0, 150.0, 200.0]])
    box2.conf = np.array([0.85])
    box2.cls = np.array([1])
    mock_pred_res2.boxes = box2

    def mock_predict_side_effect(source, **kwargs):
        if "000001" in str(source) or (isinstance(source, Image.Image) and source is dummy_kitti_dataset[1].image):
            return [mock_pred_res2]
        return [mock_pred_res1]

    mock_model.predict.side_effect = mock_predict_side_effect

    # 1. Config & Wrapper setup
    config = YoloConfig(
        model_name="yolov8n.pt",
        epochs=5,
        project_directory=tmp_path / "outputs",
        experiment_name="integration_test",
    )
    wrapper = YoloWrapper(config=config)

    # 2. Trainer workflow
    trainer = YoloTrainer(wrapper=wrapper, config=config)
    train_summary = trainer.train(dummy_kitti_dataset)
    assert train_summary["status"] == "success"

    # 3. Predictor workflow on clean and attacked datasets
    predictor = YoloPredictor(wrapper=wrapper, config=config)
    clean_predictions = predictor.predict_dataset(dummy_kitti_dataset)
    assert len(clean_predictions) == 2

    # 4. Evaluator workflow
    evaluator = YoloEvaluator()
    clean_report = evaluator.evaluate_dataset(
        dataset=dummy_kitti_dataset,
        predictor=predictor,
        dataset_name="Clean KITTI Split",
    )
    assert clean_report.metrics.mAP50 > 0.0

    # 5. Visualizer workflow
    visualizer = YoloVisualizer()
    ann_img = visualizer.visualize_prediction(
        image=dummy_kitti_dataset[0].image_path,
        detection_result=clean_predictions[0],
        output_path=tmp_path / "outputs" / "visualized_000000.png",
    )
    assert ann_img is not None
    assert (tmp_path / "outputs" / "visualized_000000.png").exists()
