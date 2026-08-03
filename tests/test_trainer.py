"""Unit tests for YoloTrainer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from models.trainer import YoloTrainer
from models.yolo_config import YoloConfig
from models.yolo_wrapper import YoloWrapper


@patch("models.trainer.prepare_yolo_dataset")
def test_trainer_train_success(mock_prepare: MagicMock, tmp_path: Path) -> None:
    """Test YoloTrainer train workflow execution."""
    mock_prepare.return_value = tmp_path / "data.yaml"
    mock_wrapper = MagicMock(spec=YoloWrapper)
    mock_results = MagicMock()
    mock_results.results_dict = {"metrics/mAP50(B)": 0.85}
    mock_wrapper.train.return_value = mock_results

    config = YoloConfig(project_directory=tmp_path, experiment_name="exp1")
    trainer = YoloTrainer(wrapper=mock_wrapper, config=config)

    summary = trainer.train(dataset="dummy_ds")

    assert summary["status"] == "success"
    assert summary["epochs"] == 50
    assert "metrics" in summary
    mock_wrapper.train.assert_called_once()


def test_trainer_resume_file_not_found(tmp_path: Path) -> None:
    """Test YoloTrainer resume raising FileNotFoundError when checkpoint is missing."""
    mock_wrapper = MagicMock(spec=YoloWrapper)
    config = YoloConfig(project_directory=tmp_path, experiment_name="exp_non_existent")
    trainer = YoloTrainer(wrapper=mock_wrapper, config=config)

    with pytest.raises(FileNotFoundError, match="Checkpoint for resuming not found"):
        trainer.resume()


def test_trainer_save_best(tmp_path: Path) -> None:
    """Test YoloTrainer save_best checkpoint file copy."""
    mock_wrapper = MagicMock(spec=YoloWrapper)
    config = YoloConfig(project_directory=tmp_path, experiment_name="exp_save")
    trainer = YoloTrainer(wrapper=mock_wrapper, config=config)

    # Create dummy best.pt in experiment directory
    best_pt = trainer.best_weights_path
    best_pt.parent.mkdir(parents=True, exist_ok=True)
    best_pt.write_text("dummy model weights")

    target_dir = tmp_path / "saved_models"
    saved_file = trainer.save_best(target_dir=target_dir)

    assert saved_file.exists()
    assert saved_file.name == "best.pt"
    assert saved_file.read_text() == "dummy model weights"
