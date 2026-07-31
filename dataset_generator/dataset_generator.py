"""DatasetGenerator orchestrator integrating KittiLoader, AttackPipeline, OutputManager, and ProgressTracker."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from attack_engine.attack_config import AttackConfig
from attack_engine.attack_pipeline import AttackPipeline
from dataset_loader.kitti_loader import KittiLoader, KittiSample
from .attack_executor import AttackExecutor
from .generator_config import GeneratorConfig
from .output_manager import OutputManager
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Main orchestrator for generating attacked object detection datasets."""

    def __init__(
        self,
        loader: Optional[KittiLoader] = None,
        pipeline: Optional[AttackPipeline] = None,
        config: Optional[GeneratorConfig] = None,
        attack_config: Optional[AttackConfig] = None,
        output_manager: Optional[OutputManager] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> None:
        """Initialize DatasetGenerator using public API components.

        Args:
            loader: Optional KittiLoader instance.
            pipeline: Optional AttackPipeline instance.
            config: Optional GeneratorConfig instance.
            attack_config: Optional AttackConfig instance for pipeline.
            output_manager: Optional OutputManager instance.
            progress_tracker: Optional ProgressTracker instance.
        """
        self.config = config or GeneratorConfig()

        # Set up logging level
        log_level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level)

        self.loader = loader or KittiLoader()
        self.pipeline = pipeline or AttackPipeline(config=attack_config)
        self.attack_executor = AttackExecutor(pipeline=self.pipeline)
        self.output_manager = output_manager or OutputManager(config=self.config)
        self.progress_tracker = progress_tracker or ProgressTracker()

    def _get_split_dir_name(self) -> str:
        """Helper to return split folder name ('testing' or 'training')."""
        split = getattr(self.loader, "split", "training")
        if split in ("test", "testing"):
            return "testing"
        return "training"

    def _process_sample(
        self,
        sample: KittiSample,
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """Process a single KittiSample through attack pipeline and output manager.

        Args:
            sample: KittiSample object to process.
            pattern_type: Optional pattern generation algorithm identifier.
            **kwargs: Additional attack parameter overrides.

        Returns:
            True if sample was processed successfully.
        """
        split_dir_name = self._get_split_dir_name()
        ptype = pattern_type or self.pipeline.config.pattern_type

        # Check resume / overwrite mode
        if not self.config.overwrite_existing and self.output_manager.is_sample_processed(
            sample.sample_id, split_dir_name
        ):
            logger.debug(f"Sample '{sample.sample_id}' already processed. Skipping.")
            return True

        # Execute attack in memory
        attacked_image, pattern, exec_meta = self.attack_executor.execute(
            sample=sample,
            pattern_type=ptype,
            **kwargs,
        )

        # Save outputs via OutputManager
        self.output_manager.save_attacked_image(attacked_image, sample.sample_id, split_dir_name)
        self.output_manager.copy_label(sample.label_path, sample.sample_id, split_dir_name)
        self.output_manager.copy_calib(sample.calib_path, sample.sample_id, split_dir_name)
        self.output_manager.save_original_copy(sample.image_path, sample.sample_id, split_dir_name)

        # Save metadata
        effective_attack_config = self.pipeline.config
        if kwargs:
            cfg_dict = {
                "laser_color": effective_attack_config.laser_color,
                "intensity": effective_attack_config.intensity,
                "alpha": effective_attack_config.alpha,
                "blur_radius": effective_attack_config.blur_radius,
                "spot_radius": effective_attack_config.spot_radius,
                "max_spots": effective_attack_config.max_spots,
                "random_seed": effective_attack_config.random_seed,
                "pattern_type": ptype,
                "output_dtype": effective_attack_config.output_dtype,
            }
            cfg_dict.update(kwargs)
            effective_attack_config = AttackConfig(**cfg_dict)

        self.output_manager.save_metadata(
            sample_id=sample.sample_id,
            pattern=pattern,
            attack_config=effective_attack_config,
            execution_metadata=exec_meta,
            split=split_dir_name,
        )

        return True

    def generate_single(
        self,
        sample_id: str,
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate attacked output for a single sample by sample ID.

        Args:
            sample_id: KITTI sample ID string (e.g. '000000').
            pattern_type: Pattern generation type.
            **kwargs: Additional attack parameters.

        Returns:
            Dict containing execution summary metrics.
        """
        split_dir_name = self._get_split_dir_name()
        self.output_manager.setup_structure(split_dir_name)
        self.progress_tracker.start(total_samples=1)

        try:
            sample = self.loader.get_sample_by_id(sample_id)
            self._process_sample(sample, pattern_type=pattern_type, **kwargs)
            self.progress_tracker.update(sample_id, success=True)
        except Exception as err:
            logger.error(f"Failed to generate sample '{sample_id}': {err}")
            self.progress_tracker.update(sample_id, success=False, error=str(err))

        self.progress_tracker.finish()
        report = self.progress_tracker.generate_report()
        self._write_summary(report)
        return report

    def generate_subset(
        self,
        count: int,
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate attacked dataset for a subset of N samples.

        Args:
            count: Number of samples to process (> 0).
            pattern_type: Pattern generation type.
            **kwargs: Additional attack parameters.

        Returns:
            Dict containing subset execution summary.

        Raises:
            ValueError: If count is non-positive.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"count must be a positive integer, got {count}.")

        sample_ids = self.loader.sample_ids[:count]
        return self._run_generation(sample_ids, pattern_type=pattern_type, **kwargs)

    def generate_dataset(
        self,
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate attacked dataset for all samples in the loaded split.

        Args:
            pattern_type: Pattern generation type.
            **kwargs: Additional attack parameters.

        Returns:
            Dict containing dataset generation summary.
        """
        sample_ids = self.loader.sample_ids
        return self._run_generation(sample_ids, pattern_type=pattern_type, **kwargs)

    def resume_generation(
        self,
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Resume dataset generation, skipping already processed samples.

        Args:
            pattern_type: Pattern generation type.
            **kwargs: Additional attack parameters.

        Returns:
            Dict containing summary of resumed execution.
        """
        split_dir_name = self._get_split_dir_name()
        all_ids = self.loader.sample_ids
        pending_ids = [
            sid for sid in all_ids if not self.output_manager.is_sample_processed(sid, split_dir_name)
        ]
        logger.info(f"Resuming generation: {len(pending_ids)} pending out of {len(all_ids)} total samples.")
        return self._run_generation(pending_ids, pattern_type=pattern_type, **kwargs)

    def _run_generation(
        self,
        sample_ids: List[str],
        pattern_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Internal helper to execute processing loop with progress tracking.

        Args:
            sample_ids: List of sample ID strings to process.
            pattern_type: Pattern generation algorithm identifier.
            **kwargs: Additional attack parameter overrides.

        Returns:
            Summary report dictionary.
        """
        split_dir_name = self._get_split_dir_name()
        self.output_manager.setup_structure(split_dir_name)
        self.progress_tracker.start(total_samples=len(sample_ids))

        for sid in sample_ids:
            try:
                sample = self.loader.get_sample_by_id(sid)
                self._process_sample(sample, pattern_type=pattern_type, **kwargs)
                self.progress_tracker.update(sid, success=True)
            except Exception as err:
                logger.error(f"Error generating sample '{sid}': {err}")
                self.progress_tracker.update(sid, success=False, error=str(err))

        self.progress_tracker.finish()
        report = self.progress_tracker.generate_report()
        self._write_summary(report)
        return report

    def _write_summary(self, report: Dict[str, Any]) -> None:
        """Write generation summary JSON report to output directory."""
        if self.config.save_metadata:
            summary_path = self.output_manager.output_dir / "generation_summary.json"
            self.output_manager.metadata_writer.write_dataset_summary(summary_path, report)
