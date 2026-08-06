"""YOLO-format dataset adapter used by the evaluation pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class YoloAnnotation:
    class_name: str
    bbox: tuple[float, float, float, float]


@dataclass
class YoloSample:
    sample_id: str
    image_path: Path
    label_path: Path
    annotations: List[YoloAnnotation] = field(default_factory=list)
    image: None = None


class YoloDataset(Sequence[YoloSample]):
    """Read an image/label split referenced by a standard YOLO data.yaml."""

    def __init__(self, samples: List[YoloSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> YoloSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[YoloSample]:
        return iter(self.samples)

    @classmethod
    def from_yaml(cls, data_yaml: Union[str, Path], split: str = "val") -> "YoloDataset":
        yaml_path = Path(data_yaml).resolve()
        with yaml_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        split_value = config.get(split)
        if not split_value:
            raise ValueError(f"YOLO data.yaml has no '{split}' split: {yaml_path}")

        root_value = config.get("path")
        if root_value:
            root_path = Path(root_value)
            root = root_path.resolve() if root_path.is_absolute() else (yaml_path.parent / root_path).resolve()
        else:
            root = yaml_path.parent
        names = config.get("names", {})
        class_names: Dict[int, str] = ({index: str(name) for index, name in enumerate(names)} if isinstance(names, list) else {int(index): str(name) for index, name in names.items()})

        images: List[Path] = []
        for entry in split_value if isinstance(split_value, list) else [split_value]:
            entry_path = Path(entry)
            path = entry_path if entry_path.is_absolute() else root / entry_path
            if path.is_dir():
                images.extend(sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS))
            elif path.is_file():
                images.extend((path.parent / line.strip()).resolve() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            else:
                raise FileNotFoundError(f"YOLO {split} path does not exist: {path}")
        return cls([cls._sample_from_image(path, root, class_names) for path in images])

    @staticmethod
    def _sample_from_image(image_path: Path, root: Path, class_names: Dict[int, str]) -> YoloSample:
        try:
            relative = image_path.relative_to(root / "images")
            label_path = root / "labels" / relative.with_suffix(".txt")
            sample_id = relative.with_suffix("").as_posix()
        except ValueError:
            label_path = image_path.with_suffix(".txt")
            sample_id = image_path.stem
        annotations: List[YoloAnnotation] = []
        if label_path.exists():
            with Image.open(image_path) as image:
                width, height = image.size
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                tokens = line.split()
                if len(tokens) != 5:
                    raise ValueError(f"Malformed YOLO label at {label_path}:{line_number}")
                class_id, x_center, y_center, box_width, box_height = map(float, tokens)
                class_index = int(class_id)
                if class_index != class_id:
                    raise ValueError(f"Invalid YOLO class id at {label_path}:{line_number}")
                box_w, box_h = box_width * width, box_height * height
                center_x, center_y = x_center * width, y_center * height
                annotations.append(YoloAnnotation(class_names.get(class_index, f"class_{class_index}"), (center_x - box_w / 2, center_y - box_h / 2, center_x + box_w / 2, center_y + box_h / 2)))
        return YoloSample(sample_id, image_path, label_path, annotations)

