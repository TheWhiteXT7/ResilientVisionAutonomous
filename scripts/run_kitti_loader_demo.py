import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dataset_loader import KittiLoader


def main():
    print("=" * 60)
    print("KITTI Loader Integration Test")
    print("=" * 60)

    try:
        # Create loader
        loader = KittiLoader()

        print(f"✓ Loader created successfully")
        print(f"✓ Total samples: {len(loader)}")

        # Get first sample
        sample = loader[0]

        print("\nFirst Sample")
        print("-" * 60)
        print(f"Sample ID      : {sample.sample_id}")
        print(f"Image Path     : {sample.image_path}")
        print(f"Label Path     : {sample.label_path}")

        if hasattr(sample, "calib_path"):
            print(f"Calibration    : {sample.calib_path}")

        # Load image
        image = sample.load_image()

        print(f"Image Size     : {image.size}")

        print(f"Objects Found  : {len(sample.annotations)}")

        if sample.annotations:
            obj = sample.annotations[0]

            print("\nFirst Object")
            print("-" * 60)
            print(f"Class          : {obj.class_name}")
            print(f"BBox           : {obj.bbox}")
            print(f"Dimensions     : {obj.dimensions}")
            print(f"Location       : {obj.location}")
            print(f"Rotation Y     : {obj.rotation_y}")

        print("\nIteration Test")
        print("-" * 60)

        for i, sample in enumerate(loader):
            print(sample.sample_id)

            if i == 4:
                break

        print("\n✓ Integration test completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()