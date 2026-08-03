from dataset_generator import DatasetGenerator


def main():
    print("=" * 60)
    print("DATASET GENERATOR INTEGRATION TEST")
    print("=" * 60)

    generator = DatasetGenerator()

    print("Generating first 10 attacked KITTI samples...\n")

    generator.generate_subset(10)

    print("\nIntegration test completed successfully.")


if __name__ == "__main__":
    main()