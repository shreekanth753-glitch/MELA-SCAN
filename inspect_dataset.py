from pathlib import Path
from PIL import Image

# The dataset folder is directly inside MELA-SCAN.
DATASET_DIR = Path("dataset")

CLASS_NAMES = ["Healthy", "Eczema"]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def inspect_class(class_name):
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"\nERROR: Could not find: {class_dir}")
        return

    files = [
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"\n{class_name}")
    print("-" * 40)
    print(f"Images found: {len(files)}")

    valid = 0
    invalid = 0

    for image_path in files:
        try:
            with Image.open(image_path) as image:
                image.verify()
            valid += 1
        except Exception:
            invalid += 1
            print(f"Invalid image: {image_path}")

    print(f"Valid images:   {valid}")
    print(f"Invalid images: {invalid}")


def main():
    print("=" * 60)
    print("MELA-SCAN DATASET INSPECTION")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print("\nERROR:")
        print(f"Could not find the dataset folder at:")
        print(DATASET_DIR.resolve())
        print("\nMake sure your structure is:")
        print("MELA-SCAN/")
        print("└── dataset/")
        print("    ├── Healthy/")
        print("    └── Eczema/")
        return

    print(f"\nDataset location:")
    print(DATASET_DIR.resolve())

    for class_name in CLASS_NAMES:
        inspect_class(class_name)

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()