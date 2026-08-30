from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB" / "2750"
OUTPUT_DIR = PROJECT_ROOT / "data" / "split"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_split_settings():
    with PARAMS_PATH.open("r", encoding="utf-8") as params_file:
        settings = yaml.safe_load(params_file)["split"]

    ratios = (
        float(settings["train_ratio"]),
        float(settings["validation_ratio"]),
        float(settings["test_ratio"]),
    )
    if any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 0.000001:
        raise ValueError("Split ratios must be positive and add up to 1.0")
    return int(settings["seed"]), ratios


def find_images():
    valid_extensions = {".jpg", ".jpeg", ".png"}

    image_paths = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in valid_extensions
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {DATASET_DIR}")

    records = []

    for image_path in image_paths:
        label = image_path.parent.name

        records.append(
            {
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "label": label,
            }
        )

    return pd.DataFrame(records)


def main():
    seed, (train_ratio, validation_ratio, test_ratio) = load_split_settings()
    dataframe = find_images()

    # Create stable numeric labels
    class_names = sorted(dataframe["label"].unique())
    class_to_id = {
        class_name: class_id for class_id, class_name in enumerate(class_names)
    }

    dataframe["label_id"] = dataframe["label"].map(class_to_id)

    temporary_ratio = validation_ratio + test_ratio
    train_df, temporary_df = train_test_split(
        dataframe,
        train_size=train_ratio,
        test_size=temporary_ratio,
        random_state=seed,
        stratify=dataframe["label"],
    )

    relative_test_ratio = test_ratio / temporary_ratio
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temporary_df["label"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        split_df = split_df.sort_values("image_path").reset_index(drop=True)
        output_path = OUTPUT_DIR / f"{split_name}.csv"
        split_df.to_csv(output_path, index=False)

        print(f"{split_name}: {len(split_df)} images")

    print(f"Classes: {class_names}")
    print(f"Total: {len(dataframe)} images")


if __name__ == "__main__":
    main()
