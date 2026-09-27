from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

DATASETS = {
    "train_source1": DATASET_DIR / "train" / "train_source1.tsv",
    "train_source2": DATASET_DIR / "train" / "train_source2.tsv",
    "train_source3": DATASET_DIR / "train" / "train_source3.tsv",
    "test_source1": DATASET_DIR / "test" / "test_source1.tsv",
    "test_source2": DATASET_DIR / "test" / "test_source2.tsv",
    "test_source3": DATASET_DIR / "test" / "test_source3.tsv",
}


def profile_file(name, path):
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    total_rows = 0
    missing_name = 0
    missing_address = 0
    country_counts = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=250_000
    ):
        total_rows += len(chunk)

        missing_name += chunk["business_name"].isna().sum()
        missing_address += chunk["business_address"].isna().sum()

        counts = chunk["country"].value_counts(dropna=False)

        for country, count in counts.items():
            country_counts[country] = country_counts.get(country, 0) + int(count)

    print(f"Rows: {total_rows:,}")
    print(f"Missing business_name: {missing_name:,}")
    print(f"Missing business_address: {missing_address:,}")

    print("\nCountry distribution:")
    for country, count in sorted(
        country_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = count / total_rows * 100
        print(f"  {country}: {count:,} ({percentage:.2f}%)")


for name, path in DATASETS.items():
    profile_file(name, path)

print("\n" + "=" * 80)
print("STEP 2 PROFILING COMPLETE")
print("=" * 80)
