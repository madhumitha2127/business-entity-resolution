from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

files = [
    DATASET_DIR / "train" / "train_source1.tsv",
    DATASET_DIR / "train" / "train_source2.tsv",
    DATASET_DIR / "train" / "train_source3.tsv",
    DATASET_DIR / "train" / "train_ground_truth.tsv",
    DATASET_DIR / "test" / "test_source1.tsv",
    DATASET_DIR / "test" / "test_source2.tsv",
    DATASET_DIR / "test" / "test_source3.tsv",
]

for file in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    if not file.exists():
        print("FILE NOT FOUND")
        continue

    size_mb = file.stat().st_size / (1024 ** 2)
    print(f"Size: {size_mb:.2f} MB")

    df = pd.read_csv(file, sep="\t")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nData types:")
    print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)
