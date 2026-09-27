import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_business_name, normalize_address


DATASET = "dataset"
OUTPUT = "experiments"

CHUNK_SIZE = 100_000


def normalize_series(series, func):
    return series.fillna("").map(func)


def make_keys(name, address, country):
    """
    Generate conservative, country-aware blocking keys.

    Multiple keys are used so that a single noisy field
    does not eliminate a true candidate.
    """
    keys = set()

    country = str(country).strip().upper()

    if name:
        tokens = name.split()

        # Exact normalized name
        keys.add(f"{country}|NEX|{name}")

        # First two tokens
        if len(tokens) >= 2:
            keys.add(f"{country}|N2|{tokens[0]}_{tokens[1]}")

        # Compact character prefix
        compact = "".join(tokens)
        if len(compact) >= 4:
            keys.add(f"{country}|NP4|{compact[:4]}")

    if address:
        tokens = address.split()

        # Exact normalized address
        keys.add(f"{country}|AEX|{address}")

        # Numeric address component
        numbers = [
            token for token in tokens
            if any(ch.isdigit() for ch in token)
        ]

        for number in numbers[:2]:
            keys.add(f"{country}|AN|{number}")

    return keys


def build_blocking_index(source_paths, index_path):
    """
    Build a compact blocking index.

    The index contains:
        blocking_key -> source entity IDs

    We process the large files in chunks.
    """

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Temporary partition files.
    partition_dir = os.path.join(OUTPUT, "blocking_partitions")
    os.makedirs(partition_dir, exist_ok=True)

    print("=" * 70)
    print("BUILDING BLOCKING INDEX")
    print("=" * 70)

    partition_count = 64

    files = [
        ("S2", source_paths["s2"]),
        ("S3", source_paths["s3"]),
    ]

    for source_name, path in files:

        print(f"\nProcessing {source_name}: {path}")

        chunk_no = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False
        ):

            chunk_no += 1

            rows = []

            for row in chunk.itertuples(index=False):

                name = normalize_business_name(row.business_name)
                address = normalize_address(row.business_address)
                country = row.country

                keys = make_keys(
                    name,
                    address,
                    country
                )

                entity_id = row.entity_id

                for key in keys:
                    partition = hash(key) % partition_count

                    rows.append(
                        (
                            partition,
                            key,
                            entity_id
                        )
                    )

            if rows:

                part_df = pd.DataFrame(
                    rows,
                    columns=["partition", "blocking_key", "entity_id"]
                )

                for partition, group in part_df.groupby("partition"):

                    output_file = os.path.join(
                        partition_dir,
                        f"part_{partition:02d}.tsv"
                    )

                    group[
                        ["blocking_key", "entity_id"]
                    ].to_csv(
                        output_file,
                        sep="\t",
                        mode="a",
                        header=not os.path.exists(output_file),
                        index=False
                    )

            print(
                f"  {source_name} chunk {chunk_no} completed"
            )

    print("\nCombining partitions...")

    index_parts = []

    for filename in sorted(os.listdir(partition_dir)):

        if not filename.endswith(".tsv"):
            continue

        path = os.path.join(
            partition_dir,
            filename
        )

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str
        )

        index_parts.append(df)

    if not index_parts:
        raise RuntimeError("No blocking records were generated.")

    index_df = pd.concat(
        index_parts,
        ignore_index=True
    )

    # Remove duplicate key/entity combinations.
    index_df = index_df.drop_duplicates(
        ["blocking_key", "entity_id"]
    )

    index_df.to_csv(
        index_path,
        sep="\t",
        index=False
    )

    print("\nBlocking index created:")
    print(index_path)

    print(f"Index rows: {len(index_df):,}")
    print(
        f"Unique blocking keys: "
        f"{index_df['blocking_key'].nunique():,}"
    )


if __name__ == "__main__":

    paths = {
        "s2": os.path.join(
            DATASET,
            "train/train_source2.tsv"
        ),
        "s3": os.path.join(
            DATASET,
            "train/train_source3.tsv"
        ),
    }

    index_path = os.path.join(
        OUTPUT,
        "blocking_index.tsv"
    )

    build_blocking_index(
        paths,
        index_path
    )
