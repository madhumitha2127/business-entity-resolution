import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization import normalize_business_name, normalize_address


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE, "dataset", "train")
EXPERIMENT_DIR = os.path.join(BASE, "experiments")

S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")

OUTPUT_PATH = os.path.join(
    EXPERIMENT_DIR,
    "train_candidate_pairs.tsv"
)

CHUNK_SIZE = 100_000

# Prevent extremely common blocking keys from producing
# enormous candidate sets.
MAX_KEY_FREQUENCY = 200


def make_block_keys(name, address, country):
    keys = set()

    country = str(country).strip().upper()

    if name:
        tokens = name.split()

        # Strongest name key
        keys.add(f"{country}|NEX|{name}")

        # First two name tokens
        if len(tokens) >= 2:
            keys.add(
                f"{country}|N2|{tokens[0]}_{tokens[1]}"
            )

        # Character prefix
        compact = "".join(tokens)

        if len(compact) >= 4:
            keys.add(
                f"{country}|NP4|{compact[:4]}"
            )

    if address:
        tokens = address.split()

        # Strongest address key
        keys.add(f"{country}|AEX|{address}")

        # Numeric address components
        numbers = [
            t for t in tokens
            if any(c.isdigit() for c in t)
        ]

        for number in numbers[:2]:
            keys.add(
                f"{country}|AN|{number}"
            )

    return keys


def normalize_chunk(df):
    df["norm_name"] = (
        df["business_name"]
        .fillna("")
        .map(normalize_business_name)
    )

    df["norm_address"] = (
        df["business_address"]
        .fillna("")
        .map(normalize_address)
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .str.upper()
        .str.strip()
    )

    return df


def build_index():

    print("=" * 70)
    print("BUILDING COMPACT BLOCKING INDEX")
    print("=" * 70)

    index = defaultdict(list)
    key_counts = defaultdict(int)

    total = 0

    for source_name, path in [
        ("S2", S2_PATH),
        ("S3", S3_PATH)
    ]:

        print(f"\nReading {source_name}")

        for chunk_no, df in enumerate(
            pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                chunksize=CHUNK_SIZE,
                keep_default_na=False
            ),
            start=1
        ):

            df = normalize_chunk(df)

            for row in df.itertuples(index=False):

                keys = make_block_keys(
                    row.norm_name,
                    row.norm_address,
                    row.country_norm
                )

                for key in keys:
                    key_counts[key] += 1

            total += len(df)

            print(
                f"  chunk {chunk_no}: "
                f"{len(df):,} rows | "
                f"total {total:,}"
            )

    print("\nCounting complete.")
    print(f"Unique keys: {len(key_counts):,}")

    # Keep only keys that are not excessively common.
    valid_keys = {
        key
        for key, count in key_counts.items()
        if count <= MAX_KEY_FREQUENCY
    }

    print(
        f"Keys retained: {len(valid_keys):,}"
    )

    # Second pass: build compact index.
    print("\nBuilding index...")

    for source_name, path in [
        ("S2", S2_PATH),
        ("S3", S3_PATH)
    ]:

        for chunk_no, df in enumerate(
            pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                chunksize=CHUNK_SIZE,
                keep_default_na=False
            ),
            start=1
        ):

            df = normalize_chunk(df)

            for row in df.itertuples(index=False):

                entity_id = row.entity_id

                keys = make_block_keys(
                    row.norm_name,
                    row.norm_address,
                    row.country_norm
                )

                for key in keys:

                    if key in valid_keys:

                        index[key].append(
                            (
                                source_name,
                                entity_id
                            )
                        )

            print(
                f"  indexed {source_name} "
                f"chunk {chunk_no}"
            )

    print(
        f"\nFinal index keys: {len(index):,}"
    )

    return index


def generate_candidates(index):

    print("\n" + "=" * 70)
    print("GENERATING TRAINING CANDIDATES")
    print("=" * 70)

    os.makedirs(
        EXPERIMENT_DIR,
        exist_ok=True
    )

    # Start fresh.
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    total_s1 = 0
    total_pairs = 0

    first_write = True

    for chunk_no, df in enumerate(
        pd.read_csv(
            S1_PATH,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False
        ),
        start=1
    ):

        df = normalize_chunk(df)

        candidates = []

        for row in df.itertuples(index=False):

            s1_id = row.entity_id

            keys = make_block_keys(
                row.norm_name,
                row.norm_address,
                row.country_norm
            )

            seen = set()

            for key in keys:

                for source_name, entity_id in index.get(
                    key,
                    []
                ):

                    pair = (
                        s1_id,
                        entity_id
                    )

                    if pair not in seen:
                        seen.add(pair)

                        candidates.append(
                            (
                                s1_id,
                                source_name,
                                entity_id
                            )
                        )

        if candidates:

            out = pd.DataFrame(
                candidates,
                columns=[
                    "source1_entity_id",
                    "source",
                    "candidate_entity_id"
                ]
            )

            out.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            total_pairs += len(out)

        total_s1 += len(df)

        print(
            f"  S1 chunk {chunk_no}: "
            f"{len(df):,} entities | "
            f"{len(candidates):,} candidates | "
            f"total candidates {total_pairs:,}"
        )

    print("\nCandidate generation complete.")
    print(f"S1 processed: {total_s1:,}")
    print(f"Candidate pairs: {total_pairs:,}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":

    index = build_index()

    generate_candidates(index)
