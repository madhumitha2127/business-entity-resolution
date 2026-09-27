import os
import re
from collections import defaultdict

import pandas as pd

# ============================================================
# STEP 13 - TEST CANDIDATE GENERATION (V3)
# ============================================================

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

OUTPUT_FILE = "output/test_candidate_pairs.tsv"

MAX_KEY_FREQUENCY = 100
CHUNK_SIZE = 100000
MAX_CHAR_KEYS = 8


def norm(value):
    if value is None:
        return ""

    value = str(value).lower()

    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    ).strip()


def compact(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        value
    )


def tokens(value):
    return [
        x for x in norm(value).split()
        if x
    ]


def numeric_tokens(value):
    return re.findall(
        r"\d+",
        str(value)
    )


def make_keys(row):

    name = norm(row["business_name"])
    address = norm(row["business_address"])
    country = str(row["country"]).upper().strip()

    nt = tokens(name)
    at = tokens(address)

    name_compact = compact(name)
    address_compact = compact(address)

    keys = []

    # --------------------------------------------------------
    # Name keys
    # --------------------------------------------------------

    if name:
        keys.append(
            f"{country}|N|{name}"
        )

        if len(nt) >= 2:
            keys.append(
                f"{country}|NS|{' '.join(sorted(nt))}"
            )

            keys.append(
                f"{country}|N2|{' '.join(sorted(nt[:2]))}"
            )

        for token in nt:
            if len(token) >= 5:
                keys.append(
                    f"{country}|NT|{token}"
                )

        if len(name_compact) >= 5:
            keys.append(
                f"{country}|CP5|{name_compact[:5]}"
            )

        if len(name_compact) >= 6:
            keys.append(
                f"{country}|CP6|{name_compact[:6]}"
            )

        if len(name_compact) >= 5:
            keys.append(
                f"{country}|CS5|{name_compact[-5:]}"
            )

        # Character 4-grams
        if len(name_compact) >= 4:
            grams = set(
                name_compact[i:i+4]
                for i in range(
                    min(
                        len(name_compact) - 3,
                        MAX_CHAR_KEYS
                    )
                )
            )

            for gram in grams:
                keys.append(
                    f"{country}|C4|{gram}"
                )

    # --------------------------------------------------------
    # Address keys
    # --------------------------------------------------------

    if address:

        keys.append(
            f"{country}|A|{address}"
        )

        if len(at) >= 1:
            keys.append(
                f"{country}|AF|{at[0]}"
            )

        if len(at) >= 2:
            keys.append(
                f"{country}|AP|{' '.join(sorted(at[:2]))}"
            )

        nums = numeric_tokens(address)

        for num in nums[:3]:
            keys.append(
                f"{country}|AN|{num}"
            )

        if len(address_compact) >= 6:
            keys.append(
                f"{country}|AC6|{address_compact[:6]}"
            )

    return set(keys)


def read_rows(path):

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def main():

    print("=" * 70)
    print("STEP 13 - TEST CANDIDATE GENERATION")
    print("=" * 70)

    os.makedirs(
        "output",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading test Source 2...")

    s2 = read_rows(S2_FILE)

    print(
        f"Source 2 rows: {len(s2):,}"
    )

    print("\nLoading test Source 3...")

    s3 = read_rows(S3_FILE)

    print(
        f"Source 3 rows: {len(s3):,}"
    )

    # --------------------------------------------------------
    # First pass: count key frequencies
    # --------------------------------------------------------

    key_counts = defaultdict(int)

    print("\nCounting blocking keys...")

    for source_name, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(
            f"Processing {source_name}..."
        )

        for start in range(
            0,
            len(df),
            CHUNK_SIZE
        ):

            chunk = df.iloc[
                start:start + CHUNK_SIZE
            ]

            for _, row in chunk.iterrows():

                for key in make_keys(row):

                    key_counts[key] += 1

    print(
        f"\nTotal unique keys: {len(key_counts):,}"
    )

    # --------------------------------------------------------
    # Keep only useful-frequency keys
    # --------------------------------------------------------

    allowed_keys = {
        key
        for key, count in key_counts.items()
        if count <= MAX_KEY_FREQUENCY
    }

    print(
        f"Allowed keys: {len(allowed_keys):,}"
    )

    # --------------------------------------------------------
    # Build candidate index
    # --------------------------------------------------------

    index = defaultdict(list)

    print("\nBuilding candidate index...")

    for source_name, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(
            f"Indexing {source_name}..."
        )

        for start in range(
            0,
            len(df),
            CHUNK_SIZE
        ):

            chunk = df.iloc[
                start:start + CHUNK_SIZE
            ]

            for _, row in chunk.iterrows():

                entity_id = row["entity_id"]

                for key in make_keys(row):

                    if key in allowed_keys:

                        index[key].append(
                            entity_id
                        )

    print(
        f"Index keys: {len(index):,}"
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    print("\nGenerating test candidates...")

    s1 = read_rows(S1_FILE)

    print(
        f"Source 1 rows: {len(s1):,}"
    )

    first_write = True
    total_candidates = 0

    for start in range(
        0,
        len(s1),
        CHUNK_SIZE
    ):

        chunk = s1.iloc[
            start:start + CHUNK_SIZE
        ]

        output_rows = []

        for _, row in chunk.iterrows():

            s1_id = row["entity_id"]

            candidate_ids = set()

            for key in make_keys(row):

                if key in index:

                    candidate_ids.update(
                        index[key]
                    )

            for candidate_id in candidate_ids:

                output_rows.append(
                    (
                        s1_id,
                        candidate_id
                    )
                )

        if output_rows:

            out_df = pd.DataFrame(
                output_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id"
                ]
            )

            out_df.to_csv(
                OUTPUT_FILE,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            total_candidates += len(out_df)

        print(
            f"Processed S1: "
            f"{min(start + CHUNK_SIZE, len(s1)):,}/"
            f"{len(s1):,} | "
            f"Candidates: {total_candidates:,}"
        )

    print("\n" + "=" * 70)
    print("TEST CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Total candidates: {total_candidates:,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    if os.path.exists(OUTPUT_FILE):

        size_mb = (
            os.path.getsize(OUTPUT_FILE)
            / (1024 * 1024)
        )

        print(
            f"File size: {size_mb:.2f} MB"
        )


if __name__ == "__main__":
    main()
