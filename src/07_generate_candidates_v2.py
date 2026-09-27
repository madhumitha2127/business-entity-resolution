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
    "train_candidate_pairs_v2.tsv"
)

CHUNK_SIZE = 100_000

# Very common keys create huge candidate sets.
MAX_KEY_FREQUENCY = 150


def name_keys(name, country):
    keys = set()

    if not name:
        return keys

    country = country.upper().strip()

    tokens = [
        t for t in name.split()
        if len(t) >= 2
    ]

    if not tokens:
        return keys

    # -----------------------------------------------------
    # 1. Sorted token signature
    # Handles:
    # "LLC Orellana Investments"
    # vs
    # "Orellana Investments LLC"
    # -----------------------------------------------------

    sorted_tokens = "_".join(sorted(tokens))

    keys.add(
        f"{country}|NS|{sorted_tokens}"
    )

    # -----------------------------------------------------
    # 2. First two significant tokens
    # -----------------------------------------------------

    if len(tokens) >= 2:

        first_two = sorted(
            tokens[:2]
        )

        keys.add(
            f"{country}|N2|{'_'.join(first_two)}"
        )

    # -----------------------------------------------------
    # 3. Token-level keys
    # -----------------------------------------------------

    for token in tokens:

        if len(token) >= 5:

            keys.add(
                f"{country}|NT|{token}"
            )

    # -----------------------------------------------------
    # 4. Character prefixes
    # -----------------------------------------------------

    compact = "".join(tokens)

    for n in (5, 6):

        if len(compact) >= n:

            keys.add(
                f"{country}|CP{n}|{compact[:n]}"
            )

    # -----------------------------------------------------
    # 5. Character suffix
    # -----------------------------------------------------

    if len(compact) >= 5:

        keys.add(
            f"{country}|CS5|{compact[-5:]}"
        )

    return keys


def address_keys(address, country):
    keys = set()

    if not address:
        return keys

    country = country.upper().strip()

    tokens = address.split()

    # -----------------------------------------------------
    # Numeric components
    # -----------------------------------------------------

    numbers = []

    for token in tokens:

        cleaned = "".join(
            c for c in token
            if c.isdigit()
        )

        if cleaned:
            numbers.append(cleaned)

    for number in numbers[:3]:

        keys.add(
            f"{country}|AN|{number}"
        )

    # -----------------------------------------------------
    # Address token signatures
    # -----------------------------------------------------

    significant = [
        t for t in tokens
        if len(t) >= 4
        and not t.isdigit()
    ]

    if significant:

        # First significant token
        keys.add(
            f"{country}|AF|{significant[0]}"
        )

        # Sorted rare-looking token pair
        if len(significant) >= 2:

            pair = sorted(
                significant[:4]
            )[:2]

            keys.add(
                f"{country}|AP|{'_'.join(pair)}"
            )

    # -----------------------------------------------------
    # Character prefix of address
    # -----------------------------------------------------

    compact = "".join(tokens)

    if len(compact) >= 6:

        keys.add(
            f"{country}|AC6|{compact[:6]}"
        )

    return keys


def make_keys(name, address, country):

    keys = set()

    keys.update(
        name_keys(
            name,
            country
        )
    )

    keys.update(
        address_keys(
            address,
            country
        )
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


def count_keys():

    print("=" * 70)
    print("V2 - COUNTING BLOCKING KEYS")
    print("=" * 70)

    counts = defaultdict(int)

    for source_name, path in [
        ("S2", S2_PATH),
        ("S3", S3_PATH)
    ]:

        total = 0

        print(f"\nProcessing {source_name}")

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

                keys = make_keys(
                    row.norm_name,
                    row.norm_address,
                    row.country_norm
                )

                for key in keys:
                    counts[key] += 1

            total += len(df)

            print(
                f"  {source_name} "
                f"chunk {chunk_no}: "
                f"{total:,} rows"
            )

    print(
        f"\nTotal unique keys: "
        f"{len(counts):,}"
    )

    valid = {
        key
        for key, count in counts.items()
        if count <= MAX_KEY_FREQUENCY
    }

    print(
        f"Keys retained: "
        f"{len(valid):,}"
    )

    return valid


def build_index(valid_keys):

    print("\n" + "=" * 70)
    print("V2 - BUILDING INDEX")
    print("=" * 70)

    index = defaultdict(list)

    for source_name, path in [
        ("S2", S2_PATH),
        ("S3", S3_PATH)
    ]:

        print(f"\nIndexing {source_name}")

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

                keys = make_keys(
                    row.norm_name,
                    row.norm_address,
                    row.country_norm
                )

                for key in keys:

                    if key in valid_keys:

                        index[key].append(
                            (
                                source_name,
                                row.entity_id
                            )
                        )

            print(
                f"  indexed "
                f"{source_name} "
                f"chunk {chunk_no}"
            )

    print(
        f"\nIndex keys: "
        f"{len(index):,}"
    )

    return index


def generate_candidates(index):

    print("\n" + "=" * 70)
    print("V2 - GENERATING CANDIDATES")
    print("=" * 70)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    first_write = True

    total_s1 = 0
    total_candidates = 0

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

        rows = []

        for row in df.itertuples(index=False):

            s1_id = row.entity_id

            keys = make_keys(
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

                        rows.append(
                            (
                                s1_id,
                                source_name,
                                entity_id
                            )
                        )

        if rows:

            output = pd.DataFrame(
                rows,
                columns=[
                    "source1_entity_id",
                    "source",
                    "candidate_entity_id"
                ]
            )

            output.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            total_candidates += len(output)

        total_s1 += len(df)

        print(
            f"  S1 chunk {chunk_no}: "
            f"{len(df):,} rows | "
            f"{len(rows):,} candidates | "
            f"total {total_candidates:,}"
        )

    print("\n" + "=" * 70)
    print("V2 COMPLETE")
    print("=" * 70)

    print(
        f"S1 processed: "
        f"{total_s1:,}"
    )

    print(
        f"Candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Output: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":

    valid_keys = count_keys()

    index = build_index(
        valid_keys
    )

    generate_candidates(
        index
    )

