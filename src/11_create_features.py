import os
import re
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio

TRAIN_PAIRS = "experiments/ml_training_pairs.tsv"

S1_FILE = "dataset/train/train_source1.tsv"
S2_FILE = "dataset/train/train_source2.tsv"
S3_FILE = "dataset/train/train_source3.tsv"

OUTPUT_FILE = "experiments/ml_features.tsv"

CHUNK_SIZE = 50000


def norm_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_text(value):
    return re.sub(r"[^a-z0-9]", "", value)


def token_overlap(a, b):
    if not a or not b:
        return 0.0

    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def numeric_tokens(value):
    if not value:
        return set()
    return set(re.findall(r"\d+", value))


def numeric_overlap(a, b):
    na = numeric_tokens(a)
    nb = numeric_tokens(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / len(na | nb)


def calculate_features(row, s1_lookup, s2_lookup, s3_lookup):

    s1 = s1_lookup.get(row["source1_entity_id"])

    if s1 is None:
        return None

    matched_id = row["candidate_entity_id"]

    if str(matched_id).upper().startswith("S2-"):
        target = s2_lookup.get(matched_id)
    else:
        target = s3_lookup.get(matched_id)

    if target is None:
        return None

    name1 = norm_text(s1["business_name"])
    name2 = norm_text(target["business_name"])

    addr1 = norm_text(s1["business_address"])
    addr2 = norm_text(target["business_address"])

    compact_name1 = compact_text(name1)
    compact_name2 = compact_text(name2)

    compact_addr1 = compact_text(addr1)
    compact_addr2 = compact_text(addr2)

    features = {
        "source1_entity_id": row["source1_entity_id"],
        "candidate_entity_id": matched_id,
        "label": int(row["label"]),

        # Country
        "country_match": int(
            str(s1["country"]).lower() ==
            str(target["country"]).lower()
        ),

        # Name features
        "name_exact": int(bool(name1) and name1 == name2),

        "name_ratio": ratio(name1, name2) / 100.0,

        "name_token_sort_ratio":
            token_sort_ratio(name1, name2) / 100.0,

        "name_token_set_ratio":
            token_set_ratio(name1, name2) / 100.0,

        "name_token_overlap":
            token_overlap(name1, name2),

        "name_compact_ratio":
            ratio(compact_name1, compact_name2) / 100.0,

        "name_length_diff":
            abs(len(name1) - len(name2)),

        "name_length_max":
            max(len(name1), len(name2), 1),

        # Address features
        "address_exact": int(bool(addr1) and addr1 == addr2),

        "address_ratio":
            ratio(addr1, addr2) / 100.0,

        "address_token_sort_ratio":
            token_sort_ratio(addr1, addr2) / 100.0,

        "address_token_set_ratio":
            token_set_ratio(addr1, addr2) / 100.0,

        "address_token_overlap":
            token_overlap(addr1, addr2),

        "address_numeric_overlap":
            numeric_overlap(addr1, addr2),

        "address_compact_ratio":
            ratio(compact_addr1, compact_addr2) / 100.0,

        "address_length_diff":
            abs(len(addr1) - len(addr2)),

        "address_length_max":
            max(len(addr1), len(addr2), 1),

        # Missing-field indicators
        "name1_missing": int(not bool(name1)),
        "name2_missing": int(not bool(name2)),
        "address1_missing": int(not bool(addr1)),
        "address2_missing": int(not bool(addr2)),
    }

    return features


def load_lookup(path):

    print(f"Loading {path} ...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df = df.set_index("entity_id")

    print(f"Loaded {len(df):,} rows")

    return df.to_dict("index")


def main():

    print("=" * 70)
    print("STEP 11 - FEATURE ENGINEERING")
    print("=" * 70)

    s1_lookup = load_lookup(S1_FILE)
    s2_lookup = load_lookup(S2_FILE)
    s3_lookup = load_lookup(S3_FILE)

    print("\nLoading training pairs...")

    total_rows = 0
    written_rows = 0

    first_write = True

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRAIN_PAIRS,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        print(
            f"\nProcessing chunk {chunk_no} "
            f"({len(chunk):,} rows)..."
        )

        rows = []

        for _, row in chunk.iterrows():

            result = calculate_features(
                row,
                s1_lookup,
                s2_lookup,
                s3_lookup
            )

            if result is not None:
                rows.append(result)

        if rows:

            feature_df = pd.DataFrame(rows)

            feature_df.to_csv(
                OUTPUT_FILE,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

            written_rows += len(feature_df)

        total_rows += len(chunk)

        print(
            f"Processed: {total_rows:,} | "
            f"Written: {written_rows:,}"
        )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"Input rows : {total_rows:,}")
    print(f"Output rows: {written_rows:,}")
    print(f"Output     : {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"File size  : {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
