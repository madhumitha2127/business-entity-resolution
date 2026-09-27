import os
import json
import re
import numpy as np
import pandas as pd
import lightgbm as lgb

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio
)

# ============================================================
# STEP 14 - TEST PREDICTION
# ============================================================

CANDIDATE_FILE = "output/test_candidate_pairs.tsv"

S1_FILE = "dataset/test/test_source1.tsv"
S2_FILE = "dataset/test/test_source2.tsv"
S3_FILE = "dataset/test/test_source3.tsv"

MODEL_FILE = "models/lightgbm_entity_matcher.txt"
THRESHOLD_FILE = "models/best_threshold.json"

MATCHING_OUTPUT = "output/matching_results.tsv"
FINAL_CANDIDATE_OUTPUT = "output/candidate_pairs.tsv"

CHUNK_SIZE = 50000


# ------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------

def norm_text(value):
    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def compact_text(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        value
    )


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

    return set(
        re.findall(
            r"\d+",
            value
        )
    )


def numeric_overlap(a, b):

    na = numeric_tokens(a)
    nb = numeric_tokens(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / len(na | nb)


# ------------------------------------------------------------
# Load source data
# ------------------------------------------------------------

def load_lookup(path):

    print(f"Loading {path} ...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Loaded {len(df):,} rows"
    )

    df = df.set_index(
        "entity_id"
    )

    return df.to_dict("index")


# ------------------------------------------------------------
# Feature creation
# ------------------------------------------------------------

def create_features(
    source1_id,
    candidate_id,
    s1_lookup,
    s2_lookup,
    s3_lookup
):

    s1 = s1_lookup.get(
        source1_id
    )

    if s1 is None:
        return None

    if str(candidate_id).upper().startswith("S2-"):
        target = s2_lookup.get(
            candidate_id
        )
    else:
        target = s3_lookup.get(
            candidate_id
        )

    if target is None:
        return None

    name1 = norm_text(
        s1["business_name"]
    )

    name2 = norm_text(
        target["business_name"]
    )

    addr1 = norm_text(
        s1["business_address"]
    )

    addr2 = norm_text(
        target["business_address"]
    )

    compact_name1 = compact_text(
        name1
    )

    compact_name2 = compact_text(
        name2
    )

    compact_addr1 = compact_text(
        addr1
    )

    compact_addr2 = compact_text(
        addr2
    )

    return [

        int(
            str(s1["country"]).lower()
            ==
            str(target["country"]).lower()
        ),

        int(
            bool(name1)
            and
            name1 == name2
        ),

        ratio(
            name1,
            name2
        ) / 100.0,

        token_sort_ratio(
            name1,
            name2
        ) / 100.0,

        token_set_ratio(
            name1,
            name2
        ) / 100.0,

        token_overlap(
            name1,
            name2
        ),

        ratio(
            compact_name1,
            compact_name2
        ) / 100.0,

        abs(
            len(name1)
            -
            len(name2)
        ),

        max(
            len(name1),
            len(name2),
            1
        ),

        int(
            bool(addr1)
            and
            addr1 == addr2
        ),

        ratio(
            addr1,
            addr2
        ) / 100.0,

        token_sort_ratio(
            addr1,
            addr2
        ) / 100.0,

        token_set_ratio(
            addr1,
            addr2
        ) / 100.0,

        token_overlap(
            addr1,
            addr2
        ),

        numeric_overlap(
            addr1,
            addr2
        ),

        ratio(
            compact_addr1,
            compact_addr2
        ) / 100.0,

        abs(
            len(addr1)
            -
            len(addr2)
        ),

        max(
            len(addr1),
            len(addr2),
            1
        ),

        int(not bool(name1)),

        int(not bool(name2)),

        int(not bool(addr1)),

        int(not bool(addr2))
    ]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("STEP 14 - TEST PREDICTION")
    print("=" * 70)

    os.makedirs(
        "output",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading LightGBM model...")

    model = lgb.Booster(
        model_file=MODEL_FILE
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold_info = json.load(f)

    threshold = float(
        threshold_info["threshold"]
    )

    feature_columns = (
        threshold_info["feature_columns"]
    )

    print(
        f"Threshold: {threshold:.4f}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load test sources
    # --------------------------------------------------------

    print("\nLoading test source files...")

    s1_lookup = load_lookup(
        S1_FILE
    )

    s2_lookup = load_lookup(
        S2_FILE
    )

    s3_lookup = load_lookup(
        S3_FILE
    )

    # --------------------------------------------------------
    # Prepare S1 IDs
    # --------------------------------------------------------

    s1_df = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"]
    )

    all_s1_ids = s1_df[
        "entity_id"
    ].tolist()

    print(
        f"\nTest Source 1 entities: "
        f"{len(all_s1_ids):,}"
    )

    # --------------------------------------------------------
    # Match storage
    # --------------------------------------------------------

    matches = {}

    for entity_id in all_s1_ids:
        matches[entity_id] = []

    # --------------------------------------------------------
    # Process candidate pairs
    # --------------------------------------------------------

    print(
        "\nProcessing candidate pairs..."
    )

    total_candidates = 0
    valid_candidates = 0
    total_matches = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            CANDIDATE_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        feature_rows = []
        pair_ids = []

        for _, row in chunk.iterrows():

            source1_id = row[
                "source1_entity_id"
            ]

            candidate_id = row[
                "candidate_entity_id"
            ]

            features = create_features(
                source1_id,
                candidate_id,
                s1_lookup,
                s2_lookup,
                s3_lookup
            )

            if features is not None:

                feature_rows.append(
                    features
                )

                pair_ids.append(
                    (
                        source1_id,
                        candidate_id
                    )
                )

        if feature_rows:

            X = pd.DataFrame(
                feature_rows,
                columns=feature_columns
            )

            probabilities = model.predict(
                X
            )

            for (
                pair,
                probability
            ) in zip(
                pair_ids,
                probabilities
            ):

                if probability >= threshold:

                    source1_id, candidate_id = pair

                    matches[
                        source1_id
                    ].append(
                        candidate_id
                    )

                    total_matches += 1

            valid_candidates += len(
                feature_rows
            )

        total_candidates += len(
            chunk
        )

        print(
            f"Chunk {chunk_no} | "
            f"Processed: {total_candidates:,} | "
            f"Matches: {total_matches:,}"
        )

    # --------------------------------------------------------
    # Remove duplicate matches
    # --------------------------------------------------------

    print(
        "\nRemoving duplicate matches..."
    )

    for source1_id in matches:

        if matches[source1_id]:

            matches[source1_id] = sorted(
                set(
                    matches[source1_id]
                )
            )

    # --------------------------------------------------------
    # Write matching_results.tsv
    # --------------------------------------------------------

    print(
        "\nWriting matching_results.tsv..."
    )

    output_rows = []

    for source1_id in all_s1_ids:

        matched_ids = matches[
            source1_id
        ]

        output_rows.append(
            {
                "source1_entity_id":
                    source1_id,

                "matched_entity_ids":
                    ",".join(
                        matched_ids
                    )
            }
        )

    result_df = pd.DataFrame(
        output_rows
    )

    result_df.to_csv(
        MATCHING_OUTPUT,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Copy exact candidate set
    # --------------------------------------------------------

    print(
        "\nCreating final candidate_pairs.tsv..."
    )

    if os.path.exists(
        FINAL_CANDIDATE_OUTPUT
    ):
        os.remove(
            FINAL_CANDIDATE_OUTPUT
        )

    first_write = True

    for chunk in pd.read_csv(
        CANDIDATE_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    ):

        chunk.to_csv(
            FINAL_CANDIDATE_OUTPUT,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    singleton_count = sum(
        1
        for values in matches.values()
        if len(values) == 0
    )

    matched_s1_count = (
        len(all_s1_ids)
        -
        singleton_count
    )

    print("\n" + "=" * 70)
    print("STEP 14 - TEST PREDICTION COMPLETE")
    print("=" * 70)

    print(
        f"Candidate pairs processed: "
        f"{total_candidates:,}"
    )

    print(
        f"Valid feature pairs: "
        f"{valid_candidates:,}"
    )

    print(
        f"Predicted matches: "
        f"{total_matches:,}"
    )

    print(
        f"S1 entities with matches: "
        f"{matched_s1_count:,}"
    )

    print(
        f"S1 entities with no matches: "
        f"{singleton_count:,}"
    )

    print(
        f"\nMatching output:"
    )

    print(
        MATCHING_OUTPUT
    )

    print(
        f"\nCandidate output:"
    )

    print(
        FINAL_CANDIDATE_OUTPUT
    )


if __name__ == "__main__":
    main()
