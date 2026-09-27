import os
import hashlib
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_PATH = os.path.join(
    BASE_DIR, "dataset", "train", "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    BASE_DIR, "experiments", "train_candidate_pairs_v3.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "experiments", "ml_training_pairs.tsv"
)

CHUNK_SIZE = 500_000

TARGET_POSITIVES = 300_000
TARGET_NEGATIVES = 900_000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def pair_hash(s1, s2):
    value = str(s1) + "\x1f" + str(s2)

    return hashlib.blake2b(
        value.encode("utf-8"),
        digest_size=16
    ).digest()


print("=" * 70)
print("STEP 10 - BUILDING GROUND-TRUTH LOOKUP")
print("=" * 70)

positive_hashes = set()

gt_rows = 0
gt_pairs = 0

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=100_000
):

    chunk = chunk.fillna("")

    for _, row in chunk.iterrows():

        s1 = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        gt_rows += 1

        if not matched:
            continue

        for entity_id in matched.split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            positive_hashes.add(
                pair_hash(s1, entity_id)
            )

            gt_pairs += 1

print(f"Ground-truth S1 rows : {gt_rows:,}")
print(f"Ground-truth pairs    : {gt_pairs:,}")
print(f"Unique positive keys  : {len(positive_hashes):,}")

print()
print("=" * 70)
print("STEP 10 - SAMPLING CANDIDATE PAIRS")
print("=" * 70)

positive_samples = []
negative_samples = []

processed = 0
found_positive = 0
found_negative = 0

# Reservoir sampling counters.
positive_seen = 0
negative_seen = 0

for chunk in pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):

    chunk = chunk.fillna("")

    for s1, candidate in zip(
        chunk["source1_entity_id"],
        chunk["candidate_entity_id"]
    ):

        h = pair_hash(s1, candidate)

        if h in positive_hashes:

            found_positive += 1
            positive_seen += 1

            item = (s1, candidate, 1)

            if len(positive_samples) < TARGET_POSITIVES:
                positive_samples.append(item)

            else:
                j = random.randint(1, positive_seen)

                if j <= TARGET_POSITIVES:
                    positive_samples[j - 1] = item

        else:

            found_negative += 1
            negative_seen += 1

            item = (s1, candidate, 0)

            if len(negative_samples) < TARGET_NEGATIVES:
                negative_samples.append(item)

            else:
                j = random.randint(1, negative_seen)

                if j <= TARGET_NEGATIVES:
                    negative_samples[j - 1] = item

    processed += len(chunk)

    print(
        f"Processed candidates: {processed:,} | "
        f"Positive seen: {found_positive:,} | "
        f"Negative seen: {found_negative:,}"
    )

    if (
        len(positive_samples) >= TARGET_POSITIVES
        and len(negative_samples) >= TARGET_NEGATIVES
        and processed >= 50_000_000
    ):
        # We already have enough training examples.
        # Continue a little further for better reservoir diversity.
        pass


print()
print("=" * 70)
print("STEP 10 - SAVING TRAINING DATA")
print("=" * 70)

rows = positive_samples + negative_samples

random.shuffle(rows)

training = pd.DataFrame(
    rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "label"
    ]
)

training.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print(f"Training rows       : {len(training):,}")
print(f"Positive rows       : {(training['label'] == 1).sum():,}")
print(f"Negative rows       : {(training['label'] == 0).sum():,}")
print(f"Output              : {OUTPUT_PATH}")

print()
print("=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)
