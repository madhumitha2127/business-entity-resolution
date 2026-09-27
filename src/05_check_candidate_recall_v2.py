import pandas as pd

GT_PATH = "dataset/train/train_ground_truth.tsv"
CAND_PATH = "experiments/train_candidate_pairs_v2.tsv"

print("=" * 70)
print("CANDIDATE RECALL CHECK")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load ground truth
# ---------------------------------------------------------

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

truth = set()

for row in gt.itertuples(index=False):

    ids = row.matched_entity_ids.strip()

    if not ids:
        continue

    for entity_id in ids.split(","):

        entity_id = entity_id.strip()

        if entity_id:
            truth.add(
                (row.source1_entity_id, entity_id)
            )

print(f"Ground-truth pairs: {len(truth):,}")

# ---------------------------------------------------------
# 2. Stream candidate file
# ---------------------------------------------------------

found = set()

chunk_size = 500_000
processed = 0

for chunk_no, chunk in enumerate(
    pd.read_csv(
        CAND_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_entity_id",
            "candidate_entity_id"
        ],
        chunksize=chunk_size,
        keep_default_na=False
    ),
    start=1
):

    for row in chunk.itertuples(index=False):

        pair = (
            row.source1_entity_id,
            row.candidate_entity_id
        )

        if pair in truth:
            found.add(pair)

    processed += len(chunk)

    print(
        f"Processed candidates: "
        f"{processed:,} | "
        f"true matches found: "
        f"{len(found):,}"
    )

# ---------------------------------------------------------
# 3. Results
# ---------------------------------------------------------

missing = len(truth) - len(found)

recall = (
    len(found) / len(truth)
    if truth
    else 0
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Ground-truth matches      : {len(truth):,}")
print(f"Found in candidates       : {len(found):,}")
print(f"Missing from candidates   : {missing:,}")
print(f"Candidate recall          : {recall:.4%}")

