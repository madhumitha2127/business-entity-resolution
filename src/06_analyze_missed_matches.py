import pandas as pd

GT_PATH = "dataset/train/train_ground_truth.tsv"
S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
CAND_PATH = "experiments/train_candidate_pairs.tsv"

print("=" * 70)
print("ANALYZING MISSED TRUE MATCHES")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load candidate pairs in chunks
# ---------------------------------------------------------

candidate_pairs = set()

for chunk in pd.read_csv(
    CAND_PATH,
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id", "candidate_entity_id"],
    chunksize=500_000,
    keep_default_na=False
):
    candidate_pairs.update(
        zip(
            chunk["source1_entity_id"],
            chunk["candidate_entity_id"]
        )
    )

print(f"Candidate pairs loaded: {len(candidate_pairs):,}")

# ---------------------------------------------------------
# 2. Find missed ground-truth pairs
# ---------------------------------------------------------

missed = []

for chunk in pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    chunksize=200_000,
    keep_default_na=False
):

    for row in chunk.itertuples(index=False):

        if not row.matched_entity_ids:
            continue

        for entity_id in row.matched_entity_ids.split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            pair = (
                row.source1_entity_id,
                entity_id
            )

            if pair not in candidate_pairs:
                missed.append(pair)

                if len(missed) >= 100:
                    break

        if len(missed) >= 100:
            break

    if len(missed) >= 100:
        break

print(f"Collected missed examples: {len(missed)}")

# ---------------------------------------------------------
# 3. Collect required IDs
# ---------------------------------------------------------

s1_ids = {x[0] for x in missed}

s2_ids = {
    x[1]
    for x in missed
    if str(x[1]).startswith("S2")
}

s3_ids = {
    x[1]
    for x in missed
    if str(x[1]).startswith("S3")
}

# ---------------------------------------------------------
# 4. Load required S1 records
# ---------------------------------------------------------

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

s1 = s1[
    s1["entity_id"].isin(s1_ids)
].set_index("entity_id")

# ---------------------------------------------------------
# 5. Load required S2/S3 records in chunks
# ---------------------------------------------------------

records = {}

for path, wanted in [
    (S2_PATH, s2_ids),
    (S3_PATH, s3_ids)
]:

    if not wanted:
        continue

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
        keep_default_na=False
    ):

        selected = chunk[
            chunk["entity_id"].isin(wanted)
        ]

        for row in selected.itertuples(index=False):
            records[row.entity_id] = row

# ---------------------------------------------------------
# 6. Display missed examples
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MISSED MATCH EXAMPLES")
print("=" * 70)

for i, (s1_id, target_id) in enumerate(
    missed,
    start=1
):

    s1_row = s1.loc[s1_id]

    target = records.get(target_id)

    print(f"\n--- Example {i} ---")

    print(f"S1 ID        : {s1_id}")
    print(f"True Match   : {target_id}")

    print(f"S1 Name      : {s1_row['business_name']}")
    print(f"S1 Address   : {s1_row['business_address']}")
    print(f"S1 Country   : {s1_row['country']}")

    if target is not None:

        print(f"Match Name   : {target.business_name}")
        print(f"Match Address: {target.business_address}")
        print(f"Match Country: {target.country}")

print("\nDone.")
