import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

S1_FILE = BASE / "dataset" / "test" / "test_source1.tsv"
RAW_CANDIDATES = BASE / "output" / "test_candidate_pairs.tsv"
TEMP_OUTPUT = BASE / "output" / "candidate_pairs_formatted.tsv"
FINAL_OUTPUT = BASE / "output" / "candidate_pairs.tsv"


print("Loading test Source-1 IDs...")

valid_s1 = set()

with open(S1_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        valid_s1.add(row["entity_id"])

print(f"Test S1 entities: {len(valid_s1):,}")
print("Formatting candidate groups...")


seen_s1 = set()
total_candidate_rows = 0
group_count = 0


with open(RAW_CANDIDATES, "r", encoding="utf-8", newline="") as f, \
     open(TEMP_OUTPUT, "w", encoding="utf-8", newline="") as out:

    reader = csv.DictReader(f, delimiter="\t")
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")

    writer.writerow(["source1_entity_id", "candidate_entity_ids"])

    current_s1 = None
    candidates = []

    for row in reader:
        s1 = row["source1_entity_id"]
        candidate = row["candidate_entity_id"]

        if s1 not in valid_s1:
            raise ValueError(f"Invalid S1 ID found: {s1}")

        if current_s1 is None:
            current_s1 = s1

        if s1 != current_s1:

            writer.writerow([
                current_s1,
                ",".join(candidates)
            ])

            seen_s1.add(current_s1)
            group_count += 1

            current_s1 = s1
            candidates = []

        candidates.append(candidate)
        total_candidate_rows += 1

        if total_candidate_rows % 5_000_000 == 0:
            print(
                f"Candidate rows processed: "
                f"{total_candidate_rows:,} | "
                f"S1 groups written: {group_count:,}"
            )

    # Write final group
    if current_s1 is not None:
        writer.writerow([
            current_s1,
            ",".join(candidates)
        ])

        seen_s1.add(current_s1)
        group_count += 1


missing_s1 = valid_s1 - seen_s1

print()
print(f"Candidate rows processed: {total_candidate_rows:,}")
print(f"S1 groups with candidates: {group_count:,}")
print(f"S1 entities without candidates: {len(missing_s1):,}")


# Append empty candidate rows for S1 entities with no candidates
if missing_s1:

    with open(TEMP_OUTPUT, "a", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")

        for s1 in missing_s1:
            writer.writerow([s1, ""])


print("Replacing old candidate_pairs.tsv...")

if FINAL_OUTPUT.exists():
    FINAL_OUTPUT.unlink()

TEMP_OUTPUT.rename(FINAL_OUTPUT)

print()
print("Formatting completed successfully.")
print(f"Final output: {FINAL_OUTPUT}")
