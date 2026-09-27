import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

MATCHING = BASE / "output" / "matching_results.tsv"
CANDIDATES = BASE / "output" / "candidate_pairs.tsv"

print("Loading candidate sets...")

candidate_map = {}

with open(CANDIDATES, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        s1 = row["source1_entity_id"]
        ids = row["candidate_entity_ids"]

        if ids:
            candidate_map[s1] = set(ids.split(","))

print(f"Candidate S1 entities loaded: {len(candidate_map):,}")

print("Checking matching results...")

total_s1 = 0
matched_s1 = 0
total_matches = 0
invalid_matches = 0

with open(MATCHING, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        total_s1 += 1

        s1 = row["source1_entity_id"]
        ids = row["matched_entity_ids"]

        if not ids:
            continue

        matched_s1 += 1

        matches = ids.split(",")
        total_matches += len(matches)

        candidates = candidate_map.get(s1, set())

        for entity_id in matches:
            if entity_id not in candidates:
                invalid_matches += 1
                if invalid_matches <= 10:
                    print(
                        f"INVALID: {s1} -> {entity_id} "
                        f"not present in candidate set"
                    )

print()
print("========== VALIDATION ==========")
print(f"Matching S1 rows:       {total_s1:,}")
print(f"S1 with matches:        {matched_s1:,}")
print(f"Total predicted matches:{total_matches:,}")
print(f"Invalid matches:        {invalid_matches:,}")
print("================================")

if invalid_matches == 0:
    print("PASS: Every predicted match is in the candidate set.")
else:
    print("FAIL: Some predicted matches are outside the candidate set.")
