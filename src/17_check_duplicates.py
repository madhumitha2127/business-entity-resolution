import csv

MATCHING = "output/matching_results.tsv"
CANDIDATES = "output/candidate_pairs.tsv"

def check_file(path, id_column):
    rows = 0
    duplicate_s1 = 0
    duplicate_ids = 0
    seen_s1 = set()

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows += 1
            s1 = row["source1_entity_id"]

            if s1 in seen_s1:
                duplicate_s1 += 1
            seen_s1.add(s1)

            ids = row[id_column]

            if ids:
                values = ids.split(",")
                if len(values) != len(set(values)):
                    duplicate_ids += 1

    return rows, len(seen_s1), duplicate_s1, duplicate_ids


print("Checking matching_results.tsv...")
m = check_file(MATCHING, "matched_entity_ids")

print("Checking candidate_pairs.tsv...")
c = check_file(CANDIDATES, "candidate_entity_ids")

print()
print("========== DUPLICATE CHECK ==========")
print(f"Matching rows:              {m[0]:,}")
print(f"Unique matching S1 IDs:     {m[1]:,}")
print(f"Duplicate matching S1 rows: {m[2]:,}")
print(f"Duplicate matching ID lists:{m[3]:,}")
print()
print(f"Candidate rows:             {c[0]:,}")
print(f"Unique candidate S1 IDs:    {c[1]:,}")
print(f"Duplicate candidate S1 rows:{c[2]:,}")
print(f"Duplicate candidate ID lists:{c[3]:,}")
print("======================================")

if (
    m[0] == 1732544
    and m[1] == 1732544
    and m[2] == 0
    and m[3] == 0
    and c[0] == 1732544
    and c[1] == 1732544
    and c[2] == 0
    and c[3] == 0
):
    print("PASS: No duplicate S1 rows or duplicate IDs found.")
else:
    print("CHECK FAILED: Review the values above.")
