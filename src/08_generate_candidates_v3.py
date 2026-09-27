import os
import re
from collections import defaultdict, Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

S1_PATH = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv")
S2_PATH = os.path.join(BASE_DIR, "dataset", "train", "train_source2.tsv")
S3_PATH = os.path.join(BASE_DIR, "dataset", "train", "train_source3.tsv")

OUTPUT_PATH = os.path.join(
    BASE_DIR, "experiments", "train_candidate_pairs_v3.tsv"
)

CHUNK_SIZE = 100_000

# Keep very common keys out of the index.
# This prevents huge candidate explosions.
MAX_KEY_FREQUENCY = 100

# Maximum number of character shingles generated per field.
MAX_CHAR_KEYS = 8


def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text):
    return re.sub(r"[^a-z0-9]", "", text)


def tokens(text):
    return [x for x in normalize(text).split() if x]


def char_ngrams(text, n=4):
    text = compact(text)

    if len(text) < n:
        return []

    grams = set()

    for i in range(len(text) - n + 1):
        grams.add(text[i:i+n])

    # Deterministic selection to control index size.
    grams = sorted(grams)

    if len(grams) > MAX_CHAR_KEYS:
        # Take evenly distributed shingles.
        idx = [
            round(i * (len(grams) - 1) / (MAX_CHAR_KEYS - 1))
            for i in range(MAX_CHAR_KEYS)
        ]
        grams = [grams[i] for i in idx]

    return grams


def numeric_key(text):
    nums = re.findall(r"\d+", normalize(text))

    if not nums:
        return None

    return "|".join(sorted(set(nums)))


def name_keys(name):
    n = normalize(name)

    if not n:
        return set()

    t = tokens(n)
    keys = set()

    # Exact normalized name.
    keys.add("N:" + n)

    # Word-order independent signature.
    if t:
        keys.add("NS:" + " ".join(sorted(t)))

    # First important tokens.
    if len(t) >= 2:
        keys.add("N2:" + " ".join(sorted(t[:2])))

    # Long informative tokens.
    for tok in t:
        if len(tok) >= 6:
            keys.add("NT:" + tok)

    # Character shingles for typo tolerance.
    for gram in char_ngrams(n, 4):
        keys.add("C4:" + gram)

    # Compact prefixes.
    c = compact(n)

    if len(c) >= 6:
        keys.add("CP6:" + c[:6])

    if len(c) >= 8:
        keys.add("CP8:" + c[:8])

    return keys


def address_keys(address):
    a = normalize(address)

    if not a:
        return set()

    t = tokens(a)
    keys = set()

    # Exact address.
    keys.add("A:" + a)

    # Numeric components.
    nk = numeric_key(a)
    if nk:
        keys.add("AN:" + nk)

    # First significant address tokens.
    meaningful = [
        x for x in t
        if len(x) >= 3 and not x.isdigit()
    ]

    if meaningful:
        keys.add("AF:" + meaningful[0])

    if len(meaningful) >= 2:
        keys.add(
            "AP:" + "|".join(sorted(meaningful[:2]))
        )

    # Address character shingles.
    for gram in char_ngrams(a, 4):
        keys.add("AC4:" + gram)

    # Compact address prefix.
    c = compact(a)

    if len(c) >= 7:
        keys.add("ACP7:" + c[:7])

    return keys


def make_keys(row):
    country = normalize(row["country"])
    keys = set()

    for k in name_keys(row["business_name"]):
        keys.add(country + "|" + k)

    for k in address_keys(row["business_address"]):
        keys.add(country + "|" + k)

    return keys


def count_keys(path):
    counts = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")

        for _, row in chunk.iterrows():
            for key in make_keys(row):
                counts[key] += 1

    return counts


def build_index(path, allowed_keys):
    index = defaultdict(list)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        chunk = chunk.fillna("")

        for _, row in chunk.iterrows():
            entity_id = row["entity_id"]

            for key in make_keys(row):
                if key in allowed_keys:
                    index[key].append(entity_id)

    return index


print("=" * 70)
print("V3 - COUNTING BLOCKING KEYS")
print("=" * 70)

count_s2 = count_keys(S2_PATH)
count_s3 = count_keys(S3_PATH)

combined_counts = count_s2 + count_s3

allowed_keys = {
    key
    for key, count in combined_counts.items()
    if count <= MAX_KEY_FREQUENCY
}

print(f"Total unique keys      : {len(combined_counts):,}")
print(f"Allowed keys           : {len(allowed_keys):,}")
print(f"Max key frequency      : {MAX_KEY_FREQUENCY:,}")

del count_s2
del count_s3
del combined_counts

print()
print("=" * 70)
print("V3 - BUILDING SOURCE INDEX")
print("=" * 70)

index_s2 = build_index(S2_PATH, allowed_keys)
index_s3 = build_index(S3_PATH, allowed_keys)

print(f"S2 index keys          : {len(index_s2):,}")
print(f"S3 index keys          : {len(index_s3):,}")

print()
print("=" * 70)
print("V3 - GENERATING CANDIDATES")
print("=" * 70)

if os.path.exists(OUTPUT_PATH):
    os.remove(OUTPUT_PATH)

total_candidates = 0
total_s1 = 0

first_write = True

for chunk in pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    chunksize=CHUNK_SIZE
):
    chunk = chunk.fillna("")

    output_rows = []

    for _, row in chunk.iterrows():

        s1_id = row["entity_id"]

        keys = make_keys(row)

        candidates = set()

        for key in keys:
            if key in index_s2:
                candidates.update(index_s2[key])

            if key in index_s3:
                candidates.update(index_s3[key])

        for candidate_id in candidates:
            output_rows.append(
                (s1_id, candidate_id)
            )

    if output_rows:

        out = pd.DataFrame(
            output_rows,
            columns=[
                "source1_entity_id",
                "candidate_entity_id"
            ]
        )

        out.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False
        total_candidates += len(out)

    total_s1 += len(chunk)

    print(
        f"Processed S1: {total_s1:,} | "
        f"Candidates: {total_candidates:,}"
    )

print()
print("=" * 70)
print("V3 COMPLETE")
print("=" * 70)

print(f"S1 processed       : {total_s1:,}")
print(f"Candidate pairs    : {total_candidates:,}")
print(f"Output             : {OUTPUT_PATH}")
