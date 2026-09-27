import re
import unicodedata


# Common legal/business suffix normalization.
# We keep this deliberately conservative so that
# meaningful business-name information is not removed.
LEGAL_SUFFIX_MAP = {
    "private limited": "pvt ltd",
    "private ltd": "pvt ltd",
    "privatelimited": "pvt ltd",
    "pvt limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "limited": "ltd",
    "ltd": "ltd",
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "llc": "llc",
    "llp": "llp",
}


def unicode_normalize(text):
    """Normalize Unicode while preserving non-Latin scripts."""
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    return text


def normalize_basic(text):
    """
    General normalization for business names and addresses.

    - Unicode normalization
    - lowercase
    - punctuation -> spaces
    - whitespace normalization
    """
    text = unicode_normalize(text)

    text = text.lower()

    # Replace punctuation/symbols with spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(text):
    """
    Conservative business-name normalization.

    Keeps meaningful tokens while normalizing common legal suffixes.
    """
    text = normalize_basic(text)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    i = 0

    while i < len(tokens):
        # Handle "private limited"
        if (
            i + 1 < len(tokens)
            and tokens[i] == "private"
            and tokens[i + 1] == "limited"
        ):
            normalized_tokens.append("pvt")
            normalized_tokens.append("ltd")
            i += 2
            continue

        # Handle "private ltd"
        if (
            i + 1 < len(tokens)
            and tokens[i] == "private"
            and tokens[i + 1] == "ltd"
        ):
            normalized_tokens.append("pvt")
            normalized_tokens.append("ltd")
            i += 2
            continue

        token = LEGAL_SUFFIX_MAP.get(tokens[i], tokens[i])
        normalized_tokens.append(token)

        i += 1

    return " ".join(normalized_tokens)


def normalize_address(text):
    """
    General address normalization.

    We intentionally do not use external geocoding or
    external address databases.
    """
    text = normalize_basic(text)

    if not text:
        return ""

    # Common address abbreviations.
    replacements = {
        " road ": " rd ",
        " street ": " st ",
        " avenue ": " ave ",
        " boulevard ": " blvd ",
        " highway ": " hwy ",
        " apartment ": " apt ",
        " suite ": " ste ",
        " building ": " bldg ",
    }

    padded = f" {text} "

    for old, new in replacements.items():
        padded = padded.replace(old, new)

    return re.sub(r"\s+", " ", padded).strip()


def tokenize(text):
    """Return normalized whitespace-separated tokens."""
    if not text:
        return []

    return text.split()


def name_tokens(text):
    """Normalize a business name and return tokens."""
    return tokenize(normalize_business_name(text))


def address_tokens(text):
    """Normalize an address and return tokens."""
    return tokenize(normalize_address(text))
