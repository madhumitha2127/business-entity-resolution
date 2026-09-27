import re

def first_token(text):
    if not text:
        return ""
    return text.split()[0]

def first_n_chars(text, n=4):
    if not text:
        return ""
    return text.replace(" ", "")[:n]

def name_block_keys(name):
    """
    Generate multiple blocking keys from normalized business name.
    Multiple keys improve recall.
    """
    if not name:
        return []

    tokens = name.split()
    keys = set()

    # Exact normalized name
    keys.add(f"name_exact:{name}")

    # First token
    if tokens:
        keys.add(f"name_first:{tokens[0]}")

    # First two tokens
    if len(tokens) >= 2:
        keys.add(f"name_first2:{tokens[0]}_{tokens[1]}")

    # Character prefix
    compact = "".join(tokens)
    if compact:
        keys.add(f"name_prefix4:{compact[:4]}")

    return list(keys)


def address_block_keys(address):
    """
    Generate multiple blocking keys from normalized address.
    """
    if not address:
        return []

    tokens = address.split()
    keys = set()

    # Exact normalized address
    keys.add(f"addr_exact:{address}")

    # Numeric tokens are often strong address signals
    numbers = [t for t in tokens if any(c.isdigit() for c in t)]

    for number in numbers[:2]:
        keys.add(f"addr_num:{number}")

    # First address token
    if tokens:
        keys.add(f"addr_first:{tokens[0]}")

    # First 2 tokens
    if len(tokens) >= 2:
        keys.add(f"addr_first2:{tokens[0]}_{tokens[1]}")

    return list(keys)


def combined_block_keys(name, address, country):
    """
    Country-aware combined blocking keys.
    """
    keys = set()

    country = (country or "").strip().upper()

    for key in name_block_keys(name):
        keys.add(f"{country}|{key}")

    for key in address_block_keys(address):
        keys.add(f"{country}|{key}")

    return list(keys)
