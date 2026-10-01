import re

def extract_claims(answer: str) -> list[str]:
    cleaned = re.sub(r"(?im)^\s*(disclaimer|note|warning):.*$", "", answer)
    bits = re.split(r"(?<=[.!?])\s+|\n+|;\s*", cleaned)
    return [b.strip(" •-*\t") for b in bits if len(b.strip(" •-*\t")) > 12 and not b.strip().lower().startswith(("hello", "hi "))]
