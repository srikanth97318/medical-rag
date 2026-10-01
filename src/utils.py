import re
import uuid
from datetime import datetime, timezone
from typing import Any

def new_id() -> str: return str(uuid.uuid4())
def normalize_text(value: Any) -> str:
    if value is None: return ""
    return re.sub(r"\s+", " ", str(value)).strip()
def utc_now() -> str: return datetime.now(timezone.utc).isoformat()
def is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or not normalize_text(value)
