import json
import logging
from pathlib import Path
from .utils import utc_now

def log_result(result: dict, path: str | Path) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    row={"query_id":result.get("query_id"),"timestamp":utc_now(),"status":result.get("status"),"timing":result.get("timing"),"correction_count":result.get("correction_count"),"error":result.get("error")}
    with p.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")

def configure_logging(level=logging.INFO): logging.basicConfig(level=level,format="%(asctime)s %(levelname)s %(message)s")
