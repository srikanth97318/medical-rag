def correction_feedback(claims: list[dict], passages: list[dict]) -> str:
    problems = [c for c in claims if c.get("status") in {"UNSUPPORTED","CONTRADICTED"}]
    evidence = "\n".join(p["text"] for p in passages)
    return "Remove or qualify the unsupported claims listed here; do not add facts beyond the evidence. Claims: " + "; ".join(c["claim"] for c in problems) + "\nAvailable evidence: " + evidence
