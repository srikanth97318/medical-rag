import re

MEDICAL_TERMS = re.compile(r"\b(symptom|disease|diagnos|treatment|medical|health|pain|fever|medication|drug|dose|doctor|patient|blood|infection|cancer|heart|lung|therapy|pregnan|vaccine|headache|rash|injury|condition|syndrome)\w*\b", re.I)
class TopicGuard:
    """Conservative lexical guard; this is a configurable heuristic, not a clinical classifier."""
    def classify(self, query: str) -> str:
        q = query.strip()
        if not q: return "INVALID_QUERY"
        return "MEDICAL" if MEDICAL_TERMS.search(q) else "OFF_TOPIC"
