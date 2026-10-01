class SelfCheckVerifier:
    """SelfCheck-inspired consistency check using optional sampled generations; not official SelfCheckGPT."""
    def __init__(self, samples: int = 3): self.samples = samples
    def verify(self, answer: str, samples: list[str]) -> dict:
        if not samples: return {"status":"NOT_EVALUATED","consistency":None,"method":"SelfCheck-inspired lexical consistency; not official SelfCheckGPT"}
        words = set(answer.lower().split())
        scores = [len(words & set(s.lower().split())) / max(1,len(words)) for s in samples]
        return {"status":"EVALUATED","consistency":sum(scores)/len(scores),"method":"SelfCheck-inspired lexical consistency; not official SelfCheckGPT"}
