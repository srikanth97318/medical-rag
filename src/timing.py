import time
import torch

def sync_cuda(enabled: bool=True) -> None:
    if enabled and torch.cuda.is_available(): torch.cuda.synchronize()
class StageTimer:
    def __init__(self, synchronize: bool=False): self.synchronize=synchronize
    def start(self):
        if self.synchronize: sync_cuda()
        return time.perf_counter()
    def stop(self, start: float) -> float:
        if self.synchronize: sync_cuda()
        return time.perf_counter()-start
