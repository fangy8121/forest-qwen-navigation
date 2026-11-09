"""Bounded lossless scan output; compression never touches the simulator."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np


class ScanWriter:
    def __init__(self, capacity=16):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='forest-scans')
        self.pending = deque()
        self.capacity = capacity

    def submit(self, path, frame):
        while self.pending and (self.pending[0].done() or len(self.pending) >= self.capacity):
            self.pending.popleft().result()
        snapshot = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in frame.items()}
        self.pending.append(self.pool.submit(np.savez_compressed, path, **snapshot))

    def close(self):
        try:
            while self.pending: self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)
