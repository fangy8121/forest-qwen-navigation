"""Buffered append-only measured traces, recoverable after an interrupted run."""
import json
from pathlib import Path

import numpy as np


class TraceJournal:
    columns = 21

    def __init__(self, folder):
        self.folder = Path(folder)
        self.rows = 0
        self.stream = (self.folder/'trajectory.partial.f64').open('xb', buffering=65536)
        self._metadata(False)

    def _metadata(self, closed):
        temporary = self.folder/'trajectory_journal.json.tmp'
        temporary.write_text(json.dumps(dict(
            schema='measured_flight_trace_journal_v1', dtype='<f8', columns=self.columns,
            closed_cleanly=closed, rows_written=self.rows if closed else None,
            scope='measured prefix only; never a complete-flight or success certificate',
            durability='buffered OS writes; restart may lose the final buffered records'))+'\n')
        temporary.replace(self.folder/'trajectory_journal.json')

    def append(self, row):
        values = np.asarray(row, dtype='<f8')
        if values.shape != (self.columns,) or not np.isfinite(values).all():
            raise ValueError('Trace journal requires one finite 21-column measured row')
        self.stream.write(values.tobytes())
        self.rows += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        self._metadata(True)


def read_trace_journal(folder):
    """Recover complete records only; callers must still label them partial."""
    folder = Path(folder)
    metadata = json.loads((folder/'trajectory_journal.json').read_text())
    if (metadata.get('schema') != 'measured_flight_trace_journal_v1'
            or metadata.get('dtype') != '<f8' or metadata.get('columns') != 21):
        raise ValueError('Unsupported trace journal schema')
    raw = (folder/'trajectory.partial.f64').read_bytes()
    width = 21*8
    usable = len(raw)//width*width
    rows = np.frombuffer(raw[:usable], dtype='<f8').reshape(-1, 21).copy()
    if len(rows) and (not np.isfinite(rows).all() or (np.diff(rows[:, 0]) <= 0).any()):
        raise ValueError('Trace journal contains invalid measured records')
    return rows, dict(metadata, recovered_rows=len(rows), trailing_bytes=len(raw)-usable,
                     trajectory_complete=False)
