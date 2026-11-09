"""An interrupted prefix must remain useful without becoming a full-flight pass."""
import json

import numpy as np
import pytest

from omni_drones.navigation.forest.trace_journal import TraceJournal, read_trace_journal


def test_clean_journal_matches_measured_rows_and_stays_partial_evidence(tmp_path):
    expected = np.arange(84, dtype=float).reshape(4, 21)
    with TraceJournal(tmp_path) as journal:
        for row in expected:
            journal.append(row)
    actual, metadata = read_trace_journal(tmp_path)
    np.testing.assert_array_equal(actual, expected)
    assert metadata['closed_cleanly'] and metadata['rows_written'] == 4
    assert metadata['trajectory_complete'] is False


def test_restart_recovers_full_records_and_ignores_an_unfinished_record(tmp_path):
    journal = TraceJournal(tmp_path)
    expected = np.arange(42, dtype=float).reshape(2, 21)
    for row in expected:
        journal.append(row)
    journal.stream.close()  # Simulate interruption before final metadata.
    with (tmp_path/'trajectory.partial.f64').open('ab') as stream:
        stream.write(b'partial')
    actual, metadata = read_trace_journal(tmp_path)
    np.testing.assert_array_equal(actual, expected)
    assert metadata['closed_cleanly'] is False
    assert metadata['rows_written'] is None and metadata['trailing_bytes'] == 7
    assert metadata['trajectory_complete'] is False


def test_corrupt_or_nonmonotonic_journal_cannot_supply_coverage_evidence(tmp_path):
    with TraceJournal(tmp_path) as journal:
        journal.append(np.zeros(21))
        with pytest.raises(ValueError, match='finite'):
            journal.append(np.full(21, np.nan))
        journal.append(np.zeros(21))
    with pytest.raises(ValueError, match='invalid measured'):
        read_trace_journal(tmp_path)
    metadata = json.loads((tmp_path/'trajectory_journal.json').read_text())
    metadata['columns'] = 22
    (tmp_path/'trajectory_journal.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='schema'):
        read_trace_journal(tmp_path)
