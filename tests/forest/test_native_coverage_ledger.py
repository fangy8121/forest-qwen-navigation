import numpy as np

from omni_drones.navigation.forest.coverage import CoverageLedger, completion_evidence


def test_within_corridor_detour_counts_without_crediting_the_next_aisle():
    corridors = [dict(id=str(i), direction=[1., 0.], normal=[0., 1.],
        offset_m=5.*i, span=[0., 10.], inspection_half_width_m=1.75) for i in range(2)]
    ledger = CoverageLedger()
    for index, x in enumerate(np.arange(0., 10.01, .05)):
        ledger.observe(index*.1, [x, 1.3*np.sin(np.pi*x/10.), 2.5], corridors)
    assert ledger.gaps(corridors[0]) == []
    assert ledger.gaps(corridors[1]) == [[0., 10.]]


def test_motion_in_the_tree_row_buffer_cannot_earn_corridor_coverage():
    corridor = dict(id='c', direction=[1., 0.], normal=[0., 1.],
        offset_m=0., span=[0., 10.], inspection_half_width_m=1.75)
    ledger = CoverageLedger()
    for index, x in enumerate(np.arange(0., 10.01, .05)):
        ledger.observe(index*.1, [x, 2.2, 2.5], [corridor])
    assert ledger.gaps(corridor) == [[0., 10.]]


def test_missing_row_frontier_prevents_completion_after_known_corridors_are_covered():
    corridor = dict(id='c', direction=np.array([1., 0.]), normal=np.array([0., 1.]),
        offset_m=0., span=[0., 10.],
        ends={side:dict(confirmed=True) for side in ('low', 'high')})
    ledger = CoverageLedger(covered={'c':[[0., 10.]]})
    structure = dict(corridors=[corridor],
        outer_sides={side:dict(confirmed=True) for side in ('low', 'high')},
        unresolved_row_gaps=[dict(bordering_rows=['R01', 'R03'])])
    assert completion_evidence(structure, ledger)['unresolved'] == ['UNRESOLVED_ROW_GAPS']
    structure['unresolved_row_gaps'] = []
    assert completion_evidence(structure, ledger)['eligible']
