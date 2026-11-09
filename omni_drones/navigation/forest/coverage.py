"""Task evidence independent of the model, simulator and hidden forest map."""
from dataclasses import dataclass, field
import numpy as np


def union(intervals, tolerance=.05):
    merged = []
    for lo, hi in sorted((min(a, b), max(a, b)) for a, b in intervals):
        if hi-lo < 1e-8:
            continue
        if merged and lo <= merged[-1][1]+tolerance:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([float(lo), float(hi)])
    return merged


def subtract(required, covered, tolerance=.15):
    result = []
    for low, high in union(required):
        cursor = low
        for a, b in union(covered):
            if b <= cursor or a >= high:
                continue
            if a-cursor > tolerance:
                result.append([cursor, min(a, high)])
            cursor = max(cursor, b)
        if high-cursor > tolerance:
            result.append([cursor, high])
    return result


@dataclass
class CoverageLedger:
    """Intervals are earned by measured motion in the identified corridor.

    A goal, planned spline, transit in an adjacent corridor or verbal 'done'
    cannot add coverage. Required intervals may grow after later observations.
    """
    covered: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    previous: tuple | None = None
    directions: dict = field(default_factory=dict)

    def required(self, corridor):
        d = np.asarray(self.directions.setdefault(corridor['id'], corridor['direction']), float)
        points = np.asarray(corridor['span'])[:, None]*corridor['direction']+np.asarray(corridor['normal'])*corridor['offset_m']
        along = points@d
        return [float(along.min()), float(along.max())]

    def observe(self, now, position, corridors, *, enabled=True, sensor_valid=True,
                lateral_tolerance=.8, max_sample_gap=.5):
        p = np.asarray(position, float)
        match = []
        if enabled and sensor_valid:
            for c in corridors:
                d, n = np.asarray(c['direction']), np.asarray(c['normal'])
                if abs(p[:2]@n-c['offset_m']) <= c.get('inspection_half_width_m', lateral_tolerance):
                    fixed_d = np.asarray(self.directions.setdefault(c['id'], c['direction']))
                    s = float(p[:2]@fixed_d)
                    lo, hi = self.required(c)
                    if lo-.5 <= s <= hi+.5:
                        match.append((c['id'], s))
        current = (float(now), p.copy(), match[0]) if len(match) == 1 else None
        if current and self.previous:
            t0, p0, (identifier, s0) = self.previous
            if (current[2][0] == identifier and 0 < now-t0 <= max_sample_gap
                    and np.linalg.norm(p-p0) <= 1.2*(now-t0)+.02):
                self.covered[identifier] = union(self.covered.get(identifier, [])+[[s0, current[2][1]]])
        self.previous = current

    def gaps(self, corridor):
        lo, hi = self.required(corridor)
        gaps = subtract([[lo, hi]], self.covered.get(corridor['id'], []), tolerance=.03)
        # Arrival tolerance applies only at the two outer endpoints. Many small
        # interior sensor outages must not silently disappear from coverage.
        return [[a, b] for a, b in gaps if not (b-a <= .25 and
            (abs(a-lo) < 1e-6 or abs(b-hi) < 1e-6))]

    def snapshot(self, corridors):
        rows = []
        for c in corridors:
            gaps = self.gaps(c)
            rows.append(dict(corridor_id=c['id'], required=[self.required(c)],
                along_direction=self.directions[c['id']],
                covered=self.covered.get(c['id'], []), uncovered=gaps,
                uncovered_length_m=sum(b-a for a, b in gaps)))
        return dict(schema='observed_corridor_coverage_v1', corridors=rows,
            denominator='discovered supported corridor spans; not hidden full-forest area',
            total_uncovered_length_m=sum(r['uncovered_length_m'] for r in rows))


def completion_evidence(structure, ledger):
    corridors = structure.get('corridors', [])
    unresolved = []
    if not corridors:
        unresolved.append('NO_CORRIDORS')
    for c in corridors:
        if ledger.gaps(c):
            unresolved.append(c['id']+':UNCOVERED')
        for end in ('low', 'high'):
            if not c.get('ends', {}).get(end, {}).get('confirmed'):
                unresolved.append(c['id']+':'+end+':END_UNCONFIRMED')
    for side in ('low', 'high'):
        if not structure.get('outer_sides', {}).get(side, {}).get('confirmed'):
            unresolved.append('OUTER_'+side.upper()+'_UNCONFIRMED')
    if structure.get('identity_ambiguous'):
        unresolved.append('IDENTITY_AMBIGUOUS')
    if structure.get('unresolved_row_gaps'):
        unresolved.append('UNRESOLVED_ROW_GAPS')
    return dict(eligible=not unresolved, unresolved=unresolved,
        status='COMPLETE_UNDER_ASSUMPTIONS' if not unresolved else 'UNRESOLVED',
        coverage_scope=(structure.get('public_work_area') or {}).get('coverage_scope', 'UNKNOWN_BOUNDARY'),
        assumptions=structure.get('assumptions', {}))
