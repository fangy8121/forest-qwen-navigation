"""Bounded, asynchronous tree-row evidence from raw onboard scans only.

This first implementation assumes a finite approximately parallel plantation.
Boundary tests are conditional on explicit gap bounds, visibility and successful
stem extraction. It is not a completeness proof for an arbitrary natural forest.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from .geometry import quat_matrix
from .observed_terrain import ObservedTerrain, query_terrain
from .row_map import RowMap


def _public_work_area(bounds):
    if bounds is None:
        return None
    b = np.asarray(bounds, float)
    if b.shape != (4,) or not np.isfinite(b).all() or b[0] >= b[1] or b[2] >= b[3]:
        raise ValueError('work_area_bounds must be finite ordered [xmin,xmax,ymin,ymax]')
    return dict(xy_bounds=b.tolist(), coverage_scope='BOUND_PUBLIC_WORK_AREA',
                source='operator supplied XY task rectangle')


# Global decision image raster over the work area (+margin): bit 1 = free
# evidence near the flight height, bit 2 = height-band returns (trunks, branches).
OVERVIEW_RESOLUTION, OVERVIEW_MARGIN, OVERVIEW_FREE, OVERVIEW_BAND = .4, 4., 1, 2


# Rows are at least 2.5 m apart, so parallel tracks closer than the 1.2 m stem
# association band are one row split by the tracker (run 8e03d728: a sparse row
# became R06 and R07 0.2 m apart; the structure stayed ambiguous at rest).
ROW_MERGE_DISTANCE = 1.2

# A previously observed aisle that vanishes keeps the structure ambiguous (the
# mission holds) for this long; if it is still gone, it was a refit, not a
# flicker, and is forgotten. Without a bound one early refit held the vehicle
# until observation_no_progress (val/000 run ef65ee46: an aisle seen 6.5-9 s;
# train/016 run 62fbc380: R06_R05 vanished at 137.5 s).
MISSING_CORRIDOR_SETTLE_SECONDS = 10.


def _merge_split_rows(rows):
    """Merge offset-sorted family rows closer than ROW_MERGE_DISTANCE.

    The older track id (lower number) is kept so corridor ids stay stable when
    the newer track later gains more support. Returns rows and [kept, merged].
    """
    result, merged = [], []
    for row in rows:
        if not result or row['offset_m']-result[-1]['offset_m'] >= ROW_MERGE_DISTANCE:
            result.append(row)
            continue
        keep, drop = sorted((result[-1], row), key=lambda r:(len(r['id']), r['id']))
        combined = deepcopy(keep)
        combined['spans'] = list(keep['spans'])+list(drop['spans'])
        combined['support'] = keep.get('support', 1)+drop.get('support', 1)
        for key in ('support_xy', 'extent_support_xy'):
            if key in keep or key in drop:
                combined[key] = list(keep.get(key, []))+list(drop.get(key, []))
        if 'last_observed_s' in keep or 'last_observed_s' in drop:
            combined['last_observed_s'] = max(keep.get('last_observed_s', -np.inf),
                                              drop.get('last_observed_s', -np.inf))
        support = (np.asarray(combined['support_xy'], float).reshape(-1, 2) if combined.get('support_xy')
                   else np.asarray(combined['spans'], float).reshape(-1, 2))
        combined['offset_m'] = float(np.median(support@np.asarray(combined['normal'])))
        result[-1] = combined
        merged.append([keep['id'], drop['id']])
    return sorted(result, key=lambda r:r['offset_m']), merged


class _Accumulator:
    def __init__(self, output, *, row_gap_bound=7., stem_gap_bound=5.5,
                 update_seconds=1., max_voxels=180000, max_free_cells=1500000,
                 work_area_bounds=None, terrain_truth=None):
        # Same ground source as the mission's AGL anchors (sim stand-in for the
        # airframe's own AGL). Row-end probes otherwise wait for LiDAR ground
        # that an upright MID-360 rarely sees nearby (run f830a311).
        self.terrain_truth = terrain_truth
        self.public_work_area = _public_work_area(work_area_bounds)
        self.work_area_bounds = None if self.public_work_area is None else np.asarray(work_area_bounds, float)
        self.output = Path(output)
        self.folder = self.output/'structure_scans'
        self.folder.mkdir()
        self.row_map = RowMap(update_seconds=update_seconds, max_voxels=max_voxels, row_gap_bound=row_gap_bound)
        self.terrain = ObservedTerrain(update_seconds=update_seconds)
        self.row_gap_bound, self.stem_gap_bound = row_gap_bound, stem_gap_bound
        self.max_voxels, self.max_free_cells = max_voxels, max_free_cells
        self.frames = 0
        self.free = set()
        self.views = {}
        self.last_error = None
        self.corridor_ids_seen = set()
        self.corridor_missing_since = {}
        self.row_frames = {}
        self.family, self.family_resets = None, 0
        self.overview = None
        if self.work_area_bounds is not None:
            xmin, xmax, ymin, ymax = self.work_area_bounds
            # Boundaries half a cell off the 0.4 m free-evidence lattice: lattice
            # points never sit on a boundary where floor() would skip cells.
            self.overview_origin = (np.floor((np.array([xmin, ymin])-OVERVIEW_MARGIN)/OVERVIEW_RESOLUTION)-.5)*OVERVIEW_RESOLUTION
            shape = np.ceil((np.array([xmax-xmin, ymax-ymin])+2*OVERVIEW_MARGIN)/OVERVIEW_RESOLUTION).astype(int)+1
            self.overview = np.zeros(shape, np.uint8)

    def _mark_overview(self, xy, flag):
        if self.overview is None or not len(xy):
            return
        cells = np.floor((np.asarray(xy, float)[:, :2]-self.overview_origin)/OVERVIEW_RESOLUTION).astype(int)
        cells = cells[((cells >= 0) & (cells < self.overview.shape)).all(axis=1)]
        self.overview[cells[:, 0], cells[:, 1]] |= flag

    def _in_work_area(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.work_area_bounds is None:
            return np.ones(len(xy), bool)
        xmin, xmax, ymin, ymax = self.work_area_bounds
        return ((xy[:, 0] >= xmin) & (xy[:, 0] <= xmax)
                & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax))

    def _add(self, frame):
        began = time.perf_counter()
        self.row_map.add(frame)
        if len(self.row_map.voxels) > self.max_voxels:
            raise RuntimeError('STRUCTURE_STORAGE_BUDGET: stop or submap; never silently discard coverage evidence')
        self.frames += 1
        scan_id = f'scan_{self.frames:05d}'
        np.savez_compressed(self.folder/(scan_id+'.npz'), **frame)
        hits = np.asarray(frame['points_sensor'])[frame['valid']]
        hits = hits@quat_matrix(frame['sensor_quaternion']).T+frame['sensor_position']
        self.terrain.add(hits, float(frame['timestamp']), frame['position'])
        terrain = self.terrain.snapshot()
        # Finite measured returns only: free samples stop BEFORE the return.
        # Invalid/no-return rays are not converted to range-limit free evidence.
        valid = np.flatnonzero(frame['valid'])
        ids = valid[::max(1, len(valid)//1800)]
        lengths = np.asarray(frame['distances'])[ids]-.2
        rays = np.asarray(frame['directions_world'])[ids]
        origin = np.asarray(frame['sensor_position'])
        distances = np.arange(.4, float(frame['max_range']), .4)
        p = origin+rays[:, None, :]*distances[None, :, None]
        p = p[distances[None, :] < lengths[:, None]]
        cells = np.unique(np.rint(p/.4).astype(np.int32), axis=0)
        self.free.update(map(tuple, cells.tolist()))
        self._mark_overview(cells[np.abs(cells[:, 2]*.4-float(frame['position'][2])) <= 1.5, :2]*.4,
                            OVERVIEW_FREE)
        if len(self.free) > self.max_free_cells:
            raise RuntimeError('FREE_EVIDENCE_STORAGE_BUDGET')
        view_key = tuple(np.floor(np.asarray(frame['position'])/.75).astype(int))
        self.views[view_key] = dict(position=np.asarray(frame['position']).tolist(),
            timestamp=float(frame['timestamp']), scan_id=scan_id)
        if len(self.views) > 20000:
            raise RuntimeError('STRUCTURE_VIEW_STORAGE_BUDGET')
        structure, _ = self.row_map.snapshot()
        # RowMap uses its first body position as origin, not world coordinates.
        shift = self.row_map.origin[:2]
        band = np.asarray(structure.get('height_band_xy', []), float).reshape(-1, 2)+shift
        self._mark_overview(band, OVERVIEW_BAND)
        structure_xy = np.unique(np.rint(band/.4).astype(np.int32), axis=0)*.4
        if len(structure_xy) > 3000:
            structure_xy = structure_xy[np.linspace(0, len(structure_xy)-1, 3000, dtype=int)]
        stems = [dict(stem, xy=(np.asarray(stem['xy'])+shift).tolist())
                 for stem in structure.get('stems', [])]
        outside_stems = [stem for stem in stems if not self._in_work_area([stem['xy']])[0]]
        rows = []
        frame_ambiguous = False

        def family_direction(candidates):
            # Seeded by the best-supported row; only rows within 8 deg vote, so
            # one off-family track cannot tilt the true rows out of the family.
            if not candidates:
                return np.array([1., 0.])
            seed = np.asarray(max(candidates, key=lambda r:r.get('support', 1))['direction'], float)
            total = np.sum([np.asarray(r['direction'])*r.get('support', 1)*np.sign(seed@r['direction'] or 1.)
                            for r in candidates
                            if abs(seed@np.asarray(r['direction'])) >= np.cos(np.deg2rad(8))], axis=0)
            return total/np.linalg.norm(total)

        def consistent(row, d):
            # A stale track through scattered stems (run 5312eadd: 4 points over
            # 15 m diagonally across aisle 1) is not a plantation row; it tilted
            # the shared frame and made the whole structure ambiguous forever.
            # Spread of the central 90%: track support only grows, and one stem
            # 0.6 m outside a 114-stem row's 1.9 m span dropped the row and both
            # its aisles for the rest of the run (run c1c3147f, 1.15 m central).
            n = np.array([-d[1], d[0]])
            support = (np.asarray(row['support_xy'], float).reshape(-1, 2)+shift if row.get('support_xy')
                       else np.asarray(row['spans'], float).reshape(-1, 2)+shift)
            return (abs(d@np.asarray(row['direction'])) >= np.cos(np.deg2rad(8))
                    and np.subtract(*np.percentile(support@n, [95, 5])) <= 2.5)

        raw_rows = list(structure['rows'])
        family = family_direction(raw_rows)
        rejected_rows = [dict(id=r['id'], support=r.get('support'), reason='NOT_PARALLEL_ROW_FAMILY')
                         for r in raw_rows if not consistent(r, family)]
        if len(rejected_rows) < len(raw_rows):
            raw_rows = [r for r in raw_rows if consistent(r, family)]
        else:
            rejected_rows = []
        common_direction = family_direction(raw_rows)
        family_reset = False
        if raw_rows and self.family is not None and abs(self.family@common_direction) < np.cos(np.deg2rad(45)):
            # One plantation has one row family. A flip means the first fit took
            # tree columns for rows (run 5d434c21: y-direction "rows" at 8 s, the
            # true family at 30 s); identities of the wrong family cannot be missed.
            self.corridor_ids_seen, self.row_frames, family_reset = set(), {}, True
            self.corridor_missing_since = {}
            self.family_resets += 1
        if raw_rows:
            self.family = common_direction
        for raw in raw_rows:
            row = deepcopy(raw)
            row['spans'] = (np.asarray(row['spans'])+shift).tolist()
            d = common_direction
            previous = np.asarray(self.row_frames.get(row['id'], d))
            if abs(previous@d) < np.cos(np.deg2rad(8)) or abs(d@row['direction']) < np.cos(np.deg2rad(8)):
                frame_ambiguous = True
            self.row_frames[row['id']] = d.tolist()
            n = np.array([-d[1], d[0]])
            support = (np.asarray(row['support_xy'])+shift if row.get('support_xy')
                       else np.asarray(row['spans']).reshape(-1, 2))
            if 'support_xy' in row:
                row['support_xy'] = support.tolist()
            if 'extent_support_xy' in row:
                row['extent_support_xy'] = (np.asarray(row['extent_support_xy'])+shift).tolist()
            row.update(direction=d.tolist(), normal=n.tolist(),
                offset_m=float(np.median(support@n)))
            rows.append(row)
        # Tree queries depend on the cell set, not its ordering. Sorting up to
        # max_free_cells Python tuples at every scan adds no evidence.
        free_tree = cKDTree(np.asarray(list(self.free), float).reshape(-1, 3)*.4)
        corridors, unresolved_gaps = [], []
        rows.sort(key=lambda r:r['offset_m'])
        rows, merged_rows = _merge_split_rows(rows)
        ambiguous = frame_ambiguous or any(abs(np.dot(r['direction'], rows[0]['direction'])) < .98 for r in rows)
        for a, b in zip(rows[:-1], rows[1:]):
            gap = b['offset_m']-a['offset_m']
            if gap < 2.5:
                ambiguous = True
                continue
            d, n = np.asarray(a['direction']), np.asarray(a['normal'])
            aa, bb = np.asarray(a['spans']).reshape(-1, 2)@d, np.asarray(b['spans']).reshape(-1, 2)@d
            if gap > self.row_gap_bound:
                along = float(np.clip(np.asarray(frame['position'])[:2]@d,
                    min(aa.min(), bb.min()), max(aa.max(), bb.max())))
                unresolved_gaps.append(dict(id='gap_'+a['id']+'_'+b['id'],
                    bordering_rows=[a['id'], b['id']], direction=d.tolist(), normal=n.tolist(),
                    lateral_span=[a['offset_m'], b['offset_m']], observed_gap_m=float(gap),
                    span=[float(min(aa.min(), bb.min())), float(max(aa.max(), bb.max()))],
                    observation_target_xy=(d*along+n*(a['offset_m']+b['offset_m'])/2).tolist(),
                    status='UNRESOLVED_MISSING_ROW_OR_OPEN_SPACE', last_scan_id=scan_id,
                    target_semantics='observation intention; not a corridor or traversability claim'))
                continue
            bilateral = [float(max(aa.min(), bb.min())), float(min(aa.max(), bb.max()))]
            if bilateral[1]-bilateral[0] < 4.:
                continue
            low, high = min(aa.min(), bb.min()), max(aa.max(), bb.max())
            offset = (a['offset_m']+b['offset_m'])/2
            c = dict(id=a['id']+'_'+b['id'], bordering_rows=[a['id'], b['id']],
                direction=d.tolist(), normal=n.tolist(), offset_m=offset,
                inspection_half_width_m=max(.1, (b['offset_m']-a['offset_m'])/2-.75),
                span=[float(low), float(high)], bilateral_span=bilateral,
                support_semantics='union of observed bordering rows, including single-sided tails', ends={})
            for name, side in [('low', -1), ('high', 1)]:
                edge = min(aa.min(), bb.min()) if side < 0 else max(aa.max(), bb.max())
                center = d*edge+n*offset
                # Probe beyond BOTH rows; not merely beyond the shorter row.
                probes = np.array([d*(edge+side*s)+n*(off+w)
                    for off in (a['offset_m'], b['offset_m'])
                    for s in np.arange(1., self.stem_gap_bound+.01, .8) for w in (-.4, 0., .4)])
                counterexamples = [stem for stem in stems
                    if .25 < side*(np.asarray(stem['xy'])@d-edge) <= self.stem_gap_bound+.65
                    and min(abs(np.asarray(stem['xy'])@n-off) for off in (a['offset_m'], b['offset_m'])) <= 1.2]
                c['ends'][name] = self._evidence(probes, free_tree, frame, center, scan_id, terrain, counterexamples)
                c['ends'][name].update(s=float(edge), observation_target_xy=(center+side*d*(self.stem_gap_bound/2)).tolist())
            corridors.append(c)
        outer = {}
        for name, side, row in [('low', -1, rows[0] if rows else None),
                                ('high', 1, rows[-1] if rows else None)]:
            if row is None:
                outer[name] = dict(confirmed=False)
                continue
            d, n = np.asarray(row['direction']), np.asarray(row['normal'])
            s = np.asarray(row['spans']).reshape(-1, 2)@d
            probes = np.array([d*x+n*(row['offset_m']+side*y)
                for x in np.linspace(s.min(), s.max(), max(3, int(np.ptp(s)/2)+1))
                for y in np.arange(1., self.row_gap_bound+.01, 1.)])
            counterexamples = [stem for stem in stems
                if 1.2 < side*(np.asarray(stem['xy'])@n-row['offset_m']) <= self.row_gap_bound+.65
                and s.min()-.5 <= np.asarray(stem['xy'])@d <= s.max()+.5]
            outer[name] = self._evidence(probes, free_tree, frame,
                d*np.mean(s)+n*row['offset_m'], scan_id, terrain, counterexamples)
        present = {c['id'] for c in corridors}
        now = float(frame['timestamp'])
        for cid in present:
            self.corridor_missing_since.pop(cid, None)
        forgotten = []
        for cid in sorted(self.corridor_ids_seen-present):
            since = self.corridor_missing_since.setdefault(cid, now)
            if now-since >= MISSING_CORRIDOR_SETTLE_SECONDS:
                forgotten.append(cid)
                self.corridor_ids_seen.discard(cid)
                del self.corridor_missing_since[cid]
        missing = sorted(self.corridor_ids_seen-present)
        self.corridor_ids_seen.update(present)
        ambiguous = ambiguous or bool(missing)
        value = dict(timestamp=float(frame['timestamp']), rows=rows, stem_candidates=stems,
            public_work_area=self.public_work_area,
            coverage_scope='UNKNOWN_STAND_BOUNDARY' if self.public_work_area is None else 'BOUND_PUBLIC_WORK_AREA',
            outside_public_work_area_stem_candidates=outside_stems,
            structure_points_xy=tuple(map(tuple, structure_xy.tolist())),
            corridors=corridors, unresolved_row_gaps=unresolved_gaps, terrain=terrain,
            rejected_rows=rejected_rows, merged_rows=merged_rows,
            family_reset=family_reset, family_resets=self.family_resets,
            overview=None if self.overview is None else dict(origin=self.overview_origin.tolist(),
                resolution=OVERVIEW_RESOLUTION, flags=self.overview.copy()),
            worker_pid=os.getpid(),
            outer_sides=outer, identity_ambiguous=ambiguous, scans=self.frames,
            missing_previously_observed_corridor_ids=missing,
            forgotten_corridor_ids=forgotten,
            free_evidence_cells=len(self.free), structure_voxels=len(self.row_map.voxels),
            compute_seconds=time.perf_counter()-began,
            assumptions=dict(parallel_rows=True, max_row_spacing_m=self.row_gap_bound,
                min_row_spacing_m=2.5, median_in_row_spacing_less_than_row_spacing=True,
                max_in_row_gap_m=self.stem_gap_bound,
                row_extent_association_m=1.2, positive_vertical_stems_veto_absence=True,
                connected_finite_stand=True, successful_stem_detection_required=True,
                boundary_rule='sampled absence evidence, conditional on detection and gap assumptions'))
        # Raw scans preserve the reconstruction evidence. Repeating the entire
        # persistent terrain in every JSON event would grow quadratically.
        event = dict(value, terrain={k:v for k, v in terrain.items()
                                     if k not in ('vertices', 'timestamps') and not k.startswith('_')})
        event.pop('structure_points_xy')
        event.pop('overview')
        event['structure_points_xy_count'] = len(structure_xy)
        event['terrain']['vertex_count'] = len(terrain['vertices'])
        with (self.output/'structure_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        return value

    def _evidence(self, xy, tree, frame, center, scan_id, terrain, counterexamples=()):
        # Visibility at two stem-height bands is evaluated against accumulated
        # finite-return rays. This is a sampled detector, not a solid-free proof.
        original = np.asarray(xy, float).reshape(-1, 2)
        in_area = self._in_work_area(original)
        xy = original[in_area]
        required_counterexamples = [stem for stem in counterexamples if self._in_work_area([stem['xy']])[0]]
        outside_counterexamples = [stem for stem in counterexamples if not self._in_work_area([stem['xy']])[0]]
        if self.terrain_truth is not None:
            from forest_generator.terrain import grid_height
            t = self.terrain_truth
            ground, supported = grid_height(xy, t['xs'], t['ys'], t['heights']), np.ones(len(xy), bool)
        else:
            ground, supported = query_terrain(terrain, xy)
        points = np.concatenate([np.c_[xy, ground+h] for h in (.8, 1.6)])
        points[~np.isfinite(points)] = 1e12
        distances = tree.query(points)[0]
        fraction = float(np.mean((distances <= .65) & np.tile(supported, 2))) if len(points) else 0.
        nearby = [v for v in self.views.values() if np.linalg.norm(np.asarray(v['position'])[:2]-center) < 12.]
        diverse = bool(nearby and max(np.linalg.norm(np.asarray(v['position'])-nearby[0]['position']) for v in nearby) >= 1.)
        clipped = bool(np.any(~in_area))
        return dict(confirmed=bool(len(xy) and fraction >= .85 and diverse and not required_counterexamples),
            observed_fraction=fraction,
            terrain_supported_fraction=float(supported.mean()) if len(supported) else 0.,
            positive_stem_counterexamples=[dict(stem, reported_at_scan_id=scan_id) for stem in required_counterexamples],
            outside_public_work_area_stem_candidates=outside_counterexamples,
            outside_public_work_area=(None if self.public_work_area is None else dict(
                classification='OUTSIDE_PUBLIC_WORK_AREA', count=int(np.sum(~in_area)),
                xy=original[~in_area].tolist(), free_space_credit=0)),
            public_work_area=self.public_work_area,
            proof_kind=('OBSERVED_ABSENCE_TO_PUBLIC_WORK_AREA_BOUNDARY' if clipped
                        else 'FINITE_RAY_ABSENCE_UNDER_GAP_ASSUMPTIONS'),
            required_probe_count=len(xy), original_probe_count=len(original),
            unresolved_reason='NO_IN_DOMAIN_PROBES' if not len(xy) else None,
            independent_position_baseline_m=1., diverse_views=diverse,
            last_scan_id=scan_id, samples=len(points), rule='finite-return-ray-sampled absence; gap/detection assumptions apply')


_WORKER_ACCUMULATOR = None


def _initialize_structure(output, parameters):
    global _WORKER_ACCUMULATOR
    _WORKER_ACCUMULATOR = _Accumulator(output, **parameters)


def _accumulate_frame(frame):
    try:
        return _WORKER_ACCUMULATOR._add(frame)
    except Exception as error:
        import traceback
        (_WORKER_ACCUMULATOR.output/'structure_error.json').write_text(json.dumps(dict(
            timestamp=float(frame['timestamp']), worker_pid=os.getpid(),
            error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()), indent=2))
        raise


class OnlineStructure:
    """One bounded process: structural Python work cannot hold the control GIL.

    ``inline`` is explicitly for offline logic/replay diagnostics only. It is
    never selected by the physical integration entry point.
    """
    def __init__(self, output, *, execution='process', **parameters):
        self.output = Path(output).resolve()
        self.public_work_area = _public_work_area(parameters.get('work_area_bounds'))
        if parameters.get('terrain_truth') is not None:
            parameters['terrain_truth'] = {k:np.asarray(v, float) for k, v in parameters['terrain_truth'].items()}
        self.execution = execution
        self.future, self.last_submitted, self.latest = None, -np.inf, None
        self.last_error, self.dropped = None, 0
        if execution == 'inline':
            self._inline = _Accumulator(self.output, **parameters)
            self.pool = None
        elif execution == 'process':
            self._inline = None
            self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_structure, initargs=(str(self.output), parameters))
        else:
            raise ValueError('structure execution must be process or inline')

    @property
    def row_map(self):
        return self._inline.row_map

    @property
    def free(self):
        return self._inline.free

    def _add(self, frame):
        if self._inline is None:
            raise RuntimeError('synchronous structure updates are only allowed in offline inline mode')
        return self._inline._add(frame)

    def submit(self, frame):
        if self.pool is None:
            raise RuntimeError('inline mode is for explicit offline _add calls')
        if self.future is not None:
            if not self.future.done():
                self.dropped += 1
                return
            try:
                self.latest = self.future.result()
            except Exception as error:
                self.last_error = f'{type(error).__name__}: {error}'
            self.future = None
        now = float(frame['timestamp'])
        if now-self.last_submitted < .45 or self.last_error:
            return
        self.last_submitted = now
        self.future = self.pool.submit(_accumulate_frame, frame)

    def snapshot(self):
        if self.latest is None:
            return dict(corridors=[], rows=[], outer_sides={}, public_work_area=deepcopy(self.public_work_area),
                coverage_scope='UNKNOWN_STAND_BOUNDARY' if self.public_work_area is None else 'BOUND_PUBLIC_WORK_AREA')
        value = deepcopy({k:v for k, v in self.latest.items()
                          if k not in ('terrain', 'structure_points_xy', 'overview')})
        # Terrain vertices/times are immutable tuples, shared safely between
        # readers; avoid copying thousands of vertices at control frequency.
        if 'terrain' in self.latest:
            value['terrain'] = dict(self.latest['terrain'])
        if 'structure_points_xy' in self.latest:
            value['structure_points_xy'] = self.latest['structure_points_xy']
        # The overview raster is replaced, never mutated, by each worker update.
        if 'overview' in self.latest:
            value['overview'] = self.latest['overview']
        return value

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        if self.future is not None and self.future.done():
            try:
                self.latest = self.future.result()
            except Exception as error:
                self.last_error = str(error)
