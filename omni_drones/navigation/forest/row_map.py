"""Causal, measurement-only tree-row abstraction; never a collision map.

Ground is a locally supported lower surface, stems require vertical persistence,
and rows are finite observed spans. No map manifest or future scan is accepted.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree
from .geometry import quat_matrix

HEIGHT_BAND = (.65, 2.65)


def components(points, radius):
    points = np.asarray(points)
    if not len(points): return []
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    parent = np.arange(len(points))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in pairs:
        a, b = root(a), root(b)
        if a != b: parent[b] = a
    groups = {}
    for i in range(len(points)): groups.setdefault(root(i), []).append(i)
    return [np.asarray(v) for v in groups.values()]


def extract_stems(points):
    """Return compact candidates supported in >=3 vertical bands above ground."""
    if len(points) < 30: return [], dict(voxels=len(points), ground_cells=0)
    cell, inverse = np.unique(np.floor(points[:, :2]).astype(int), axis=0, return_inverse=True)
    low = np.full(len(cell), np.inf); np.minimum.at(low, inverse, points[:, 2])
    centers = cell+.5
    distances, neighbors = cKDTree(centers).query(centers, k=min(9, len(cell)))
    # Isolated elevated cells are not accepted as ground. Use neighboring lower
    # surface estimates, with support limited to 2.5 m (no terrain extrapolation).
    values = np.where(distances <= 2.5, low[neighbors], np.nan)
    surface = np.nanpercentile(values, 25, axis=1)
    supported = (np.sum(distances <= 2.5, axis=1) >= 5) & (low <= surface+.45)
    if np.count_nonzero(supported) < 5:
        return [], dict(voxels=len(points), ground_cells=int(supported.sum()))
    ground_xy, ground_z = centers[supported], low[supported]
    distance, neighbor = cKDTree(ground_xy).query(points[:, :2], k=4)
    weight = 1/np.maximum(distance, .2)**2
    ground = np.sum(weight*ground_z[neighbor], axis=1)/weight.sum(axis=1)
    height = points[:, 2]-ground
    mask = (distance[:, 3] <= 2.8) & (height >= HEIGHT_BAND[0]) & (height <= HEIGHT_BAND[1])
    selected, h = points[mask], height[mask]
    if not len(selected): return [], dict(voxels=len(points), ground_cells=int(supported.sum()))
    # Persistent XY columns suppress ground and horizontal branch fragments.
    xy, inv = np.unique(np.rint(selected[:, :2]/.2).astype(int), axis=0, return_inverse=True)
    bits = np.zeros(len(xy), np.uint8)
    band = np.minimum(3, ((h-.65)/.5).astype(int))
    np.bitwise_or.at(bits, inv, (1 << band).astype(np.uint8))
    population = np.array([int(i).bit_count() for i in range(16)])
    columns = xy[population[bits] >= 3]*.2
    stems = []
    for ids in components(columns, .43):
        xy_group = columns[ids]
        if len(ids) < 2 or np.linalg.norm(np.ptp(xy_group, axis=0)) > 1.6: continue
        center = (xy_group.min(axis=0)+xy_group.max(axis=0))/2
        radius = max(.12, float(np.linalg.norm(xy_group-center, axis=1).max()))
        stems.append(dict(xy=center.tolist(), radius_m=radius,
                          support_columns=len(ids), confidence='candidate'))
    return stems, dict(voxels=len(points), ground_cells=int(supported.sum()),
                      height_band_xy=(np.unique(np.rint(selected[:, :2]/.2).astype(int), axis=0)*.2).tolist(),
                      height_filtered_voxels=len(selected), persistent_columns=len(columns))


def fit_rows(stems, *, min_row_spacing=2.5, max_row_spacing=7.):
    if len(stems) < 8: return []
    xy = np.array([s['xy'] for s in stems])
    # Like catkin row_search, search a common orientation using transverse
    # concentration. Search a full half-circle rather than assuming world X.
    best = None
    for angle in np.deg2rad(np.arange(0., 180., 1.)):
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        lateral = xy @ normal
        order = np.argsort(lateral)
        groups = np.split(order, np.flatnonzero(np.diff(lateral[order]) > .65)+1)
        # Shorter row hypotheses help disambiguate the plantation axis even
        # before enough of each row is visible to promote its permanent ID.
        valid = [g for g in groups if len(g) >= 3 and np.ptp(xy[g]@direction) >= 4
                 and np.ptp(lateral[g]) <= 2.5]
        # One accidental diagonal through stems from several rows is not a
        # plantation direction. Require independently supported parallel rows.
        if len(valid) < 2:
            continue
        offsets = np.array([np.median(lateral[g]) for g in valid])
        stem_spacing = []
        for group in valid:
            along_gaps = np.diff(np.sort(xy[group]@direction))
            along_gaps = along_gaps[along_gaps >= 1.]  # Suppress split stem detections.
            stem_spacing.append(float(np.median(along_gaps)) if len(along_gaps) else np.inf)
        # Select the best supported compatible family. A weak short off-row
        # cluster must not disqualify both neighboring, strongly supported rows.
        weights = [len(g)*(1-float(np.std(lateral[g]))/max_row_spacing) for g in valid]
        scores, paths = list(weights), [[i] for i in range(len(valid))]
        for j in range(len(valid)):
            for i in range(j):
                gap = offsets[j]-offsets[i]
                if (min_row_spacing <= gap <= max_row_spacing
                        and max(stem_spacing[i], stem_spacing[j]) < gap
                        and scores[i]+weights[j] > scores[j]):
                    scores[j], paths[j] = scores[i]+weights[j], paths[i]+[j]
        families = [i for i, path in enumerate(paths) if len(path) >= 2]
        if not families:
            continue
        family = paths[max(families, key=lambda i:scores[i])]
        valid = [valid[i] for i in family]
        # Normalize scatter by observed row spacing. An unscaled metre-valued
        # penalty can discard a true end; pure support count can tilt the whole
        # plantation to absorb a few off-row stems.
        scale = float(np.median(np.diff([np.median(lateral[g]) for g in valid])))
        score = sum(len(g) for g in valid)-sum(float(np.std(lateral[g]))*len(g) for g in valid)/scale
        promoted = [g for g in valid if len(g) >= 4 and np.ptp(xy[g]@direction) >= 8]
        if best is None or score > best[0]: best = (score, direction, normal, promoted)
    if best is None:
        return []
    _, direction, normal, groups = best
    if len(groups) < 2:
        return []
    rows = []
    for ids in groups:
        points = xy[ids]; along = points @ direction
        # Local normal-form regression tolerates arbitrary world orientation.
        slope, intercept = np.polyfit(along, points@normal, 1)
        residual = points@normal-(slope*along+intercept)
        keep = np.abs(residual) <= .65
        if keep.sum() < 4: continue
        points, along = points[keep], along[keep]
        slope, intercept = np.polyfit(along, points@normal, 1)
        d = direction+slope*normal; d /= np.linalg.norm(d)
        if d[0] < 0: d = -d
        n = np.array([-d[1], d[0]])
        offset = float(np.median(points@n)); s = points@d
        order = np.argsort(s); s = s[order]; points = points[order]
        # A gap larger than 12 m is not drawn as a continuously observed row.
        pieces = np.split(np.arange(len(s)), np.flatnonzero(np.diff(s) > 12.)+1)
        spans = [[(d*s[p[0]]+n*offset).tolist(), (d*s[p[-1]]+n*offset).tolist()]
                 for p in pieces if len(p) >= 2]
        if not spans: continue
        rows.append(dict(direction=d.tolist(), normal=n.tolist(), offset_m=offset,
            spans=spans, support=len(points), support_xy=points.tolist(),
            residual_m=float(np.sqrt(np.mean((points@n-offset)**2))),
            endpoints='observed_support_limits_not_confirmed_row_ends'))
    return rows


class RowMap:
    """Fixed-cadence updates make results independent of sample selection."""
    def __init__(self, update_seconds=5., *, local_radius=24., max_voxels=180000, row_gap_bound=7.):
        self.origin = None; self.latest = -np.inf; self.trail = []; self.voxels = set()
        self.update_seconds = update_seconds; self.next_update = 0.; self.next_id = 1
        self.local_radius, self.max_voxels = float(local_radius), int(max_voxels)
        self.row_gap_bound = float(row_gap_bound)
        self.pending_direction, self.coherent_updates = None, 0
        self.tracks = {}; self.structure = dict(rows=[], stems=[], height_band_xy=[], timestamp_s=0., diagnostics={})

    def add(self, frame):
        now = float(frame['timestamp'])
        if now <= self.latest: raise ValueError('scan timestamps must strictly increase')
        self.latest = now
        if self.origin is None: self.origin = frame['position'].copy()
        hits = frame['points_sensor'][frame['valid']]
        hits = hits @ quat_matrix(frame['sensor_quaternion']).T + frame['sensor_position']-self.origin
        cells = np.unique(np.rint(hits/.2).astype(np.int32), axis=0)
        self.voxels.update(map(tuple, cells.tolist()))
        self.trail.append((frame['position']-self.origin).copy())
        if now+1e-5 >= self.next_update:
            cells = np.asarray(list(self.voxels), dtype=np.int32).reshape(-1, 3)
            center = (frame['position']-self.origin)[:2]
            keep = np.linalg.norm(cells[:, :2]*.2-center, axis=1) <= self.local_radius
            self.voxels = set(map(tuple, cells[keep].tolist()))
            if len(self.voxels) > self.max_voxels:
                raise RuntimeError('ROW_LOCAL_STORAGE_BUDGET')
            points = cells[keep]*.2
            stems, diagnostics = extract_stems(points)
            band_xy = diagnostics.pop('height_band_xy', [])
            rows = fit_rows(stems, max_row_spacing=self.row_gap_bound)
            hypotheses = copy.deepcopy(rows)
            if rows:
                seed = max(rows, key=lambda r:r['support'])['direction']
                rows = [r for r in rows if np.dot(r['direction'], seed) >= np.cos(np.deg2rad(8))]
            if not self.tracks:
                if len(rows) >= 2:
                    direction = np.sum([np.asarray(r['direction'])*r['support'] for r in rows], axis=0)
                    direction /= np.linalg.norm(direction)
                    consistent = (self.pending_direction is not None and
                        np.dot(direction, self.pending_direction) >= np.cos(np.deg2rad(8)))
                    self.coherent_updates = self.coherent_updates+1 if consistent else 1
                    self.pending_direction = direction
                else:
                    self.coherent_updates = 0
                if self.coherent_updates < 2:
                    rows = []
            unused = set(self.tracks)
            for row in sorted(rows, key=lambda r:r['offset_m']):
                options = []
                for rid in unused:
                    old = self.tracks[rid]
                    angle = abs(np.dot(old['direction'], row['direction']))
                    center = np.mean(np.array(row['spans']).reshape(-1, 2), axis=0)
                    delta = abs(np.dot(center, old['normal'])-old['offset_m'])
                    if angle > np.cos(np.deg2rad(8)) and delta < 1.2: options.append((delta, rid))
                if options:
                    rid = min(options)[1]; unused.remove(rid)
                    old = self.tracks[rid]
                    # Keep measured stem support while refining its line.
                    # Occlusion and a rolling window cannot erase an already
                    # observed section or freeze an early angular error.
                    support = np.asarray(old.get('support_xy', np.asarray(old['spans']).reshape(-1, 2).tolist())
                        + row.get('support_xy', np.asarray(row['spans']).reshape(-1, 2).tolist()))
                    _, ids = np.unique(np.rint(support/.3).astype(int), axis=0, return_index=True)
                    support = support[ids]
                    center = support.mean(axis=0)
                    _, _, vectors = np.linalg.svd(support-center, full_matrices=False)
                    d = vectors[0]
                    if np.dot(d, old['direction']) < 0:
                        d = -d
                    n = np.array([-d[1], d[0]])
                    offset = float(np.median(support@n))
                    along = np.sort(support@d)
                    pieces = np.split(along, np.flatnonzero(np.diff(along) > 12.)+1)
                    row.update(direction=d.tolist(), normal=n.tolist(), offset_m=offset,
                        support_xy=support.tolist(), support=len(support),
                        spans=[[(d*s[0]+n*offset).tolist(), (d*s[-1]+n*offset).tolist()]
                               for s in pieces if len(s) >= 2])
                else:
                    rid = f'R{self.next_id:02d}'; self.next_id += 1
                row.update(id=rid, last_observed_s=float(now))
                self.tracks[rid] = copy.deepcopy(row)
            # Central direction uses robust line inliers, while a uniquely
            # associated, vertically supported stem may extend a jittered row
            # beyond that central fit. Its extent must not vanish as an outlier.
            tracks = list(self.tracks.values())
            extent = {r['id']:list(r.get('extent_support_xy', []))+r.get('support_xy', []) for r in tracks}
            for stem in stems:
                point = np.asarray(stem['xy'])
                distances = sorted((abs(point@r['normal']-r['offset_m']), i) for i, r in enumerate(tracks))
                if not distances:
                    break
                distance, index = distances[0]
                band = min(1.2, .45*(distances[1][0]+distance)) if len(distances) > 1 else 1.2
                if distance <= band and (len(distances) == 1 or distances[1][0]-distance > .25):
                    extent[tracks[index]['id']].append(point.tolist())
            for row in tracks:
                support = np.asarray(extent[row['id']], float).reshape(-1, 2)
                if not len(support):
                    continue
                _, ids = np.unique(np.rint(support/.3).astype(int), axis=0, return_index=True)
                support = support[ids]
                row['extent_support_xy'] = support.tolist()
                d, n = np.asarray(row['direction']), np.asarray(row['normal'])
                along = np.sort(support@d)
                pieces = np.split(along, np.flatnonzero(np.diff(along) > 12.)+1)
                row['spans'] = [[(d*s[0]+n*row['offset_m']).tolist(), (d*s[-1]+n*row['offset_m']).tolist()]
                                for s in pieces if len(s) >= 2]
            self.structure = dict(rows=list(copy.deepcopy(self.tracks).values()), stems=stems, height_band_xy=band_xy,
                row_hypotheses=hypotheses, coherent_direction_updates=self.coherent_updates,
                height_band_above_estimated_ground_m=list(HEIGHT_BAND),
                timestamp_s=now, diagnostics=diagnostics,
                semantics='estimated_tree_rows_not_free_space_or_confirmed_endpoints')
            self.next_update = (np.floor(now/self.update_seconds)+1)*self.update_seconds

    def snapshot(self):
        structure = copy.deepcopy(self.structure)
        # Bracketing observed row supports is a corridor hypothesis, never a
        # clearance/connection claim. No ground-truth lane numbering is used.
        structure['current_corridor'] = None
        position = self.trail[-1][:2] if self.trail else np.zeros(2)
        near = []
        for row in structure['rows']:
            d, n = np.array(row['direction']), np.array(row['normal'])
            along = float(position@d)
            if any(min(np.array(s)@d) <= along <= max(np.array(s)@d) for s in row['spans']):
                near.append((float(position@n-row['offset_m']), row))
        left = sorted((x for x in near if x[0] < 0), key=lambda x:abs(x[0]))
        right = sorted((x for x in near if x[0] >= 0), key=lambda x:abs(x[0]))
        if left and right:
            a, b = right[0][1], left[0][1]
            if np.dot(a['direction'], b['direction']) > .98:
                structure['current_corridor'] = dict(id=a['id']+'_'+b['id'],
                    bordering_rows=[a['id'], b['id']], status='geometric_hypothesis_not_verified_free')
        return structure, np.asarray(self.trail).copy()


def draw_rows(ax, structure, trail, yaw, chinese=False):
    """Metric height-band cells dominate; fitted rows are faint annotations.

    Display every band cell, including branches and rejected stem clusters.
    Empty cells remain unclassified, not free. No dilation/closing fills gaps.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.set_facecolor('#f0f1ef')
    band = np.asarray(structure['height_band_xy']).reshape(-1, 2)
    if len(band):
        cells = np.rint(band/.2).astype(int)
        lower, upper = cells.min(axis=0), cells.max(axis=0)
        pixels = np.zeros((upper[1]-lower[1]+1, upper[0]-lower[0]+1, 4))
        ij = cells-lower
        pixels[ij[:,1], ij[:,0]] = [45/255, 57/255, 48/255, 1.]
        ax.imshow(pixels, origin='lower', interpolation='nearest', zorder=2,
                  extent=[(lower[0]-.5)*.2, (upper[0]+.5)*.2,
                          (lower[1]-.5)*.2, (upper[1]+.5)*.2])
    for row in structure['rows']:
        for span in row['spans']:
            p = np.asarray(span)
            ax.plot(*p.T, color='#699477', linewidth=.65, linestyle=(0,(5,5)), alpha=.5, zorder=1)
        p = row['spans'][0][0]
        ax.annotate(row['id'], p, xytext=(-12, 7), textcoords='offset points',
                    ha='right', fontsize=8, color='#5b7864')
    ax.plot(trail[:, 0], trail[:, 1], color='#237bc0', linewidth=1.3, zorder=3)
    ax.scatter(*trail[0, :2], marker='s', c='#38784d', edgecolors='white', linewidths=.8, s=40, zorder=4)
    ax.scatter(*trail[-1, :2], c='#d64e3d', edgecolors='white', linewidths=1., s=55, zorder=5)
    ax.annotate('', xy=trail[-1,:2]+1.8*np.array([np.cos(yaw),np.sin(yaw)]),
                xytext=trail[-1,:2], arrowprops=dict(arrowstyle='-|>', color='#d64e3d', lw=1.7), zorder=5)
    labels = (['高度带命中', '树行辅助线', '已飞轨迹', '当前位置'] if chinese else
              ['Height-band returns', 'Estimated rows', 'Flown trail', 'Drone'])
    handles = [Patch(facecolor='#2d3930',label=labels[0]),
               Line2D([],[],color='#699477',lw=.8,ls='--',label=labels[1]),
               Line2D([],[],color='#237bc0',lw=1.3,label=labels[2]),
               Line2D([],[],color='#d64e3d',marker='o',ls='',label=labels[3])]
    t = structure['timestamp_s']
    title = (f'全局地图 · {t:.1f} 秒  |  离地 {HEIGHT_BAND[0]:g}–{HEIGHT_BAND[1]:g} 米高度带' if chinese else
             f'Global map · {t:.1f} s | Height band {HEIGHT_BAND[0]:g}–{HEIGHT_BAND[1]:g} m above ground')
    ax.set(title=title, xlabel='距起点 X / 米' if chinese else 'World X from start (m)',
           ylabel='距起点 Y / 米' if chinese else 'World Y from start (m)', aspect='equal')
    # Bounds depend only on the current prefix; reserve room for row labels.
    bounds = np.vstack([band, np.asarray(trail)[:,:2]])
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    ax.set_xlim(lo[0]-5,hi[0]+4); ax.set_ylim(lo[1]-4,hi[1]+5)
    ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=4, framealpha=.95)
    ax.tick_params(colors='#58605c', labelsize=9)
    for spine in ax.spines.values(): spine.set_color('#c4cac5')
