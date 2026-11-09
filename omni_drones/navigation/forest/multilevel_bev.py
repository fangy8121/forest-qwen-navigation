"""Paper-inspired multilevel BEV input, without teacher labels or route choices.

Reuses the collision/evidence semantics of ``structured_bev``. Heights are
sampled drone-center positions, NOT the manuscript's disjoint point bins.
Each height checks the existing body envelope, so density slabs can overlap.
The gap/pass feature families follow manuscript 1157, III-B; unknown masks,
body inflation and meter-valued descriptors are project adaptations. Neither
cross-section gaps nor free fractions establish a continuous 3D flight path.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import Z_BOUNDS
from .structured_bev import (
    PASS_BANDS, StructuredBEV, _cell_edges, extract_structured_bev,
)


DEFAULT_HEIGHTS = (-.5, 0., .5)


@dataclass
class MultilevelBEV:
    layers: tuple[StructuredBEV, ...]

    @property
    def heights(self):
        return np.array([layer.height for layer in self.layers])

    def tensors(self):
        """Dense, lossless grid export; meter coordinates are NOT normalized.

        F_occ is [K+5,X,Y]: K slab densities, maximum, x,y,dx,dy.
        F_gap is [X,K,9]: four widths, widest center (NaN when absent),
        two same-row extensions, x,dx. F_pass is [B+2,X,K]: B observed-free
        fractions, x,dx. Separate masks preserve unknown/outside/body support.
        """
        base = self.layers[0]
        xedges = _cell_edges(base.xs, base.bounds[:2], base.resolution)
        yedges = _cell_edges(base.ys, base.bounds[2:], base.resolution)
        xx, yy = np.meshgrid(base.xs, base.ys, indexing='ij')
        dx = np.broadcast_to(np.diff(xedges)[:, None], xx.shape)
        dy = np.broadcast_to(np.diff(yedges)[None, :], xx.shape)
        density = np.stack([b.occupancy_density for b in self.layers])
        gaps = np.stack([
            np.column_stack((b.gap_widths, b.gap_centers[:, 0],
                             [s['center_connects_left'] for s in b.sections],
                             [s['center_connects_right'] for s in b.sections],
                             b.xs, np.diff(xedges))) for b in self.layers
        ], axis=1)
        passes = np.stack([
            np.vstack((b.pass_free.T, b.xs, np.diff(xedges)))
            for b in self.layers
        ], axis=2)
        arrays = dict(
            F_occ=np.concatenate((density, density.max(axis=0)[None],
                                  np.stack((xx, yy, dx, dy)))),
            F_gap=gaps, F_pass=passes,
            xs=base.xs, ys=base.ys, heights=self.heights,
            x_edges=xedges, y_edges=yedges,
            pass_bands=np.array(base.pass_bands),
        )
        for name in ('blocked', 'free', 'unknown', 'allowed', 'observed',
                     'ray_observed', 'actual_body_supported', 'gap_edges',
                     'pass_unknown', 'pass_observed'):
            arrays[name] = np.stack([getattr(b, name) for b in self.layers])
        return arrays

    def spec(self):
        return dict(
            name='multilevel_observed_structured_bev_v1',
            attribution='Manuscript 1157 III-B: occupancy, lateral gaps, band free ratios',
            heights_m=self.heights.tolist(),
            height_semantics='sampled drone-center heights relative to current body; not disjoint bins or actions',
            geometry_units='meters, unnormalized; fractions in [0,1]',
            gap_names=['widest_width', 'center_width', 'left_width', 'right_width',
                       'widest_center_y', 'center_extends_left', 'center_extends_right', 'x', 'dx'],
            gap_missing_center='NaN only in dense NPZ; missing gaps serialize as empty lists',
            occupancy_channels=['slab_density_at_each_height', 'maximum_slab_density', 'x', 'y', 'dx', 'dy'],
            pass_channels=['free_fraction_in_each_band', 'x', 'dx'],
            dense_shapes={k: list(v.shape) for k, v in self.tensors().items()},
            shared_grid=self.layers[0].spec(),
            vertical_connection='not inferred: inter-layer motion needs continuous 3D validation',
        )


def extract_multilevel_bev(points, evidence, heights=DEFAULT_HEIGHTS, **kwargs):
    heights = np.asarray(heights, dtype=float)
    if (heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all()
            or np.any(np.diff(heights) <= 0)):
        raise ValueError('heights must be finite, nonempty, and strictly increasing')
    if heights[0] < Z_BOUNDS[0] or heights[-1] > Z_BOUNDS[1]:
        raise ValueError('sampled heights must lie inside project Z_BOUNDS')
    if 'height' in kwargs:
        raise ValueError('use heights, not a single height')
    tree = evidence if isinstance(evidence, cKDTree) else cKDTree(
        np.asarray(evidence, dtype=float).reshape(-1, 3))
    return MultilevelBEV(tuple(extract_structured_bev(
        points, tree, height=float(z), **kwargs) for z in heights))


def measured_sections(bev, anchors=(1., 2., 3., 4., 5., 6.)):
    """Fixed x anchors, independent of teacher labels and obstacle arrangement."""
    base = bev.layers[0]
    anchors = np.asarray(anchors, dtype=float)
    if anchors.ndim != 1 or not np.isfinite(anchors).all():
        raise ValueError('anchors must be finite one-dimensional coordinates')
    indices = sorted({int(np.argmin(abs(base.xs-x))) for x in anchors
                      if base.xs[0] <= x <= base.xs[-1]})
    result = []
    for b in bev.layers:
        rows = []
        for i in indices:
            s = b.sections[i]
            rows.append(dict(
                x_m=round(float(b.xs[i]), 3),
                gaps=[dict(y_edges_m=[round(g['y_min'], 3), round(g['y_max'], 3)],
                           width_m=round(g['width'], 3), center_y_m=round(g['center'], 3))
                      for g in s['intervals']],
                free_fractions=np.round(b.pass_free[i], 3).tolist(),
                unknown_fractions=np.round(b.pass_unknown[i], 3).tolist(),
            ))
        result.append(dict(height_m=b.height, sections=rows))
    return result


def _number_list(values):
    """Compact already-rounded measurements without dropping numeric precision."""
    return '['+','.join(_number_list(v) if isinstance(v, (list, tuple)) else str(int(v))
                        if float(v).is_integer() else str(v) for v in values)+']'


def input_prompt(bev, goal, velocity, acceleration, allow_vertical=False,
                 anchors=(1., 2., 3., 4., 5., 6.), corridor=None, verified_stop_only=False,
                 goal_horizon=None):
    """One image + compact text; preserves the existing six-XYZ-point protocol."""
    state = {}
    for name, value in [('目标XYZ', goal), ('速度XYZ', velocity), ('加速度XYZ', acceleration)]:
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f'{name} must contain three finite numbers')
        state[name] = _number_list(np.round(value, 3).tolist())
    base = bev.layers[0]
    horizon = ''
    if goal_horizon is not None:
        if not np.isfinite(goal_horizon) or goal_horizon <= 0:
            raise ValueError('goal horizon must be positive and finite')
        # Goal horizon is a planning scale, not a mandatory x coordinate or length.
        horizon = f'规划尺度{float(goal_horizon):g}m；近目标可缩短。'
    lines = [
        horizon+'规划安全、平滑的局部绕障路线，向目标推进并保留制动空间。',
        '当前机体中心为原点；x前/y左/z上，单位m。输出6个按行进顺序排列的机体中心XYZ。',
        f'BEV中心高度z={_number_list(bev.heights.tolist())}，共用XY；格宽{base.resolution:g}m。各层不是点云分箱。',
        '橙色Blocked=含机体与安全余量的中心禁区；浅色=已观测自由；浅蓝=当前机体补充的自由证据；灰色=未知；深灰=通道外。',
        '中心可视为点，空隙不再减机体直径。六点及从原点出发重建的整条连续轨迹，不得触碰或穿越对应高度的Blocked、未知或通道外。',
        '蓝圈在z=0表示机体，其他层仅为XY投影；绿星=目标XY投影。目标被挡须先绕行，终点不必等于目标。',
        ('本次允许左右及上下绕行，升降须有观测支持。' if allow_vertical else '仅水平绕行，所有输出点z=0。')+
        '层间及未显示高度不能假定自由；整段运动须经三维路径检查。',
        '；'.join(f'{k}={v}' for k, v in state.items())+'。速度m/s，加速度m/s²。',
        f'XY边界={_number_list(list(base.bounds))}；z范围={_number_list(list(Z_BOUNDS))}。',
    ]
    if corridor is not None:
        lines.append('通道多边形XY='+_number_list(np.round(corridor, 3).tolist())+'。')
    lines += [
        '青线=截面空隙；截面空隙和下列自由比例均不保证前后连通。',
        '每行：x；空隙[y_min,y_max,宽度]（[]=无已观测自由空隙）；F=自由比例；U=未知比例。',
        f'F/U按|y|≤{_number_list(list(base.pass_bands))}m排列；其余x见图。',
    ]
    for layer in measured_sections(bev, anchors):
        lines.append(f'z={layer["height_m"]:+g}米：')
        for row in layer['sections']:
            gaps = [[*g['y_edges_m'], g['width_m']] for g in row['gaps']]
            lines.append(f'x={row["x_m"]:g}; 空隙={_number_list(gaps)}; F={_number_list(row["free_fractions"])}; U={_number_list(row["unknown_fractions"])}')
    lines += [
        '仅输出JSON，不附解释。GO恰好6点：',
        '{"status":"GO","points":[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x5,y5,z5],[x6,y6,z6]]}',
        ('仅到达目标且可安全停止时输出{"status":"HOLD","points":[]}；路线不确定不能作为HOLD依据。'
         if verified_stop_only else '无法确定可行路线则输出{"status":"HOLD","points":[]}。'),
    ]
    return '\n'.join(lines)+'\n'


def render_multilevel_bev(bev, goal, output, anchors=(1., 2., 3., 4., 5., 6.)):
    """Three height panels without heatmaps; corridor/goal come from the caller."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pathlib import Path

    k = len(bev.layers)
    fig = plt.figure(figsize=(4.5*k, 4.8), dpi=110)
    grid = fig.add_gridspec(1, k)
    samples = measured_sections(bev, anchors)
    for column, (b, sample) in enumerate(zip(bev.layers, samples)):
        ax = fig.add_subplot(grid[0, column])
        colors = np.full((*b.free.shape, 3), .57)
        colors[b.free] = [.91, .95, .97]
        colors[b.actual_body_supported & b.free] = [.66, .81, 1.]
        colors[b.blocked] = [1., .59, .22]
        colors[~b.allowed] = [.25, .27, .30]
        xe = _cell_edges(b.xs, b.bounds[:2], b.resolution)
        ye = _cell_edges(b.ys, b.bounds[2:], b.resolution)
        # pcolormesh honors clipped boundary cells rather than stretching imshow.
        ax.pcolormesh(xe, ye, colors.transpose(1, 0, 2), shading='flat', rasterized=True)
        for row in sample['sections']:
            for gap in row['gaps']:
                lo, hi = gap['y_edges_m']
                ax.plot([row['x_m']]*2, [lo, hi], color='teal', lw=1.5, alpha=.8)
                if gap['width_m'] >= .4:
                    ax.text(row['x_m']+.06, (lo+hi)/2, f'{gap["width_m"]:g}',
                            fontsize=7, va='center', color='teal',
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.4))
        ax.add_patch(plt.Circle((0, 0), .5, fill=False, color='royalblue',
                               lw=1.5, linestyle='-' if abs(b.height) < 1e-9 else '--'))
        ax.arrow(0, 0, .75, 0, width=.025, color='royalblue')
        ax.scatter(*np.asarray(goal)[:2], marker='*', s=110, color='green', zorder=6)
        ax.set(title=f'Drone-center height z={b.height:+g} m',
               xlabel='x forward (m)', ylabel='y left (m)',
               xlim=b.bounds[:2], ylim=b.bounds[2:])
        ax.set_aspect('equal')
        ax.grid(alpha=.12)
    fig.suptitle(f'Local multi-height structured BEV | Goal XYZ={np.round(goal, 2).tolist()} m', fontsize=13, y=.985)
    fig.legend(handles=[Patch(color=c, label=t) for c, t in [
        ((1., .59, .22), 'Blocked'), ((.91, .95, .97), 'Observed free'),
        ((.66, .81, 1.), 'Actual body support'), ((.57,)*3, 'Unknown'),
        ((.25, .27, .30), 'Outside corridor')]], loc='upper center',
        bbox_to_anchor=(.5, .95), ncol=5, fontsize=9, frameon=False)
    fig.subplots_adjust(left=.055, right=.98, top=.80, bottom=.16, wspace=.27)
    fig.text(.5, .025, 'Widths: available body-center space (m). Unknown is not free. Layer changes need 3D path checks.',
             ha='center', fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
