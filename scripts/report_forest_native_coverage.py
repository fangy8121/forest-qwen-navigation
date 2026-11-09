"""Independently score and plot a saved native forest coverage flight."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_drones.navigation.forest.native_coverage_evaluation import evaluate_native_coverage
from omni_drones.navigation.forest.evaluation import truth_collision, sample_polyline
from omni_drones.navigation.forest.geometry import INFLATION, VERTICAL_RADIUS
from omni_drones.navigation.forest.runs import file_hash


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def report(folder):
    folder = Path(folder).resolve()
    cfg, original, mission = [read(folder/name, {}) for name in ('config.json', 'result.json', 'mission.json')]
    if cfg.get('coverage', {}).get('scene_kind') != 'native':
        raise ValueError('this report requires a native forest flight')
    rows = np.load(folder/'trajectory.npy')
    if not len(rows):
        raise ValueError('flight has no measured trajectory')
    source = read(folder/'map_source.json', {})
    truth = read(folder/'scene_truth.json')
    truth['evaluation_terrain_file'] = str(Path(source['path'])/'terrain.npz')
    truth['evaluation_protocol'] = dict(agl_bounds=cfg['coverage']['agl_bounds'],
        nominal_agl=cfg['coverage']['desired_agl'],
        work_area_bounds=(read(folder/'public_work_area.json', {}) or {}).get('xy_bounds'))
    home = np.asarray(mission['start'])
    with np.load(folder/'scene_mesh.npz') as mesh:
        violation = bool(truth_collision(sample_polyline(np.vstack([home, rows[:, 8:11]])),
            dict(mesh=(mesh['vertices'], mesh['faces'])), INFLATION, VERTICAL_RADIUS))
    status = original.get('mission', {})
    state = SimpleNamespace(coverage_result=mission['coverage_result'],
        status=status.get('status', 'FAILED'), stop_requested=status.get('stop_requested', False),
        command_results=mission.get('commands'))
    evaluation = evaluate_native_coverage(rows[:, 8:11], rows[-1, 15:18], home,
        original.get('physical_stop_verified', False), state, violation, truth,
        trace=rows, decisions=folder)
    hashes = read(folder/'source_hashes.json', {})
    root = Path(__file__).resolve().parents[1]
    snapshots = {name:bool((folder/'source'/name).is_file()
        and file_hash(folder/'source'/name) == digest) for name, digest in hashes.items()}
    current = {name:bool((root/name).is_file() and file_hash(root/name) == digest)
        for name, digest in hashes.items()}
    map_checks = {name:bool((Path(source['path'])/name).is_file()
        and file_hash(Path(source['path'])/name) == digest) for name, digest in source['sha256'].items()}
    audit = dict(evaluation=evaluation, source_snapshots=snapshots, current_sources=current,
        source_unchanged=original.get('source_unchanged'), changed_sources=original.get('changed_sources'),
        map_checks=map_checks, actual_path_length_m=float(np.linalg.norm(np.diff(rows[:, 8:11], axis=0), axis=1).sum()),
        report_source_sha256=file_hash(__file__))
    audit['passed'] = bool(evaluation['full_coverage_passed'] and snapshots and all(snapshots.values())
        and map_checks and all(map_checks.values()) and original.get('reason') == 'mission_reached'
        and original.get('failure_reason') is None and original.get('source_unchanged') is True)
    (folder/'native_independent_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    plot(folder, rows, truth, home)
    lines = ['# 原生树林全覆盖航次复核', '',
        f'完整通过：**{audit["passed"]}**；终止原因：`{original.get("reason")}`。', '',
        f'航次源码未变：`{audit["source_unchanged"]}`；航次检测到的变动：'
        f'`{json.dumps(audit["changed_sources"], ensure_ascii=False)}`。', '',
        f'完整实际路径 {audit["actual_path_length_m"]:.2f} m；仿真时间 {rows[-1, 0]:.2f} s。', '',
        f'任务范围：`{evaluation["coverage_scope"]}`；公开作业区包含完整原林：'
        f'`{evaluation["work_area_contains_entire_forest"]}`（旧航次未提供边界时为 `None`）。', '',
        '| 真值通道（包含不等长尾段） | 路线覆盖 | 漏巡长度 / m |', '|---|---:|---:|']
    for c in evaluation['truth_route_coverage']:
        lines.append(f'| {c["truth_corridor_id"]} | {c["actual_route_coverage_fraction"]:.2%} | {c["uncovered_length_m"]:.3f} |')
    lines += ['', f'S 形次序通过：`{evaluation["s_pattern"]["passed"]}`；'
        f'实际净空通过：`{evaluation["truth_margin_clear"]}`；'
        f'实测返航停稳：`{evaluation["final_home"] and evaluation["final_measured_stop"]}`。', '',
        f'地形高度通过：`{evaluation["terrain_following_passed"]}`；'
        f'Qwen 决策来源通过：`{evaluation["decision_provenance"]["passed"]}`。', '',
        '![完整树林与实际轨迹](native_coverage_overview.png)', '',
        '[独立数据与决策来源审计](native_independent_audit.json) · [原始结果](result.json)', '',
        '评价使用原始完整地图和实际轨迹；包含全部内部通道及单侧延长尾段。'
        '仿真采用理想定位与射线传感，结果不代表实机或树面/病害检查完成。']
    (folder/'native_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(run=str(folder), passed=audit['passed'],
        coverage=[round(c['actual_route_coverage_fraction'], 4) for c in evaluation['truth_route_coverage']]), ensure_ascii=False))
    return audit


def plot(folder, rows, truth, home):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    xy = np.array([t['position'][:2] for t in truth['trees']])
    axes[0].scatter(xy[:, 0], xy[:, 1], s=14, color='#547c48', label='Original native stems')
    axes[0].plot(rows[:, 8], rows[:, 9], lw=1.2, color='#246a99', label='Measured flight')
    axes[0].scatter(*home[:2], marker='*', s=90, color='#d79219', label='HOME')
    axes[0].set(xlabel='World X / m', ylabel='World Y / m', title='Complete unchanged native forest')
    axes[0].axis('equal'); axes[0].legend(fontsize=8)
    from forest_generator.terrain import grid_height
    with np.load(truth['evaluation_terrain_file']) as terrain:
        ground = grid_height(rows[:, 8:10], terrain['xs'], terrain['ys'], terrain['heights'])
        inside = ((rows[:, 8] >= terrain['xs'][0]) & (rows[:, 8] <= terrain['xs'][-1])
            & (rows[:, 9] >= terrain['ys'][0]) & (rows[:, 9] <= terrain['ys'][-1]))
    agl = np.where(inside, rows[:, 10]-ground, np.nan)
    axes[1].plot(rows[:, 0], agl, lw=1)
    for bound in truth['evaluation_protocol']['agl_bounds']:
        axes[1].axhline(bound, color='#bd7043', lw=.8, ls='--')
    axes[1].set(xlabel='Simulation time / s', ylabel='True AGL / m', title='Post-flight true-ground altitude')
    error_path = folder/'tracking_errors.npy'
    if error_path.is_file():
        errors = np.load(error_path)
        axes[2].plot(errors[:, 0], errors[:, 1], lw=.8)
    axes[2].set(xlabel='Simulation time / s', ylabel='Tracking error / m', title='Measured position vs reference')
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(folder/'native_coverage_overview.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    report(parser.parse_args().run_directory)
