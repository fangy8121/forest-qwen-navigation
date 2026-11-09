"""Reproducible executor diagnostics and native-forest previews, kept separate."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from .generator import DEFAULT_CONFIG, SOURCE, sample_forest, mesh_forest, forest_scene_id
from .maps import write_usd
from omni_drones.navigation.forest.mesh import tapered_tube


def diagnostic_tasks():
    return [dict(goal_id=name, revision=1, target_world=target, arrival_yaw=yaw)
            for name, target, yaw in [
                ('forward_stop', [1.2, 0., 2.], 0.),
                ('left_heading', [0., 2.5, 2.], np.pi/2),
                ('right_heading', [0., -2.5, 2.], -np.pi/2),
                ('bypass_rejoin', [5.5, 0., 2.], 0.)]]


def diagnostic_scene():
    vertices = [np.array([[-8., -8., 0.], [12., -8., 0.], [12., 8., 0.], [-8., 8., 0.]])]
    faces = [np.array([[0, 1, 2], [0, 2, 3]])]
    trees = [(x, y, .2, 4.) for x in (2., 4., 6., 8.) for y in (-3., 3.)]
    trees.append((2.8, 0., .35, 4.))
    offset = 4
    for x, y, radius, height in trees:
        v, f = tapered_tube([x, y, 0.], [x, y, height], radius, radius, sides=16)
        vertices.append(v); faces.append(f+offset); offset += len(v)
    info = dict(source='vln_pilot', scene_id='vln_executor_diagnostic_v1', initial_yaw=0.,
                start=[0., 0., 2.], goal=[5.5, 0., 2.], reference=[[0., 0., 2.], [5.5, 0., 2.]],
                reference_use='spawn only; not a feasible route or executor input', trees=trees,
                validation=dict(route_bounds=[-4., 10., -6., 6.]),
                limitations='Fixed cylinders and flat ground; no branches, terrain or language policy. '
                             'Known goal coordinates diagnose the executor only.')
    return info, np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32), None


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def preview(folder, info, vertices, faces, terrain=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(121)
    view = fig.add_subplot(122, projection='3d')
    if terrain is not None:
        ax.contourf(terrain.xs, terrain.ys, terrain.heights.T, levels=16, cmap='terrain', alpha=.6)
        centers = np.array([t['position'] for t in info['trees']])
        ax.scatter(centers[:, 0], centers[:, 1], s=[120*t['diameter'] for t in info['trees']], color='#214c2b')
        ax.set_xlim(-2., 15.); ax.set_ylim(-2., 15.)
        title = f'Native forest seed={info["parameters"]["seed"]}; geometry preview, tasks pending'
    else:
        from matplotlib.patches import Circle
        for x, y, radius, _ in info['trees']:
            ax.add_patch(Circle((x, y), radius, color='#425c36'))
        for task in diagnostic_tasks():
            x, y, _ = task['target_world']
            ax.scatter(x, y, marker='*', s=110)
            ax.annotate(task['goal_id'], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.scatter(0., 0., color='black', label='Start')
        ax.set_xlim(-2., 9.); ax.set_ylim(-4., 4.)
        title = 'Fixed executor diagnostic; flat ground + cylinders'
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', title='Top view / public diagnostic targets')
    ax.set_aspect('equal'); ax.grid(alpha=.2)
    # Ground has a large extent; keep the 3D view focused on the native stand.
    triangles = vertices[faces]
    selected = triangles[np.any(triangles[:, :, 2] > (terrain.heights.min()+.5 if terrain else .1), axis=1)]
    collection = Poly3DCollection(selected[::max(1, int(np.ceil(len(selected)/5000)))], alpha=.6, linewidth=0.)
    collection.set_facecolor('#607e46'); view.add_collection3d(collection)
    view.set(xlim=ax.get_xlim(), ylim=ax.get_ylim(), zlim=(float(vertices[:, 2].min()), float(vertices[:, 2].max())),
             xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Mesh preview (display subsampling only)')
    view.view_init(elev=25, azim=-65)
    fig.suptitle(title)
    fig.savefig(folder/'overview.png', dpi=150)
    plt.close(fig)


def build_executor_maps(output, seeds=(202610240, 202610241, 202610242)):
    from omni_drones.navigation.forest.evaluation import truth_collision, sample_polyline
    from omni_drones.navigation.forest.geometry import INFLATION, VERTICAL_RADIUS
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    folder = output/'diagnostic/000'; folder.mkdir(parents=True)
    info, vertices, faces, _ = diagnostic_scene()
    witness = np.array([[0., 0., 2.], [1., 1.5, 2.], [4.5, 1.5, 2.], [5.5, 0., 2.]])
    truth = dict(mesh=(vertices, faces))
    checks = dict(endpoints_clear=not truth_collision(np.array([info['start']]+[t['target_world'] for t in diagnostic_tasks()]),
                                                    truth, INFLATION, VERTICAL_RADIUS),
                  bypass_witness_clear=not truth_collision(sample_polyline(witness), truth, INFLATION, VERTICAL_RADIUS),
                  direct_bypass_blocked=truth_collision(sample_polyline(np.array([info['start'], info['goal']])),
                                                       truth, INFLATION, VERTICAL_RADIUS))
    info.update(generator='forest_generator.executor_pilot', profile='diagnostic', task='executor_primitives', split='development',
                mesh_file='scene_mesh.npz', usd_file='scene.usd')
    info['validation'].update(passed=all(checks.values()), scope='endpoint and geometric witness only; not flight validation', **checks)
    if not info['validation']['passed']:
        dump(folder/'rejection.json', checks)
        raise ValueError('Diagnostic geometry check failed; no replacement seed was chosen')
    np.savez_compressed(folder/'scene_mesh.npz', vertices=vertices, faces=faces)
    write_usd(folder/'scene.usd', vertices, faces)
    dump(folder/'map.json', info); dump(folder/'tasks.json', diagnostic_tasks())
    dump(folder/'offline_witness.json', dict(purpose='offline feasibility audit only; never executor input', points=witness.tolist()))
    preview(folder, info, vertices, faces)
    records.append(dict(map_id='diagnostic/000', profile='diagnostic', ready='executor_diagnostics', tasks=4))
    for seed in seeds:
        folder = output/f'native/{seed}'; folder.mkdir(parents=True)
        config = json.loads(DEFAULT_CONFIG.read_text())
        config['forest_size'] = [12., 12.]
        # Keep all source tree/branch/terrain distributions. Only the footprint is shortened.
        trees, rows, terrain = sample_forest(seed, config)
        vertices, faces, uv, normals = mesh_forest(trees, terrain)
        info = dict(scene_id=forest_scene_id(seed, config), generator='forest_generator.executor_pilot',
                    profile='native_preview', parameters=dict(seed=seed, config=config), trees=trees, rows=rows,
                    mesh_file='scene_mesh.npz', usd_file='scene.usd', split='development',
                    validation=dict(passed=False, scope='geometry preview only; task feasibility and flight not checked'),
                    native_distributions_preserved=True, branch_count=sum(len(t['branches']) for t in trees),
                    modifications_from_default=dict(forest_size=[12., 12.]))
        np.savez_compressed(folder/'scene_mesh.npz', vertices=vertices, faces=faces)
        np.savez_compressed(folder/'terrain.npz', xs=terrain.xs, ys=terrain.ys, heights=terrain.heights)
        shutil.copy2(SOURCE/'tex.jpg', folder/'tex.jpg')
        write_usd(folder/'scene.usd', vertices, faces, uv, 'tex.jpg', normals)
        dump(folder/'map.json', info)
        dump(folder/'tasks.json', dict(status='pending_task_generation', tasks=[]))
        preview(folder, info, vertices, faces, terrain)
        records.append(dict(map_id=f'native/{seed}', profile='native_preview', ready=False,
                            trees=len(trees), branches=info['branch_count'], tasks=0))
    paths = [Path(__file__), DEFAULT_CONFIG, Path(__file__).with_name('generator.py'),
             Path(__file__).with_name('terrain.py'), Path(__file__).with_name('maps.py'), SOURCE/'untitled.dae', SOURCE/'tex.jpg']
    manifest = dict(schema='forest_executor_maps_v1', maps=records,
                    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    files_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.rglob('*')) if p.is_file()})
    dump(output/'manifest.json', manifest)
    dump(output/'index.json', [dict(map_id='diagnostic/000', scene_id='vln_executor_diagnostic_v1',
                                   task='executor_primitives', split='development')])
    lines = ['# 执行器地图索引', '', '圆柱诊断地图和原生森林预览分开保存；原生预览尚无可运行任务，不计入实验分母。', '',
             '| 地图 | 状态 | 预览 | 参数与几何 |', '|---|---|---|---|']
    for r in records:
        p=output/r['map_id']
        lines.append(f'| {r["map_id"]} | {r["ready"]} | [overview.png]({p}/overview.png) | [map.json]({p}/map.json) |')
    lines += ['', '`scene_mesh.npz` 与 `scene.usd` 是完整几何，预览图的显示抽样不改变物理/射线模型。',
              '原生森林保留原树模板、枝干分布和起伏地形，仅将林分尺寸设为 12×12 米。没有通过删枝、压平地面或扩大通道确保通过。',
              '所有输出带种子/参数和文件哈希；命令拒绝覆盖已有目录。', '']
    (output/'README.md').write_text('\n'.join(lines))
    print(output/'README.md')
    return output
