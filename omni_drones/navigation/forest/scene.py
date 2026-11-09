"""任务化随机森林；场景真值只用于采集路线和独立评测。"""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from .geometry import INFLATION, astar_grid
from .mesh import tapered_tube, horizontal_section, projected_clearance, section_grid

DEFAULT_ASSETS = Path(__file__).resolve().parents[3]/'catkin_ws/src/survey_simulator/models/out_test'
TASKS = ('straight', 'left', 'right', 'both', 'slalom', 'curve', 'stop')


def load_scene(seed, map_dir, map_id=None, split=None, task='random', route_mode='default', **legacy):
    """只选择并加载已验证的地图，不在仿真启动时生成或规划。"""
    root = Path(map_dir).expanduser().resolve()
    if map_id:
        manifest = root / map_id / 'map.json'
        info = json.loads(manifest.read_text())
    else:
        candidates = [entry for entry in json.loads((root/'index.json').read_text())
                      if (split is None or entry['split'] == split)
                      and (task == 'random' or entry['task'] == task)]
        if not candidates:
            raise ValueError(f'no validated maps for task={task}, split={split} in {root}')
        manifest = root / candidates[seed % len(candidates)]['map_id'] / 'map.json'
        info = json.loads(manifest.read_text())
    if route_mode not in ('default', 'continuous', 'stop'):
        raise ValueError(f'Unknown route_mode: {route_mode}')
    if route_mode != 'default':
        variant = info.get('route_variants', {}).get(route_mode)
        if variant is None: raise ValueError(f'Map does not export route_mode={route_mode}')
        info = json.loads((manifest.parent/variant).read_text())
    if not info['validation']['passed']:
        raise ValueError(f'map is a geometry reference, not a validated flight task: {manifest}')
    info['map_path'] = str(manifest.parent)
    with np.load(manifest.parent / info['mesh_file']) as mesh:
        vertices, faces = mesh['vertices'], mesh['faces']
    return info, vertices, faces, None


def create_scene(seed, source='offline', **config):
    if source == 'vln_pilot':
        from .pilot_scene import make_scene
        return make_scene()
    if source == 'empty':
        start, goal = [0., 0., 2.], [8., 0., 2.]
        info = dict(start=start, goal=goal, reference=[start, goal], initial_yaw=0.,
                    source='empty', validation=dict(route_bounds=[-20., 20., -20., 20.]))
        vertices = np.array([[-25,-25,0], [25,-25,0], [25,25,0], [-25,25,0]], dtype=np.float32)
        return info, vertices, np.array([[0,1,2], [0,2,3]], dtype=np.int32), None
    if source == 'offline':
        return load_scene(seed, **config)
    for key in ('map_dir', 'map_id', 'split'):
        config.pop(key, None)
    return make_scene(seed, **config)


@lru_cache(maxsize=4)
def tree_templates(asset_dir):
    """借用原项目枝杈的高度/长度比例；CSV 中未说明的角度语义不作物理标定。"""
    files = sorted(Path(asset_dir).glob('scene*.csv'))
    if not files:
        raise FileNotFoundError(f'no tree parameter CSV in {asset_dir}')
    templates, digest = [], hashlib.sha256()
    for file in files:
        digest.update(file.read_bytes())
        trees = {}
        with file.open(newline='') as stream:
            for row in csv.DictReader(stream):
                trees.setdefault(row['tree_id'], []).append([float(row['branchHeight']), float(row['branchLength'])])
        templates.extend(np.asarray(branches) for branches in trees.values())
    return templates, digest.hexdigest()


def tree_mesh(position, radius, height, branches, rng):
    trunk_v, trunk_f = tapered_tube([*position,0], [*position,height], radius, radius*.18)
    vertices, faces, offset = [trunk_v], [trunk_f], len(trunk_v)
    phase = rng.uniform(-np.pi,np.pi)
    for i, (level, length) in enumerate(branches):
        z = float(np.clip(level/branches[:,0].max()*height*.92, .9, height*.95))
        reach = float(np.clip(length/branches[:,0].max()*height*.65, .45, 1.45))
        angle = phase+i*2.39996+rng.uniform(-.25,.25)
        start = np.array([*position,z])
        end = start+[reach*np.cos(angle),reach*np.sin(angle),rng.uniform(.15,.55)]
        v, f = tapered_tube(start,end,radius*.25,.018,sides=8)
        vertices.append(v); faces.append(f+offset); offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def collection_route(start, goal, section, bounds, clearance=INFLATION+.25):
    """真值 A* 仅安排采集飞行；额外 0.25 m 是采集跟踪余量，不修改教师的 0.65 m 定义。"""
    resolution = .2
    occupied, origin = section_grid(section,bounds,resolution)
    distances = distance_transform_edt(~occupied)*resolution if occupied.any() else np.full(occupied.shape,np.inf)
    blocked = distances <= clearance+resolution/np.sqrt(2)
    indices = np.rint((np.asarray([start,goal])-origin)/resolution).astype(int)
    if ((indices < 0) | (indices >= blocked.shape)).any():
        return None
    route = astar_grid(blocked,*indices)
    if route is None:
        return None
    path = np.vstack([start,origin+route*resolution,goal])
    simple, i = [path[0]], 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and projected_clearance(path[[i,j]],section) <= clearance:
            j -= 1
        if np.linalg.norm(path[j]-simple[-1]) > 1e-6:
            simple.append(path[j])
        i = j
    simple = np.asarray(simple)
    if len(simple) < 2 or projected_clearance(simple,section) <= clearance:
        return None
    return simple


def make_scene(seed, asset_dir=DEFAULT_ASSETS, length=12., width=14., buffer=4., height=2., task='random'):
    if length < 8 or width < 10 or buffer < 2 or not .8 <= height <= 3:
        raise ValueError('forest requires length>=8, width>=10, buffer>=2 and flight height in [0.8,3]')
    if task != 'random' and task not in TASKS:
        raise ValueError(f'task must be random or one of {TASKS}')
    templates, source_hash = tree_templates(str(Path(asset_dir).resolve()))
    rng = np.random.default_rng(seed)
    kind = str(rng.choice(TASKS,p=[.2,.15,.15,.1,.2,.1,.1])) if task == 'random' else task
    bounds = (-buffer,length+buffer,-width/2,width/2)
    # 同一随机种子的任务变体、不同高度和重复飞行始终归入同一集合。
    scene_id = hashlib.sha256(f'forest-v2:{seed}:{source_hash}'.encode()).hexdigest()[:20]
    for attempt in range(40):
        start = np.array([0.,rng.uniform(-.25,.25)])
        goal = np.array([length,rng.uniform(-.45,.45)])
        centers = []
        spacing = rng.uniform(2.4,3.4)
        for x in np.arange(-buffer,length+buffer+.01,spacing):
            bend = 1.8*np.sin(np.pi*np.clip(x,0,length)/length) if kind == 'curve' else 0.
            inner = rng.uniform(1.5,1.8) if kind == 'curve' else rng.uniform(2.6,3.3)
            for y in (-width/2+.3,-inner,inner,width/2-.3):
                centers.append([x+rng.uniform(-.3,.3),y+bend+rng.uniform(-.15,.15)])
        barrier_x = length*rng.uniform(.43,.57)
        if kind in ('left','right','slalom','stop'):
            barriers = [(barrier_x,kind)] if kind != 'slalom' else [(length*.33,'left'),(length*.68,'right')]
            for x, side in barriers:
                low, high = (-width/2, .1) if side == 'left' else (-.1,width/2)
                if side == 'stop': low, high = -width/2,width/2
                for y in np.linspace(low,high,int(np.ceil((high-low)/1.1))+1):
                    centers.append([x+rng.uniform(-.08,.08),y])
        elif kind == 'both':
            centers.append([barrier_x,rng.uniform(-.25,.25)])
        vertices = [np.array([[bounds[0],bounds[2],0],[bounds[1],bounds[2],0],
                              [bounds[1],bounds[3],0],[bounds[0],bounds[3],0]],float)]
        faces, offset, trees = [np.array([[0,1,2],[0,2,3]],np.int32)], 4, []
        for center in centers:
            radius, tree_height = rng.uniform(.24,.4), rng.uniform(4.5,7.5)
            template_id = int(rng.integers(len(templates)))
            v, f = tree_mesh(center,radius,tree_height,templates[template_id],rng)
            vertices.append(v); faces.append(f+offset); offset += len(v)
            trees.append(dict(position=center,radius=radius,height=tree_height,template=template_id))
        vertices, faces = np.concatenate(vertices).astype(np.float32), np.concatenate(faces)
        section = horizontal_section(vertices,faces,height-.5,height+.5)
        if min(projected_clearance([start],section),projected_clearance([goal],section)) <= INFLATION+.3:
            continue
        route = collection_route(start,goal,section,bounds)
        if kind == 'stop':
            if route is not None:
                continue
            # 目标仍在障碍后方；采集飞机只到安全观察点，不穿过封闭通道。
            observation_goal = np.array([barrier_x-2.2,start[1]])
            route = collection_route(start,observation_goal,section,bounds)
        if route is None:
            continue
        direct = projected_clearance([start,goal],section) #删除中间点
        if kind == 'straight' and direct <= INFLATION+.25:
            continue
        if kind in ('left','right','both','slalom','curve') and direct > INFLATION:
            continue
        break
    else:
        raise ValueError(f'could not generate a safe {kind} collection route for seed {seed}')
    geometry_id = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()[:20]
    info = dict(scene_id=scene_id,geometry_id=geometry_id,geometry='triangle_mesh',generator='random_forest_v2',
                task=kind,source=str(Path(asset_dir).resolve()),source_sha256=source_hash,
                mesh_file='scene_mesh.npz',vertices=len(vertices),triangles=len(faces),trees=trees,
                parameters=dict(seed=int(seed),length=length,width=width,buffer=buffer,height=height,task=task),
                reference=np.c_[route,np.full(len(route),height)].tolist(),goal=[*goal,height],
                initial_yaw=float(rng.uniform(-.35,.35)),bounds=list(bounds),generation_attempts=attempt+1,
                collection_clearance=projected_clearance(route,section),
                simplification='procedural tapered trunk/branch meshes using source CSV proportions; no leaves, wind or calibrated vegetation')
    return info, vertices, faces, np.zeros((*faces.shape,2),dtype=np.float32)


def scene_split(scene_ids):
    # 固定哈希分组：增加新森林不会把原来的训练森林挪到测试集。小数据缺集合时如实报告。
    result = {}
    for scene_id in sorted(set(scene_ids)):
        bucket = int(hashlib.sha256(scene_id.encode()).hexdigest()[:8],16)%100
        result[scene_id] = 'test' if bucket < 15 else 'val' if bucket < 30 else 'train'
    return result


def recorded_scene_splits(scenes):
    """沿用记录中的划分；仅无划分的旧场景使用固定哈希，同一森林不得跨集合。"""
    scenes = list(scenes)
    assigned = {}
    for scene in scenes:
        scene_id, split = scene['scene_id'], scene.get('split')
        if split is None:
            continue
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'invalid split {split!r} for scene {scene_id}')
        previous = assigned.setdefault(scene_id, split)
        if previous != split:
            raise ValueError(f'conflicting splits for scene {scene_id}: {previous} and {split}; use different seeds for separate splits')
    result = scene_split(scene['scene_id'] for scene in scenes)
    result.update(assigned)
    return result
