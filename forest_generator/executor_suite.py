"""Freeze development episodes before flight; geometry witnesses stay offline."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .executor_pilot import diagnostic_scene, diagnostic_tasks, dump
from .maps import write_usd
from omni_drones.navigation.forest.subgoal import Subgoal
from omni_drones.navigation.forest.evaluation import truth_collision, sample_polyline
from omni_drones.navigation.forest.geometry import INFLATION, VERTICAL_RADIUS


def episode(goal_id, start, target, initial_yaw, arrival_yaw):
    start, target = np.asarray(start), np.asarray(target)
    lo, hi = np.minimum(start, target), np.maximum(start, target)
    return dict(goal=Subgoal(goal_id, 1, target, arrival_yaw).to_dict(), start=start.tolist(),
                initial_yaw=float(initial_yaw),
                operating_bounds=[float(lo[0]-4), float(hi[0]+4), float(lo[1]-4), float(hi[1]+4)],
                height_bounds=[float(lo[2]-.8), float(hi[2]+.8)])


def export(folder, info, vertices, faces, episodes, witnesses):
    folder.mkdir(parents=True, exist_ok=True)
    truth = dict(mesh=(vertices, faces))
    valid = [not truth_collision(sample_polyline(np.asarray(path)), truth, INFLATION, VERTICAL_RADIUS)
             for path in witnesses]
    if not all(valid):
        dump(folder/'rejection.json', dict(witness_clear=valid))
        raise ValueError(f'Invalid frozen geometry witness: {folder}')
    first = episodes[0]
    info.update(generator='forest_generator.executor_suite', task='executor_development', split='development',
                start=first['start'], goal=first['goal']['target_world'],
                reference=[first['start'], first['goal']['target_world']],
                reference_use='spawn only; offline witnesses are not executor inputs',
                initial_yaw=first['initial_yaw'], mesh_file='scene_mesh.npz', usd_file='scene.usd',
                validation=dict(passed=True, scope='geometric witnesses only; not flight validation',
                                route_bounds=[-10., 20., -10., 20.], witness_clear=valid))
    np.savez_compressed(folder/'scene_mesh.npz', vertices=vertices, faces=faces)
    if not (folder/'scene.usd').exists():
        write_usd(folder/'scene.usd', vertices, faces)
    dump(folder/'map.json', info)
    dump(folder/'tasks.json', [e['goal'] for e in episodes])
    dump(folder/'episodes.json', [{k:v for k,v in e.items() if k!='goal'} | {'goal_id':e['goal']['goal_id']}
                                  for e in episodes])
    dump(folder/'offline_witnesses.json', dict(scope='offline feasibility only; never executor input', paths=witnesses))
    # Every frozen start/target is visible without opening binary data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if info['profile']=='native_executor':
        trees=info['trees']; ax.scatter([t['position'][0] for t in trees], [t['position'][1] for t in trees],
                                      s=[160*t['diameter'] for t in trees], color='#345b38', label='Tree trunks')
    else:
        from matplotlib.patches import Circle
        for x,y,r,_ in info['trees']: ax.add_patch(Circle((x,y),r,color='#345b38'))
    for e, path in zip(episodes,witnesses):
        path=np.asarray(path); line,=ax.plot(path[:,0],path[:,1],alpha=.6)
        ax.scatter(*path[0,:2],color=line.get_color(),marker='o')
        ax.scatter(*path[-1,:2],color=line.get_color(),marker='*',s=100)
        ax.annotate(e['goal']['goal_id'],path[-1,:2],fontsize=7)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Frozen tasks / offline witnesses (not flight paths)')
    ax.set_aspect('equal'); ax.grid(alpha=.2); fig.savefig(folder/'tasks.png',dpi=160); plt.close(fig)


def build_executor_suite(output, source_maps):
    output, source_maps = Path(output).resolve(), Path(source_maps).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records=[]
    # Three coordinate frames and two lateral offsets of the blocking trunk.
    for index, (angle, shift, obstacle_y) in enumerate([(0., [0.,0.,0.], 0.),
            (np.pi/2, [2.,-1.,0.], .25), (-np.pi/3, [-1.,2.,0.], -.25)]):
        info,v,f,_=diagnostic_scene()
        # Move only the middle trunk; the first four vertices are the ground and each trunk has equal size.
        per_tree=(len(v)-4)//len(info['trees'])
        v[-per_tree:,1] += obstacle_y
        trees=np.asarray(info['trees'],float); trees[-1,1]+=obstacle_y
        rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        def transform(points): return (np.asarray(points)@rot.T+shift).tolist()
        v=np.asarray(transform(v),np.float32)
        trees[:,:2]=np.asarray(transform(np.c_[trees[:,:2],np.zeros(len(trees))]))[:,:2]
        info.update(profile='diagnostic',scene_id=f'executor_dev_diag_{index}',trees=trees.tolist(),
                    variation=dict(rotation_radians=angle,translation=shift,blocking_trunk_y=obstacle_y))
        definitions=[(t['goal_id'],[0.,0.,2.],t['target_world'],0.,t['arrival_yaw']) for t in diagnostic_tasks()]
        definitions += [('forward_far',[0.,0.,2.],[1.5,0.,2.],-.35,0.),
                        ('ascending',[0.,0.,2.],[1.,0.,2.3],.35,0.),
                        ('descending',[0.,0.,2.],[1.,0.,1.7],0.,0.),
                        ('reverse_bypass',[5.5,0.,2.],[0.,0.,2.],np.pi/2,np.pi)]
        entries=[]; witnesses=[]
        for name,start,target,yaw,end_yaw in definitions:
            entries.append(episode(name,transform(start),transform(target),yaw+angle,end_yaw+angle))
            path=[start,target]
            if 'bypass' in name:
                path=[[0.,0.,2.],[1.,-1.5,2.],[4.5,-1.5,2.],[5.5,0.,2.]]
                if name=='reverse_bypass': path=path[::-1]
            witnesses.append(transform(path))
        map_id=f'diagnostic/{index:03d}'
        export(output/map_id,info,v,f,entries,witnesses)
        records.append(dict(map_id=map_id,profile='diagnostic',tasks=8))
    # Native geometry is copied byte-for-byte. Selection uses geometry only, before any flight.
    for seed in (202610240,202610241):
        source=source_maps/f'native/{seed}'
        info=json.loads((source/'map.json').read_text())
        with np.load(source/'scene_mesh.npz') as mesh: v,f=mesh['vertices'],mesh['faces']
        with np.load(source/'terrain.npz') as t:
            terrain=RegularGridInterpolator((t['xs'],t['ys']),t['heights'])
        truth=dict(mesh=(v,f)); entries=[]; witnesses=[]; candidates=[]
        mids=[(a['y']+b['y'])/2 for a,b in zip(info['rows'][:-1],info['rows'][1:])]
        for n in range(48):
            row=n%len(mids); x=2.+(n//len(mids)%6)*1.25; distance=1.5 if n<24 else 2.
            a=np.array([x,mids[row]]); b=a+[distance,0.]
            # Alternate direction without changing the forest or removing branches.
            if n%2: a,b=b,a
            start=np.r_[a,float(terrain(a).item())+2.]
            target=np.r_[b,float(terrain(b).item())+2.]
            path=sample_polyline(np.array([start,target]))
            clear=not truth_collision(path,truth,INFLATION,VERTICAL_RADIUS)
            candidates.append(dict(candidate=n,row_gap=row,start=start.tolist(),target=target.tolist(),witness_clear=clear))
            if clear:
                yaw=float(np.arctan2(*(target-start)[:2][::-1]))
                name=f'native_{len(entries):02d}'
                entries.append(episode(name,start,target,yaw+(.35 if n%3==0 else 0.),yaw))
                witnesses.append([start.tolist(),target.tolist()])
                if len(entries)==8: break
        folder=output/f'native/{seed}'; folder.mkdir(parents=True)
        dump(folder/'candidate_selection.json',dict(policy='first eight clear straight witnesses in fixed candidate order; no flight outcomes used',candidates=candidates))
        if len(entries)!=8: raise ValueError(f'Only {len(entries)}/8 native witnesses for frozen seed {seed}; no replacement seed')
        for name in ('scene.usd','tex.jpg','terrain.npz'): shutil.copy2(source/name,folder/name)
        info.update(profile='native_executor',source_preview=str(source),
                    source_mesh_sha256=hashlib.sha256((source/'scene_mesh.npz').read_bytes()).hexdigest(),
                    limitations='Short geometry-feasible segments; no language, full row transition or withheld-scene generalization')
        export(folder,info,v,f,entries,witnesses)
        # Preserve the original binary as well as numerical geometry.
        shutil.copy2(source/'scene_mesh.npz',folder/'scene_mesh.npz')
        records.append(dict(map_id=f'native/{seed}',profile='native_executor',tasks=8))
    dump(output/'suite.json',dict(schema='executor_development_40_v1',expected_tasks=40,
        maps=records,split='development',gate=dict(minimum_passed=36,maximum_truth_margin_violations=0),
        scope='24 cylinder diagnostics + 16 native short segments; five maps, not forty independent scenes',
        files_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(output.rglob('*')) if p.is_file()}))
    dump(output/'index.json',[r|dict(task='executor_development',split='development') for r in records])
    lines=['# 冻结的 40 个执行器开发任务','','24 个圆柱诊断任务 + 16 个原生森林短距离任务。共 5 张图，不是 40 个独立场景。',
           '原生森林保留枝干和地形；先按固定候选顺序选择几何可行的短段，再运行全部任务。可行性不代表飞行成功。','',
           '| 地图 | 任务数 | 图示 | 任务 |','|---|---:|---|---|']
    for r in records:
        folder=output/r['map_id'];lines.append(f'| {r["map_id"]} | 8 | [tasks.png]({folder}/tasks.png) | [episodes.json]({folder}/episodes.json) |')
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return output
