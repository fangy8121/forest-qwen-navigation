"""移植 V3 树行采样和 spawnTree，直接使用原始树模板。"""
import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import numpy as np
from omni_drones.navigation.forest.mesh import read_collada
from .terrain import Terrain

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'assets'
DEFAULT_CONFIG=ROOT/'config.json'


def layout_config(config, layout):
    """短路线保留山地起伏与周边树林；长路线沿用输入配置。"""
    config=deepcopy(config)
    if layout=='short':
        config['forest_size'][0]=36.
        config['flight_height']=2.5
        config['forest_margin']=6.
        config['terrain'].update(amplitude=1.2,amplitude2=4.,frequency2=.08,slope_range=.06,padding=8.)
        for difficulty,count in [('easy',[2,2]),('medium',[2,3]),('hard',[3,3])]:
            config['difficulty'][difficulty]['count']=count
    return config


def clipped(rng,specification,gamma=False): #clipped 采样，返回原始值和裁剪值。
    a,b,low,high=specification
    raw=float(rng.gamma(a,b) if gamma else rng.normal(a,b))
    return raw,float(np.clip(raw,low,high))


def truncated(rng,specification,gamma=False): 
    a,b,low,high=specification
    while True:
        value=float(rng.gamma(a,b) if gamma else rng.normal(a,b))
        if low<=value<=high:
            return value


@lru_cache(maxsize=1)
def template():
    vertices,faces,uv,normals=read_collada(SOURCE/'untitled.dae',with_normals=True)
    vertices[:,2]+=5.  # 原 main() 将 [-5,5] 修正为 [0,10]。
    return vertices,faces,uv,normals


def branch_parameters(rng,azimuth,config): #生成枝干参数，返回字典列表。
    count=int(np.floor(truncated(rng,config['branch_count'])+.5))
    return [dict(angle=truncated(rng,config['branch_angle']),
        length=truncated(rng,config['branch_length_gamma'],True),
        height=truncated(rng,config['branch_height']),
        azimuth=float(azimuth.integers(0,3601)*np.pi/1800)) for _ in range(count)]


def sample_forest(seed,config):
    # 原代码给各分布独立随机流；任务插树不改变基础树林的随机序列。
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    rx,ry,dy,stem,branches,azimuth,terrain_rng,border_rng=streams
    layout=config.get('row_layout')
    # A separate stream leaves all legacy samples exactly unchanged when disabled.
    extent_rng=np.random.default_rng(np.random.SeedSequence([seed,47119]))
    previous_length=None
    if layout:
        low,high=layout['length_range']
        if not (0<low<=high and 0<=layout.get('correlation',.35)<1
                and layout.get('start_jitter',2.)>=0):
            raise ValueError('Invalid row_layout length, correlation or start jitter')
    terrain=Terrain(config['terrain'],config['forest_size'],terrain_rng)
    margin=config.get('forest_margin',0.)
    trees=[];rows=[];y=0.
    while y<config['forest_size'][1]:
        row=len(rows);x=0.;end=config['forest_size'][0]
        if layout:
            length=float(extent_rng.uniform(low,high))
            if previous_length is not None:
                correlation=layout.get('correlation',.35)
                length=correlation*previous_length+(1-correlation)*length
            previous_length=length
            x=float(extent_rng.uniform(0,2*layout.get('start_jitter',2.)))
            start=x;end=start+length
        while x<end:
            xraw,xstep=clipped(rx,config['tree_spacing_gamma'],True)
            yraw,ydev=clipped(dy,config['row_offset'])
            draw,diameter=clipped(stem,config['stem_diameter'])
            x+=xstep  # 原程序保留最后一棵跨越 xsize 的树。
            # Variable rows have an explicit nominal end; do not overshoot it.
            if layout and x>end:break
            trees.append(dict(row=row,position=[x,y+ydev],diameter=diameter,radius=diameter/2,
                height=10.,inserted=False,spacing_raw=xraw,spacing=xstep,offset_raw=yraw,offset=ydev,
                diameter_raw=draw,branches=branch_parameters(branches,azimuth,config)))
        raw,step=clipped(ry,config['row_spacing'])
        record=dict(row=row,y=y,spacing_raw=raw,spacing=step)
        if layout:record.update(start_x=start,end_x=end,length_m=length)
        rows.append(record);y+=step
    # 独立随机流补齐外围树林，保留任务区原有的树行和种子对应关系。
    if margin:
        y=-margin
        while y<config['forest_size'][1]+margin:
            x=-margin
            while x<config['forest_size'][0]+margin:
                raw,step=clipped(border_rng,config['tree_spacing_gamma'],True)
                x+=step
                if x>config['forest_size'][0]+margin:
                    break
                yraw,offset=clipped(border_rng,config['row_offset'])
                yy=y+offset
                if -1.5<x<config['forest_size'][0]+1.5 and -1.5<yy<config['forest_size'][1]+1.5:
                    continue
                if min(np.linalg.norm(np.array(t['position'])-[x,yy]) for t in trees)<1.5:
                    continue
                draw,diameter=clipped(border_rng,config['stem_diameter'])
                trees.append(dict(row=None,position=[x,yy],diameter=diameter,radius=diameter/2,
                    height=10.,inserted=False,spacing_raw=raw,spacing=step,offset_raw=yraw,offset=offset,
                    diameter_raw=draw,branches=branch_parameters(border_rng,border_rng,config)))
            y+=clipped(border_rng,config['row_spacing'])[1]
    return trees,rows,terrain


def mesh_forest(trees,terrain,obstacles=()):
    tv,tf,uv,normal=template()
    vertices=[terrain.vertices];faces=[terrain.faces]
    # 与源代码一样，地面使用模板选定的四个 UV；树干和枝干保持完整原 UV。
    guv=np.array([[uv[0,0],uv[1,1],uv[2,2]],[uv[4,0],uv[2,2],uv[1,1]]])
    tex=[np.concatenate([np.tile(guv[0],(len(terrain.faces)//2,1,1)),
                         np.tile(guv[1],(len(terrain.faces)//2,1,1))])]
    triangle=terrain.vertices[terrain.faces]
    gn=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0])
    gn/=np.linalg.norm(gn,axis=1)[:,None]
    normals=[np.repeat(gn[:,None,:],3,axis=1)]
    offset=len(vertices[0])
    def add(scale,rotation,position):
        nonlocal offset
        vertices.append((tv*scale)@rotation.T+position);faces.append(tf+offset)
        n=(normal/scale)@rotation.T;n/=np.linalg.norm(n,axis=-1,keepdims=True)
        normals.append(n);tex.append(uv);offset+=len(tv)
    for tree in trees:
        x,y=tree['position'];z=float(terrain([[x,y]])[0]);tree['ground_z']=z
        for branch in tree['branches']:
            a,r=branch['angle'],branch['azimuth'];ca,sa=np.cos(a),np.sin(a);cr,sr=np.cos(r),np.sin(r)
            rx=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
            rz=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
            add([.2,.2,branch['length']/10],rz@rx,[x,y,z+branch['height']])
        add([tree['diameter'],tree['diameter'],tree.get('height',10.)/10.],np.eye(3),[x,y,z])
    for obstacle in obstacles:
        start,end=np.asarray(obstacle['start']),np.asarray(obstacle['end'])
        axis=end-start;length=np.linalg.norm(axis);axis/=length
        tangent=np.cross(axis,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
        rotation=np.column_stack([tangent,np.cross(axis,tangent),axis])
        add([2*obstacle['radius'],2*obstacle['radius'],length/10],rotation,start)
    return (np.concatenate(vertices).astype(np.float32),np.concatenate(faces).astype(np.int32),
            np.concatenate(tex).astype(np.float32),np.concatenate(normals).astype(np.float32))


def add_task(trees,rows,config,task,corridor,seed,difficulty=None,variant='center',terrain=None,reverse=False):
    """自主绕行的开口由几何决定；blocked 才封闭整条通道。"""
    trees=deepcopy(trees)
    y=(rows[corridor]['y']+rows[corridor+1]['y'])/2
    start=np.array([2.,y]);goal=np.array([config['forest_size'][0]-2,y])
    checkpoints=[];rng=np.random.default_rng(np.random.SeedSequence([seed,10000+corridor]))
    o=config['task_obstacles']
    profile=config['difficulty'][difficulty] if difficulty else None
    if profile:
        count=int(rng.integers(profile['count'][0],profile['count'][1]+1))
        if task in ('slalom','vertical_slalom','lateral','vertical'):count=max(2,count)
        if task=='mixed':count=3
        if task=='choice':count=2
        if task=='blocked':count=1
        length=config['forest_size'][0]
        jitter=min(2.,length*.025)
        xs=(rng.uniform(.25*length,.75*length,1) if count==1 else
            np.linspace(.2*length,.8*length,count)+rng.uniform(-jitter,jitter,count))
        fractions=(xs-start[0])/(goal[0]-start[0])
    else:
        fractions=o['fractions']
    if task=='survey':fractions=[]
    def insert(x,y,diameter,branches):
        trees.append(dict(row=None,position=[float(x),float(y)],diameter=float(diameter),radius=float(diameter/2),
            height=10.,inserted=True,branches=branches))
    for i,fraction in enumerate(fractions):
        kind=task
        if task=='slalom':kind='left' if i%2==0 else 'right'
        if task=='vertical_slalom':kind='over' if i%2==0 else 'under'
        if task=='lateral':kind='left' if (i+seed)%2==0 else 'right'
        if task=='vertical':kind='over' if (i+seed+reverse)%2==0 else 'under'
        if task=='mixed':kind=(('left','over','under') if seed%2==0 else ('right','under','over'))[i%3]
        side=1 if kind=='left' else -1 if kind=='right' else 0
        diameter=rng.uniform(*profile['diameter']) if profile else o['diameter']
        if task=='choice':diameter=rng.uniform(*o['choice_diameter'])
        offset=rng.uniform(*profile['offset']) if profile else o['side_offset']
        x=float(start[0]+fraction*(goal[0]-start[0]));obstacle_y=y-side*offset
        check=dict(x=x,obstacle_y=obstacle_y,side=side,kind=kind)
        if kind in ('over','under'):
            low_y,high_y=rows[corridor]['y']-.8,rows[corridor+1]['y']+.8
            height=float(terrain([[x,y]])[0]+config['flight_height'])
            direction=-1 if kind=='over' else 1
            skew=rng.uniform(-.3,.3)
            # 单根横木让名义高度需要调整，但上下两侧都允许按实际净空绕行。
            level=height+direction*.65
            check['logs']=[dict(start=[x-skew,low_y,level],
                end=[x+skew,high_y,level],radius=.25)]
            for edge in (rows[corridor]['y'],rows[corridor+1]['y']):
                insert(x,edge,.65,[])
            check['reference_height']=height
        elif task=='blocked':
            for yy in np.linspace(rows[corridor]['y'],rows[corridor+1]['y'],
                                  int(np.ceil((rows[corridor+1]['y']-rows[corridor]['y'])/1.2))+1):
                insert(x,yy,diameter,[])
        else:
            insert(x,obstacle_y,diameter,[] if task=='choice' else branch_parameters(rng,rng,config))
            if profile and side:
                edge=rows[corridor if side>0 else corridor+1]['y']
                for yy in np.linspace(edge,obstacle_y,int(np.ceil(abs(edge-obstacle_y)/1.2))+1)[:-1]:
                    insert(x,yy,diameter,[])
        checkpoints.append(check)
    for t in config['insert_trees']:
        diameter=t.get('diameter',o['diameter'])
        trees.append(dict(row=None,position=list(t['position']),diameter=diameter,radius=diameter/2,
            height=10.,inserted=True,branches=branch_parameters(rng,rng,config)))
    if variant!='center':
        pose_rng=np.random.default_rng(np.random.SeedSequence([seed,20000+corridor]))
        start[0]=(checkpoints[0]['x']-4. if variant=='approach' and checkpoints else pose_rng.uniform(2.,6.))
        start[1]+=pose_rng.uniform(-.3,.3)
    return trees,start,goal,checkpoints


def forest_scene_id(seed,config):
    return hashlib.sha256(json.dumps(dict(seed=seed,config=config),sort_keys=True).encode()).hexdigest()[:20]


def describe(seed,config,trees,rows,terrain,task,corridor,start,goal,checkpoints):
    plane=np.linalg.lstsq(np.c_[terrain.vertices[:,:2],np.ones(len(terrain.vertices))],terrain.vertices[:,2],rcond=None)[0]
    residual=terrain.vertices[:,2]-np.c_[terrain.vertices[:,:2],np.ones(len(terrain.vertices))]@plane
    source_hashes=json.loads((SOURCE/'provenance.json').read_text())
    source_hashes.update({name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                         for name in ('untitled.dae','tex.jpg')})
    group=forest_scene_id(seed,config)
    return dict(scene_id=group,task=task,parameters=dict(seed=seed,config=config),trees=trees,rows=rows,
        corridor=corridor,checkpoints=checkpoints,start_xy=start.tolist(),goal_xy=goal.tolist(),
        bounds=[0.,config['forest_size'][0],0.,config['forest_size'][1]],height_mode='terrain_relative',
        initial_yaw=0.,generator='v6_thin_level_branches',source='forest_generator/generator.py',
        source_files_sha256=source_hashes,
        terrain_statistics=dict(height_min=float(terrain.heights.min()),height_max=float(terrain.heights.max()),
            height_rms=float(np.sqrt(np.mean(residual**2))),plane=plane.tolist(),
            realized_slope=terrain.slope.tolist(),realized_amplitude2=terrain.amplitude2),
        spacing_statistics=dict(row_mean=float(np.mean(np.diff([r['y'] for r in rows]))),
            row_std=float(np.std(np.diff([r['y'] for r in rows]))),
            within_row_mean=float(np.mean([t['spacing'] for t in trees if not t['inserted']])),
            offset_std=float(np.std([t['offset'] for t in trees if not t['inserted']]))))
