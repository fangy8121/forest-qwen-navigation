"""离线导出原模板森林、地形跟随采集路线和可核对的采样参数。"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from omni_drones.navigation.forest.mesh import read_collada
from omni_drones.navigation.forest.scene import recorded_scene_splits
from omni_drones.navigation.forest.esdf import ESDF,gate_crossings
from omni_drones.navigation.forest.geometry import REFERENCE_CLEARANCE
from .generator import SOURCE,sample_forest,add_task,mesh_forest,describe,forest_scene_id

TASKS=('lateral','vertical','choice','mixed')


def write_index(root):
    root=Path(root);records=[]
    for file in sorted(root.rglob('map.json')):
        info=json.loads(file.read_text())
        if info['validation']['passed']:
            records.append(dict(map_id=str(file.parent.relative_to(root)),scene_id=info['scene_id'],
                task=info['task'],split=info['split']))
    (root/'index.json').write_text(json.dumps(records,indent=2)+'\n')


def write_usd(path, vertices, faces, uvs=None, texture=None, normals=None):
    def tuples(array):
        return ',\n'.join('('+', '.join(f'{float(v):.7g}' for v in row)+')' for row in array)
    material = ''
    binding = ''
    if texture:
        binding = 'rel material:binding = </Forest/Bark>'
        material = '''def Material "Bark" {
            token outputs:surface.connect = </Forest/Bark/Surface.outputs:surface>
            def Shader "Surface" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Forest/Bark/Texture.outputs:rgb>
                token outputs:surface
            }
            def Shader "Texture" {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @'''+texture+'''@
                float2 inputs:st.connect = </Forest/Bark/UV.outputs:result>
                float3 outputs:rgb
            }
            def Shader "UV" {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }
        }'''
    st = '' if uvs is None else 'texCoord2f[] primvars:st = ['+tuples(uvs.reshape(-1,2))+'] (interpolation = "faceVarying")'
    normal_text = '' if normals is None else 'normal3f[] normals = ['+tuples(normals.reshape(-1,3))+'] (interpolation = \"faceVarying\")'
    path.write_text('''#usda 1.0
(
    defaultPrim = "Forest"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Forest" {
    def Mesh "Mesh" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]) {
        point3f[] points = ['''+tuples(vertices)+''']
        int[] faceVertexCounts = ['''+', '.join(['3']*len(faces))+''']
        int[] faceVertexIndices = ['''+', '.join(map(str,faces.ravel()))+''']
        uniform token subdivisionScheme = "none"
        uniform bool doubleSided = true
        bool physics:collisionEnabled = true
        token physics:approximation = "none"
        color3f[] primvars:displayColor = [(0.38, 0.43, 0.29)]
        '''+st+'\n'+normal_text+'\n'+binding+'\n    }\n'+material+'\n}\n')


class Clearance:
    def __init__(self,vertices,faces,terrain,required=REFERENCE_CLEARANCE):
        import trimesh
        self.mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        self.terrain,self.required=terrain,float(required)

    def distance(self,points):
        # 对完整三角网格查询，不把随地形升降的路线当作固定高度 XY 切片。
        values=[]
        for batch in np.array_split(points,max(1,int(np.ceil(len(points)/256)))):
            values.append(self.mesh.nearest.on_surface(batch)[1].min())
        return float(min(values))

    def __call__(self,points):
        return bool(np.min(points[:,2]-self.terrain(points))>.8 and self.distance(points)>self.required)


def crossings_on_route(route,checkpoints): 
    crossings=[]
    for check in checkpoints:
        values=[]
        for a,b in zip(route[:-1],route[1:]): 
            if (a[0]-check['x'])*(b[0]-check['x'])<=0 and abs(b[0]-a[0])>1e-9:
                y=a[1]+(b[1]-a[1])*(check['x']-a[0])/(b[0]-a[0]);values.append(float(y))
        if not values or any(check['side']*(y-check['obstacle_y'])<=.8 for y in values):
            raise ValueError('route does not pass obstacles on the requested side')
        crossings.append(values)
    return crossings


def route_for_map(trees,rows,terrain,vertices,faces,config,task,corridor,start,goal,checkpoints,flight,autonomous=False):
    from omni_drones.navigation.forest.planning import smooth_path
    bounds=[0.,config['forest_size'][0],rows[corridor]['y'],rows[corridor+1]['y']]
    center=(bounds[2]+bounds[3])/2
    active_checks=[] if task=='blocked' else checkpoints
    gates=[]
    vertical=any('logs' in c for c in checkpoints)
    for index,check in enumerate(active_checks):
        low,high=bounds[2:]
        if check['side']>0:low=check['obstacle_y']+.8
        elif check['side']<0:high=check['obstacle_y']-.8
        gate=dict(x=check['x'],y_min=low,y_max=high,kind='obstacle')
        gates.append(gate)
        # 回中平面随相邻障碍间距缩短，避免短路线的约束互相交叉。
        previous=active_checks[index-1]['x'] if index else start[0]
        following=active_checks[index+1]['x'] if index+1<len(active_checks) else goal[0]
        for x in (check['x']-min(6.,.4*(check['x']-previous)),
                  check['x']+min(6.,.4*(following-check['x']))):
            gate=dict(x=x,y_min=center-.4,y_max=center+.4,kind='center')
            if vertical:
                z=float(terrain([[x,center]])[0]+config['flight_height'])
                gate.update(z_min=z-.3,z_max=z+.3)
            gates.append(gate)
    gates.sort(key=lambda g:g['x'])
    height=config['flight_height']
    esdf=ESDF(vertices,faces,terrain,bounds,height=height)
    openings=[]
    ys=esdf.origin[1]+np.arange(esdf.free.shape[1])*esdf.resolution
    zs=esdf.origin[2]+np.arange(esdf.free.shape[2])*esdf.resolution
    for check in checkpoints:
        ix=int(round((check['x']-esdf.origin[0])/esdf.resolution))
        plane=esdf.free[ix]
        left=bool(plane[ys>check['obstacle_y']].any())
        right=bool(plane[ys<check['obstacle_y']].any())
        opening=dict(x=check['x'],left=left,right=right)
        if 'logs' in check:
            above=bool(plane[:,zs>check['reference_height']].any())
            below=bool(plane[:,zs<=check['reference_height']].any())
            opening.update(above=above,below=below,layout=check['kind'])
            # 开口如实记录；有多个安全方向时不按任务名封掉其中一个。
            if plane[:,abs(zs-check['reference_height'])<=esdf.resolution/2].any():
                raise ValueError('height obstacle leaves a level-flight opening')
        openings.append(opening)
        if task=='blocked' and plane.any():
            raise ValueError('blocked task still has an opening')
        if task=='choice' and not (left and right):
            raise ValueError('choice task needs openings on both sides')
        if autonomous and check['side'] and (right if check['side']>0 else left):
            raise ValueError('geometry does not close the unwanted side')
    collection_goal=np.array([checkpoints[0]['x']-3.,center]) if task=='blocked' else goal
    endpoints=np.c_[[start,collection_goal],terrain(np.array([start,collection_goal]))+height]
    if task=='choice':
        for side in (1,-1):
            alternatives=[dict(g) for g in gates]
            for gate,check in zip((g for g in alternatives if g['kind']=='obstacle'),checkpoints):
                gate['y_min' if side>0 else 'y_max']=check['obstacle_y']+side*.8
            esdf.route(*endpoints,alternatives)
    reference=esdf.route(*endpoints,gates)
    side_checks=[c for c in active_checks if c['side']]
    crossings=crossings_on_route(reference,side_checks)
    clearance=Clearance(vertices,faces,terrain)
    def safe(points):
        return esdf.clear(points) and gate_crossings(points,gates,require_all=False) is not None
    path=smooth_path(reference,flight,safe)
    xyz=path.remaining(0.,reference[0])
    curve_clearance=clearance.distance(xyz)
    if curve_clearance<=REFERENCE_CLEARANCE:
        raise ValueError('3D trajectory fails mesh clearance')
    gate_values=gate_crossings(xyz,gates)
    if gate_values is None:
        raise ValueError('trajectory fails local bypass/return gates')
    task_distance=np.min(abs(xyz[:,0,None]-[c['x'] for c in checkpoints]),axis=1) if checkpoints else np.full(len(xyz),np.inf)
    height_crossings=[]
    for check in checkpoints:
        if 'logs' not in check:continue
        heights=[]
        for a,b in zip(xyz[:-1],xyz[1:]):
            if (a[0]-check['x'])*(b[0]-check['x'])<=0 and abs(b[0]-a[0])>1e-9:
                heights.append(float(a[2]+(b[2]-a[2])*(check['x']-a[0])/(b[0]-a[0])))
        height_crossings.append(dict(x=check['x'],layout=check['kind'],
            actions=['over' if z>check['reference_height'] else 'under' for z in heights],
            z=heights,reference_height=check['reference_height']))
    return reference,path,dict(passed=True,crossings=crossings,curve_crossings=crossings_on_route(xyz[:,:2],side_checks),curve_clearance=curve_clearance,
        minimum_height_above_ground=float(np.min(xyz[:,2]-terrain(xyz))),
        maximum_height_above_ground=float(np.max(xyz[:,2]-terrain(xyz))),
        route_length=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),duration=path.duration,
        route_bounds=bounds,task_gates=gates,gate_crossings=gate_values,
        openings=openings,expected_outcome='stop_before_barrier' if task=='blocked' else 'reach_goal',
        planner='3d_esdf_astar',resolution=esdf.resolution,
        obstacle_count=sum(t['inserted'] for t in trees)+sum(len(c.get('logs',[])) for c in checkpoints),
        task_obstacle_groups=len(checkpoints),task_zone_fraction=float(np.mean(task_distance<=4.)),
        height_crossings=height_crossings)


def export(folder,info,vertices,faces,uvs=None,normals=None,texture=None):
    split=recorded_scene_splits([info])[info['scene_id']]
    folder.mkdir(parents=True,exist_ok=False)
    info.update(geometry_id=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()[:20],
        geometry='triangle_mesh',mesh_file='scene_mesh.npz',usd_file='scene.usd',
        source_sha256=info.get('original_file_sha256',hashlib.sha256(Path(__file__).with_name('generator.py').read_bytes()).hexdigest()),
        split=split)
    if texture:
        shutil.copy2(texture,folder/'tex.jpg');texture='tex.jpg'
    write_usd(folder/'scene.usd',vertices,faces,uvs,texture,normals)
    np.savez_compressed(folder/'scene_mesh.npz',vertices=vertices,faces=faces)
    (folder/'map.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')


def build_map(root,seed,config,task,flight,difficulty=None,variant='center',reverse=False,split=None):
    name=f'{seed:06d}'+(f'_{variant}' if variant!='center' else '')+('_reverse' if reverse else '')
    folder=Path(root)/(difficulty or '')/task/name
    if folder.exists():raise FileExistsError(folder)
    scene_id=forest_scene_id(seed,config)
    existing=[json.loads(file.read_text()) for file in sorted(Path(root).rglob('map.json'))]
    split=recorded_scene_splits([*existing,dict(scene_id=scene_id,split=split)])[scene_id]
    base,rows,terrain=sample_forest(seed,config)
    # 按距中心的远近选择可巡检的通道；不重采样森林来迎合任务。
    candidates=sorted((i for i in range(len(rows)-1)
        if rows[i]['y']>=0 and rows[i+1]['y']<=config['forest_size'][1]),
        key=lambda i:abs((rows[i]['y']+rows[i+1]['y'])/2-config['forest_size'][1]/2))
    errors=[]
    for corridor in candidates:
        trees,start,goal,checks=add_task(base,rows,config,task,corridor,seed,difficulty,variant,terrain,reverse)
        vertices,faces,uvs,normals=mesh_forest(trees,terrain,[log for c in checks for log in c.get('logs',[])])
        try:
            reference,path,validation=route_for_map(trees,rows,terrain,vertices,faces,config,task,
                corridor,start,goal,checks,flight,autonomous=difficulty is not None)
            # 旧 over/under 接口保留配对检查；vertical 只验证种子指定的排列。
            if task in ('over','under'):
                other='under' if task=='over' else 'over'
                paired,ps,pg,pc=add_task(base,rows,config,other,corridor,seed,difficulty,variant,terrain,not reverse)
                pv,pf,_,_=mesh_forest(paired,terrain,[log for c in pc for log in c.get('logs',[])])
                route_for_map(paired,rows,terrain,pv,pf,config,other,corridor,ps,pg,pc,flight)
        except ValueError as error:
            errors.append(f'row {corridor}: {error}');continue
        info=describe(seed,config,trees,rows,terrain,task,corridor,start,goal,checks)
        task_goal=np.r_[goal,terrain([goal])[0]+config['flight_height']]
        yaw_offset=(np.random.default_rng(seed).uniform(-.17,.17) if variant!='center' else 0.)
        info.update(start=reference[0].tolist(),goal=task_goal.tolist(),reference=reference.tolist(),
            collection_goal=reference[-1].tolist(),difficulty=difficulty or 'legacy',collection_variant=variant,
            obstacle_sequence=[c['kind'] for c in checks],reverse_pattern=reverse,
            initial_yaw=float(np.arctan2(*(reference[1,:2]-reference[0,:2])[::-1])+yaw_offset),
            validation=validation,flight_parameters=dict(flight),trajectory_file='reference_trajectory.npz',
            terrain_file='terrain.npz',split=split)
        export(folder,info,vertices,faces,uvs,normals,SOURCE/'tex.jpg')
        path.save(folder/info['trajectory_file'])
        np.savez_compressed(folder/'terrain.npz',xs=terrain.xs,ys=terrain.ys,heights=terrain.heights)
        for name,records in [('rows',rows),('trees',[dict(x=t['position'][0],y=t['position'][1],**{k:v for k,v in t.items() if k not in ('branches','position')}) for t in trees])]:
            keys=list(dict.fromkeys(k for r in records for k in r))
            with (folder/(name+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)
        write_index(root)
        print(json.dumps(dict(task=task,variant=variant,corridor=corridor,
            obstacle_groups=validation['task_obstacle_groups'],route_length=validation['route_length'],
            duration=path.duration,clearance=validation['curve_clearance'])),flush=True)
        return folder
    raise ValueError('; '.join(errors))


def convert_reference(source,folder):
    source=Path(source).resolve();v,f,uv,n=read_collada(source,with_normals=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    info=dict(scene_id=digest[:20],task='reference',source=str(source),parameters={},
        validation=dict(passed=False,reason='geometry reference'),generator='original_dae',original_file_sha256=digest)
    texture=source.parent/'tex.jpg'
    export(Path(folder),info,v,f,uv,n,texture if texture.is_file() else SOURCE/'tex.jpg')
    return dict(vertices=len(v),triangles=len(f),bounds=[v.min(0).tolist(),v.max(0).tolist()])
