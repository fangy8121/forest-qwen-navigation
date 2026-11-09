"""短路线的障碍密度、上下几何对照及连续 XYZ 采集约束。"""
import json
import numpy as np
import pytest
from forest_generator.generator import DEFAULT_CONFIG, layout_config, sample_forest, add_task, mesh_forest, describe
from forest_generator.maps import route_for_map, Clearance
from forest_generator.terrain import Terrain
from omni_drones.navigation.forest.esdf import ESDF, gate_crossings
from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
from omni_drones.navigation.forest.evaluation import mesh_collision
from omni_drones.navigation.forest.geometry import INFLATION, VERTICAL_RADIUS


def short_scene(task, variant='center', reverse=False, seed=2):
    config=layout_config(json.loads(DEFAULT_CONFIG.read_text()),'short')
    config['terrain'].update(amplitude=0.,amplitude2=0.,slope_range=0.)
    terrain=Terrain(config['terrain'],config['forest_size'],np.random.default_rng(0))
    rows=[dict(y=0.,spacing=6.),dict(y=6.,spacing=6.)]
    trees,start,goal,checks=add_task([],rows,config,task,0,seed,'hard',variant,terrain,reverse)
    return config,terrain,rows,trees,start,goal,checks


def test_short_layout_keeps_source_and_places_obstacles_inside_route():
    original=json.loads(DEFAULT_CONFIG.read_text())
    config=layout_config(original,'short')
    assert original['forest_size'][0]==100 and config['forest_size'][0]==36
    assert layout_config(original,'long')==original
    for task in ('lateral','vertical','choice','mixed'):
        config,terrain,rows,trees,start,goal,checks=short_scene(task)
        xs=np.array([c['x'] for c in checks])
        assert 2<=len(xs)<=3
        assert xs.min()>start[0]+4 and xs.max()<goal[0]-4
        assert np.diff(xs).min()>8
        variant=add_task([],rows,config,task,0,2,'hard','approach',terrain)
        assert checks==variant[3] and trees==variant[0]
        assert variant[1][0]==xs[0]-4


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_vertical_seed_determines_alternating_height_pattern(seed):
    scene=short_scene('vertical',seed=seed)
    checks=scene[-1]
    assert [check['kind'] for check in checks] == (['over','under','over'] if seed%2==0 else ['under','over','under'])
    for check in checks:
        level=check['logs'][0]['start'][2]
        assert level-check['reference_height'] == pytest.approx(-.65 if check['kind']=='over' else .65)
    again=short_scene('vertical',seed=seed)
    assert checks==again[-1] and scene[3]==again[3]


def test_legacy_vertical_reverse_has_same_xy_background_and_split_identity():
    over,under=short_scene('vertical'),short_scene('vertical',reverse=True)
    for a,b in zip(over[-1],under[-1]):
        assert 'solid' not in a and 'solid' not in b
        assert len(a['logs'])==len(b['logs'])==1
        for first,second in zip(a['logs'],b['logs']):
            for key in ('start','end'):
                np.testing.assert_array_equal(first[key][:2],second[key][:2])
                assert first[key][2]+second[key][2]==pytest.approx(2*a['reference_height'])
            assert first['radius']==second['radius']
            assert first['radius']==.25 and first['start'][2]==first['end'][2]
        assert a['kind']!=b['kind']
    assert over[3]==under[3]  # 承托横枝的竖直树干也保持相同。
    identities=[]
    for task,scene in [('vertical',over),('vertical',under)]:
        config,terrain,rows,trees,start,goal,checks=scene
        # describe also records sampled spacing statistics.
        background=[dict(position=[1.,0.],inserted=False,spacing=5.,offset=0.)]
        info=describe(2,config,background,rows,terrain,task,0,start,goal,checks)
        identities.append(info['scene_id'])
    assert identities[0]==identities[1]


@pytest.mark.parametrize('task,variant,reverse',[
    ('vertical','center',False),('vertical','center',True),('vertical','approach',False),('mixed','offset',False)])
def test_short_route_has_real_height_choices_and_safe_continuous_crossings(task,variant,reverse):
    config,terrain,rows,trees,start,goal,checks=short_scene(task,variant,reverse)
    logs=[log for c in checks for log in c.get('logs',[])]
    v,f,uv,n=mesh_forest(trees,terrain,logs)
    assert len(f)==len(uv)==len(n)
    np.testing.assert_allclose(np.linalg.norm(n,axis=-1),1.,atol=1e-6)
    reference,path,validation=route_for_map(trees,rows,terrain,v,f,config,task,0,start,goal,checks,DEFAULT_FLIGHT,True)
    assert validation['passed'] and validation['curve_clearance']>.75
    assert validation['route_length']<38 and validation['task_zone_fraction']>.55
    curve=path.remaining(0,reference[0])
    assert gate_crossings(curve,validation['task_gates']) is not None
    for opening in validation['openings']:
        if opening.get('layout')=='over':assert opening['above']
        if opening.get('layout')=='under':assert opening['below']
    for crossing in validation['height_crossings']:
        delta=np.array(crossing['z'])-crossing['reference_height']
        assert np.all((delta>.2)&(delta<.7)) if crossing['layout']=='over' else np.all((delta<-.2)&(delta>-.7))
        assert set(crossing['actions'])=={crossing['layout']}
    for check in checks:
        if 'logs' in check:
            nominal=np.array([[check['x'],3.,check['reference_height']]])
            assert mesh_collision(nominal,dict(mesh=(v,f)),INFLATION,VERTICAL_RADIUS,padding=.05)
    # 障碍处没有强制高度门；名义高度直行被真实网格阻挡。
    assert all('z_min' not in g and 'z_max' not in g for g in validation['task_gates'] if g['kind']=='obstacle')
    straight=np.c_[np.linspace(start[0],goal[0],100),np.full(100,3.),np.full(100,2.5)]
    assert Clearance(v,f,terrain).distance(straight)<.75
    if task=='vertical':assert gate_crossings(straight,validation['task_gates']) is not None


@pytest.mark.parametrize('seed',[0,1,2])
def test_short_terrain_has_relief_and_wooded_borders(seed):
    config=layout_config(json.loads(DEFAULT_CONFIG.read_text()),'short')
    trees,rows,terrain=sample_forest(seed,config)
    core,core_rows,_=sample_forest(seed,dict(config,forest_margin=0.))
    assert rows==core_rows and trees[:len(core)]==core
    xy=np.array([t['position'] for t in trees])
    size=np.array(config['forest_size'])
    assert (xy.min(0)<-3).all() and (xy.max(0)>size+3).all()
    assert xy[:,0].min()>terrain.xs[0] and xy[:,0].max()<terrain.xs[-1]
    assert xy[:,1].min()>terrain.ys[0] and xy[:,1].max()<terrain.ys[-1]
    inside=(terrain.vertices[:,0]>=0)&(terrain.vertices[:,0]<=size[0])&(terrain.vertices[:,1]>=0)&(terrain.vertices[:,1]<=size[1])
    xyz=terrain.vertices[inside]
    plane=np.c_[xyz[:,:2],np.ones(len(xyz))]
    residual=xyz[:,2]-plane@np.linalg.lstsq(plane,xyz[:,2],rcond=None)[0]
    assert np.ptp(xyz[:,2])>1. and np.std(residual)>.15


def test_single_branch_keeps_both_safe_vertical_choices():
    config,terrain,rows,trees,start,goal,checks=short_scene('vertical')
    for check in checks:
        check['logs']=check['logs'][:1]
        log=check['logs'][0]
        log['start'][2]=log['end'][2]=check['reference_height']
        log['radius']=.3
    v,f,_,_=mesh_forest(trees,terrain,[log for c in checks for log in c['logs']])
    _,_,validation=route_for_map(trees,rows,terrain,v,f,config,'vertical',0,start,goal,checks,DEFAULT_FLIGHT,True)
    assert all(o['above'] and o['below'] for o in validation['openings'])
    # 两侧不仅有零散空体素，还各有一条从起点到终点的完整路线。
    field=ESDF(v,f,terrain,validation['route_bounds'],height=config['flight_height'])
    for limit,z in [('z_min',3.3),('z_max',1.7)]:
        gates=[dict(g) for g in validation['task_gates']]
        for gate in gates:
            if gate['kind']=='obstacle':gate[limit]=z
        route=field.route(np.r_[start,2.5],np.r_[goal,2.5],gates)
        assert gate_crossings(route,gates) is not None


def test_height_gate_checks_every_crossing_including_reverse_motion():
    gate=dict(x=0.,y_min=-1.,y_max=1.,z_min=2.,z_max=3.)
    assert gate_crossings([[-1,0,2.5],[1,0,2.5]],[gate]) is not None
    assert gate_crossings([[-1,0,2.5],[1,0,2.5],[-1,0,0.]],[gate]) is None
    assert gate_crossings([[-2,0,1],[-1,0,1]],[gate],require_all=False)==[[]]
