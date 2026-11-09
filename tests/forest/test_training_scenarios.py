import json
import numpy as np
import pytest
from forest_generator.generator import DEFAULT_CONFIG, sample_forest, add_task
from forest_generator.maps import write_index
from omni_drones.navigation.forest.geometry import corridor_polygon, inside_polygon, local_to_world
from omni_drones.navigation.forest.teacher import teacher_path


def test_random_tasks_and_pose_variants_are_reproducible():
    config=json.loads(DEFAULT_CONFIG.read_text())
    base,rows,_=sample_forest(1,config)
    a=add_task(base,rows,config,'left',1,1,'easy')
    b=add_task(base,rows,config,'left',1,1,'easy','approach')
    assert a[0]==b[0] and a[3]==b[3]
    assert b[1][0]==a[3][0]['x']-4.
    assert abs(b[1][1]-a[1][1])<=.3
    assert len(a[3])==1
    hard=add_task(base,rows,config,'slalom',1,1,'hard')
    assert 3<=len(hard[3])<=5
    assert all(c['side']==(1 if i%2==0 else -1) for i,c in enumerate(hard[3]))
    blocked=add_task(base,rows,config,'blocked',1,1,'easy')
    inserted=[t for t in blocked[0] if t['inserted']]
    assert min(t['position'][1] for t in inserted)==rows[1]['y']
    assert max(t['position'][1] for t in inserted)==rows[2]['y']
    assert not any(t['inserted'] for t in base)
    choice=add_task(base,rows,config,'choice',1,1,'easy')
    obstacle=next(t for t in choice[0] if t['inserted'])
    assert .3<=obstacle['diameter']<=.4 and not obstacle['branches']


@pytest.mark.parametrize('difficulty',['easy','medium','hard'])
def test_choice_has_two_reproducible_obstacles(difficulty):
    config=json.loads(DEFAULT_CONFIG.read_text())
    base,rows,_=sample_forest(1,config)
    trees,_,_,checks=add_task(base,rows,config,'choice',1,1,difficulty)
    obstacles=[t for t in trees if t['inserted']]
    assert len(obstacles)==len(checks)==2
    assert all(c['side']==0 for c in checks)
    assert all(.3<=t['diameter']<=.4 and not t['branches'] for t in obstacles)
    assert np.all(np.abs(np.array([c['x'] for c in checks])-[20.,80.])<=2.)
    again=add_task(base,rows,config,'choice',1,1,difficulty,'approach')
    assert trees==again[0] and checks==again[3]


def test_teacher_cannot_escape_a_blocked_corridor():
    y,z=np.meshgrid(np.linspace(-1.,1.,25),np.arange(-2,2.1,.2))
    wall=np.c_[np.full(y.size,2.),y.ravel(),z.ravel()]
    x,y,z=np.meshgrid(np.arange(-2,10.1,.2),np.arange(-2,2.1,.2),np.arange(-2,2.1,.2))
    evidence=np.c_[x.ravel(),y.ravel(),z.ravel()]
    corridor=np.array([[-2.,-1.],[10.,-1.],[10.,1.],[-2.,1.]])
    answer,reason=teacher_path(wall,[6.,0.,0.],corridor=corridor,evidence=evidence)
    # 可以靠近墙，但不能穿过去；无剩余进展空间时才悬停。
    if answer['status'] == 'GO':
        assert max(p[0] for p in answer['points']) < 1.35
    else:
        assert answer == {'status':'HOLD','points':[]}


def test_rotated_task_boundary_round_trip():
    pos=np.array([3.,2.,1.]); yaw=.7
    polygon=corridor_polygon([0.,100.,0.,4.],pos,yaw)
    assert inside_polygon([[0.,0.]],polygon)[0]
    world=local_to_world(np.c_[polygon,np.zeros(4)],pos,yaw)
    np.testing.assert_allclose(world[:,:2],[[0.,0.],[100.,0.],[100.,4.],[0.,4.]],atol=1e-12)


def test_nested_map_index(tmp_path):
    folder=tmp_path/'easy'/'blocked'/'000001_approach'; folder.mkdir(parents=True)
    (folder/'map.json').write_text(json.dumps(dict(validation=dict(passed=True),scene_id='forest',task='blocked',split='train')))
    write_index(tmp_path)
    assert json.loads((tmp_path/'index.json').read_text())[0]['map_id']=='easy/blocked/000001_approach'


def test_evaluation_rejects_switching_rows():
    from omni_drones.navigation.forest.evaluation import evaluate_path
    raw=json.dumps(dict(status='GO',points=[[x,2.,0.] for x in range(1,7)]))
    observation=dict(points=np.array([[5.,5.,0.]]),position=np.zeros(3),yaw=0.,goal=np.array([6.,0.,0.]),
                     corridor=np.array([[-2.,-1.],[10.,-1.],[10.,1.],[-2.,1.]]))
    result=evaluate_path(raw,observation,dict(trees=[[5.,5.,.3,10.]]))
    assert result['corridor_violation'] and not result['safety_filter_accept']
