import json
import numpy as np
import pytest
import torch
from omni_drones.navigation.forest.geometry import (
    RADIUS, MARGIN, INFLATION, clearance, local_to_world, world_to_local, parse_path, quat_matrix)
from omni_drones.navigation.forest.observation import History, grid_observation
from omni_drones.navigation.forest.trajectory import QuinticPath
from omni_drones.navigation.forest.teacher import teacher_path, resample_six
from omni_drones.navigation.forest.scene import make_scene, scene_split
from omni_drones.navigation.forest.qwen import assistant_labels, lora_targets
from omni_drones.navigation.forest.evaluation import evaluate_path, summarize


def free_volume():
    x,y,z=np.meshgrid(np.arange(-2,10.1,.2),np.arange(-6,6.1,.2),np.arange(-2,2.1,.2))
    return np.c_[x.ravel(),y.ravel(),z.ravel()]


def frame(t, hit=(1,0,0)):
    return dict(timestamp=t, points_sensor=np.array([hit,[np.nan]*3]), valid=np.array([True,False]),
        sensor_position=np.array([0,0,2.]),sensor_quaternion=np.array([1.,0,0,0]),
        directions_world=np.array([[1.,0,0],[0,1.,0]]),distances=np.array([1.,np.inf]),max_range=2.)


def test_coordinate_roundtrip():
    rng=np.random.default_rng(8); p=rng.normal(size=(50,3)); pos=np.array([3,-2,2])
    for yaw in [0,1.2,-np.pi]:
        np.testing.assert_allclose(local_to_world(world_to_local(p,pos,yaw),pos,yaw),p,atol=1e-14)
    np.testing.assert_allclose(local_to_world([[1,0,0]],[0,0,0],np.pi/2),[[0,1,0]],atol=1e-14)
    np.testing.assert_allclose(quat_matrix([np.sqrt(.5),0,0,np.sqrt(.5)]) @ [1,0,0],[0,1,0],atol=1e-14)


def test_history_window_future_reset():
    h=History(3)
    for t in [0.,1.,3.,4.]: h.add(frame(t, (t+1,0,0)))
    p,_=h.snapshot(3.,[0,0,2.],0)
    assert set(p[:,0]) == {2,4}  # 已过期 0 与未来 4 都不能进入快照
    h.reset(); assert len(h.snapshot(0,[0,0,2],0)[0]) == 0
    h.add(frame(0)); h.add(frame(1))
    with pytest.raises(ValueError): h.add(frame(0))


def test_sensor_pose_and_unknown():
    h=History(); f=frame(0); f['sensor_quaternion']=np.array([np.sqrt(.5),0,0,np.sqrt(.5)])
    h.add(f); p,e=h.snapshot(0,[0,0,2],0)
    np.testing.assert_allclose(p,[[0,1,0]],atol=1e-14)
    assert np.isfinite(p).all() and len(p)==1
    _,_,_,seen=grid_observation(p,e)
    assert len(e) > 0 and not seen.all()  # 一条射线不能证明整个机体体积可通行
    assert not grid_observation(np.empty((0,3)),np.empty((0,3)))[3].any()


def test_dimensions_segments():
    assert RADIUS==.5 and MARGIN==.15 and INFLATION==.65
    assert clearance([[0,0],[2,0]],[[1,.6]]) < INFLATION
    assert clearance([[0,0],[2,0]],[[1,.66]]) > INFLATION
    assert clearance([[0,0],[1,0],[1,2]],[[1,1]])==0
    assert clearance([[0,0],[0,0]],[[1,0]])==1
    assert clearance([[0,0],[2,0]],[[1,.7]],.22) == pytest.approx(.48)


@pytest.mark.parametrize('raw',[
    '```json\n{"status":"STOP","points":[]}\n```',
    '{"status":"STOP","points":[],"extra":1}',
    '{"status":"STOP","status":"STOP","points":[]}',
    '{"status":"GO","points":[[0,0]]}',
    '{"status":"STOP","points":[[0,0]]}',
    json.dumps({'status':'GO','points':[[True,0]]*6}),
    json.dumps({'status':'GO','points':[[float('nan'),0]]*6}),
    json.dumps({'status':'GO','points':[[11,0]]*6}),
    '[]', 'null', '{"status":"GO","points":null}',
    '{"status":"STOP","points":[]} trailing',
])
def test_strict_protocol(raw):
    with pytest.raises(ValueError): parse_path(raw)


def test_valid_protocol():
    assert parse_path('{"status":"HOLD","points":[]}')['status']=='HOLD'
    assert parse_path('{"status":"STOP","points":[]}',protocol_version=2)['status']=='STOP'
    assert len(parse_path(json.dumps(dict(status='GO',points=[[x,0,0] for x in range(1,7)])))['points'])==6


def test_teacher_empty_and_collision_after_sampling():
    assert teacher_path(np.empty((0,3)),[6,0,0])[0]['status']=='HOLD'
    # 重采样可跨越拐角，必须以最终六点折线为准。
    route=np.array([[0,0],[1,0],[1,1],[4,1]])
    six=resample_six(route)
    obstacle=np.array([[.90,.25]])
    assert clearance(np.vstack([[0,0],six]),obstacle) < clearance(route,obstacle)
    points=np.c_[np.arange(-2,10,.15),np.full(80,2),np.zeros(80)]
    result,reason=teacher_path(points,[6,0,0],evidence=free_volume())
    assert result['status']=='GO'
    from omni_drones.navigation.forest.planning import PathSafety
    from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
    assert PathSafety(points,free_volume(),None,DEFAULT_FLIGHT).clear(np.vstack([[0,0,0],result['points']]))
    for seed in range(8):
        p=np.c_[np.random.default_rng(seed).uniform([0,-3],[7,3],size=(30,2)),np.zeros(30)]
        result,_=teacher_path(p,[6,0,0],evidence=free_volume())
        if result['status']=='GO':
            assert PathSafety(p,free_volume(),None,DEFAULT_FLIGHT).clear(np.vstack([[0,0,0],result['points']]))


def test_scene_split_reproducible_and_grouped():
    a=make_scene(5)[0]; b=make_scene(5)[0]; assert a==b
    assert make_scene(6)[0]['scene_id']!=a['scene_id']
    ids=['scene'+str(i) for i in range(10)]
    split=scene_split(ids+ids)
    assert split==scene_split(ids[::-1]) and set(split.values())=={'train','val','test'}
    # 固定分组不能随数据扩容改变；小集合允许缺少训练/验证/测试中的某些集合。
    assert scene_split(['one'])['one']==scene_split(['one','two'])['one']
    assert all(scene_split(ids+['new'])[s]==split[s] for s in ids)


def test_assistant_only_labels():
    ids=torch.tensor([[5,9,9,9,10,21,22,2]])
    prefix=ids[:,:5]
    labels=assistant_labels(ids,prefix)
    assert (labels[:,:5]==-100).all()
    assert torch.equal(labels[:,5:],ids[:,5:])
    with pytest.raises(ValueError): assistant_labels(ids,torch.tensor([[4,5]]))
    with pytest.raises(ValueError): assistant_labels(ids,ids)
    mask=torch.ones_like(ids);mask[:,-1]=0
    assert assistant_labels(ids,prefix,mask)[0,-1]==-100


def test_time_parameterization():
    path=QuinticPath([[0,0,2],[2,1,2],[3,0,2]],1.,.8)
    for i,t in enumerate(path.times):
        p,v,a=path.sample(t)
        np.testing.assert_allclose(p,path.points[i],atol=1e-12)
        if i in (0, len(path.times)-1):
            np.testing.assert_allclose(v,0,atol=1e-12)
            np.testing.assert_allclose(a,0,atol=1e-12)
        else:
            assert np.linalg.norm(v) > 1e-3  # 小范围转弯允许减速，但不中途归零。
        for dt in [-1e-7,1e-7]:
            pp,vv,aa=path.sample(t+dt)
            np.testing.assert_allclose(pp,p,atol=1e-5)
            np.testing.assert_allclose(vv,v,atol=1e-5)
            np.testing.assert_allclose(aa,a,atol=1e-5)
    values=[path.sample(t) for t in np.linspace(0,path.duration,5000)]
    assert max(np.linalg.norm(v) for p,v,a in values)<=1+1e-9
    assert max(np.linalg.norm(a) for p,v,a in values)<=.8+1e-9


def test_mesh_and_metrics():
    info,v,f,uv=make_scene(101,length=8,width=32,buffer=16,height=2)
    assert info['geometry']=='triangle_mesh' and len(v)>1000 and len(f)>1000
    assert f.max()<len(v) and f.min()==0 and uv.shape[:2]==f.shape
    raw=json.dumps(dict(status='GO',points=[[x,0,0] for x in range(1,7)]))
    obs=dict(points=np.array([[0,2,0]]),position=np.array([0,0,2]),yaw=0,goal=np.array([6,0,0]),evidence=free_volume())
    truth=dict(mesh=(np.array([[.5,-.2,2],[.8,.2,2],[.4,.2,2]]),np.array([[0,1,2]])))
    r=evaluate_path(raw,obs,truth)
    assert r['truth_collision'] and r['safety_filter_accept']  # 隐藏真值不影响观测筛选
    stop=evaluate_path('{"status":"HOLD","points":[]}',obs,truth)
    summary=summarize([dict(stop,latency_seconds=.1)])
    assert summary['go_coverage']==0 and summary['truth_collision_go_mean'] is None


def test_goal_stays_visible_and_preserves_direction():
    from omni_drones.navigation.forest.flight import local_goal
    from omni_drones.navigation.forest.geometry import inside
    for goal in [[-20,10,2],[30,2,2],[0,-30,2]]:
        local=local_goal(goal,[0,0,2],0)
        assert inside([local[:2]])
        assert abs(np.cross(local[:2],np.array(goal)[:2]))<1e-9


def test_protocol_huge_integer():
    with pytest.raises(ValueError): parse_path(json.dumps(dict(status='GO',points=[[10**400,0]]*6)))


def test_teacher_rejects_real_resampling_corner_cut():
    p=np.c_[np.random.default_rng(11).uniform([1,-3],[5,3],size=(6,2)),np.zeros(6)]
    result,reason=teacher_path(p,[6,0,0],evidence=free_volume())
    if result['status']=='GO':
        from omni_drones.navigation.forest.teacher import DEFAULT_FLIGHT
        from omni_drones.navigation.forest.planning import smooth_path
        from omni_drones.navigation.forest.planning import PathSafety
        xyz=np.vstack([[0,0,0],result['points']])
        safety=PathSafety(p,free_volume(),None,DEFAULT_FLIGHT)
        rebuilt=smooth_path(xyz,DEFAULT_FLIGHT,safety.clear)
        assert safety.clear(rebuilt.remaining(0.,xyz[0]))
    else:
        assert reason in ('smoothing_failure','reconstruction_failure')


def test_actual_qwen_module_names():
    # 仅实例化缩小的随机网络核对模块名，不是预训练模型推理或 LoRA 训练验证。
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    config=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=16,
        rope_scaling={'rope_type':'default','mrope_section':[3,3,2]}),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
        out_hidden_size=32,num_position_embeddings=16,deepstack_visual_indexes=[0]))
    model=Qwen3VLForConditionalGeneration(config)
    targets=lora_targets(model)
    assert len(targets)==4 and all('visual' not in t for t in targets)


def test_mesh_clearance_reports_the_closest_surface_in_the_body_band():
    import trimesh
    from omni_drones.navigation.forest.evaluation import mesh_clearance
    trunk = trimesh.creation.cylinder(radius=.2, height=4., sections=32)
    trunk.apply_translation([2., 1., 2.])
    truth = dict(mesh=(np.asarray(trunk.vertices), np.asarray(trunk.faces)))
    path = np.array([[0., 0., 2.], [4., 0., 2.]])
    assert abs(mesh_clearance(path, truth, .4)-.8) < .02
    # Out of the height band, or beyond the horizon, nothing is seen.
    assert mesh_clearance(path+[0., 0., 5.], truth, .4) == 2.
    assert mesh_clearance(path-[0., 3., 0.], truth, .4) == 2.


def test_downward_cone_covers_the_space_under_the_body():
    # Emulated M300 downward sensing: an upright MID-360 (-7..+52 deg) never
    # sees under its own body (run 6244d06b).
    import pytest
    from omni_drones.navigation.forest.lidar import downward_cone
    cone = downward_cone(300, 60.)
    assert cone.shape == (300, 3)
    np.testing.assert_allclose(np.linalg.norm(cone, axis=1), 1.)
    tilt = np.degrees(np.arccos(-cone[:, 2]))
    assert tilt.max() <= 60.+1e-9 and tilt.min() < 3.
    # From 0.26 m above the body bottom the cone reaches the 0.5 m body radius.
    assert (np.tan(np.radians(tilt.max()))*.26) > .44
    azimuth = np.degrees(np.arctan2(cone[:, 1], cone[:, 0]))
    assert np.histogram(azimuth, bins=8, range=(-180, 180))[0].min() >= 25
    with pytest.raises(ValueError):
        downward_cone(10, 90.)
