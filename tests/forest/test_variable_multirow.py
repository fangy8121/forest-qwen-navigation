"""Variable forest boundaries, continuous turns and complete phase partitions."""
from copy import deepcopy
import json
import numpy as np
import pytest
from omegaconf import OmegaConf
from forest_generator.generator import DEFAULT_CONFIG,layout_config,sample_forest
from forest_generator.multirow_layout import layout_options,configure,translate_rows,aisle_endpoints,row_envelopes
from forest_generator.multirow_routes import continuous_reference,reference_phases,validate_phases,motion_audit
from omni_drones.navigation.forest.planning import smooth_path


def test_variable_rows_reproducible_in_bounds_and_preserve_native_distributions():
    base=layout_config(json.loads(DEFAULT_CONFIG.read_text()),'short')
    config,offset,_=configure(base,layout_options())
    for key in ('branch_count','branch_length_gamma','branch_height','branch_angle','stem_diameter','terrain'):
        assert config[key]==base[key]
    trees,rows,terrain=sample_forest(20261700,config)
    again,other,_=sample_forest(20261700,config)
    assert trees==again and rows==other
    assert np.ptp([r['length_m'] for r in rows])>1.
    assert np.ptp([r['start_x'] for r in rows])>1.
    assert np.ptp(terrain.heights)>1.
    for r in rows:
        assert 65<=r['length_m']<=75
        ts=[t for t in trees if t['row']==r['row']]
        assert all(r['start_x']<t['position'][0]<=r['end_x'] for t in ts)
        assert all(t['height']==10. for t in ts)
    for t in trees:t['position']=(np.array(t['position'])+offset).tolist()
    translate_rows(rows,offset)
    ends=aisle_endpoints(trees,rows,layout_options());envelope=row_envelopes(trees,rows)
    for i,(left,right) in enumerate(ends):
        assert left<=envelope[i:i+2,0].min()-2+1e-8
        assert right>=envelope[i:i+2,1].max()+2-1e-8
    for i in range(len(ends)-1):
        side=1 if i%2==0 else 0
        assert ends[i,side]==ends[i+1,side]


def test_continuous_turn_faster_at_same_limits_and_roundtrips(tmp_path):
    from omni_drones.navigation.forest.trajectory import QuinticPath
    flight=OmegaConf.load('cfg/task/ForestSurvey.yaml').flight
    lanes=[dict(reference=[[0.,0.,2.5],[20.,0.,2.5]]),
           dict(reference=[[20.,5.,2.5],[0.,5.,2.5]])]
    path,intervals,audit=continuous_reference(lanes,flight,lambda p:True)
    phases=reference_phases(path,intervals);validate_phases(phases,path.duration,2)
    assert [s['phase'] for s in phases]==['INSPECT','EXIT','TURN','ENTRY','INSPECT']
    assert audit['passed'] and motion_audit(path,flight,samples=97)['passed']
    theta=np.linspace(-np.pi/2,np.pi/2,15)
    arc=np.c_[20+2.5*np.cos(theta),2.5+2.5*np.sin(theta),np.full(15,2.5)]
    old=smooth_path(np.vstack([lanes[0]['reference'],arc[1:],lanes[1]['reference'][1:]]),flight,lambda p:True)
    assert path.duration<old.duration-10.,(path.duration,old.duration)
    turn=next(s for s in phases if s['phase']=='TURN')
    assert turn['duration_s']<15.
    for q in (intervals[1]['q_start'],intervals[1]['q_end']):
        for order in (0,1,2):
            np.testing.assert_allclose(path.geometry(q-1e-7,nu=order),path.geometry(q+1e-7,nu=order),atol=1e-5)
    filename=tmp_path/'reference.npz';path.save(filename)
    loaded=QuinticPath.load(filename);ts=np.linspace(0,path.duration,501)
    for a,b in zip(path.sample(ts),loaded.sample(ts)):np.testing.assert_allclose(a,b)
    invalid=deepcopy(phases);invalid[1]['start_s']+=.1
    with pytest.raises(ValueError,match='gap or overlap'):validate_phases(invalid,path.duration,2)


def test_turn_geometry_does_not_bypass_clearance_veto():
    flight=OmegaConf.load('cfg/task/ForestSurvey.yaml').flight
    lanes=[dict(reference=[[0.,0.,2.5],[20.,0.,2.5]]),
           dict(reference=[[20.,5.,2.5],[0.,5.,2.5]])]
    with pytest.raises(ValueError,match='clearance'):
        continuous_reference(lanes,flight,lambda p:bool((p[:,0]<=20.5).all()))


@pytest.mark.parametrize('length,jitter', [([75,65],2),([20,30],2),([65,75],-1),([65,float('nan')],2)])
def test_invalid_layout_rejected(length,jitter):
    with pytest.raises(ValueError):layout_options('variable',length,jitter)
