"""原始模板、裁剪采样和地形移植的关键对应关系。"""
import json
import numpy as np

from forest_generator.generator import DEFAULT_CONFIG,sample_forest,mesh_forest,template
from forest_generator.terrain import Terrain


def test_original_template_and_spawn_transform():
    vertices,faces,uv,normals=template()
    assert len(vertices)==64 and len(faces)==124
    np.testing.assert_allclose(np.ptp(vertices,axis=0),[1.,1.,10.],atol=1e-6)
    config=json.loads(DEFAULT_CONFIG.read_text())
    terrain=Terrain(config['terrain'],[4,4],np.random.default_rng(4))
    branch=dict(angle=.4,azimuth=.7,length=3.,height=4.)
    tree=dict(position=[1.,2.],diameter=.6,branches=[branch])
    v,f,u,n=mesh_forest([tree],terrain)
    root=np.r_[tree['position'],tree['ground_z']]
    np.testing.assert_allclose(v[-64:],vertices*[.6,.6,1.]+root,atol=1e-6)
    ca,sa=np.cos(.4),np.sin(.4);cr,sr=np.cos(.7),np.sin(.7)
    rotation=np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])@np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    expected=(vertices*[.2,.2,.3])@rotation.T+root+[0,0,4.]
    np.testing.assert_allclose(v[-128:-64],expected,atol=2e-6)
    np.testing.assert_allclose(u[-124:],uv,atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(n,axis=-1),1.,atol=1e-6)


def test_actual_positions_use_recorded_clipped_draws():
    c=json.loads(DEFAULT_CONFIG.read_text());trees,rows,terrain=sample_forest(0,c)
    np.testing.assert_allclose(np.diff([r['y'] for r in rows]),[r['spacing'] for r in rows[:-1]])
    assert abs(np.mean([r['spacing'] for r in rows])-c['row_spacing'][0])<.2
    for r in rows:
        assert r['spacing']==np.clip(r['spacing_raw'],*c['row_spacing'][2:])
        ts=[t for t in trees if t['row']==r['row']]
        np.testing.assert_allclose(np.diff([0.]+[t['position'][0] for t in ts]),[t['spacing'] for t in ts])
        for t in ts:
            assert t['spacing']==np.clip(t['spacing_raw'],*c['tree_spacing_gamma'][2:])
            assert t['offset']==np.clip(t['offset_raw'],*c['row_offset'][2:])
            assert t['position'][1]==r['y']+t['offset']
    x,y=terrain.vertices[:,:2].T
    expected=(2*terrain.noise.noise((x+23)*.52,(y+23)*.52)+
        terrain.amplitude2*terrain.noise.noise(x*.05+1245,y*.05+1245,54)+x*terrain.slope[0]+y*terrain.slope[1])
    np.testing.assert_allclose(terrain.vertices[:,2],expected)
    np.testing.assert_allclose(terrain(terrain.vertices),terrain.vertices[:,2])
    assert np.ptp(terrain.heights)>10.
