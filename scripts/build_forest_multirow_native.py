"""Multi-row review map using the ORIGINAL forest sampler and tree parameters.

Only site extent, translation and end-of-row task space are new. No branch
shortening, height filtering, replacement tree sampler, or seed retries.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/forest-multirow-mpl')
import numpy as np
from omegaconf import OmegaConf
from forest_generator.generator import DEFAULT_CONFIG,SOURCE,layout_config,sample_forest,mesh_forest
from forest_generator.terrain import Terrain
from forest_generator.maps import export,Clearance
from omni_drones.navigation.forest.esdf import ESDF,gate_crossings
from omni_drones.navigation.forest.planning import smooth_path
from omni_drones.navigation.forest.geometry import REFERENCE_PROFILE
from omni_drones.navigation.forest.evaluation import reference_clearance_audit
from omni_drones.navigation.forest.runs import file_hash
from forest_generator.multirow_layout import layout_options,configure,translate_rows,aisle_endpoints
from forest_generator.multirow_routes import continuous_reference,reference_phases,validate_phases
from forest_generator.clearance_profiles import clearance_profile


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def full_curve_lane_checks(dense,lanes):
    cursor=0;checks=[]
    for lane in lanes:
        first=cursor+int(np.argmin(np.linalg.norm(dense[cursor:]-lane['reference'][0],axis=1)))
        last=first+int(np.argmin(np.linalg.norm(dense[first:]-lane['reference'][-1],axis=1)))
        crossings=gate_crossings(dense[first:last+1],lane['task_gates'])
        if crossings is None:return None
        checks.append((first,last,crossings));cursor=last
    return checks


def branch_axis(tree,b):
    a,r=b['angle'],b['azimuth']
    start=np.r_[tree['position'],tree['ground_z']+b['height']]
    end=start+b['length']*np.array([np.sin(a)*np.sin(r),-np.sin(a)*np.cos(r),np.cos(a)])
    return np.array([start,end])


def plot_mesh(out,vertices,faces,terrain,bounds=(0,60,0,36)):
    """Render the exported triangles, not substitute tree symbols."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    triangles=vertices[faces]
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-8)
    light=np.array([-.4,-.5,1.]);light/=np.linalg.norm(light)
    brightness=.45+.55*np.abs(normal@light)
    colors=np.tile([.48,.30,.16,1.],(len(faces),1))
    groundfaces=len(terrain.faces)
    z=triangles[:groundfaces,:,2].mean(1)
    colors[:groundfaces]=plt.get_cmap('terrain')((z-z.min())/max(float(z.max()-z.min()),1e-8))
    colors[:,:3]*=brightness[:,None]
    fig=plt.figure(figsize=(15,9),dpi=140);ax=fig.add_subplot(111,projection='3d')
    ax.add_collection3d(Poly3DCollection(triangles,facecolors=colors,edgecolors='none',rasterized=True))
    ax.set(xlim=bounds[:2],ylim=bounds[2:],zlim=(float(terrain.heights.min()),float(vertices[:,2].max())))
    ax.set_box_aspect((bounds[1]-bounds[0],bounds[3]-bounds[2],18));ax.view_init(24,-65)
    ax.set_axis_off();ax.set_position([-.08,-.12,1.16,1.20])
    fig.text(.5,.95,'Actual triangle mesh from the original forest generator',ha='center',fontsize=17)
    fig.text(.5,.035,'Original stem/branch templates and distributions. Flat shading, not Isaac rendering.',ha='center',fontsize=10)
    fig.savefig(out/'03_实际网格视图.png');plt.close(fig)


def plot_native(out,info,terrain,route=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,9),dpi=150)
    im=ax.pcolormesh(terrain.xs,terrain.ys,terrain.heights.T,cmap='terrain',shading='auto',alpha=.65)
    fig.colorbar(im,ax=ax,label='Ground elevation (m)')
    for t in info['trees']:
        for b in t['branches']:
            line=branch_axis(t,b)
            ax.plot(*line[:,:2].T,color='saddlebrown',alpha=.65,lw=1.2)
        ax.add_patch(plt.Circle(t['position'],t['radius'],color='darkgreen'))
    if route is not None:ax.plot(*route[:,:2].T,color='royalblue',lw=1.7,label='Verified reference')
    for i,y in enumerate(info['lane_centers_y']):ax.text(info['bounds'][0]+1,y,f'{i+1}',color='blue',fontsize=11)
    ax.set(xlim=info['bounds'][:2],ylim=info['bounds'][2:],xlabel='World X (m)',ylabel='World Y (m)',
           title=f'Native generator | {len(info["trees"])} trees, {sum(len(t["branches"]) for t in info["trees"])} branches')
    ax.set_aspect('equal')
    fig.text(.5,.015,'Green: stems. Brown: branch axes projected to XY (all heights). Full mesh used for safety.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/'01_原生成器俯视.png');plt.close(fig)
    fig=plt.figure(figsize=(15,9),dpi=150);ax=fig.add_subplot(111,projection='3d')
    xx,yy=np.meshgrid(terrain.xs,terrain.ys,indexing='ij')
    ax.plot_surface(xx,yy,terrain.heights,cmap='terrain',alpha=.55,rstride=2,cstride=2,linewidth=0)
    for t in info['trees']:
        x,y=t['position'];z=t['ground_z']
        ax.plot([x,x],[y,y],[z,z+10],color='saddlebrown',lw=1.4)
        for b in t['branches']:ax.plot(*branch_axis(t,b).T,color='saddlebrown',lw=.8,alpha=.8)
    if route is not None:ax.plot(*route.T,color='royalblue',lw=2)
    ax.set(xlim=info['bounds'][:2],ylim=info['bounds'][2:],xlabel='World X (m)',ylabel='World Y (m)',zlabel='Z (m)',
           title='Original tree/branch geometry + original short-layout terrain\nAxes preview; unchanged template mesh exported as scene.usd')
    ax.set_box_aspect((info['bounds'][1]-info['bounds'][0],info['bounds'][3]-info['bounds'][2],18));ax.view_init(24,-60)
    fig.tight_layout();fig.savefig(out/'02_原生成器立体.png');plt.close(fig)


def build(out,seed,resume=False,split='train',task_profile=None,flat=False,mesh_preview=True,
          continue_after_task_mismatch=False,task_attempt=0,layout=None,clearance='legacy'):
    layout=layout_options('legacy') if layout is None else layout
    profile=clearance_profile(clearance);required=profile['clearance_m']
    variable=layout['mode']=='variable'
    folder=out if flat else out/'map'
    previous=None
    if (folder/'map.json').exists():
        if not resume:raise FileExistsError(folder)
        previous=json.loads((folder/'map.json').read_text())
        if previous['validation']['passed']:raise ValueError('Preserve already validated map')
        if previous['parameters']['seed']!=seed:raise ValueError('Resume seed differs')
    if flat:
        out.parent.mkdir(parents=True,exist_ok=True)
    else:
        out.mkdir(parents=True,exist_ok=True)
    base=json.loads(DEFAULT_CONFIG.read_text())
    config=layout_config(base,'short')
    config,translation,site_size=configure(config,layout)
    original,source_rows,_=sample_forest(seed,config)
    trees=deepcopy(original);rows=deepcopy(source_rows)
    for tree in trees:tree['position']=(np.array(tree['position'])+translation).tolist()
    translate_rows(rows,translation)
    # Strict equality of all biological/random attributes before translation.
    for source,tree in zip(original,trees):
        assert {k:v for k,v in source.items() if k!='position'}=={k:v for k,v in tree.items() if k!='position'}
    terrain_rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(8)[6])
    terrain=Terrain(config['terrain'],site_size,terrain_rng)
    tasks=None
    if task_profile is not None:
        from forest_generator.multirow_tasks import add_tasks
        native_vertices,native_faces,_,_=mesh_forest(trees,terrain)
        native_clearance=Clearance(native_vertices,native_faces,terrain,required)
        tasks=add_tasks(trees,rows,config,seed,task_profile,native_clearance,terrain,task_attempt,required)
        gaps=[float(np.linalg.norm(np.array(t['position'])-b['position'])-t['radius']-b['radius'])
              for j,t in enumerate(trees) if t.get('inserted')
              for k,b in enumerate(trees) if k!=j]
        minimum=min(gaps,default=None)
        assert minimum is None or minimum>=tasks['minimum_added_stem_gap_m']-1e-8
        tasks['stem_placement_audit']=dict(passed=True,minimum_surface_gap_m=minimum,
            scope='added vertical stem enclosing disks versus all stems; not branch intersection certification')
        branch_gaps=[t['native_branch_clearance_lower_bound_m'] for t in trees if t.get('inserted')
                     and t.get('native_branch_clearance_lower_bound_m') is not None]
        assert all(g>=.05 for g in branch_gaps)
        tasks['branch_placement_audit']=dict(passed=True,minimum_clearance_lower_bound_m=min(branch_gaps,default=None),
            scope='added stem capsules versus native branch capsules, including sample-spacing bound; >= 0.05 m')
    vertices,faces,uvs,normals=mesh_forest(trees,terrain)
    centers=np.array([(a['y']+b['y'])/2 for a,b in zip(rows[:-1],rows[1:])])
    endpoints=aisle_endpoints(trees,rows,layout)
    flight=OmegaConf.load(ROOT/'cfg/task/ForestSurvey.yaml').flight
    flight.height=2.5;flight.safety_profile=REFERENCE_PROFILE;flight.ray_traversal='voxel_dda_v1'
    branch_stats={}
    for key in ('height','length','angle'):
        values=np.array([b[key] for t in trees for b in t['branches']])
        branch_stats[key]=dict(minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max()))
    info=dict(scene_id=f'native_multirow_{seed}_'+('variable_v2' if variable else 'v1'),task='multirow_native',split=split,
        purpose='review_only_not_training_or_blind_test',parameters=dict(seed=seed,config=dict(config,forest_size=site_size),source_sampling_config=config,multirow_layout=layout,
            clearance_profile=profile),
        trees=trees,rows=rows,lane_centers_y=centers.tolist(),bounds=[0.,site_size[0],0.,site_size[1]],
        validation=dict(passed=False,reason='reference_not_yet_checked',route_bounds=None),
        provenance=dict(sampler='forest_generator.generator.sample_forest',layout='short',translation_xy=translation.tolist(),
            exact_source_tree_attributes_preserved=True,source_tree_count=len(original),
            sampler_sha256=file_hash(ROOT/'forest_generator/generator.py'),config_sha256=file_hash(DEFAULT_CONFIG),
            branch_statistics=branch_stats,
            differences=[f'Native sampled stand translated by {translation.tolist()} into a {site_size} m site',
                        ('Variable row ends sampled before tree placement; native tree/branch distributions unchanged' if variable else
                         'Original sampler keeps the final tree past nominal x=40; it is NOT trimmed'),
                        'No additional perimeter forest: end space is explicitly reserved for the demo task',
                        'No artificial task trees yet; natural sampled branches and terrain remain unchanged']),
        flight_parameters=OmegaConf.to_container(flight),trajectory_file='reference_trajectory.npz',terrain_file='terrain.npz')
    if tasks is not None:
        info.update(task='multirow_tasks',tasks=tasks,purpose='map_geometry_stage_not_scan_or_training_ready')
        info['provenance']['differences'][-1]='Task trees added separately; all original trees and branches preserved'
    if previous is None:
        export(folder,info,vertices,faces,uvs,normals,SOURCE/'tex.jpg')
        np.savez_compressed(folder/'terrain.npz',xs=terrain.xs,ys=terrain.ys,heights=terrain.heights)
        if not flat:plot_native(out,info,terrain)
        if mesh_preview:plot_mesh(out,vertices,faces,terrain,info['bounds'])
    else:
        with np.load(folder/'scene_mesh.npz') as saved:
            assert np.array_equal(vertices,saved['vertices']) and np.array_equal(faces,saved['faces'])
        for key in ('geometry_id','geometry','mesh_file','usd_file','source_sha256'):
            info[key]=previous[key]
        info['previous_reference_checks']=previous.get('previous_reference_checks',[])+[previous['validation']]
    print(json.dumps(dict(stage='original_geometry_exported',trees=len(trees),branches=sum(len(t['branches']) for t in trees),rows=len(rows),branch_stats=branch_stats)),flush=True)
    reference=[];turns=[];lanes=[];task_reviews=[]
    try:
        for i,y in enumerate(centers):
            # Endpoints must be inside the search bounds: rounding an endpoint
            # on the grid boundary can select an outside, forbidden voxel.
            xlo,xhi=endpoints[i]
            bounds=[float(xlo-2),float(xhi+2),rows[i]['y'],rows[i+1]['y']]
            events=[] if tasks is None else [e for e in tasks['events'] if e['lane']==i+1]
            gates=[]
            if tasks is not None:
                from forest_generator.multirow_tasks import lane_problem,verify_openings
                bounds,gates=lane_problem(rows,i,events,bounds[:2])
            print(f'{seed}: aisle {i+1}/{len(centers)}',flush=True)
            xy=np.array([[xlo,y],[xhi,y]])
            if i%2:xy=xy[::-1]
            ends=np.c_[xy,terrain(xy)+2.5]
            candidates=[];attempts=[]
            for resolution in (.2,.1):
                print(f'{seed}: aisle {i+1} grid {resolution} m',flush=True)
                if profile.get('agl_band_m') is None:
                    esdf=ESDF(vertices,faces,terrain,bounds,height=2.5,resolution=resolution,clearance=required)
                else:
                    # Search in the online planner's cylinder and height band.
                    from forest_generator.nav_esdf import NavigationESDF
                    esdf=NavigationESDF(vertices,faces,terrain,bounds,height=2.5,resolution=resolution,
                        horizontal=profile['horizontal_m'],vertical=profile['vertical_m'],agl_band=profile['agl_band_m'],
                        agl_check=profile['agl_check_m'])
                note=None
                try:
                    openings=verify_openings(esdf,rows,i,events,ends,gates) if tasks is not None else []
                except ValueError as error:
                    note=str(error);openings=[dict(passed=False,reason=note)]
                try:
                    route=esdf.route(ends[0],ends[1],gates)
                    lane_curve=smooth_path(route,flight,lambda p: esdf.clear(p) and gate_crossings(p,gates,False) is not None)
                    lane_points=lane_curve.remaining(0.,route[0])
                    if gate_crossings(lane_points,gates) is None:raise ValueError('smoothed task gates failed')
                    lane_audit=reference_clearance_audit(lane_points,dict(mesh=(vertices,faces)),required)
                    if not lane_audit['passed']:raise ValueError('full-mesh veto')
                    candidates.append((route,lane_audit,openings,note,resolution))
                    attempts.append(dict(resolution=resolution,reference_passed=True,task_review=note))
                    if note is None:break
                except ValueError as error:
                    attempts.append(dict(resolution=resolution,reference_passed=False,reason=str(error),task_review=note))
            if not candidates:raise ValueError(f'aisle {i+1}: no validated reference: {attempts}')
            route,lane_audit,openings,note,resolution=candidates[-1]
            if note:
                if not continue_after_task_mismatch:raise ValueError(note)
                task_reviews.append(dict(lane=i+1,reason=note))
            lanes.append(dict(lane=i+1,reference=route.tolist(),audit=lane_audit,task_openings=openings,
                              task_gates=gates,search_resolution_m=resolution,search_attempts=attempts))
            reference.extend(route.tolist() if i==0 else route[1:].tolist())
            if i<len(centers)-1:
                side=1 if i%2==0 else -1;edge=float(xhi if side>0 else xlo)
                radius=(centers[i+1]-y)/2;yc=(centers[i+1]+y)/2
                theta=np.linspace(-np.pi/2,np.pi/2,15)
                arc=np.c_[edge+side*radius*np.cos(theta),yc+radius*np.sin(theta)]
                turn_ref=np.c_[arc,terrain(arc)+2.5]
                reference.extend(turn_ref[1:].tolist())
                turns.append(dict(id=i+1,from_lane=i+1,to_lane=i+2,side='east' if side>0 else 'west',
                                  center=[edge,float(yc)],radius_m=float(radius),
                                  from_center_y=float(y),to_center_y=float(centers[i+1])))
        reference=np.array(reference)
        clearance=Clearance(vertices,faces,terrain,required)
        lane_checks=None
        connection_failures=[]
        for corner_scale in (1.,.5,.25):
            local_flight=OmegaConf.create(OmegaConf.to_container(flight))
            local_flight.corner_distance*=corner_scale
            try:
                if variable:
                    path,intervals,motion=continuous_reference(lanes,local_flight,clearance)
                else:
                    path=smooth_path(reference,local_flight,clearance)
            except ValueError as error:
                connection_failures.append(str(error));continue
            dense=path.remaining(0.,reference[0])
            lane_checks=full_curve_lane_checks(dense,lanes)
            if lane_checks is not None:break
        if lane_checks is None:raise ValueError('full-curve connections/gates failed: '+str(connection_failures))
        for lane,(first,last,crossings) in zip(lanes,lane_checks):
            lane['full_curve_gate_crossings']=crossings
            lane_xyz=dense[first:last+1]
            lane['event_route_metrics']=[]
            for event in ([] if tasks is None else tasks['events']):
                if event['lane']!=lane['lane']:continue
                near=lane_xyz[np.abs(lane_xyz[:,0]-event['x'])<.25]
                if not len(near):raise ValueError('event crossing absent from dense reference')
                center=centers[lane['lane']-1]
                offset=float(np.median(near[:,1])-center)
                if event['kind']!='detour' and abs(offset)<.5:
                    task_reviews.append(dict(lane=lane['lane'],event=event['id'],
                                             reason='reference lateral deviation below 0.5 m'))
                lane['event_route_metrics'].append(dict(event=event['id'],kind=event['kind'],
                    centerline_offset_world_y_m=offset,
                    lateral_offset_local_m=offset*(1 if (lane['lane']-1)%2==0 else -1)))
        agl_limits=(.8,4.5) if profile.get('agl_check_m') is None else tuple(profile['agl_check_m'])
        audit=reference_clearance_audit(dense,dict(mesh=(vertices,faces),terrain=dict(xs=terrain.xs,ys=terrain.ys,heights=terrain.heights),agl_limits=agl_limits),required)
        if not audit['passed']:raise ValueError('full multi-row curve veto: '+str(audit))
        envelope=None
        if profile['horizontal_m'] is not None:
            # The 3-D sphere above is not the planner's geometry. Veto in its own
            # terms: horizontal clearance to surfaces within the body height band.
            from omni_drones.navigation.forest.evaluation import mesh_clearance
            minimum=mesh_clearance(dense,dict(mesh=(vertices,faces)),profile['vertical_m'],
                                   horizon=profile['horizontal_m']+1.)
            envelope=dict(profile=profile['name'],required_horizontal_m=profile['horizontal_m'],
                          vertical_band_m=profile['vertical_m'],minimum_horizontal_m=minimum,
                          passed=bool(minimum>profile['horizontal_m']))
            if not envelope['passed']:raise ValueError('navigation envelope veto: '+str(envelope))
        if variable and not ((dense[:,0]>=info['bounds'][0])&(dense[:,0]<=info['bounds'][1])&
                             (dense[:,1]>=info['bounds'][2])&(dense[:,1]<=info['bounds'][3])).all():
            raise ValueError('Reference leaves the reserved site bounds')
        if variable:
            phases=reference_phases(path,intervals)
            validate_phases(phases,path.duration,len(lanes))
            for t,phase in zip(turns,[s for s in phases if s['phase']=='TURN']):
                t.update({k:phase[k] for k in ('start_s','end_s','duration_s','length_m','mean_speed_mps')})
                t.update(geometry='single_c2_quintic',nominal_half_aisle_spacing_m=t.pop('radius_m'))
            info.update(reference_phases=phases,phase_schema=dict(version=1,time_origin='reference_start_after_settle',
                interval_convention='[start_s,end_s), final endpoint included',teacher_only=True,
                default_navigation_training_phases=['INSPECT'],
                note='Collection must retain all scans; phase metadata does not certify training labels'))
            # Export the actual continuous geometry as the display polyline.
            reference=path.geometry(np.linspace(0,path.geometry.x[-1],int(path.geometry.x[-1]/.2)+2))
        else:
            times=np.linspace(0,path.duration,int(np.ceil(path.duration/.05))+1);position=path.sample(times)[0]
            for t in turns:
                edge,yc=t['center'];radius=t['radius_m']
                indices=np.flatnonzero((position[:,1]>=yc-radius-.1)&(position[:,1]<=yc+radius+.1)&
                        ((position[:,0]>=edge-.1) if t['side']=='east' else (position[:,0]<=edge+.1)))
                t.update(start_s=float(times[indices[0]]),end_s=float(times[indices[-1]]))
        info.update(start=reference[0].tolist(),goal=reference[-1].tolist(),reference=reference.tolist(),
            collection_goal=reference[-1].tolist(),initial_yaw=float(np.arctan2(*(reference[1,:2]-reference[0,:2])[::-1])),
            turns=turns,lanes=lanes,
            validation=dict(passed=not task_reviews,continuous_reference_passed=True,
                task_design_passed=not task_reviews,task_design_reviews=task_reviews,
                full_mesh=audit,navigation_envelope=envelope,duration=path.duration,route_bounds=None,full_curve_corner_scale=corner_scale,
                route_length=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).sum()),
                planner='native_esdf_aisles_c2_headlands' if variable else 'original_esdf_per_aisle_plus_headland_connections',
                observation_coverage_verified=False,rolling_six_point_verified=False))
        if variable:info['validation'].update(motion=motion,phase_partition_passed=True,
            turn_duration_s=sum(t['duration_s'] for t in turns))
        if task_reviews:info['validation']['reason']='reference_passed_but_task_design_needs_review'
        path.save(folder/'reference_trajectory.npz')
        if not flat:plot_native(out,info,terrain,dense)
        print(json.dumps(info['validation']),flush=True)
    except ValueError as error:
        info['validation'].update(reason=str(error),completed_lane_checks=len(lanes),
                                  continuous_reference_passed=False,task_design_reviews=task_reviews)
        info['lanes']=lanes
        print('REVIEW: '+str(error),flush=True)
    dump(folder/'map.json',info)
    if flat:plot_dataset_map(out,info,terrain)
    return info


def plot_dataset_map(out,info,terrain):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6.5),dpi=120)
    im=ax.pcolormesh(terrain.xs,terrain.ys,terrain.heights.T,cmap='terrain',shading='auto',alpha=.55)
    fig.colorbar(im,ax=ax,label='Ground Z (m)')
    for tree in info['trees']:
        for b in tree['branches']:
            ax.plot(*branch_axis(tree,b)[:,:2].T,color='saddlebrown',lw=.6,alpha=.5)
        ax.add_patch(plt.Circle(tree['position'],tree['radius'],color='darkorange' if tree.get('inserted') else 'darkgreen'))
    if info['validation'].get('continuous_reference_passed',info['validation']['passed']):
        from omni_drones.navigation.forest.trajectory import QuinticPath
        path=QuinticPath.load(out/'reference_trajectory.npz');xyz=path.remaining(0.,info['start'])
        ax.plot(*xyz[:,:2].T,color='royalblue',lw=1,label='Inspection reference')
        for i,turn in enumerate(info['turns']):
            p=path.sample(np.linspace(turn['start_s'],turn['end_s'],101))[0]
            ax.plot(*p[:,:2].T,color='magenta',lw=1.6,label='Headland turn' if i==0 else None)
        ax.scatter(*np.array(info['start'])[:2],marker='>',s=35,color='blue',label='Start')
    for row in info['rows']:
        if 'length_m' in row:
            ax.text(row['start_x']-1,row['y'],f'{row["length_m"]:.1f} m',fontsize=6,ha='right',color='darkgreen')
    for e in info.get('tasks',{}).get('events',[]):
        ax.text(e['x']+.3,e['obstacle_y']+.4,e['kind'],fontsize=6,color='darkred')
    status='PASS' if info['validation']['passed'] else 'REVIEW'
    ax.set(xlim=info['bounds'][:2],ylim=info['bounds'][2:],xlabel='World X (m)',ylabel='World Y (m)',
        title=f'{info["split"]} | seed {info["parameters"]["seed"]} | {info.get("tasks",{}).get("profile","natural")} | {status}')
    ax.set_aspect('equal')
    if info['validation'].get('continuous_reference_passed'):ax.legend(fontsize=7,loc='upper right')
    fig.tight_layout();fig.savefig(out/'overview.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/11_原生成器多行地图')
    p.add_argument('--seed',type=int,default=20261016)
    p.add_argument('--resume',action='store_true',help='retry planning only; verify identical existing mesh')
    a=p.parse_args();build(a.output.resolve(),a.seed,a.resume)
