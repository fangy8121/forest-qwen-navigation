"""Continuous headland geometry, motion audit and explicit reference phases.

All geometry here is offline teacher metadata, not an online perception map.
"""
import numpy as np
from scipy.interpolate import BPoly
from scipy.optimize import brentq
from omni_drones.navigation.forest.trajectory import QuinticPath


def motion_audit(path,flight,samples=65):
    knots=path.clock.x
    ts=(knots[:-1,None]+np.diff(knots)[:,None]*np.linspace(1e-7,1-1e-7,samples)).ravel()
    _,v,a,j=path._base(ts,True)
    vv=np.maximum(np.sum(v[:,:2]**2,axis=1),1e-8)
    yaw=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/vv
    yaw_acc=(v[:,0]*j[:,1]-v[:,1]*j[:,0])/vv-2*yaw*np.sum(v[:,:2]*a[:,:2],axis=1)/vv
    maxima=dict(max_speed=float(np.linalg.norm(v,axis=1).max()),
                max_acceleration=float(np.linalg.norm(a,axis=1).max()),
                max_jerk=float(np.linalg.norm(j,axis=1).max()),
                max_yaw_rate=float(np.abs(yaw).max()),
                max_yaw_acceleration=float(np.abs(yaw_acc).max()))
    orders=dict(max_speed=1,max_acceleration=2,max_jerk=3,max_yaw_rate=1,max_yaw_acceleration=2)
    ratio=max((value/flight[key])**(1/orders[key]) for key,value in maxima.items())
    return dict(passed=all(v<=flight[k]*1.001 for k,v in maxima.items()),maxima=maxima,time_scale_required=ratio)


def continuous_reference(lanes,flight,check,corner_scale=1.):
    geometries=[QuinticPath._geometry(np.asarray(lane['reference']),
        flight.corner_distance*corner_scale,flight.max_path_deviation,check) for lane in lanes]
    pieces=[];intervals=[];q=0.
    for i,geometry in enumerate(geometries):
        width=float(geometry.x[-1]);pieces.append(geometry)
        intervals.append(dict(phase='INSPECT',lane=i+1,q_start=q,q_end=q+width));q+=width
        if i==len(geometries)-1:continue
        other=geometries[i+1];start=geometry(geometry.x[-1]);end=other(0.)
        if abs(start[0]-end[0])>1e-6:raise ValueError('Headland endpoints must share an outer edge')
        width=1.6*abs(end[1]-start[1])
        # One quintic turn, with exact position, tangent and second-derivative
        # continuity to BOTH aisle curves. No tiny polyline corners in the arc.
        turn=BPoly.from_derivatives([0.,width],
            [[start,geometry(geometry.x[-1],nu=1),geometry(geometry.x[-1],nu=2)],
             [end,other(0.,nu=1),other(0.,nu=2)]])
        if not check(turn(np.linspace(0,width,max(101,int(width/.02)+1)))):
            raise ValueError(f'Headland {i+1} continuous curve lacks clearance')
        pieces.append(turn)
        intervals.append(dict(phase='TURN',turn_id=i+1,from_lane=i+1,to_lane=i+2,q_start=q,q_end=q+width));q+=width
    knots=[0.];coefficients=[]
    for piece in pieces:
        knots.extend((piece.x[1:]+knots[-1]).tolist());coefficients.append(piece.c)
    path=QuinticPath.from_geometry(BPoly(np.concatenate(coefficients,axis=1),knots),flight)
    audit=motion_audit(path,flight)
    # The generation audit is denser than the timing solver. Apply a tiny
    # conservative correction before publication, never relax motion limits.
    if audit['time_scale_required']>1:
        factor=audit['time_scale_required']*1.002
        path.clock=BPoly(path.clock.c.copy(),path.clock.x*factor);path._finish()
        audit=motion_audit(path,flight)
    if not audit['passed']:raise ValueError('Dense motion audit failed')
    return path,intervals,audit


def reference_phases(path,intervals,transition_m=3.):
    """Partition the entire trajectory; phase timing excludes takeoff settling."""
    phases=[]
    def at(q):
        if q<=0:return 0.
        if q>=path.geometry.x[-1]:return path.duration
        return float(brentq(lambda t:float(path.clock(t))-q,0.,path.duration))
    def add(record,phase,start,end):
        a,b=at(start),at(end)
        if b-a<1e-7:return
        ts=np.linspace(a,b,max(2,int(np.ceil((b-a)/.05))+1))
        xyz=path.sample(ts)[0]
        length=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
        phases.append(dict({k:v for k,v in record.items() if not k.startswith('q_')},
            id=f'S{len(phases):03d}',phase=phase,start_s=a,end_s=b,duration_s=b-a,
            start=xyz[0].tolist(),end=xyz[-1].tolist(),length_m=length,
            mean_speed_mps=length/(b-a),default_navigation_training=phase=='INSPECT'))
    for i,record in enumerate(intervals):
        a,b=record['q_start'],record['q_end']
        if record['phase']=='TURN':add(record,'TURN',a,b);continue
        # Locate transition boundaries by actual spatial arc distance, not time.
        qs=np.linspace(a,b,max(2,int(np.ceil((b-a)/.05))+1));xyz=path.geometry(qs)
        arc=np.r_[0.,np.linalg.norm(np.diff(xyz,axis=0),axis=1).cumsum()]
        entry=min(transition_m,arc[-1]/3) if i else 0.
        exit=min(transition_m,arc[-1]/3) if i<len(intervals)-1 else 0.
        left=float(np.interp(entry,arc,qs));right=float(np.interp(arc[-1]-exit,arc,qs))
        add(record,'ENTRY',a,left);add(record,'INSPECT',left,right);add(record,'EXIT',right,b)
    return phases


def validate_phases(phases,duration,lane_count):
    if not phases or abs(phases[0]['start_s'])>1e-6 or abs(phases[-1]['end_s']-duration)>1e-6:
        raise ValueError('Reference phases do not cover the full trajectory')
    if any(abs(a['end_s']-b['start_s'])>1e-6 for a,b in zip(phases[:-1],phases[1:])):
        raise ValueError('Reference phases contain a gap or overlap')
    if any(s['end_s']<=s['start_s'] for s in phases):raise ValueError('Empty phase')
    if [s['lane'] for s in phases if s['phase']=='INSPECT']!=list(range(1,lane_count+1)):
        raise ValueError('Reference phases omit a required aisle')
    if sum(s['phase']=='TURN' for s in phases)!=lane_count-1:raise ValueError('Missing turn')
