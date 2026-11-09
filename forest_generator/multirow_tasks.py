"""Task geometry over an unchanged native forest; all labels are teacher-only."""
import numpy as np
from scipy.spatial import cKDTree

PROFILES=('light','balanced','sequences','mixed_bypass','neighbor_detour')
STEM_GAP=.15


def stem_conflicts(trees,xy,radius,gap=STEM_GAP):
    """Exact enclosing disks for the native vertical stem template (radius .5).

    This prevents added stem intersections. It does not claim that projected
    branches are disjoint, or replace the full 3-D trajectory check.
    """
    return [i for i,t in enumerate(trees)
            if np.linalg.norm(np.asarray(t['position'])-xy)<t['radius']+radius+gap]


def native_branch_cloud(trees):
    """Native branch axes sampled at <= 5 cm; template branch radius is .1 m."""
    samples=[]
    for tree in trees:
        if tree.get('inserted'):continue
        for branch in tree['branches']:
            a,r=branch['angle'],branch['azimuth']
            start=np.r_[tree['position'],tree['ground_z']+branch['height']]
            end=start+branch['length']*np.array([np.sin(a)*np.sin(r),-np.sin(a)*np.cos(r),np.cos(a)])
            samples.append(np.linspace(start,end,int(np.ceil(branch['length']/.05))+1))
    return cKDTree(np.concatenate(samples)) if samples else None


def add_tasks(trees,rows,config,seed,profile,clearance=None,terrain=None,attempt=0,required_clearance=.75):
    if profile not in PROFILES:raise ValueError(profile)
    # The probe keeps 0.10 m above the reference clearance and bypasses at a
    # further 0.25 m, as the original 0.85 m / 0.75+0.25 m rule did for 0.75 m.
    probe_threshold=float(required_clearance)+.10
    rng=np.random.default_rng(np.random.SeedSequence([seed,90173] if attempt==0 else [seed,90173,attempt]))
    count=len(rows)-1
    assignments=['natural']*count
    templates={
        'light':['choice','lateral'],
        'balanced':['choice','lateral','lateral','sequence'],
        'sequences':['sequence','sequence','choice'],
        'mixed_bypass':['choice','choice','sequence','lateral'],
        'neighbor_detour':['detour','choice','lateral'],
    }
    # At least two aisles retain only the original sampled forest.
    kinds=templates[profile][:max(0,count-2)]
    order=rng.permutation(count)
    for lane,kind in zip(order,kinds):assignments[int(lane)]=kind
    records=[];omitted=[]
    branch_index=native_branch_cloud(trees) if terrain is not None else None
    def insert(x,y,diameter,branches,task_id):
        # Preserve original branches. Choose shorter deadwood or a nearby stem
        # position instead of letting added solids intersect them.
        conflicts=stem_conflicts(trees,np.array([x,y]),diameter/2)
        if conflicts:
            omitted.append(dict(task_id=task_id,position=[float(x),float(y)],
                diameter=float(diameter),existing_tree_indices=conflicts,
                reason='existing_stem_or_required_stem_gap'))
            return
        requested=[float(x),float(y)];chosen=None
        desired_height=float(rng.uniform(5.5,8.))
        for dx in (0.,-.2,.2,-.4,.4):
            px=x+dx
            if stem_conflicts(trees,np.array([px,y]),diameter/2):continue
            ground=float(terrain([[px,y]])[0]) if terrain is not None else 0.
            for height in (desired_height,5.5,5.,4.5):
                if branch_index is None:gap=None
                else:
                    axis=np.c_[np.full(int(np.ceil(height/.05))+1,px),np.full(int(np.ceil(height/.05))+1,y),
                               np.linspace(ground,ground+height,int(np.ceil(height/.05))+1)]
                    # Half sample spacings total <= .05 m. Require another .05
                    # physical margin outside the enclosing stem/branch radii.
                    gap=float(branch_index.query(axis)[0].min()-diameter/2-.1-.05)
                    if gap<.05:continue
                chosen=(px,height,gap);break
            if chosen is not None:break
        if chosen is None:
            omitted.append(dict(task_id=task_id,position=requested,diameter=float(diameter),
                                reason='no_nonintersecting_stem_branch_placement'))
            return
        px,height,gap=chosen
        trees.append(dict(row=None,position=[float(px),float(y)],diameter=float(diameter),
                          radius=float(diameter/2),height=float(height),inserted=True,branches=branches,task_id=task_id,
                          requested_position=requested,native_branch_clearance_lower_bound_m=gap))
    for lane,kind in enumerate(assignments):
        if kind=='natural':continue
        lo,hi=rows[lane]['y'],rows[lane+1]['y'];center=(lo+hi)/2
        n=1 if kind=='detour' else (3 if kind=='sequence' else 2)
        xs=(np.linspace(18.,44.,n)+rng.uniform(-.5,.5,n) if n>1 else rng.uniform(25.,37.,1))
        if 'start_x' in rows[lane]:
            # Keep each event and its return gates inside the overlapping stand.
            involved=rows[lane:lane+2]
            if kind=='detour':
                neighbor=lane+1 if lane+1<count else lane-1
                involved=rows[min(lane,neighbor):max(lane,neighbor)+2]
            first=max(r['start_x'] for r in involved)+11.
            last=min(r['end_x'] for r in involved)-11.
            if first>=last:raise ValueError('Rows too short for obstacle and return gates')
            xs=(np.linspace(first,last,n)+rng.uniform(-.5,.5,n) if n>1
                else rng.uniform(first,last,1))
        first_side=int(rng.choice([-1,1]))
        for j,x in enumerate(xs):
            event_kind='lateral' if kind=='sequence' else kind
            task_id=f'L{lane+1:02d}_E{j+1:02d}'
            diameter=float(rng.uniform(.5,.7))
            local_side=first_side*(-1 if j%2 else 1);side=local_side*(1 if lane%2==0 else -1)
            offset=float(rng.uniform(.10,.25))
            neighbor=lane+1 if lane+1<count else lane-1
            nominal_x=float(x);placement=None;requested_kind=event_kind;bypass_line=None
            obstacle_y=center-side*offset if event_kind=='lateral' else center
            # Teacher-side scene design only: find a local native-mesh opening,
            # retaining the same forest seed and all original branches.
            if clearance is not None:
                radius=diameter/2;half=(hi-lo)/2
                shifts=[0.]+[sign*float(d) for d in np.arange(.3,3.91,.3) for sign in (-1,1)]
                cache={}
                def native(trace,key):
                    """Best 2-3 m AGL terrain-following clearance of one probe trace."""
                    if key in cache:return cache[key]
                    samples=np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.1))+1))
                                            for a,b in zip(trace[:-1],trace[1:])])
                    best=(-np.inf,2.5)
                    window=np.maximum(0.,1.-((samples[:,0]-trace[1:-1,0].mean())/6.)**2)
                    for agl in (2.5,2.,3.):
                        score=clearance.distance(np.c_[samples,terrain(samples)+2.5+(agl-2.5)*window])
                        if score>best[0]:best=(score,agl)
                        if best[0]>probe_threshold:break
                    cache[key]=best
                    return best
                def line(cx,y):
                    # Native clearance does not depend on the inserted stem, so a
                    # bypass line is scored once and reused for every placement.
                    return native(np.c_[[cx-3.5,cx-1.2,cx+1.2,cx+3.5],[center,y,y,center]],(round(cx,3),round(y,3)))
                def lines(cx,sign,inner):
                    """Feasible bypass lines on one side, nearest the centre first."""
                    for d in np.arange(inner,half-.3+1e-9,.1):
                        score,agl=line(cx,center+sign*d)
                        if score>probe_threshold:return float(center+sign*d),score,agl
                    return None
                chosen=None
                for dx in shifts:
                    cx=nominal_x+dx
                    if event_kind=='detour':
                        adjacent=(rows[neighbor]['y']+rows[neighbor+1]['y'])/2
                        score,agl=native(np.c_[[cx-6,cx-4,cx-2,cx+2,cx+4,cx+6],
                                               [center,center,adjacent,adjacent,center,center]],('detour',round(cx,3)))
                        if score>probe_threshold:chosen=dict(x=cx,score=score,agl=agl);break
                    elif event_kind=='choice':
                        # Both sides must clear the centred stem by the threshold too.
                        found=[lines(cx,s,max(.6,radius+probe_threshold+.05)) for s in (-1,1)]
                        if all(found):
                            chosen=dict(x=cx,score=min(f[1] for f in found),agl=found[0][2],lines=[f[0] for f in found])
                            break
                if event_kind=='choice' and chosen is None:
                    # A centred stem cannot leave the envelope on both sides of
                    # this aisle (about 0.95 m each in 5 m rows): keep the event as
                    # a one-sided bypass with the stem pushed toward a row.
                    event_kind='lateral'
                if event_kind=='lateral':
                    for dx in shifts:
                        cx=nominal_x+dx
                        for sign in (side,-side):
                            found=lines(cx,sign,.6)
                            if found is None:continue
                            y,score,agl=found
                            # Stem surface threshold+0.05 m behind the open line, on
                            # the blocked side; the wall fills from that row to it.
                            stem=y-sign*(radius+probe_threshold+.05)
                            if sign*(center-stem)<.1:continue
                            chosen=dict(x=cx,score=score,agl=agl,side=sign,obstacle_y=stem,lines=[y])
                            break
                        if chosen is not None:break
                # Never place the best of failed candidates: the map would ship a
                # bypass the navigation envelope cannot fly (24/40 maps did).
                if chosen is None:
                    best=max((v[0] for v in cache.values()),default=-np.inf)
                    raise ValueError(f'{task_id}: no {requested_kind} placement with native clearance > '
                                     f'{probe_threshold:.2f} m (best {best:.2f} m over {len(cache)} probes)')
                x=chosen['x']
                if event_kind=='lateral':side,obstacle_y=chosen['side'],chosen['obstacle_y']
                bypass_line=chosen.get('lines')
                placement=dict(native_local_probe_passed=True,minimum_native_clearance_m=float(chosen['score']),
                               probe_threshold_m=probe_threshold,
                               nominal_x=nominal_x,x_shift_m=float(x-nominal_x),tested_candidates=len(cache),
                               probe_peak_agl_m=chosen['agl'],bypass_lines_world_y=bypass_line,
                               scope='native full mesh with 2-3 m AGL terrain-following probes; inserted geometry checked separately')
                local_side=side*(1 if lane%2==0 else -1)
            start=len(trees)
            if event_kind=='detour':
                for y in np.linspace(lo,hi,int(np.ceil((hi-lo)/.9))+1):
                    insert(x+float(rng.uniform(-.4,.4)),y,diameter,[],task_id)
            else:
                insert(x,obstacle_y,diameter,[],task_id)
                if event_kind=='lateral':
                    edge=lo if side>0 else hi
                    for y in np.linspace(edge,obstacle_y,int(np.ceil(abs(edge-obstacle_y)/.9))+1)[:-1]:
                        insert(x+float(rng.uniform(-.4,.4)),y,diameter,[],task_id)
            event=dict(id=task_id,lane=lane+1,kind=event_kind,sequence=kind=='sequence',requested_kind=requested_kind,
                       x=float(x),obstacle_y=float(obstacle_y),side_world_y=side if event_kind=='lateral' else 0,
                       local_side='left' if local_side>0 else 'right',
                       tree_indices=list(range(start,len(trees))),placement_probe=placement)
            if event_kind=='detour':event['neighbor_lane']=lane+2 if lane+1<count else lane
            records.append(event)
    return dict(profile=profile,assignments=assignments,events=records,
                inserted_trees=sum(t.get('inserted',False) for t in trees),
                original_trees_modified=False,student_input=False,
                placement_version=4,minimum_added_stem_gap_m=STEM_GAP,
                placement_attempt=attempt,
                added_obstacle_form='irregular standing-deadwood clusters using original stem mesh',
                omitted_redundant_stems=omitted)


def lane_problem(rows,lane,events,x_bounds=None):
    """Search bounds and visitation gates; never sent as student corridor masks."""
    lo,hi=rows[lane]['y'],rows[lane+1]['y'];center=(lo+hi)/2
    lower,upper=lo,hi;gates=[]
    for event in events:
        x=event['x'];kind=event['kind']
        if kind=='lateral':
            side=event['side_world_y'];y=event['obstacle_y']
            gates.append(dict(x=x,y_min=y+.8 if side>0 else lo,
                              y_max=hi if side>0 else y-.8,kind='task'))
        elif kind=='detour':
            other=event['neighbor_lane']-1
            lower=min(lower,rows[other]['y']);upper=max(upper,rows[other+1]['y'])
            # Visit original aisle on BOTH sides of the obstruction.
            gates.extend([dict(x=x+d,y_min=center-.65,y_max=center+.65,kind='visit_original_aisle')
                          for d in (-6.,6.)])
            gates.append(dict(x=x,y_min=rows[other]['y'],y_max=rows[other+1]['y'],kind='neighbor_detour'))
    return [*(x_bounds if x_bounds is not None else [4.,57.]),lower,upper],gates


def verify_openings(esdf,rows,lane,events,ends,gates):
    xs=esdf.origin[0]+np.arange(esdf.free.shape[0])*esdf.resolution
    ys=esdf.origin[1]+np.arange(esdf.free.shape[1])*esdf.resolution
    lo,hi=rows[lane]['y'],rows[lane+1]['y']
    results=[]
    for e in events:
        if not e['tree_indices']:raise ValueError(f'{e["id"]}: no added obstacle survived placement checks')
        plane=esdf.free[int(np.argmin(abs(xs-e['x'])))]
        positive=bool(plane[(ys>e['obstacle_y'])&(ys<hi)].any())
        negative=bool(plane[(ys<e['obstacle_y'])&(ys>lo)].any())
        r=dict(event=e['id'],positive_y_open=positive,negative_y_open=negative)
        if e['kind']=='choice':
            if not (positive and negative):raise ValueError(f'{e["id"]}: requested choice lacks two openings')
            for sign in (1,-1):
                gate=dict(x=e['x'],y_min=e['obstacle_y']+.8 if sign>0 else lo,
                          y_max=hi if sign>0 else e['obstacle_y']-.8)
                try:
                    esdf.route(ends[0],ends[1],gates+[gate])
                except ValueError as error:
                    raise ValueError(f'{e["id"]}: choice side {sign:+d} reference search failed: {error}') from error
            r['both_sides_connected']=True
        elif e['kind']=='lateral':
            desired=positive if e['side_world_y']>0 else negative
            undesired=negative if e['side_world_y']>0 else positive
            if not desired or undesired:raise ValueError(f'{e["id"]}: one-sided geometry mismatch')
            r['one_sided_verified']=True
        else:
            if plane[(ys>lo)&(ys<hi)].any():raise ValueError(f'{e["id"]}: detour barrier leaves original aisle open')
            r['original_aisle_blocked']=True
        results.append(r)
    return results
