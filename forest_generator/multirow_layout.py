"""Variable native row extents and teacher-only aisle endpoint geometry."""
from copy import deepcopy
import numpy as np


def layout_options(mode='variable',length_range=(65.,75.),start_jitter=2.):
    low,high=map(float,length_range)
    if mode not in ('variable','legacy') or not np.isfinite([low,high,start_jitter]).all():
        raise ValueError('Invalid multirow layout')
    if low<30 or high<low or start_jitter<0:
        raise ValueError('Row lengths must satisfy 30 <= min <= max; jitter must be nonnegative')
    return dict(mode=mode,length_range=[low,high],start_jitter=float(start_jitter),correlation=.35)


def configure(config,options):
    config=deepcopy(config)
    if options['mode']=='legacy':
        config.update(forest_size=[40.,32.],forest_margin=0.)
        return config,np.array([10.,3.]),[60.,36.]
    high=options['length_range'][1];jitter=options['start_jitter']
    config.update(forest_size=[high+2*jitter,32.],forest_margin=0.,
                  row_layout={k:v for k,v in options.items() if k!='mode'})
    # Native branch lengths reach 10 m. Reserve land, never trim the branches.
    return config,np.array([20.,3.]),[high+2*jitter+40.,36.]


def translate_rows(rows,translation):
    for row in rows:
        row['y']+=float(translation[1])
        if 'start_x' in row:
            row['start_x']+=float(translation[0]);row['end_x']+=float(translation[0])


def row_envelopes(trees,rows):
    """Conservative XY projection of stems AND all branches, at every height."""
    extents=[]
    for row in rows:
        xs=[]
        for tree in trees:
            if tree['row']!=row['row']:continue
            x=tree['position'][0];xs.extend([x-tree['radius'],x+tree['radius']])
            for branch in tree['branches']:
                end=x+branch['length']*np.sin(branch['angle'])*np.sin(branch['azimuth'])
                xs.extend([min(x,end)-.1,max(x,end)+.1])
        if not xs:raise ValueError('Empty tree row')
        extents.append([min(xs),max(xs)])
    return np.asarray(extents)


def aisle_endpoints(trees,rows,options):
    if options['mode']=='legacy':return np.tile([6.,55.],(len(rows)-1,1))
    extent=row_envelopes(trees,rows)
    # Space for the straight approach and 0.75 m flight-center clearance.
    ends=np.array([[min(a[0],b[0])-2.,max(a[1],b[1])+2.]
                   for a,b in zip(extent[:-1],extent[1:])])
    for lane in range(len(ends)-1):
        side=1 if lane%2==0 else 0
        edge=(max if side else min)(ends[lane,side],ends[lane+1,side])
        ends[lane:lane+2,side]=edge
    return ends
