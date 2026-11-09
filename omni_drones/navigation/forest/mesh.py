"""原项目 Collada 网格读取与固定高度水平净空；不依赖 Isaac 的 CPU 几何。"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def tapered_tube(start, end, radius, tip_radius, sides=10):
    """有封口的渐细枝干三角网格；法线朝外，同一表面用于显示、碰撞和扫描。"""
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end-start
    axis /= np.linalg.norm(axis)
    helper = np.array([1.,0,0]) if abs(axis[0]) < .9 else np.array([0.,1,0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    angles = np.arange(sides)*2*np.pi/sides
    ring = np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
    vertices = np.vstack([start+radius*ring, end+tip_radius*ring, start, end])
    faces = []
    for i in range(sides):
        j = (i+1)%sides
        faces.extend([[i,j,sides+j], [i,sides+j,sides+i],
                      [2*sides,j,i], [2*sides+1,sides+i,sides+j]])
    return vertices, np.asarray(faces, dtype=np.int32)


def section_grid(triangles, bounds, resolution=.2):
    """将高度带三角形保守覆盖到栅格；细枝不能因为没穿过格心而消失。"""
    from .geometry import segment_distances
    origin = np.array([bounds[0], bounds[2]], dtype=float)
    shape = np.rint((np.array([bounds[1],bounds[3]])-origin)/resolution).astype(int)+1
    occupied = np.zeros(shape, dtype=bool)
    padding = resolution/np.sqrt(2)
    for tri in triangles:
        lo = np.maximum(0, np.floor((tri.min(axis=0)-padding-origin)/resolution).astype(int))
        hi = np.minimum(shape-1, np.ceil((tri.max(axis=0)+padding-origin)/resolution).astype(int))
        if (hi < lo).any():
            continue
        ix, iy = np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij')
        points = origin+np.c_[ix.ravel(),iy.ravel()]*resolution
        edges = np.roll(tri,-1,axis=0)-tri
        delta = points[:,None,:]-tri
        signs = edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
        area = abs(edges[0,0]*edges[1,1]-edges[0,1]*edges[1,0])
        inside = ((signs >= -1e-10).all(axis=1) | (signs <= 1e-10).all(axis=1)) & (area > 1e-12)
        near = np.minimum.reduce([segment_distances(points,a,b) for a,b in zip(tri,np.roll(tri,-1,axis=0))]) <= padding
        occupied[ix.ravel(),iy.ravel()] |= inside | near
    return occupied, origin


def read_collada(path, with_normals=False):
    root = ET.parse(path).getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    meshes = root.findall('c:library_geometries/c:geometry/c:mesh', ns)
    nodes = root.findall('c:library_visual_scenes/c:visual_scene/c:node', ns)
    # 当前导入器针对仓库 Potato 导出的单网格；未知格式明确失败，不默默丢对象。
    if len(meshes) != 1 or len(nodes) != 1 or root.findtext('c:asset/c:up_axis', namespaces=ns) != 'Z_UP':
        raise ValueError(f'expected one Z_UP forest mesh: {path}')
    node = nodes[0]
    if any(x.tag.rsplit('}', 1)[-1] not in {'matrix', 'instance_geometry'} for x in node):
        raise ValueError('unsupported Collada node transform')
    matrices = node.findall('c:matrix', ns)
    if len(matrices) != 1:
        raise ValueError('forest must have one explicit transform matrix')
    transform = np.fromstring(matrices[0].text, sep=' ').reshape(4, 4).T
    mesh = meshes[0]
    sources = {}
    for source in mesh.findall('c:source', ns):
        accessor = source.find('c:technique_common/c:accessor', ns)
        sources[source.get('id')] = np.fromstring(source.findtext('c:float_array', namespaces=ns), sep=' ').reshape(-1, int(accessor.get('stride')))
    position = mesh.find('c:vertices/c:input[@semantic="POSITION"]', ns)
    vertices = sources[position.get('source')[1:]]
    vertices = (np.c_[vertices, np.ones(len(vertices))] @ transform.T)[:, :3]
    vertices *= float(root.find('c:asset/c:unit', ns).get('meter'))
    faces, uvs, normals = [], [], []
    for triangles in mesh.findall('c:triangles', ns):
        inputs = triangles.findall('c:input', ns)
        stride = max(int(x.get('offset')) for x in inputs)+1
        indices = np.fromstring(triangles.findtext('c:p', namespaces=ns), sep=' ', dtype=np.int64).reshape(-1, 3, stride)
        vertex = next(x for x in inputs if x.get('semantic') == 'VERTEX')
        tex = next(x for x in inputs if x.get('semantic') == 'TEXCOORD')
        faces.append(indices[:, :, int(vertex.get('offset'))])
        uvs.append(sources[tex.get('source')[1:]][indices[:, :, int(tex.get('offset'))]])
        if with_normals:
            normal = next(x for x in inputs if x.get('semantic') == 'NORMAL')
            normals.append(sources[normal.get('source')[1:]][indices[:, :, int(normal.get('offset'))]])
    if not faces:
        raise ValueError('forest contains no triangle primitives')
    faces, uvs = np.concatenate(faces), np.concatenate(uvs)
    if not np.isfinite(vertices).all() or not np.isfinite(uvs).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('invalid forest mesh data')
    if with_normals:
        normals = np.concatenate(normals) @ np.linalg.inv(transform[:3, :3])
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return vertices, faces, uvs, normals
    return vertices, faces, uvs


def ground_heights(points, triangles):
    """取竖直射线最下方交点作为地面，避免把树枝顶面当地面。"""
    lo, hi = triangles[:, :, :2].min(axis=1), triangles[:, :, :2].max(axis=1)
    result = []
    for point in np.asarray(points):
        tri = triangles[((lo <= point).all(axis=1)) & ((hi >= point).all(axis=1))]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac, ap = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], point-a[:, :2]
        determinant = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
        good = np.abs(determinant) > 1e-12
        a, b, c, ab, ac, ap, determinant = [x[good] for x in (a,b,c,ab,ac,ap,determinant)]
        u = (ap[:, 0]*ac[:, 1]-ap[:, 1]*ac[:, 0])/determinant
        v = (ab[:, 0]*ap[:, 1]-ab[:, 1]*ap[:, 0])/determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
        heights = a[:, 2]+u*(b[:, 2]-a[:, 2])+v*(c[:, 2]-a[:, 2])
        result.append(float(heights[inside].min()) if inside.any() else np.nan)
    return np.asarray(result)


def horizontal_section(vertices, faces, lower, upper):
    """先裁到机体高度带再投影；不能把高处树枝全投到飞行平面上。"""
    triangles = np.asarray(vertices)[faces]
    triangles = triangles[(triangles[:, :, 2].max(axis=1) >= lower) & (triangles[:, :, 2].min(axis=1) <= upper)]
    projected = []
    for triangle in triangles:
        polygon = list(triangle)
        for boundary, sign in ((lower, 1), (upper, -1)):
            clipped = []
            for i, current in enumerate(polygon):
                previous = polygon[i-1]
                cin, pin = sign*(current[2]-boundary) >= 0, sign*(previous[2]-boundary) >= 0
                if cin != pin:
                    clipped.append(previous+(current-previous)*(boundary-previous[2])/(current[2]-previous[2]))
                if cin:
                    clipped.append(current)
            polygon = clipped
        for i in range(1, len(polygon)-1):
            projected.append([polygon[0][:2], polygon[i][:2], polygon[i+1][:2]])
    return np.asarray(projected, dtype=float).reshape(-1, 3, 2)


def projected_clearance(path, triangles):
    """折线到投影三角形的全线段距离，包含端点落在面内部的情况。"""
    path, triangles = np.asarray(path, dtype=float), np.asarray(triangles, dtype=float)
    if not len(triangles):
        return float('inf')
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    edges = np.stack([triangles, np.roll(triangles, -1, axis=1)], axis=2).reshape(-1, 2, 2)
    starts, ends = edges[:, 0], edges[:, 1]
    delta = ends-starts
    norm2 = np.sum(delta*delta, axis=1)
    area = cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    best = float('inf')
    for point in path:
        signs = cross(np.roll(triangles, -1, axis=1)-triangles, point-triangles)
        if (((signs >= -1e-10).all(axis=1) | (signs <= 1e-10).all(axis=1)) & (np.abs(area) > 1e-12)).any():
            return 0.
        alpha = np.clip(np.sum((point-starts)*delta, axis=1)/np.maximum(norm2, 1e-20), 0, 1)
        best = min(best, float(np.linalg.norm(point-starts-alpha[:, None]*delta, axis=1).min()))
    for a, b in zip(path[:-1], path[1:]):
        direction = b-a
        denominator = cross(direction, delta)
        usable = np.abs(denominator) > 1e-12
        t = cross(starts[usable]-a, delta[usable])/denominator[usable]
        u = cross(starts[usable]-a, direction)/denominator[usable]
        if ((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)).any():
            return 0.
        for points in (starts, ends):
            alpha = np.clip(np.sum((points-a)*direction, axis=1)/max(float(direction@direction), 1e-20), 0, 1)
            best = min(best, float(np.linalg.norm(points-a-alpha[:, None]*direction, axis=1).min()))
    return best
