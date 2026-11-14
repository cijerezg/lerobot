"""Visual case from the existing scan footprint and slot masks; collision hulls untouched."""
import numpy as np
import cv2
import trimesh
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter1d
from matplotlib.path import Path as PolyPath


def build_case(source, assets, write_mesh):
    scan = trimesh.load(source / 'assets/clipper_case/clipper_case_visual.obj', process=False)
    bounds = scan.bounds
    half = (bounds[1, :2] - bounds[0, :2]) / 2
    original = cv2.imread(str(source / 'nail_ep5/materials/clipper_case_photo.png'))
    x0, y1 = bounds[0, 0] - .0005, bounds[1, 1] + .0005
    hh, ww = original.shape[:2]
    slots = (np.max(np.abs(original.astype(float) - [154, 151, 158]), axis=2) < 2).astype(np.uint8)
    contours, _ = cv2.findContours(slots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    holes = []
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        p = contour[:, 0, :].astype(float)
        p = np.stack([x0 + (p[:, 0] + .5) / 2000, y1 - (p[:, 1] + .5) / 2000], 1)
        p = gaussian_filter1d(p, 1.7, axis=0, mode='wrap')
        if np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)) < 0:
            p = p[::-1]
        holes.append(p)

    def rounded(hx, hy, radius):
        result = []
        for cx, cy, start in [(hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                              (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)]:
            for angle in np.linspace(start, start+90, 24, endpoint=False):
                a = np.deg2rad(angle)
                result.append((cx+radius*np.cos(a), cy+radius*np.sin(a)))
        return np.array(result)

    def densify(poly):
        points = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            n = max(1, int(np.ceil(np.linalg.norm(b-a) / .001)))
            points.extend(a+(b-a)*t for t in np.arange(n)/n)
        return np.array(points)

    outer = densify(rounded(half[0]-.0018, half[1]-.0018, .0065))
    shell = densify(rounded(half[0]-.0005, half[1]-.0005, .007))
    paths, outer_path = [PolyPath(h) for h in holes], PolyPath(outer)
    xx, yy = np.meshgrid(np.arange(-half[0], half[0], .002), np.arange(-half[1], half[1], .002))
    grid = np.c_[xx.ravel(), yy.ravel()]
    valid = outer_path.contains_points(grid)
    for hole in paths:
        valid &= ~hole.contains_points(grid)
    points = np.concatenate([outer, *holes, grid[valid]])
    triangles = Delaunay(points).simplices
    samples = [points[triangles].mean(1)] + [(points[triangles[:, i]]+points[triangles[:, (i+1) % 3]])*.5 for i in range(3)]
    valid = np.ones(len(triangles), bool)
    for sample in samples:
        valid &= outer_path.contains_points(sample, radius=1e-7)
        for hole in paths:
            # Shrink by 0.1 micron so exact contour edges remain part of the top surface.
            valid &= ~hole.contains_points(sample, radius=-1e-7)
    mesh = trimesh.Trimesh(np.c_[points, np.full(len(points), .0186)], triangles[valid], process=False)
    uv = np.c_[(points[:, 0]-x0)*2000/ww, 1-(y1-points[:, 1])*2000/hh]
    write_mesh('case_foam_top', mesh, uv, np.tile([0, 0, 1.], (len(points), 1)))

    def wall(poly, zlo, zhi):
        n = len(poly)
        vertices = np.concatenate([np.c_[poly, np.full(n, zlo)], np.c_[poly, np.full(n, zhi)]])
        faces = []
        for i in range(n):
            j = (i+1) % n
            faces.extend([(i, j, n+i), (j, n+j, n+i)])
        return trimesh.Trimesh(vertices, faces, process=False)

    surfaces = [wall(outer, .006, .0186)]
    floors = []
    for hole in holes:
        inner = wall(hole, .0098, .0186)
        inner.faces = inner.faces[:, ::-1]  # Cavity walls face into the cutout.
        surfaces.append(inner)
        tri = Delaunay(hole).simplices
        valid = PolyPath(hole).contains_points(hole[tri].mean(1))
        floors.append(trimesh.Trimesh(np.c_[hole, np.full(len(hole), .0098)], tri[valid], process=False))
    write_mesh('case_foam_sides', trimesh.util.concatenate(surfaces), fix_normals=False)
    write_mesh('case_slot_floors', trimesh.util.concatenate(floors), fix_normals=False)
    tin = wall(shell, .0003, .0202)
    centres = tin.triangles_center
    band_y = y1-82/2000
    for name, mask in [('case_tin', centres[:, 1] < band_y), ('case_dark', centres[:, 1] >= band_y)]:
        write_mesh(name, tin.submesh([np.flatnonzero(mask)], append=True), fix_normals=False)
    vertices, faces = [], []
    n, k = len(shell), 10
    for i, point in enumerate(shell):
        tangent = shell[(i+1) % n]-shell[(i-1) % n]
        normal = np.array([tangent[1], -tangent[0]])
        normal /= np.linalg.norm(normal)
        for a in np.linspace(0, 2*np.pi, k, endpoint=False):
            vertices.append([*(point+normal*.00055*np.cos(a)), .0200+.00055*np.sin(a)])
    for i in range(n):
        for j in range(k):
            a, b = i*k+j, i*k+(j+1) % k
            c, d = ((i+1) % n)*k+j, ((i+1) % n)*k+(j+1) % k
            faces.extend([(a, b, c), (b, d, c)])
    write_mesh('case_rolled_rim', trimesh.Trimesh(vertices, faces, process=False))
    rng = np.random.default_rng(61)
    texture = np.full((hh*4, ww*4, 3), [193, 189, 198], float)
    texture[:82*4] = [54, 63, 63]
    noise = cv2.GaussianBlur(rng.normal(0, 1, texture.shape[:2]), (0, 0), .6)
    texture += noise[:, :, None]*2.5
    cv2.imwrite(str(assets/'case_surface.png'), np.clip(texture, 0, 255).astype(np.uint8))
    return dict(slot_count=len(holes), visual_dimensions_m=(half*2).tolist(), foam_top_m=.0186,
                rim_top_m=.02055, slot_floor_m=.0098,
                source='Existing scan footprint and material-class slot masks; dimensions regularized for appearance')
