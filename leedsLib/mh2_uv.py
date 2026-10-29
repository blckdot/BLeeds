# MH2 UV splitter. GPL-3.0-or-later

"""Split triangles at UV tile boundaries and wrap UVs into 0-1, preserving all UV layers."""

import math

def split_repeat_triangle(corners):
    """corners: three (bary, [uv0, uv1, ...]) tuples with matching UV layer counts."""
    if len(corners) != 3 or len({len(c) for c in corners}) != 1:
        raise ValueError('UV splitting requires three corners with matching layers')
    polygons = [[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]]
    def value(bary, layer, axis):
        return sum(bary[i]*corners[i][layer][axis] for i in range(3))
    for layer in range(len(corners[0])):
        for axis in range(2):
            values = [c[layer][axis] for c in corners]
            if not all(math.isfinite(v) and abs(v) < 10000 for v in values):
                raise ValueError('Invalid UV coordinate')
            low, high = min(values), max(values)
            if high-low > 64:
                raise ValueError('A triangle spans more than 64 UV tiles; reduce its UV range before exporting')
            for boundary in range(math.floor(low)+1, math.ceil(high)):
                result = []
                for poly in polygons:
                    distances = [value(b, layer, axis)-boundary for b in poly]
                    if min(distances) >= -1e-12 or max(distances) <= 1e-12:
                        result.append(poly)
                        continue
                    for sign in (-1, 1):
                        clipped = []
                        for i, a in enumerate(poly):
                            b = poly[(i+1) % len(poly)]
                            da, db = distances[i], distances[(i+1) % len(poly)]
                            if sign*da >= 0:
                                clipped.append(a)
                            if da*db < 0:
                                t = da/(da-db)
                                clipped.append(tuple(a[j]+t*(b[j]-a[j]) for j in range(3)))
                        if len(clipped) >= 3:
                            result.append(clipped)
                polygons = result
                if len(polygons) > 4096:
                    raise ValueError('UV splitting creates too many pieces; reduce the triangle UV range')
    triangles = []
    for poly in polygons:
        tiles = [[math.floor(sum(value(b, layer, axis) for b in poly)/len(poly)) for axis in range(2)]
                 for layer in range(len(corners[0]))]
        for i in range(1, len(poly)-1):
            barys = [poly[0], poly[i], poly[i+1]]
            a, b, c = barys
            area = abs((b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]))
            if area <= 1e-12:
                continue
            triangle = []
            for bary in barys:
                uvs = [tuple(max(0., min(1., value(bary, layer, axis)-tiles[layer][axis])) for axis in range(2))
                       for layer in range(len(corners[0]))]
                triangle.append((bary, uvs))
            triangles.append(triangle)
    return triangles
