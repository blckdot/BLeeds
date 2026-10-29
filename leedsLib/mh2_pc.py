# BLeeds MH2 PC core. GPL-3.0-or-later.

"""Read and write MH2 PC PMLC models, including object-info chains and vertex streams."""

import math
import struct
import zlib


LAYOUTS = {
    0x52: (24, None, 12, 20, ()),
    0x152: (32, None, 12, 20, (24,)),
    0x252: (40, None, 12, 20, (24, 32)),
    0x115E: (52, 12, 32, 40, (44,)),
    0x125E: (60, 12, 32, 40, (44, 52)),
}


def close_values(a, b, tolerance=2e-6):
    return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))


def normalized_weights(indices, weights):
    merged = {}
    for index, weight in zip(indices, weights):
        if weight > 1e-6:
            merged[index] = merged.get(index, 0.0) + weight
    total = sum(merged.values())
    return {i: w / total for i, w in merged.items()} if total else {}


def same_weights(a, b):
    return set(a) == set(b) and all(abs(a[i] - b[i]) <= 2e-6 for i in a)


class PcMdl:
    def __init__(self, data):
        if data[:4] == b'Z2HM':
            declared = struct.unpack_from('<I', data, 4)[0]
            data = zlib.decompress(data[8:])
            if declared != len(data):
                raise ValueError('Z2HM decompressed size mismatch')
        self.data = bytes(data)
        if len(data) < 40 or data[:4] != b'PMLC':
            raise ValueError('Expected an MH2 PC PMLC model')
        if self.u32(8) != len(data):
            raise ValueError('PMLC file size mismatch')
        self.table_start, self.table_count = self.u32(16), self.u32(20)
        self.range(self.table_start, self.table_count * 4)
        if self.u32(12) != self.table_start:
            raise ValueError('Unsupported PC data/relocation boundary')
        if self.table_start + self.table_count * 4 != len(data):
            raise ValueError('Unsupported trailing relocation data')
        self.relocations = struct.unpack_from('<{}I'.format(self.table_count), data, self.table_start)
        for field in self.relocations:
            self.range(field, 4, self.table_start)
            target = self.u32(field)
            if target and target >= self.table_start:
                raise ValueError('Relocation points outside model data')
        first, last = self.u32(32), self.u32(36)
        if first != last or self.u32(first) != 32 or self.u32(first + 4) != 32:
            raise ValueError('Export currently supports single-entry PC skins only')
        self.entry = self.u32(first + 8)
        self.range(self.entry, 32, self.table_start)
        self.root_bone = self.u32(self.entry)
        self.parts = self._parts()

    def range(self, offset, size, limit=None):
        if offset < 0 or size < 0 or offset + size > (len(self.data) if limit is None else limit):
            raise ValueError('PMLC range outside file: 0x{:X}+{}'.format(offset, size))

    def u32(self, offset):
        self.range(offset, 4)
        return struct.unpack_from('<I', self.data, offset)[0]

    def _parts(self):
        # PC entry object-info sentinel is Entry + 20.
        sentinel = self.entry + 20
        current, last = self.u32(sentinel), self.u32(sentinel + 4)
        previous, visited, parts = sentinel, set(), []
        while current != sentinel:
            if current in visited:
                raise ValueError('Cycle in object-info list')
            visited.add(current)
            self.range(current, 28, self.table_start)
            if self.u32(current + 4) != previous:
                raise ValueError('Broken object-info backward link')
            obj = self.u32(current + 12)
            self.range(obj, 180, self.table_start)
            matids, indices, vertices = self.u32(obj + 44), self.u32(obj + 48), self.u32(obj + 80)
            stride, vet = self.u32(obj + 96), self.u32(obj + 144)
            if vet not in LAYOUTS or stride != LAYOUTS[vet][0] or indices % 3:
                raise ValueError('Unsupported PC vertex layout or face count at 0x{:X}'.format(obj))
            face_start = obj + 180 + matids * 44
            vertex_start = face_start + indices * 2
            self.range(obj, vertex_start + vertices * stride - obj, self.table_start)
            faces = tuple(struct.iter_unpack('<3H', self.data[face_start:vertex_start]))
            if any(i >= vertices for f in faces for i in f):
                raise ValueError('Face references an invalid vertex')
            face_materials = [0] * len(faces)
            for index in range(matids):
                field = obj + 180 + index * 44
                count, material, start, _ = struct.unpack_from('<4H', self.data, field + 24)
                if count % 3 or start % 3 or start + count > indices:
                    raise ValueError('Invalid material face range')
                face_materials[start // 3:(start + count) // 3] = [material] * (count // 3)
            parts.append(dict(info=current, offset=obj, parent=self.u32(current + 8),
                              vertices=vertices, stride=stride, vet=vet,
                              vertex_start=vertex_start, faces=faces,
                              face_materials=face_materials, material_count=self.u32(obj + 4)))
            previous, current = current, self.u32(current)
        if not parts or previous != last:
            raise ValueError('Incomplete PC object list')
        return parts

    def row(self, part, index):
        start = part['vertex_start'] + index * part['stride']
        return self.data[start:start + part['stride']]

    def rebuild(self, geometry):
        """Write new mesh data to a validated PC PMLC model, retaining all other records."""
        if len(geometry) != len(self.parts):
            raise ValueError('All original mesh slots must be supplied')
        output = bytearray(self.data[:self.table_start])
        relocations = list(self.relocations)
        for part, geo in zip(self.parts, geometry):
            vertices, faces, materials = geo['vertices'], geo['faces'], geo['materials']
            if not vertices or not faces or len(vertices) > 65536 or len(faces) * 3 > 65535:
                raise ValueError('Mesh needs 1-65536 split vertices and 1-21845 triangles')
            if len(materials) != len(faces):
                raise ValueError('Each triangle needs one material ID')
            max_material = max(1, len(geo.get('material_names', [])) or part['material_count'])
            if any(not isinstance(m, int) or not 0 <= m < max_material for m in materials):
                raise ValueError('Invalid material ID')
            if any(len(f) != 3 or len(set(f)) != 3 or any(not isinstance(i, int) or not 0 <= i < len(vertices) for i in f) for f in faces):
                raise ValueError('Invalid triangle vertex index')
            order = sorted(range(len(faces)), key=lambda i: materials[i])
            faces = [faces[i] for i in order]
            materials = [materials[i] for i in order]
            records = bytearray()
            for material in sorted(set(materials)):
                start = materials.index(material)
                count = materials.count(material)
                ids = {i for f in faces[start:start + count] for i in f}
                positions = [vertices[i]['position'] for i in ids]
                bounds = [min(p[a] for p in positions) for a in range(3)]
                bounds += [max(p[a] for p in positions) for a in range(3)]
                records.extend(struct.pack('<6f4H12x', *bounds, count * 3, material, start * 3, 0x91))
            header = bytearray(self.data[part['offset']:part['offset'] + 180])
            vertex_type = geo.get('vertex_type', part['vet'])
            if vertex_type not in LAYOUTS:
                raise ValueError('Unsupported rebuilt vertex format')
            struct.pack_into('<I', header, 96, LAYOUTS[vertex_type][0])
            struct.pack_into('<I', header, 144, vertex_type)
            names = geo.get('material_names')
            if names:
                if part['material_count'] == 0:
                    raise ValueError('Cannot add material slots to a mesh part without an original material table')
                if len(names) > 65535:
                    raise ValueError('Too many material slots')
                output.extend(b'\0' * (-len(output) % 4))
                material_table = len(output)
                original_table = self.u32(part['offset'])
                for i in range(len(names)):
                    record = original_table + min(i, part['material_count'] - 1) * 12
                    output.extend(self.data[record:record + 12])
                for i, name in enumerate(names):
                    encoded = name.encode('ascii')
                    if not encoded or len(encoded) > 31 or b'\0' in encoded:
                        raise ValueError('Texture name must be 1-31 ASCII characters')
                    field = material_table + i * 12
                    struct.pack_into('<I', output, field, len(output))
                    relocations.append(field)
                    output.extend(encoded + b'\0')
                struct.pack_into('<I', header, 0, material_table)
                struct.pack_into('<I', header, 4, len(names))
            struct.pack_into('<II', header, 44, len(set(materials)), len(faces) * 3)
            struct.pack_into('<I', header, 80, len(vertices))
            positions = [v['position'] for v in vertices]
            center = [(min(p[a] for p in positions) + max(p[a] for p in positions)) / 2 for a in range(3)]
            radius = max(math.sqrt(sum((p[a] - center[a]) ** 2 for a in range(3))) for p in positions)
            struct.pack_into('<4f', header, 52, *center, radius)
            buffer = bytearray()
            for vertex in vertices:
                buffer.extend(pack_vertex(vertex_type, vertex))
            block = header + records + b''.join(struct.pack('<3H', *f) for f in faces) + buffer
            struct.pack_into('<I', block, 36, len(block) - 32)
            output.extend(b'\0' * (-len(output) % 16))
            offset = len(output)
            output.extend(block)
            struct.pack_into('<I', output, part['info'] + 12, offset)
            # Duplicate the original header's pointer locations for the new
            # header; every target remains at its original metadata address.
            for field in self.relocations:
                if part['offset'] <= field < part['offset'] + 180:
                    relocations.append(offset + field - part['offset'])
        output.extend(b'\0' * (-len(output) % 4))
        table = len(output)
        relocations = sorted(set(relocations))
        output.extend(struct.pack('<{}I'.format(len(relocations)), *relocations))
        struct.pack_into('<4I', output, 8, len(output), table, table, len(relocations))
        result = bytes(output)
        PcMdl(result)
        return result

    def rewrite(self, streams):
        """Write new vertex streams to a validated PC PMLC model, retaining all other records."""
        if len(streams) != len(self.parts):
            raise ValueError('All original mesh parts must be exported')
        output = bytearray(self.data)
        changed = dict(position=0, normal=0, color=0, uv=0, skin=0)
        for part, stream in zip(self.parts, streams):
            if len(stream) != part['vertices']:
                raise ValueError('Topology changes are not supported by the PC round-trip exporter')
            _, weight_offset, normal_offset, color_offset, uv_offsets = LAYOUTS[part['vet']]
            geometry_changed = False
            for index, values in enumerate(stream):
                base = part['vertex_start'] + index * part['stride']
                raw = self.row(part, index)
                pos = tuple(values['position'])
                old_pos = struct.unpack_from('<3f', raw)
                if not all(math.isfinite(x) and abs(x) < 100000 for x in pos):
                    raise ValueError('Invalid vertex position')
                if not close_values(pos, old_pos):
                    struct.pack_into('<3f', output, base, *pos)
                    changed['position'] += 1
                    geometry_changed = True
                normal = tuple(values['normal'])
                length = math.sqrt(sum(x * x for x in normal))
                if not math.isfinite(length) or length < 1e-8:
                    raise ValueError('Invalid vertex normal')
                normal = tuple(x / length for x in normal)
                packed = struct.unpack_from('<3h', raw, normal_offset)
                old_length = math.sqrt(sum(x * x for x in packed))
                old_normal = tuple(x / old_length for x in packed) if old_length else (0, 0, 0)
                if not close_values(normal, old_normal, 1e-4):
                    struct.pack_into('<3h', output, base + normal_offset,
                                     *(max(-32768, min(32767, round(x * 32768))) for x in normal))
                    changed['normal'] += 1
                color = tuple(values['color'])
                if len(color) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in color):
                    raise ValueError('Invalid RGBA vertex color')
                old_bgra = struct.unpack_from('<4B', raw, color_offset)
                old_rgba = tuple(old_bgra[i] / 255 for i in (2, 1, 0, 3))
                # BYTE_COLOR stores linear values through an sRGB quantization;
                # preserve the source if it survives the importer within 1 byte.
                if not close_values(color, old_rgba, 1 / 255):
                    rgba = [round(x * 255) for x in color]
                    struct.pack_into('<4B', output, base + color_offset, rgba[2], rgba[1], rgba[0], rgba[3])
                    changed['color'] += 1
                if len(values['uvs']) != len(uv_offsets):
                    raise ValueError('Missing original UV layer (including blood UV2)')
                for uv, offset in zip(values['uvs'], uv_offsets):
                    if len(uv) != 2 or not all(math.isfinite(x) and abs(x) < 10000 for x in uv):
                        raise ValueError('Invalid UV coordinate')
                    old_uv = struct.unpack_from('<2f', raw, offset)
                    if not close_values(uv, old_uv):
                        struct.pack_into('<2f', output, base + offset, *uv)
                        changed['uv'] += 1
                if weight_offset is not None:
                    weights = values['skin']
                    if not 1 <= len(weights) <= 4 or not all(
                            isinstance(i, int) and 0 <= i <= 255 and math.isfinite(w) and w > 0
                            for i, w in weights.items()):
                        raise ValueError('Every skinned vertex requires 1-4 valid influences')
                    total = sum(weights.values())
                    if abs(total - 1) > 1e-4:
                        raise ValueError('Skin weights must sum to 1')
                    old_weights = struct.unpack_from('<4f', raw, weight_offset)
                    old_indices = struct.unpack_from('<4B', raw, weight_offset + 16)
                    if not same_weights(weights, normalized_weights(old_indices, old_weights)):
                        ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
                        ordered += [(0, 0.0)] * (4 - len(ordered))
                        struct.pack_into('<4f', output, base + weight_offset, *(w / total for _, w in ordered))
                        struct.pack_into('<4B', output, base + weight_offset + 16, *(i for i, _ in ordered))
                        changed['skin'] += 1
            if geometry_changed:
                # Bounds must be source-space, like the original vertex buffer.
                positions = [struct.unpack_from('<3f', output, part['vertex_start'] + i * part['stride'])
                             for i in range(part['vertices'])]
                low = [min(p[a] for p in positions) for a in range(3)]
                high = [max(p[a] for p in positions) for a in range(3)]
                center = [(a + b) * 0.5 for a, b in zip(low, high)]
                radius = max(math.sqrt(sum((p[a] - center[a]) ** 2 for a in range(3))) for p in positions)
                struct.pack_into('<4f', output, part['offset'] + 52, *center, radius)
                for mi in range(self.u32(part['offset'] + 44)):
                    record = part['offset'] + 180 + mi * 44
                    count, _, start, _ = struct.unpack_from('<4H', output, record + 24)
                    ids = {v for face in part['faces'][start // 3:(start + count) // 3] for v in face}
                    if ids:
                        bounds = [min(positions[v][a] for v in ids) for a in range(3)]
                        bounds += [max(positions[v][a] for v in ids) for a in range(3)]
                        struct.pack_into('<6f', output, record, *bounds)
        # Independent parse of the complete output catches structural corruption.
        PcMdl(bytes(output))
        return bytes(output), changed


def pack_vertex(vet, values):
    stride, weight_offset, normal_offset, color_offset, uv_offsets = LAYOUTS[vet]
    row = bytearray(stride)
    position, normal, color = values['position'], values['normal'], values['color']
    if len(position) != 3 or not all(math.isfinite(x) and abs(x) < 100000 for x in position):
        raise ValueError('Invalid vertex position')
    if len(normal) != 3 or not all(math.isfinite(x) for x in normal):
        raise ValueError('Invalid vertex normal')
    length = math.sqrt(sum(x * x for x in normal))
    if length < 1e-8:
        raise ValueError('Zero-length normal')
    if len(color) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in color):
        raise ValueError('Invalid vertex color')
    struct.pack_into('<3f', row, 0, *position)
    struct.pack_into('<4h', row, normal_offset,
                     *(max(-32768, min(32767, round(x / length * 32768))) for x in normal), 0)
    rgba = [round(x * 255) for x in color]
    struct.pack_into('<4B', row, color_offset, rgba[2], rgba[1], rgba[0], rgba[3])
    if len(values['uvs']) != len(uv_offsets):
        raise ValueError('Missing UV layer')
    for uv, offset in zip(values['uvs'], uv_offsets):
        if len(uv) != 2 or not all(math.isfinite(x) and abs(x) < 10000 for x in uv):
            raise ValueError('Invalid UV coordinate')
        struct.pack_into('<2f', row, offset, *uv)
    if weight_offset is not None:
        weights = values['skin']
        if not 1 <= len(weights) <= 4 or not all(isinstance(i, int) and 0 <= i < 256 and math.isfinite(w) and w > 0 for i, w in weights.items()):
            raise ValueError('Skinned vertices require 1-4 valid influences')
        total = sum(weights.values())
        if abs(total - 1) > 1e-4:
            raise ValueError('Skin weights must sum to 1')
        ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        ordered += [(0, 0.0)] * (4 - len(ordered))
        struct.pack_into('<4f4B', row, weight_offset, *(w / total for _, w in ordered), *(i for i, _ in ordered))
    return bytes(row)
