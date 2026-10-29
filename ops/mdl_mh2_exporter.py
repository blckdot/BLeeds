# BLeeds MH2 PC exporter. GPL-3.0-or-later.

"""Export Manhunt 2 PC PMLC models, including object-info chains and vertex streams."""

import io
import os
import tempfile
import struct
import shutil
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from ..leedsLib.mh2_pc import PcMdl, LAYOUTS, close_values, pack_vertex
from ..leedsLib.native_mdl import Manhunt2MdlReader


def find_root(context):
    candidates = [context.active_object] + list(context.selected_objects)
    for candidate in candidates:
        while candidate is not None:
            if candidate.get('bleeds_model_game') == 'MH2' and candidate.get('bleeds_is_mdl_root'):
                return candidate
            candidate = candidate.parent
    roots = [o for o in context.scene.objects
             if o.get('bleeds_model_game') == 'MH2' and o.get('bleeds_is_mdl_root')]
    if len(roots) == 1:
        return roots[0]
    raise ValueError('Select an imported Manhunt 2 PC model or one of its mesh parts')


def read_reference(path, context, model):
    reader = Manhunt2MdlReader(str(path), context, layout_mode='PC_RETAIL')
    reader.file_data = model.data
    reader.file_size = len(model.data)
    reader.file = io.BytesIO(model.data)
    reader.read_header()
    reader.read_entry()
    reader.read_bones()
    reader.complete_missing_local_bone_matrices()
    reader.read_animation_bone_bindings()
    return reader


def validate_skeleton(root, reader, allow_rest_changes=False):
    if root.type != 'ARMATURE':
        raise ValueError('Initial MH2 PC export requires the imported armature')
    if len(root.data.bones) != len(reader.bone_map):
        raise ValueError('Bone additions/removals require a full skeleton rebuild')
    rebuilt = {}

    def world(offset):
        if offset not in rebuilt:
            info = reader.bone_map[offset]
            parent = info['parent_offset']
            rebuilt[offset] = world(parent) @ info['local_matrix'] if parent in reader.bone_map else info['matrix']
        return rebuilt[offset]

    for offset, info in reader.bone_map.items():
        bone = root.data.bones.get(info['name'])
        if bone is None or int(bone.get('bleeds_mh2_bone_offset', -1)) != offset:
            raise ValueError('Original bone name/identity changed: ' + info['name'])
        parent = reader.bone_map.get(info['parent_offset'])
        if (bone.parent.name if bone.parent else None) != (parent['name'] if parent else None):
            raise ValueError('Bone hierarchy changed: ' + bone.name)
        if 'anim_bone_id' in info and int(bone.get('bleeds_mh2_anim_bone_id', -999)) != info['anim_bone_id']:
            raise ValueError('Animation BoneID changed: ' + bone.name)
        if allow_rest_changes:
            import math
            if not all(math.isfinite(v) for row in bone.matrix_local for v in row):
                raise ValueError('Nonfinite bone rest matrix: ' + bone.name)
            if abs(bone.matrix_local.to_3x3().determinant() - 1) > 1e-4:
                raise ValueError('Bone rest matrix must be rigid: ' + bone.name)
            continue
        matrix = world(offset)
        y = (matrix.to_3x3() @ Vector((0, 1, 0))).normalized()
        z = matrix.to_3x3() @ Vector((0, 0, 1))
        z = (z - y * z.dot(y)).normalized()
        if not close_values(tuple(bone.head_local), tuple(matrix.to_translation()), 2e-5):
            raise ValueError('Bone rest position changed: ' + bone.name)
        if (not close_values(tuple(bone.matrix_local.col[1][:3]), tuple(y), 2e-4)
                or not close_values(tuple(bone.matrix_local.col[2][:3]), tuple(z), 2e-4)):
            raise ValueError('Bone rest orientation changed: ' + bone.name)
    identity = Matrix.Identity(4)
    if any(abs(root.matrix_world[r][c] - identity[r][c]) > 1e-6 for r in range(4) for c in range(4)):
        raise ValueError('Keep the imported armature object transform unchanged')


def _corner_values(mesh, loop_index, uv_count):
    if hasattr(mesh, 'corner_normals'):
        normal = tuple(mesh.corner_normals[loop_index].vector)
    else:
        normal = tuple(mesh.loops[loop_index].normal)
    uvs = [tuple(layer.data[loop_index].uv) for layer in list(mesh.uv_layers)[:uv_count]]
    # MH2 prelight adds illumination. Use the original color stream if present, or black/opaque otherwise.
    color = (0.0, 0.0, 0.0, 1.0)
    layer = mesh.color_attributes.get('Color') if hasattr(mesh, 'color_attributes') else None
    if layer is not None:
        if layer.domain == 'CORNER':
            color = tuple(layer.data[loop_index].color)
        elif layer.domain == 'POINT':
            color = tuple(layer.data[mesh.loops[loop_index].vertex_index].color)
        else:
            raise ValueError('Unsupported color attribute domain')
    elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors.get('Color'):
        color = tuple(mesh.vertex_colors['Color'].data[loop_index].color)
    return normal, uvs, color


def original_normals(reader, part):
    """Reproduce Blender's import-time normal encoding, including quantization."""
    positions, normals, _, _, _, _, _ = reader.read_vertex_buffer(
        part['vertex_start'], part['vet'], part['stride'], part['vertices'],
        reader.bone_map.get(part['parent']))
    baseline = bpy.data.meshes.new('_BLeeds_MH2_export_normal_reference')
    try:
        baseline.from_pydata([tuple(v) for v in positions], [], part['faces'])
        for polygon in baseline.polygons:
            polygon.use_smooth = True
        if hasattr(baseline, 'use_auto_smooth'):
            baseline.use_auto_smooth = True
        baseline.update()
        baseline.normals_split_custom_set_from_vertices([tuple(n) for n in normals])
        if hasattr(baseline, 'calc_normals_split'):
            baseline.calc_normals_split()
        if hasattr(baseline, 'corner_normals'):
            corner_normals = [tuple(n.vector) for n in baseline.corner_normals]
        else:
            corner_normals = [tuple(loop.normal) for loop in baseline.loops]
        return normals, corner_normals
    finally:
        bpy.data.meshes.remove(baseline)


def mesh_stream(obj, root, part, reader, model, rebuild=False, normalize_repeat_uvs=False):
    mesh = obj.data
    if obj.mode != 'OBJECT':
        raise ValueError('Switch to Object Mode before export')
    original_topology = len(mesh.vertices) == part['vertices'] and tuple(tuple(p.vertices) for p in mesh.polygons) == part['faces']
    if not rebuild and not original_topology:
        raise ValueError(obj.name + ': topology changed; PC exporter supports original vertex/face order only')
    if obj.get('bleeds_mh2_object_header_layout') != 'pc_retail':
        raise ValueError('Only PC object headers are supported')
    if int(obj.get('bleeds_mh2_object_data_offset', -1)) != part['offset']:
        raise ValueError('Source MDL changed since import; reimport from the intended template')
    for modifier in obj.modifiers:
        if modifier.type != 'ARMATURE' and (modifier.show_viewport or modifier.show_render):
            raise ValueError(obj.name + ': apply or remove non-armature modifiers before PC export')
        if modifier.type == 'ARMATURE' and modifier.object != root:
            raise ValueError(obj.name + ': armature modifier points at another rig')
    if mesh.shape_keys:
        raise ValueError(obj.name + ': shape keys are not supported by this exporter')
    vertex_type = part['vet']
    diffuse_only = rebuild and bool(obj.get('bleeds_mh2_diffuse_only', False))
    if diffuse_only:
        vertex_type = 0x115E if vertex_type in (0x115E, 0x125E) else 0x152
    _, weight_offset, normal_offset, color_offset, uv_offsets = LAYOUTS[vertex_type]
    if len(mesh.uv_layers) < len(uv_offsets) or (not diffuse_only and len(mesh.uv_layers) != len(uv_offsets)):
        raise ValueError(obj.name + ': preserve all original UV layers, including blood UV2')
    # Material tables and their face partitions remain byte-for-byte intact.
    material_offset = model.u32(part['offset'])
    expected_names = []
    if material_offset >= 64:
        model.range(material_offset, part['material_count'] * 12, model.table_start)
        for index in range(part['material_count']):
            ptr = model.u32(material_offset + index * 12)
            if ptr:
                model.range(ptr, 1, model.table_start)
                end = model.data.find(b'\0', ptr, model.table_start)
                if end < 0:
                    raise ValueError('Unterminated material texture name')
                expected_names.append(model.data[ptr:end].decode('ascii'))
            else:
                expected_names.append('')
    # Some original attachment objects carry header sentinel pointers rather
    # than real materials. The importer intentionally gives those no slots.
    if not rebuild and len(mesh.materials) != len(expected_names):
        raise ValueError(obj.name + ': original material slots must be retained')
    for material, expected in zip(mesh.materials, expected_names):
        if material is None or (not rebuild and material.get('bleeds_mh2_texture_name', material.name) != expected):
            raise ValueError(obj.name + ': texture dictionary name changed')
    if not rebuild and expected_names and [p.material_index for p in mesh.polygons] != part['face_materials']:
        raise ValueError(obj.name + ': material face assignments changed')

    palette = reader.get_skin_palette_names()
    if any(not name for name in palette) or any(
            entry.get('bone_offset') not in reader.bone_map
            for entry in reader.skin_palette):
        raise ValueError('Unmapped original skin palette entry; export would discard bone influences')
    if len(palette) != len(set(palette)) or len(palette) > 256:
        raise ValueError('Ambiguous or oversized original skin palette')
    palette_lookup = {name: index for index, name in enumerate(palette)}
    group_names = {g.index: g.name for g in obj.vertex_groups}
    parent_info = reader.bone_map.get(part['parent'])
    # Match the importer: missing parents leave vertices in source-Blender
    # space rather than applying a bone transform.
    parent_matrix = parent_info['matrix'] if parent_info is not None else Matrix.Identity(4)
    source_inverse = reader.blender_to_source @ parent_matrix.inverted()
    object_matrix = root.matrix_world.inverted() @ obj.matrix_world
    to_source = source_inverse @ object_matrix
    # Invert the importer's normal transform, with the mesh's normal transform.
    normal_to_source = source_inverse.to_3x3() @ object_matrix.to_3x3().inverted().transposed()
    if rebuild:
        if abs(object_matrix.to_3x3().determinant()) < 1e-10:
            raise ValueError('Mesh object transform is singular')
        mesh.calc_loop_triangles()
        vertices, faces, materials, unique = [], [], [], {}
        repeat_uvs = normalize_repeat_uvs or bool(obj.get('bleeds_mh2_wrap_uv_tiles', False))
        from ..leedsLib.mh2_uv import split_repeat_triangle
        for triangle in mesh.loop_triangles:
            corners, source_ids = [], []
            for loop_index in triangle.loops:
                vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
                source_ids.append(vertex.index)
                normal, uvs, color = _corner_values(mesh, loop_index, len(uv_offsets))
                weights = {}
                if weight_offset is not None:
                    for assignment in vertex.groups:
                        if assignment.weight <= 1e-6:
                            continue
                        name = group_names[assignment.group]
                        if name not in palette_lookup:
                            raise ValueError('Weighted group not in target skeleton: ' + name)
                        weights[palette_lookup[name]] = float(assignment.weight)
                corners.append(dict(position=tuple(to_source @ vertex.co),
                                    normal=tuple((normal_to_source @ Vector(normal)).normalized()),
                                    color=color, uvs=uvs, skin=weights))
            if repeat_uvs:
                pieces = split_repeat_triangle([c['uvs'] for c in corners])
            else:
                pieces = [[(tuple(1. if i == j else 0. for i in range(3)), corners[j]['uvs']) for j in range(3)]]
            for piece in pieces:
                face = []
                for bary, uvs in piece:
                    def blend(field, size):
                        return tuple(sum(bary[i]*corners[i][field][a] for i in range(3)) for a in range(size))
                    weights = {}
                    if weight_offset is not None:
                        for i, corner in enumerate(corners):
                            for bone, weight in corner['skin'].items():
                                weights[bone] = weights.get(bone, 0.) + bary[i]*weight
                        weights = dict(sorted(((n, w) for n, w in weights.items() if w > 1e-8), key=lambda row: (-row[1], row[0]))[:4])
                        total = sum(weights.values())
                        if total <= 0:
                            raise ValueError('Split vertex has no skin weights')
                        weights = {n: w/total for n, w in weights.items()}
                    normal = Vector(blend('normal', 3))
                    if normal.length < 1e-8:
                        normal = Vector(corners[max(range(3), key=lambda i: bary[i])]['normal'])
                    values = dict(position=blend('position', 3), normal=tuple(normal.normalized()),
                                  color=blend('color', 4), uvs=[(u, 1-v) for u, v in uvs], skin=weights)
                    identity = tuple(sorted((source_ids[i], round(w, 12)) for i, w in enumerate(bary) if w > 1e-12))
                    key = (identity, pack_vertex(vertex_type, values))
                    if key not in unique:
                        unique[key] = len(vertices)
                        vertices.append(values)
                    face.append(unique[key])
                if object_matrix.to_3x3().determinant() < 0:
                    face.reverse()
                faces.append(tuple(face))
                materials.append(int(triangle.material_index))
        return dict(vertices=vertices, faces=faces, materials=materials,
                    vertex_type=vertex_type,
                    material_names=[m.get('bleeds_mh2_texture_name', m.name) for m in mesh.materials])
    corners = {}
    source_normals, baseline_normals = original_normals(reader, part)
    if hasattr(mesh, 'calc_normals_split'):
        mesh.calc_normals_split()
    for loop in mesh.loops:
        normal, uvs, color = _corner_values(mesh, loop.index, len(uv_offsets))
        if close_values(normal, baseline_normals[loop.index], 1e-6):
            normal = tuple(source_normals[loop.vertex_index])
        values = (normal, uvs, color)
        previous = corners.get(loop.vertex_index)
        if previous is not None:
            if (not close_values(normal, previous[0], 1e-4)
                    or any(not close_values(a, b) for a, b in zip(uvs, previous[1]))
                    or not close_values(color, previous[2], 1e-4)):
                raise ValueError(obj.name + ': vertex {} has a new UV/normal/color seam; split vertices require a geometry rebuild'.format(loop.vertex_index))
        else:
            corners[loop.vertex_index] = values
    stream = []
    for vertex in mesh.vertices:
        raw = model.row(part, vertex.index)
        if vertex.index in corners:
            normal, uvs, color = corners[vertex.index]
            normal = tuple((normal_to_source @ Vector(normal)).normalized())
            uvs = [(u, 1.0 - v) for u, v in uvs]
        else:
            # Loose source vertices have no loop attributes in Blender.
            import struct
            packed = struct.unpack_from('<3h', raw, normal_offset)
            normal = tuple(Vector(packed).normalized())
            b, g, r, a = struct.unpack_from('<4B', raw, color_offset)
            color = (r / 255, g / 255, b / 255, a / 255)
            uvs = [struct.unpack_from('<2f', raw, o) for o in uv_offsets]
        weights = {}
        if weight_offset is not None:
            for assignment in vertex.groups:
                if assignment.weight <= 1e-6:
                    continue
                name = group_names[assignment.group]
                if name not in palette_lookup:
                    raise ValueError(obj.name + ': weighted group is not in the original skin palette: ' + name)
                weights[palette_lookup[name]] = float(assignment.weight)
        stream.append(dict(position=tuple(to_source @ vertex.co), normal=normal,
                           color=color, uvs=uvs, skin=weights))
    return stream


def write_rest_matrices(data, source_model, reader, root):
    """Write the rest-pose matrices to a validated PC PMLC model, retaining all other records."""
    output = bytearray(data)
    world = {}
    original_world = {}
    metadata_nodes = {source_model.root_bone} | {part['parent'] for part in source_model.parts}
    for offset, info in reader.bone_map.items():
        bone = root.data.bones[info['name']]
        old = info['source_matrix']
        # The metadata-only mesh parent nodes must retain their source basis.
        if offset in metadata_nodes:
            new = old.copy()
        else:
            new = reader.blender_to_source @ bone.matrix_local @ reader.source_to_blender
        world[offset] = new
        original_world[offset] = old

    def write_matrix(offset, matrix):
        struct.pack_into('<16f', output, offset, *(matrix[row][col] for col in range(4) for row in range(4)))

    for offset, info in reader.bone_map.items():
        parent = info['parent_offset']
        local = world[parent].inverted() @ world[offset] if parent in world else world[offset]
        write_matrix(offset + 64, local)
        write_matrix(offset + 128, world[offset])
    for part in source_model.parts:
        if LAYOUTS[part['vet']][1] is None:
            continue
        index = source_model.u32(part['offset'] + 8)
        if not index:
            continue
        count, matrices = source_model.u32(index), source_model.u32(index + 4)
        if count != len(reader.skin_palette):
            raise ValueError('Inverse bind palette count does not match animation palette')
        source_model.range(matrices, count * 64, source_model.table_start)
        for i, entry in enumerate(reader.skin_palette):
            bone_offset = entry['bone_offset']
            bind = world[bone_offset].inverted() @ world.get(part['parent'], Matrix.Identity(4))
            write_matrix(matrices + i * 64, bind)
    return bytes(output)


def _write_export_files(files):
    """Stage all outputs first; restore originals if a later replace fails."""
    paths = list(files)
    with tempfile.TemporaryDirectory(prefix='.bleeds_export_', dir=str(paths[0].parent)) as temp:
        staged, backups, committed = {}, {}, []
        for index, destination in enumerate(paths):
            stage = Path(temp) / ('new_%d' % index)
            with stage.open('wb') as handle:
                handle.write(files[destination])
                handle.flush()
                os.fsync(handle.fileno())
            staged[destination] = stage
            if destination.exists():
                backup = Path(temp) / ('old_%d' % index)
                shutil.copy2(str(destination), str(backup))
                backups[destination] = backup
        try:
            for destination in paths:
                os.replace(str(staged[destination]), str(destination))
                committed.append(destination)
        except OSError:
            for destination in reversed(committed):
                if destination in backups:
                    os.replace(str(backups[destination]), str(destination))
                else:
                    destination.unlink()
            raise


def export_mh2_pc(context, filepath, rebuild_geometry=False, export_rest_pose=False, normalize_repeat_uvs=False,
                  export_textures=False):
    root = find_root(context)
    if root.get('bleeds_mh2_platform') != 'PC' or root.get('bleeds_mh2_asset_variant') != 'PC_RETAIL':
        raise ValueError('Select a retail Manhunt 2 PC import')
    source = Path(bpy.path.abspath(str(root.get('bleeds_mdl_filepath', ''))))
    model = PcMdl(source.read_bytes())
    reader = read_reference(source, context, model)
    if export_rest_pose and not rebuild_geometry:
        raise ValueError('Edited rest skeleton export requires Rebuild Geometry')
    if normalize_repeat_uvs and not rebuild_geometry:
        raise ValueError('Repeat UV normalization requires Rebuild Geometry')
    validate_skeleton(root, reader, allow_rest_changes=export_rest_pose)
    descendants = list(root.children_recursive)
    meshes = [o for o in descendants if o.type == 'MESH']
    by_info = {}
    for obj in meshes:
        info = int(obj.get('bleeds_mh2_objinfo_offset', -1))
        if info in by_info:
            raise ValueError('Duplicate original mesh part; remove copies from the export hierarchy')
        by_info[info] = obj
    if set(by_info) != {part['info'] for part in model.parts}:
        raise ValueError('Export needs every original mesh part and no additional parts')
    streams = [mesh_stream(by_info[part['info']], root, part, reader, model, rebuild=rebuild_geometry,
                           normalize_repeat_uvs=normalize_repeat_uvs) for part in model.parts]
    if rebuild_geometry:
        output = model.rebuild(streams)
        changes = {'rebuilt_parts': len(streams)}
    else:
        output, changes = model.rewrite(streams)
    if export_rest_pose:
        output = write_rest_matrices(output, model, reader, root)
    exported_model = PcMdl(output)
    destination = Path(bpy.path.abspath(filepath)).resolve()
    if destination.suffix.lower() != '.mdl':
        raise ValueError('Choose an .mdl output filename')
    if destination == source.resolve():
        raise ValueError('Export to a different file first; retain the source MDL as the template')
    files, texture_count = {destination: output}, 0
    if export_textures:
        from .mh2_texture_export import build_material_tex
        tex_destination = destination.with_suffix('.tex')
        protected = {source.with_suffix('.tex').resolve()}
        texture_source = root.get('bleeds_mh2_texture_source_path', '')
        if texture_source:
            protected.add(Path(bpy.path.abspath(texture_source)).resolve())
        if tex_destination.resolve() in protected:
            raise ValueError('Export to a different basename; retain the source TEX')
        texture_data, texture_count = build_material_tex(meshes)
        files[tex_destination] = texture_data
    _write_export_files(files)
    return dict(source=str(source), destination=str(destination), parts=len(exported_model.parts),
                vertices=sum(p['vertices'] for p in exported_model.parts),
                triangles=sum(len(p['faces']) for p in exported_model.parts),
                byte_identical=output == model.data, changes=changes, rebuild_geometry=rebuild_geometry,
                export_rest_pose=export_rest_pose, texture_count=texture_count,
                texture_destination=str(destination.with_suffix('.tex')) if export_textures else None)
