# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect material color images and build a matching MH2 PC dictionary."""
import hashlib
import os
import numpy as np

from ..leedsLib.mh2_dds import encode_dds
from ..leedsLib.mh2_pc_tex import build_pc_tex
from ..leedsLib.tex import parse_mh2_tex_entries


def _color_image(material):
    if not material.use_nodes or material.node_tree is None:
        raise ValueError('Material %r needs an image connected to its shader color' % material.name)
    outputs = [n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    socket = outputs[0].inputs.get('Surface') if outputs else None
    seen = set()
    while socket is not None and socket.is_linked:
        if len(socket.links) != 1:
            break
        node = socket.links[0].from_node
        if node.as_pointer() in seen:
            break
        seen.add(node.as_pointer())
        if node.type == 'TEX_IMAGE' and node.image:
            if node.image.source not in {'FILE', 'GENERATED'}:
                raise ValueError('Texture %r must be a single image, not a sequence or tiled image' % node.image.name)
            return node.image
        if node.type == 'REROUTE':
            socket = node.inputs[0]
        elif node.type == 'BSDF_PRINCIPLED':
            socket = node.inputs.get('Base Color')
        elif node.type in {'BSDF_DIFFUSE', 'EMISSION'}:
            socket = node.inputs.get('Color')
        else:
            break
    raise ValueError('Material %r: connect a single Image Texture directly to shader color (bake mixed/procedural colors first)' % material.name)


def _image_dds(image):
    # Blender retains the original DDS in its packed image. Reuse it only
    # when pixels are unchanged, retaining compression, orientation and mips.
    if not image.is_dirty and image.packed_file:
        data = bytes(image.packed_file.data)
        if len(data) >= 128 and data[:4] == b'DDS ' and data[84:88] in (b'DXT1', b'DXT5'):
            return data
    width, height = image.size[:]
    if not image.has_data and len(image.pixels) == 0:
        raise ValueError('Texture image %r has no loaded pixels' % image.name)
    if any(n < 1 or n > 4096 or n & (n-1) for n in (width, height)):
        raise ValueError('Texture %r needs power-of-two dimensions from 1 to 4096' % image.name)
    pixels = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    if not np.isfinite(pixels).all():
        raise ValueError('Texture %r contains nonfinite pixels' % image.name)
    pixels = pixels.reshape(height, width, 4)
    # Byte-backed images expose encoded channels; float images store linear
    # values and need the standard linear-to-sRGB transform for color export.
    if image.is_float and image.colorspace_settings.name not in {'Non-Color', 'Raw'}:
        rgb = pixels[:, :, :3].clip(0, 1)
        pixels[:, :, :3] = np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)
    rgba = np.rint(pixels.clip(0, 1)*255).astype(np.uint8)[::-1]
    return encode_dds(rgba)


def build_material_tex(meshes):
    textures, identities, cache = {}, {}, {}
    for obj in meshes:
        for material in obj.data.materials:
            if material is None:
                raise ValueError('Mesh %r has an empty material slot' % obj.name)
            name = str(material.get('bleeds_mh2_texture_name', material.name))
            key = os.path.splitext(os.path.basename(name.strip().replace('\\', '/')))[0].casefold()
            image = _color_image(material)
            identity = image.as_pointer()
            if identity not in cache:
                cache[identity] = _image_dds(image)
            data = cache[identity]
            digest = hashlib.sha256(data).digest()
            if key in identities:
                old_name, old_digest = identities[key]
                if name != old_name or digest != old_digest:
                    raise ValueError('Conflicting images or aliases for texture %r' % name)
                continue
            identities[key] = name, digest
            textures[name] = data
    result = build_pc_tex(textures)
    parse_mh2_tex_entries(result)
    return result, len(textures)
