# BLeeds PC TEX writer. GPL-3.0-or-later.

"""Write MH2 PC TCDT texture dictionaries, including DDS payloads and descriptors."""

import os
import struct


def build_pc_tex(textures):
    """textures: ordered mapping of ASCII dictionary names to DDS bytes."""
    if not textures:
        raise ValueError('Texture dictionary must not be empty')
    output = bytearray(64)
    entries = []
    seen = set()
    for name, dds in textures.items():
        encoded = name.encode('ascii')
        normalized_name = os.path.splitext(
            os.path.basename(name.strip().replace("\\", "/"))
        )[0].casefold()
        if not encoded or len(encoded) > 31 or b'\0' in encoded or not normalized_name or normalized_name in seen:
            raise ValueError('Invalid or duplicate texture name')
        seen.add(normalized_name)
        if len(dds) < 128 or dds[:4] != b'DDS ' or struct.unpack_from('<I', dds, 4)[0] != 124:
            raise ValueError('Expected a legacy DDS header')
        height, width = struct.unpack_from('<2I', dds, 12)
        mipmaps = max(1, struct.unpack_from('<I', dds, 28)[0])
        fourcc = dds[84:88]
        if fourcc not in (b'DXT1', b'DXT5') or not 1 <= width <= 4096 or not 1 <= height <= 4096:
            raise ValueError('Unsupported DDS format or dimensions')
        if width & (width - 1) or height & (height - 1):
            raise ValueError('Texture dimensions must be powers of two')
        max_mips = max(width, height).bit_length()
        if not 1 <= mipmaps <= max_mips:
            raise ValueError('Invalid mipmap count')
        block_size = 8 if fourcc == b'DXT1' else 16
        expected = sum(max(1, (max(1, width >> m) + 3) // 4) *
                       max(1, (max(1, height >> m) + 3) // 4) * block_size for m in range(mipmaps))
        if len(dds) != expected + 128:
            raise ValueError('DDS payload length does not match its mip chain')
        output.extend(b'\0' * (-len(output) % 16))
        offset = len(output)
        output.extend(dds)
        entries.append((encoded, offset, len(dds), width, height, mipmaps, fourcc))
    output.extend(b'\0' * (-len(output) % 16))
    descriptors = len(output)
    relocations = [36, 40]
    for index, (name, data_offset, size, width, height, mipmaps, fourcc) in enumerate(entries):
        offset = len(output)
        descriptor = bytearray(128)
        next_offset = descriptors + (index + 1) * 128 if index + 1 < len(entries) else 36
        previous = descriptors + (index - 1) * 128 if index else 36
        struct.pack_into('<2I', descriptor, 0, next_offset, previous)
        descriptor[8:8 + len(name)] = name
        alpha = fourcc == b'DXT5'
        descriptor[40] = int(alpha)
        struct.pack_into('<4I', descriptor, 72, width, height, 8 if alpha else 4,
                         # PC descriptor stores top-mip bytes/width, as in retail TEX. It is NOT bytes/block-row.
                         max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * (16 if alpha else 8) // width)
        struct.pack_into('<H', descriptor, 88, 16 if alpha else 8)
        descriptor[92] = mipmaps
        struct.pack_into('<I', descriptor, 96, data_offset)
        struct.pack_into('<I', descriptor, 104, size)
        output.extend(descriptor)
        relocations.extend([offset, offset + 4, offset + 96])
    table = len(output)
    output.extend(struct.pack('<{}I'.format(len(relocations)), *relocations))
    struct.pack_into('<4s11I', output, 0, b'TCDT', 1, len(output), table, table,
                     len(relocations), 0, 0, len(entries), descriptors,
                     descriptors + (len(entries) - 1) * 128, 0)
    return bytes(output)
