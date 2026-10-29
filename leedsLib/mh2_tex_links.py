# BLeeds MH2 TCDT validator. GPL-3.0-or-later.

"""Validate the MH2 TCDT texture dictionary's circular linked list."""

import struct

def validate_mh2_tex_links(decoded):
    if len(decoded) < 48 or decoded[:4] != b'TCDT':
        raise ValueError('Expected a decoded MH2 TCDT texture dictionary')
    def u32(offset):
        if offset < 0 or offset + 4 > len(decoded):
            raise ValueError('Texture link points outside dictionary')
        return struct.unpack_from('<I', decoded, offset)[0]
    count, first, last = (u32(offset) for offset in (32, 36, 40))
    # The list sentinel is the first/last pair, not the preceding texture count.
    sentinel = 36
    if count > 65536:
        raise ValueError('Unreasonable MH2 texture count')
    current, previous, seen = first, sentinel, set()
    for _ in range(count):
        if current < 48 or current + 112 > len(decoded) or current in seen:
            raise ValueError('Invalid MH2 texture descriptor chain')
        seen.add(current)
        if u32(current + 4) != previous:
            raise ValueError('Broken MH2 texture backward link')
        previous, current = current, u32(current)
    if current != sentinel or previous != last:
        raise ValueError('Broken MH2 texture list terminator; expected sentinel 36')
    return count
