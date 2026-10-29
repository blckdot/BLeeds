# MH2 DDS encoder. GPL-3.0-or-later

"""Encode a DDS file with a complete mip chain from an RGBA8 array."""

import struct
import numpy as np


def _color_blocks(rgb):
    # Fit endpoints along each block's principal color direction.
    centered = rgb - rgb.mean(axis=1, keepdims=True)
    covariance = np.einsum('bni,bnj->bij', centered, centered)
    direction = np.eye(3, dtype=np.float32)[np.diagonal(covariance, axis1=1, axis2=2).argmax(axis=1)]
    for _ in range(6):
        direction = np.einsum('bij,bj->bi', covariance, direction)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-10)
    projection = np.einsum('bni,bi->bn', rgb, direction)
    rows = np.arange(len(rgb))
    ends = np.stack((rgb[rows, projection.argmax(axis=1)], rgb[rows, projection.argmin(axis=1)]), axis=1)
    q = np.rint(ends * np.array([31, 63, 31]) / 255).astype(np.uint16)
    packed = (q[:, :, 0] << 11) | (q[:, :, 1] << 5) | q[:, :, 2]
    c0, c1 = packed.max(axis=1), packed.min(axis=1)
    equal = c0 == c1
    c0 = np.where(equal & (c0 < 65535), c0 + 1, c0).astype(np.uint16)
    c1 = np.where(equal & (c0 == c1), c1 - 1, c1).astype(np.uint16)
    def unpack(c):
        r, g, b = (c >> 11).astype(np.uint32), ((c >> 5) & 63).astype(np.uint32), (c & 31).astype(np.uint32)
        return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), axis=1).astype(np.float32)
    a, b = unpack(c0), unpack(c1)
    palette = np.stack((a, b, np.floor((2*a+b)/3), np.floor((a+2*b)/3)), axis=1)
    distances = ((rgb[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=3)
    indices = distances.argmin(axis=2).astype(np.uint32)
    bits = np.bitwise_or.reduce(indices << (2*np.arange(16, dtype=np.uint32)), axis=1)
    result = np.empty((len(rgb), 8), dtype=np.uint8)
    result[:, :2] = c0.astype('<u2').view(np.uint8).reshape(-1, 2)
    result[:, 2:4] = c1.astype('<u2').view(np.uint8).reshape(-1, 2)
    result[:, 4:] = bits.astype('<u4').view(np.uint8).reshape(-1, 4)
    return result


def _alpha_blocks(alpha):
    a, b = alpha.max(axis=1), alpha.min(axis=1)
    palette = np.stack([a, b] + [np.floor(((7-i)*a+i*b)/7) for i in range(1, 7)], axis=1)
    indices = np.abs(alpha[:, :, None] - palette[:, None, :]).argmin(axis=2).astype(np.uint64)
    bits = np.bitwise_or.reduce(indices << (3*np.arange(16, dtype=np.uint64)), axis=1)
    result = np.empty((len(alpha), 8), dtype=np.uint8)
    result[:, 0], result[:, 1] = a, b
    result[:, 2:] = bits.astype('<u8').view(np.uint8).reshape(-1, 8)[:, :6]
    return result


def encode_dds(rgba):
    """Encode an RGBA8 HxWx4 array with a complete mip chain."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError('DDS pixels must be an HxWx4 uint8 array')
    height, width = rgba.shape[:2]
    if any(n < 1 or n > 4096 or n & (n-1) for n in (width, height)):
        raise ValueError('MH2 textures must have power-of-two dimensions from 1 to 4096')
    alpha = bool(np.any(rgba[:, :, 3] < 255))
    payload, count, image = bytearray(), 0, rgba
    while True:
        h, w = image.shape[:2]
        padded = np.pad(image, ((0, (-h) % 4), (0, (-w) % 4), (0, 0)), mode='edge')
        blocks = padded.reshape(padded.shape[0]//4, 4, padded.shape[1]//4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4)
        for offset in range(0, len(blocks), 4096):
            batch = blocks[offset:offset+4096].astype(np.float32)
            color = _color_blocks(batch[:, :, :3])
            if alpha:
                color = np.concatenate((_alpha_blocks(batch[:, :, 3]), color), axis=1)
            payload.extend(color.tobytes())
        count += 1
        if h == w == 1:
            break
        # Box-filter premultiplied colors to avoid dark fringes in alpha mips.
        nh, nw = max(1, h//2), max(1, w//2)
        pixels = image.astype(np.float32).reshape(nh, h//nh, nw, w//nw, 4)
        a = pixels[:, :, :, :, 3:4] / 255
        mean_a = a.mean(axis=(1, 3))
        rgb = (pixels[:, :, :, :, :3]*a).mean(axis=(1, 3)) / np.maximum(mean_a, 1e-8)
        image = np.rint(np.concatenate((rgb, mean_a*255), axis=2)).clip(0, 255).astype(np.uint8)
    header = bytearray(128)
    header[:4] = b'DDS '
    struct.pack_into('<7I', header, 4, 124, 0xA1007, height, width,
                     max(1, (width+3)//4)*max(1, (height+3)//4)*(16 if alpha else 8), 0, count)
    struct.pack_into('<2I', header, 76, 32, 4)
    header[84:88] = b'DXT5' if alpha else b'DXT1'
    struct.pack_into('<I', header, 108, 0x401008)
    return bytes(header) + payload
