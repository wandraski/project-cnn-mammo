"""Imagem: CLAHE e normalização com caminho rápido (cv2/PIL) e fallback puro.

O fallback existe para que o CI valide a matemática sem dependências pesadas;
em GPU usa-se cv2 quando disponível.
"""

from __future__ import annotations

from collections.abc import Sequence


def _histogram(tile: Sequence[int], bins: int) -> list[int]:
    counts = [0] * bins
    for value in tile:
        counts[value] += 1
    return counts


def clahe(
    pixels: Sequence[int],
    width: int,
    height: int,
    *,
    tiles: int = 8,
    clip_limit: float = 2.0,
    bins: int = 256,
) -> list[int]:
    """CLAHE em grade de tiles com interpolação bilinear (stdlib).

    Correto, porém O(tiles²·bins); use `clahe_fast` em produção (cv2).
    """
    if width <= 0 or height <= 0 or len(pixels) != width * height:
        raise ValueError("dimensões incompatíveis com o vetor de pixels")
    if tiles < 1 or clip_limit <= 0:
        raise ValueError("tiles e clip_limit devem ser positivos")

    tile_w = max(1, width // tiles)
    tile_h = max(1, height // tiles)
    grids_x = (width + tile_w - 1) // tile_w
    grids_y = (height + tile_h - 1) // tile_h

    lut: list[list[list[float]]] = []
    for gy in range(grids_y):
        row: list[list[float]] = []
        for gx in range(grids_x):
            x0, y0 = gx * tile_w, gy * tile_h
            x1, y1 = min(width, x0 + tile_w), min(height, y0 + tile_h)
            tile = [
                pixels[y * width + x]
                for y in range(y0, y1)
                for x in range(x0, x1)
            ]
            counts = _histogram(tile, bins)
            total = len(tile)
            clipped = 0.0
            for index, count in enumerate(counts):
                ceiling = clip_limit * total / bins
                if count > ceiling:
                    clipped += count - ceiling
                    counts[index] = int(ceiling)
            redistribute = clipped / bins
            lut_row: list[float] = []
            cumulative = 0.0
            for count in counts:
                cumulative += count + redistribute
                lut_row.append(min(bins - 1, cumulative * bins / max(total, 1)))
            row.append(lut_row)
        lut.append(row)

    output: list[int] = []
    for y in range(height):
        gy = min(y // tile_h, grids_y - 1)
        ty = (y % tile_h) / tile_h
        for x in range(width):
            gx = min(x // tile_w, grids_x - 1)
            tx = (x % tile_w) / tile_w
            value = pixels[y * width + x]
            top = lut[gy][gx][value]
            bottom = lut[min(gy + 1, grids_y - 1)][gx][value]
            left = lut[gy][min(gx + 1, grids_x - 1)][value]
            right = lut[min(gy + 1, grids_y - 1)][min(gx + 1, grids_x - 1)][value]
            interpolated = (
                top * (1 - ty) * (1 - tx)
                + bottom * ty * (1 - tx)
                + left * (1 - ty) * tx
                + right * ty * tx
            )
            output.append(int(round(interpolated)))
    return output


def clahe_fast(pixels, width: int, height: int, *, tiles: int = 8, clip_limit: float = 2.0):
    """Caminho rápido via OpenCV; retorna `None` se cv2 não estiver instalado."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    array = np.asarray(pixels, dtype=np.uint8).reshape(height, width)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    return equalizer.apply(array)


def normalize_zscore(pixels: Sequence[float]) -> list[float]:
    if not pixels:
        raise ValueError("amostra vazia")
    mean = sum(pixels) / len(pixels)
    std = (sum((value - mean) ** 2 for value in pixels) / len(pixels)) ** 0.5 or 1.0
    return [(value - mean) / std for value in pixels]


def preprocess_array(
    pixels: Sequence[int],
    width: int,
    height: int,
    *,
    normalization: str = "clahe_zscore",
    tiles: int = 8,
    clip_limit: float = 2.0,
) -> dict[str, object]:
    """Aplica o plano completo; retorna vetor e resumo para o model card."""
    equalized = clahe_fast(pixels, width, height, tiles=tiles, clip_limit=clip_limit)
    source = "cv2"
    vector: Sequence[int]
    if equalized is None:
        vector = clahe(pixels, width, height, tiles=tiles, clip_limit=clip_limit)
        source = "stdlib"
    else:
        vector = equalized.reshape(-1).tolist()
    normalized = normalize_zscore([float(value) for value in vector])
    return {
        "values": normalized,
        "source": source,
        "tiles": tiles,
        "clip_limit": clip_limit,
        "normalization": normalization,
    }
