#!/usr/bin/env python3
"""Regenerate the sparkflash-vision fixtures.

Usage (Pillow and numpy; no torch):

    python3 -I crates/sparkflash-vision/tests/fixtures/generate.py \
        --ref <checkpoint>/inference/image_processor.py

Outputs (next to this script):

    prep_cases.txt   one line per procedural image: the plan and the SHA-256 of the canvas that
                     the reference's load_image builds from it
    decode/          small image files, decode/manifest.txt saying what each must decode to

The canvases come from the reference itself: load_image runs unchanged on a PNG of each
procedural image, with ImageOps.pad wrapped so that its output is recorded and the run stops
before the torch code. torch is stubbed; nothing before the pad uses it.

Decoding follows the serving recipe, not convert("RGB"), where they differ (alpha is blended
onto #FDFDFD, 16-bit samples round to 8), so the lossless files' expected pixels are computed
here from the source arrays. JPEG and lossy WebP decoders differ across libraries; for those
the manifest names a PNG of Pillow's decode, and the test bounds the difference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import struct
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
CONFIG = WORKTREE / "reference" / "inference_config.json"
BACKGROUND = 0xFD

# width, height, seed; the Rust test builds the same images from procedural() below.
PREP_CASES = [
    (1, 1, 0),
    (2, 165, 1),
    (40, 50, 2),
    (37, 23, 3),
    (516, 517, 4),
    (640, 480, 5),
    (978, 479, 6),
    (1033, 141, 7),
    (125, 751, 8),
    (1091, 397, 9),
    (1302, 1302, 10),
    (1920, 1080, 11),
    (1080, 1920, 12),
    (8192, 8, 13),
    (3, 4000, 14),
    (2000, 2000, 15),
    (546, 546, 16),
    (1000, 1, 17),
    (743, 870, 18),
    (1107, 873, 19),
]


def procedural(width: int, height: int, seed: int) -> np.ndarray:
    """An RGB image with gradients, texture and hard edges, from integers only."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    chans = [
        (x * (3 + c) + y * (5 + 2 * c) + ((x * y) >> 5) + ((x ^ y) & 63) * (c + 1) + seed * 17 + c * 85)
        & 255
        for c in range(3)
    ]
    return np.stack(chans, -1).astype(np.uint8)


def alpha_of(width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    a = (x * 9 + y * 7) & 255
    a[0, :] = 0
    a[-1, :] = 255
    return a.astype(np.uint8)


def blend(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.uint32)
    a = a.astype(np.uint32)[..., None]
    return ((c * a + BACKGROUND * (255 - a) + 127) // 255).astype(np.uint8)


def from16(v: np.ndarray) -> np.ndarray:
    return ((v.astype(np.uint32) + 128) // 257).astype(np.uint8)


def load_reference(path: Path):
    torch = types.ModuleType("torch")
    torch.Tensor = object
    sys.modules["torch"] = torch
    spec = importlib.util.spec_from_file_location("image_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Padded(Exception):
    pass


def reference_canvas(ip, args, rgb: np.ndarray) -> Image.Image:
    """Runs the reference load_image on a PNG of `rgb` and returns the padded canvas."""
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    captured = {}

    class Spy:
        @staticmethod
        def pad(*a, **kw):
            captured["out"] = ImageOps.pad(*a, **kw)
            raise Padded

    real = ip.ImageOps
    ip.ImageOps = Spy
    try:
        ip.load_image({"data": buf.getvalue()}, args)
    except Padded:
        pass
    finally:
        ip.ImageOps = real
    return captured["out"]


def write_png(path: Path, width: int, height: int, color_type: int, depth: int, rows: bytes):
    """A minimal PNG writer, for the 16-bit colour files Pillow cannot save."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    stride = len(rows) // height
    raw = b"".join(b"\0" + rows[y * stride : (y + 1) * stride] for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).hexdigest()


def decode_fixtures(out: Path) -> list[str]:
    out.mkdir(exist_ok=True)
    lines = []

    def exact(name: str, rgb: np.ndarray):
        h, w, _ = rgb.shape
        lines.append(f"{name} exact {w} {h} {sha(rgb)}")

    def lossy(name: str):
        with Image.open(out / name) as im:
            ref = im.convert("RGB")
        ref_name = name.rsplit(".", 1)[0] + ".pillow.png"
        ref.save(out / ref_name)
        lines.append(f"{name} lossy {ref.width} {ref.height} {ref_name}")

    rgb = procedural(37, 23, 1)
    Image.fromarray(rgb, "RGB").save(out / "rgb8.png")
    exact("rgb8.png", rgb)

    rgb = procedural(29, 31, 2)
    a = alpha_of(29, 31)
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(out / "rgba8.png")
    exact("rgba8.png", blend(rgb, a))

    g = procedural(19, 17, 3)[..., 0]
    a = alpha_of(19, 17)
    Image.fromarray(np.dstack([g, a]), "LA").save(out / "la8.png")
    exact("la8.png", blend(np.dstack([g, g, g]), a))

    Image.fromarray(g, "L").save(out / "gray8.png")
    exact("gray8.png", np.dstack([g, g, g]))

    bits = (procedural(23, 9, 4)[..., 0] & 1).astype(bool)
    # fromarray(bits, "1") would read the bools as packed bytes; let Pillow pick the mode.
    one = Image.fromarray(bits)
    assert one.mode == "1"
    one.save(out / "gray1.png")
    v = np.where(bits, 255, 0).astype(np.uint8)
    exact("gray1.png", np.dstack([v, v, v]))

    idx = (procedural(21, 13, 5)[..., 0] % 16).astype(np.uint8)
    palette = procedural(16, 1, 6)[0]
    p = Image.fromarray(idx, "P")
    p.putpalette(palette.flatten().tolist())
    p.save(out / "palette_trns.png", transparency=3)
    pal = palette[idx]
    exact("palette_trns.png", blend(pal, np.where(idx == 3, 0, 255)))

    rgb16 = ((procedural(11, 7, 7).astype(np.uint32) * 257 + procedural(11, 7, 8)) & 0xFFFF).astype(np.uint16)
    write_png(out / "rgb16.png", 11, 7, 2, 16, rgb16.astype(">u2").tobytes())
    exact("rgb16.png", from16(rgb16))

    a16 = np.minimum(alpha_of(11, 7).astype(np.uint32) * 257 + 100, 65535).astype(np.uint16)
    rgba16 = np.dstack([rgb16, a16]).astype(np.uint16)
    write_png(out / "rgba16.png", 11, 7, 6, 16, rgba16.astype(">u2").tobytes())
    exact("rgba16.png", blend(from16(rgb16), from16(a16)))

    g16 = rgb16[..., 1]
    write_png(out / "gray16.png", 11, 7, 0, 16, g16.astype(">u2").tobytes())
    g8 = from16(g16)
    exact("gray16.png", np.dstack([g8, g8, g8]))

    src = Image.fromarray(procedural(45, 29, 9), "RGB")
    src.save(out / "q90_444.jpg", quality=90, subsampling=0)
    lossy("q90_444.jpg")
    src.save(out / "q75_420.jpg", quality=75, subsampling=2)
    lossy("q75_420.jpg")
    src.save(out / "q85_422_progressive.jpg", quality=85, subsampling=1, progressive=True)
    lossy("q85_422_progressive.jpg")
    src.convert("L").save(out / "gray.jpg", quality=90)
    lossy("gray.jpg")

    idx = (procedural(33, 21, 10)[..., 0] % 8).astype(np.uint8)
    gp = Image.fromarray(idx, "P")
    gp.putpalette(procedural(8, 1, 11)[0].flatten().tolist())
    second = Image.fromarray(((idx + 1) % 8).astype(np.uint8), "P")
    second.putpalette(procedural(8, 1, 11)[0].flatten().tolist())
    gp.save(out / "anim.gif", save_all=True, append_images=[second], transparency=2, optimize=False, loop=0)
    with Image.open(out / "anim.gif") as im:
        rgba = np.asarray(im.convert("RGBA"))
    exact("anim.gif", blend(rgba[..., :3], rgba[..., 3]))

    rgb = procedural(27, 19, 12)
    a = alpha_of(27, 19)
    Image.fromarray(np.dstack([rgb, a]), "RGBA").save(out / "lossless.webp", lossless=True)
    exact("lossless.webp", blend(rgb, a))

    Image.fromarray(procedural(31, 17, 13), "RGB").save(out / "lossy.webp", quality=80)
    lossy("lossy.webp")

    f0 = procedural(17, 11, 14)
    f1 = procedural(17, 11, 15)
    Image.fromarray(f0, "RGB").save(
        out / "anim.webp",
        save_all=True,
        append_images=[Image.fromarray(f1, "RGB")],
        lossless=True,
        duration=100,
    )
    exact("anim.webp", f0)
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="the checkpoint's inference/image_processor.py")
    opts = ap.parse_args()
    ip = load_reference(opts.ref)
    cfg = json.loads(CONFIG.read_text())
    args = SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("vision_")})
    ref_sha = hashlib.sha256(opts.ref.read_bytes()).hexdigest()

    lines = [
        f"# Pillow {PIL.__version__}, numpy {np.__version__}, image_processor.py sha256 {ref_sha}",
        "# width height seed n_llm_h n_llm_w best_height best_width canvas_sha256",
    ]
    for w, h, seed in PREP_CASES:
        n_llm_h, n_llm_w, best_h, best_w = ip.plan_image_grid(w, h, args)
        canvas = reference_canvas(ip, args, procedural(w, h, seed))
        assert canvas.size == (best_w, best_h) and canvas.mode == "RGB", (w, h, canvas)
        lines.append(f"{w} {h} {seed} {n_llm_h} {n_llm_w} {best_h} {best_w} {sha(np.asarray(canvas))}")
    (HERE / "prep_cases.txt").write_text("\n".join(lines) + "\n")

    decode = decode_fixtures(HERE / "decode")
    header = [
        f"# Pillow {PIL.__version__}, numpy {np.__version__}",
        "# file exact width height rgb_sha256 | file lossy width height pillow_decode.png",
    ]
    (HERE / "decode" / "manifest.txt").write_text("\n".join(header + decode) + "\n")


if __name__ == "__main__":
    main()
