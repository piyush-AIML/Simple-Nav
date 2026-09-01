"""Build a labeled contact sheet from a list of frame paths.

Used by the executor to visually label frames (baseline benchmark ground
truth, place naming, etc.). Tiles frames in row-major order with a text
label above each tile and a sheet index in the top-left corner.

Usage:
    conda run -n ML python scripts/make_contact_sheet.py <out.png> --frames f1.jpg f2.jpg ...
    conda run -n ML python scripts/make_contact_sheet.py <out.png> --json labels.json [--key place]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

TILE_W, TILE_H = 192, 144
LABEL_H = 18
MAX_COLS = 6


def build(frames: list[tuple[str, str]], out_path: Path) -> None:
    tiles: list[tuple[str, str]] = []  # (label, path)
    for label, path in frames:
        img = cv2.imread(str(path))
        if img is None:
            print(f"WARN: cannot read {path} — skipping")
            continue
        img = cv2.resize(img, (TILE_W, TILE_H))
        canvas = cv2.copyMakeBorder(img, LABEL_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(canvas, label, (2, LABEL_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        tiles.append((label, path, canvas))

    if not tiles:
        raise SystemExit("No readable frames.")

    cols = min(MAX_COLS, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * (TILE_H + LABEL_H), cols * TILE_W, 3), dtype=np.uint8)
    for i, (_, _, tile) in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * (TILE_H + LABEL_H):(r + 1) * (TILE_H + LABEL_H), c * TILE_W:(c + 1) * TILE_W] = tile

    cv2.imwrite(str(out_path), sheet)
    print(f"Wrote {out_path} ({len(tiles)} tiles, {rows}x{cols})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Labeled contact sheet from frames.")
    parser.add_argument("out", help="Output PNG path")
    parser.add_argument("--frames", nargs="+", default=None, help="frame paths (label = basename)")
    parser.add_argument("--json", default=None, help="JSON file mapping label -> frame path")
    parser.add_argument("--key", default=None, help="JSON structure: {group: {label: path}}; pass --key to select a group")
    args = parser.parse_args()

    if args.frames:
        frames = [(Path(f).name, f) for f in args.frames]
    elif args.json:
        with open(args.json) as f:
            data = json.load(f)
        if args.key is not None:
            data = data[args.key]
        frames = [(str(k), str(v)) for k, v in data.items()]
    else:
        raise SystemExit("Need --frames or --json")

    build(frames, Path(args.out))


if __name__ == "__main__":
    main()
