#!/usr/bin/env python3
"""
wall_detector.py — Wall & door detection for 2D architect floor plans.

Input : a technical floor plan (thin dark strokes on a light background,
        strict top-down view).
Output: plugin-compatible JSON (walls, doors, grid) for the
        Myxeliums Battlemap Importer module.

Dependencies: numpy + Pillow only (no scipy, no OpenCV).

Usage:
    python wall_detector.py --input plan_2d.png
    python wall_detector.py --input plan_2d.png --output battlemap.json \
        --name "Champ de la Cathédrale" --merge-lights

Pipeline:
    1. Brightness thresholding (dark strokes = walls).
    2. Binary closing (seal anti-aliased gaps).
    3. Remove components touching the image border (plan frame) and
       tiny components (text, symbols).
    4. Extract horizontal / vertical runs, merge collinear fragments.
    5. Merge close wall ends into continuous corners (L / T junctions).
    6. Detect doors & windows: gaps in a wall line — the door segment
       spans the gap exactly, so it stays connected to both walls.
    7. Emit JSON.
"""

import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image


# ── Morphology (replaces scipy.ndimage) ───────────────────────────────────
def _dilate(m, it=1):
    for _ in range(it):
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        m = out
    return m


def _erode(m, it=1):
    return ~_dilate(~m, it)


def closing(m, it=1):
    return _erode(_dilate(m, it), it)


# ── Connected components (replaces scipy.ndimage.label) ───────────────────
def label_components(m):
    """8-connected labeling. Returns (int label array, component count)."""
    H, W = m.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = {}
    next_id = 0

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(H):
        row = labels[y]
        for x in np.where(m[y])[0]:
            x = int(x)
            nbs = []
            if y > 0:
                up = labels[y - 1]
                if x > 0 and up[x - 1]:
                    nbs.append(up[x - 1])
                if up[x]:
                    nbs.append(up[x])
                if x + 1 < W and up[x + 1]:
                    nbs.append(up[x + 1])
            if x > 0 and row[x - 1]:
                nbs.append(row[x - 1])
            if nbs:
                root = min(nbs)
                row[x] = root
                for n in nbs:
                    union(root, n)
            else:
                next_id += 1
                parent[next_id] = next_id
                row[x] = next_id

    if next_id == 0:
        return labels, 0

    # Flatten unions and renumber components 1..n
    roots = {i: find(i) for i in range(1, next_id + 1)}
    uniq = sorted(set(roots.values()))
    table = np.zeros(next_id + 1, dtype=np.int32)
    for i, r in roots.items():
        table[i] = uniq.index(r) + 1
    return table[labels], len(uniq)


# ── Step 1-3: clean wall mask ─────────────────────────────────────────────
def build_wall_mask(img, dark_thr=150, min_size=200):
    """Dark strokes = walls; drop the plan frame (border) and text (small)."""
    line = np.asarray(img.convert("L"), dtype=np.uint8) < dark_thr
    line = closing(line, 1)

    lbl, n = label_components(line)
    if n == 0:
        return line

    border_ids = set(np.unique(
        np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])
    ))
    border_ids.discard(0)
    sizes = np.bincount(lbl.ravel(), minlength=n + 1)

    keep = np.zeros(n + 1, dtype=bool)
    for i in range(1, n + 1):
        if i not in border_ids and sizes[i] >= min_size:
            keep[i] = True
    return keep[lbl]


# ── Step 4-5: runs + collinear merging ────────────────────────────────────
def _runs_in_row(mask_row):
    """[(start, end), ...] runs of True in a 1-D boolean array."""
    idx = np.where(mask_row)[0]
    if len(idx) == 0:
        return []
    out = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == p + 1:
            p = i
        else:
            out.append((s, p))
            s = p = i
    out.append((s, p))
    return out


def _merge_collinear(segs, gap):
    """Merge (coord, start, end) segments that are collinear and close."""
    groups = defaultdict(list)
    for s in segs:
        groups[s[0]].append(s)
    merged = []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda s: s[1])
        cs, ce = items[0][1], items[0][2]
        for _, a, b in items[1:]:
            if a <= ce + gap:
                ce = max(ce, b)
            else:
                merged.append((key, int(cs), int(ce)))
                cs, ce = a, b
        merged.append((key, int(cs), int(ce)))
    return merged


# ── Step 5: corner merging ────────────────────────────────────────────────
def merge_corners(hsegs, vsegs, snap=6):
    """
    Merge close wall ends into one continuous corner:
    - L corner: an H end near a V end → both share the exact corner point
      (the short wall is extended, the other snapped onto it).
    - T junction: an H end near a V line → the H end snaps onto the line.
    Symmetric for V ends against H lines.
    """
    hsegs = [list(s) for s in hsegs]
    vsegs = [list(s) for s in vsegs]

    def _dist(coord, seg):
        a, b = seg[1], seg[2]
        if a <= coord <= b:
            return 0
        return min(abs(coord - a), abs(coord - b))

    # H ends against V lines
    for h in hsegs:
        y = h[0]
        for idx in (1, 2):
            cands = [v for v in vsegs if abs(v[0] - h[idx]) <= snap]
            if not cands:
                continue
            v = min(cands, key=lambda v: _dist(y, v))
            if _dist(y, v) <= snap:          # L corner: extend V to the corner
                v[1] = min(v[1], y)
                v[2] = max(v[2], y)
            h[idx] = v[0]                    # snap H end onto the V line

    # V ends against H lines (symmetric)
    for v in vsegs:
        x = v[0]
        for idx in (1, 2):
            cands = [h for h in hsegs if abs(h[0] - v[idx]) <= snap]
            if not cands:
                continue
            h = min(cands, key=lambda h: _dist(x, h))
            if _dist(x, h) <= snap:          # L corner: extend H to the corner
                h[1] = min(h[1], x)
                h[2] = max(h[2], x)
            v[idx] = h[0]                    # snap V end onto the H line

    return hsegs, vsegs


def extract_segments(line, min_len=40, gap=8):
    """H/V runs on the mask, then merge collinear fragments into full walls."""
    H, W = line.shape
    hsegs = [(y, x1, x2)
             for y in range(H)
             for x1, x2 in _runs_in_row(line[y])
             if x2 - x1 >= 6]
    vsegs = [(x, y1, y2)
             for x in range(W)
             for y1, y2 in _runs_in_row(line[:, x])
             if y2 - y1 >= 6]

    hsegs = _merge_collinear(hsegs, gap)
    vsegs = _merge_collinear(vsegs, gap)
    hsegs, vsegs = merge_corners(hsegs, vsegs)

    hsegs = [(y, a, b) for y, a, b in hsegs if b - a >= min_len]
    vsegs = [(x, a, b) for x, a, b in vsegs if b - a >= min_len]
    return hsegs, vsegs


# ── Step 6: door detection ────────────────────────────────────────────────
def detect_doors(hsegs, vsegs, line, gray, max_gap=80):
    """
    A door/window is a gap (5-`max_gap` px) in a wall line, flanked by two
    real wall pieces (>= 20 px each), with a bright interior around the gap.
    The door segment spans the gap exactly, from one wall end to the other,
    so it is always connected to the walls (never floating).
    """
    H, W = line.shape
    doors = []

    def _bright(cx, cy, r=8):
        y0, y1 = max(0, cy - r), min(H, cy + r)
        x0, x1 = max(0, cx - r), min(W, cx + r)
        return float(gray[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0

    def _gap_candidates(spans):
        """spans: list of (a, b) on one row/col → yield (a_end, b_start, gap)."""
        spans.sort()
        for i in range(len(spans) - 1):
            a_end, b_start = spans[i][1], spans[i + 1][0]
            gap = b_start - a_end
            if (5 <= gap <= max_gap
                    and (a_end - spans[i][0]) >= 20
                    and (spans[i + 1][1] - b_start) >= 20):
                yield a_end, b_start, gap

    def _near_wall(x, y, r=8):
        return bool(line[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any())

    by_row = defaultdict(list)
    for y, x1, x2 in hsegs:
        by_row[y].append((x1, x2))
    for y, spans in by_row.items():
        for a_end, b_start, gap in _gap_candidates(spans):
            cx = (a_end + b_start) // 2
            if (_bright(cx, y) > 40
                    and _near_wall(a_end, y) and _near_wall(b_start, y)):
                doors.append({"x": cx, "y": y, "orient": "h",
                              "gap": gap, "from": a_end, "to": b_start})

    by_col = defaultdict(list)
    for x, y1, y2 in vsegs:
        by_col[x].append((y1, y2))
    for x, spans in by_col.items():
        for a_end, b_start, gap in _gap_candidates(spans):
            cy = (a_end + b_start) // 2
            if (_bright(x, cy) > 40
                    and _near_wall(x, a_end) and _near_wall(x, b_start)):
                doors.append({"x": x, "y": cy, "orient": "v",
                              "gap": gap, "from": a_end, "to": b_start})

    return doors


# ── Step 7: plugin JSON ───────────────────────────────────────────────────
def build_json(hsegs, vsegs, doors, width, height, name, grid_size=32):
    def _wall(x1, y1, x2, y2, door=False):
        return {"c": [x1, y1, x2, y2], "door": 1 if door else 0, "ds": 0,
                "move": 0 if door else 20, "sound": 0 if door else 20,
                "sight": 20, "light": 0, "dir": 0}

    walls = [_wall(x1, y, x2, y) for y, x1, x2 in hsegs if x2 - x1 >= 4]
    walls += [_wall(x, y1, x, y2) for x, y1, y2 in vsegs if y2 - y1 >= 4]
    # Doors/windows span the gap exactly, from one wall end to the other →
    # always connected to the walls on both sides.
    for d in doors:
        if d["orient"] == "h":
            walls.append(_wall(d["from"], d["y"], d["to"], d["y"], door=True))
        else:
            walls.append(_wall(d["x"], d["from"], d["x"], d["to"], door=True))

    return {
        "name": name,
        "width": width,
        "height": height,
        "grid": {"size": grid_size, "type": 1, "distance": 5,
                 "units": "ft", "alpha": 0.2, "color": "#000000"},
        "shiftX": 0, "shiftY": 0,
        "walls": walls,
        "lights": [],
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Detect walls & doors in a 2D architect floor plan → plugin JSON")
    ap.add_argument("--input", default="plan_2d.png",
                    help="Input 2D floor plan image (default: plan_2d.png)")
    ap.add_argument("--output", default="battlemap.json",
                    help="Output JSON (default: battlemap.json)")
    ap.add_argument("--name", default="Battlemap Scene",
                    help="Scene name in the JSON")
    ap.add_argument("--grid", type=int, default=32,
                    help="Grid size px (default: 32)")
    ap.add_argument("--dark-threshold", type=int, default=150,
                    help="Darkness threshold 0-255, strokes darker than this "
                         "are walls (default: 150)")
    ap.add_argument("--min-len", type=int, default=40,
                    help="Minimum wall length px (default: 40)")
    ap.add_argument("--max-door-gap", type=int, default=80,
                    help="Maximum gap width (px) treated as a door/window "
                         "(default: 80)")
    ap.add_argument("--merge-lights", action="store_true",
                    help="Copy lights/globalLight/darkness from the existing "
                         "output JSON (if present)")
    args = ap.parse_args()

    img = Image.open(args.input)
    W, H = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    print(f"image {W}x{H}")

    line = build_wall_mask(img, args.dark_threshold)
    print(f"wall mask coverage: {line.mean():.3f}")

    hsegs, vsegs = extract_segments(line, min_len=args.min_len)
    print(f"segments: {len(hsegs)} horizontal, {len(vsegs)} vertical")

    doors = detect_doors(hsegs, vsegs, line, gray, max_gap=args.max_door_gap)
    print(f"doors: {len(doors)}")

    data = build_json(hsegs, vsegs, doors, W, H, args.name, args.grid)

    if args.merge_lights:
        try:
            with open(args.output, encoding="utf-8") as f:
                existing = json.load(f)
            for key in ("lights", "globalLight", "darkness"):
                if key in existing:
                    data[key] = existing[key]
            print("lighting kept from existing JSON")
        except (OSError, json.JSONDecodeError) as e:
            print(f"--merge-lights ignored: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total = len(data["walls"])
    ndoor = sum(1 for w in data["walls"] if w["door"])
    print(f"OK -> {args.output}: {total} wall segments ({ndoor} doors)")


if __name__ == "__main__":
    main()
