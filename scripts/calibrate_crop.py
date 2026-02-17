#!/usr/bin/env python3
"""Calibrate auto-crop parameters (scale, y_bias) from manual reviews.

Reads approved candidates that have both manual_crop and face_detect
in photo_meta, computes what scale and y_bias would reproduce the
user's manual crop given the MediaPipe face bbox, and outputs
recommended parameters.

Usage:
    source .env
    python scripts/calibrate_crop.py
"""

import asyncio
import json
import os
import statistics

import asyncpg


async def main():
    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set. Run: source .env")
        return

    conn = await asyncpg.connect(db_url)

    rows = await conn.fetch("""
        SELECT id, photo_meta
        FROM player_photo_assets
        WHERE asset_type = 'candidate'
          AND review_status = 'approved'
          AND photo_meta ? 'manual_crop'
          AND photo_meta ? 'face_detect'
    """)

    await conn.close()

    if not rows:
        print("No approved candidates with both manual_crop and face_detect yet.")
        print("Approve some candidates with the crop tool first.")
        return

    scales = []
    y_biases = []
    details = []

    for row in rows:
        meta = row["photo_meta"] if isinstance(row["photo_meta"], dict) else json.loads(row["photo_meta"])
        mc = meta["manual_crop"]
        fd = meta["face_detect"]

        if not fd.get("detected") or not fd.get("bbox"):
            continue

        bbox = fd["bbox"]
        sw = mc["source_width"]
        sh = mc["source_height"]

        # Face bbox in pixels
        fx = bbox["x"] * sw
        fy = bbox["y"] * sh
        fw = bbox["w"] * sw
        fh = bbox["h"] * sh
        face_size = max(fw, fh)

        if face_size < 10:
            continue

        # Face center
        cx = fx + fw / 2
        cy = fy + fh / 2

        # Manual crop params
        crop_x = mc["x"]
        crop_y = mc["y"]
        crop_size = mc["size"]

        # Reverse-engineer scale: crop_size = face_size * scale
        scale = crop_size / face_size

        # Reverse-engineer y_bias: crop_y = cy - crop_size * y_bias
        # y_bias = (cy - crop_y) / crop_size
        y_bias = (cy - crop_y) / crop_size if crop_size > 0 else 0.5

        scales.append(scale)
        y_biases.append(y_bias)
        details.append({
            "id": row["id"],
            "scale": round(scale, 3),
            "y_bias": round(y_bias, 3),
            "crop_size": crop_size,
            "face_size": round(face_size, 1),
        })

    if not scales:
        print("No valid paired data found.")
        return

    print(f"=== Crop Calibration Report ===")
    print(f"Samples: {len(scales)}")
    print()
    print(f"--- Scale (current: 1.85) ---")
    print(f"  Median: {statistics.median(scales):.3f}")
    print(f"  Mean:   {statistics.mean(scales):.3f}")
    if len(scales) > 1:
        print(f"  Stdev:  {statistics.stdev(scales):.3f}")
    print(f"  Min:    {min(scales):.3f}")
    print(f"  Max:    {max(scales):.3f}")
    print()
    print(f"--- Y-Bias (current: 0.52) ---")
    print(f"  Median: {statistics.median(y_biases):.3f}")
    print(f"  Mean:   {statistics.mean(y_biases):.3f}")
    if len(y_biases) > 1:
        print(f"  Stdev:  {statistics.stdev(y_biases):.3f}")
    print(f"  Min:    {min(y_biases):.3f}")
    print(f"  Max:    {max(y_biases):.3f}")
    print()
    print(f"--- Recommended ---")
    rec_scale = round(statistics.median(scales), 2)
    rec_ybias = round(statistics.median(y_biases), 2)
    print(f"  scale={rec_scale}, y_bias={rec_ybias}")
    print()
    print(f"Apply in app/photos/processor.py → _compute_crop_from_face_box():")
    print(f'  scale={rec_scale}, y_bias={rec_ybias}')
    print()
    print(f"--- Per-candidate details ---")
    for d in details:
        print(f"  #{d['id']:>5}  scale={d['scale']:.3f}  y_bias={d['y_bias']:.3f}  "
              f"crop={d['crop_size']}  face={d['face_size']}")


if __name__ == "__main__":
    asyncio.run(main())
