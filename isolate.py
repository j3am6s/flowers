from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

TEST = Path("test")
OUT = Path("flowers")


def flower_masks(image, masks):
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    candidates = []

    for mask in masks:
        mask = mask.astype(bool)
        area = int(mask.sum())
        if not 1600 <= area <= 40000:
            continue
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue

        ys, xs = np.where(mask)
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if min(bw, bh) < 28 or max(bw, bh) / min(bw, bh) > 3.2:
            continue
        if area / (bw * bh) < 0.40:
            continue

        green = ((hue[mask] >= 35) & (hue[mask] <= 90) & (sat[mask] > 30)).mean()
        sat_mean = float(sat[mask].mean())
        hue_mean = float(hue[mask].mean())
        if green > 0.45:
            continue
        if hue_mean >= 100 and sat_mean >= 90:
            continue
        if sat_mean < 35 or val[mask].mean() < 70:
            continue
        candidates.append(mask)

    kept = []
    for mask in sorted(candidates, key=lambda m: m.sum(), reverse=True):
        if not _inside(mask, kept):
            kept.append(mask)
    return kept


def _inside(mask, others):
    return any((mask & other).sum() / mask.sum() > 0.6 for other in others)


def isolate(image_path):
    image = cv2.imread(str(image_path))
    result = SAM("mobile_sam.pt").predict(str(image_path), verbose=False)[0]
    masks = flower_masks(image, result.masks.data.cpu().numpy())
    masks.sort(key=lambda m: (np.where(m)[0].min(), np.where(m)[1].min()))

    out = OUT / image_path.stem
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("flower_*.jpg"):
        old.unlink()

    for i, mask in enumerate(masks, start=1):
        ys, xs = np.where(mask)
        crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        cv2.imwrite(str(out / f"flower_{i:02d}.jpg"), crop)

    print(f"{image_path.name}: saved {len(masks)} flowers to {out}")


def main():
    photos = sorted(TEST.glob("*.jpg")) + sorted(TEST.glob("*.jpeg")) + sorted(TEST.glob("*.png"))
    for photo in photos:
        isolate(photo)


if __name__ == "__main__":
    main()
