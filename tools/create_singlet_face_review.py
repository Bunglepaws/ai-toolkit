from __future__ import annotations

import json
import textwrap
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


DATASET = Path(r"C:\Data\AIToolkit-StagingArea\datasets\clothing-singlet-brand-target-v3")
MODEL = Path(
    r"C:\Data\AIToolkit-StagingArea\work\clothing\singlets"
    r"\web_dataset_20260820_v3\cache\face_detection_yunet_2023mar.onnx"
)
OUTPUT = Path(r"C:\Data\AIToolkit-StagingArea\work\clothing\singlets\face_review_v3")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    no_face_rows: list[dict] = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for path in sorted(DATASET.iterdir()):
        if path.suffix.lower() not in extensions:
            continue
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            str(MODEL), "", (width, height), score_threshold=0.75, nms_threshold=0.3, top_k=5
        )
        _, faces = detector.detect(image)
        if faces is None:
            no_face_rows.append({"id": len(no_face_rows) + 1, "file": path.name})
            continue
        face = max(faces, key=lambda candidate: candidate[-1])
        x, y, box_width, box_height = map(float, face[:4])
        if box_width < 35 or box_height < 35:
            no_face_rows.append({"id": len(no_face_rows) + 1, "file": path.name})
            continue
        rows.append(
            {
                "id": len(rows) + 1,
                "file": path.name,
                "box": [round(x), round(y), round(box_width), round(box_height)],
                "score": round(float(face[-1]), 3),
            }
        )

    (OUTPUT / "faces.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "no_faces.json").write_text(
        json.dumps(no_face_rows, indent=2) + "\n", encoding="utf-8"
    )
    font = ImageFont.load_default(size=15)
    for page_start in range(0, len(rows), 20):
        page = Image.new("RGB", (1500, 1280), "white")
        draw = ImageDraw.Draw(page)
        for position, row in enumerate(rows[page_start : page_start + 20]):
            source = DATASET / row["file"]
            image = Image.open(source).convert("RGB")
            x, y, box_width, box_height = row["box"]
            padding = 0.75
            x0 = max(0, int(x - box_width * padding))
            y0 = max(0, int(y - box_height * padding))
            x1 = min(image.width, int(x + box_width * (1 + padding)))
            y1 = min(image.height, int(y + box_height * (1 + padding)))
            crop = image.crop((x0, y0, x1, y1))
            crop.thumbnail((270, 225))
            column = position % 5
            row_number = position // 5
            origin_x = column * 300 + (300 - crop.width) // 2
            origin_y = row_number * 320
            page.paste(crop, (origin_x, origin_y))
            label = f"{row['id']:03d} {Path(row['file']).stem[-31:]}"
            wrapped = "\n".join(textwrap.wrap(label, 35))
            draw.multiline_text(
                (column * 300 + 8, origin_y + 232), wrapped, fill="black", font=font, spacing=2
            )
        page.save(OUTPUT / f"faces_{page_start // 20 + 1:02d}.jpg", quality=92)

    for page_start in range(0, len(no_face_rows), 20):
        page = Image.new("RGB", (1500, 1280), "white")
        draw = ImageDraw.Draw(page)
        for position, row in enumerate(no_face_rows[page_start : page_start + 20]):
            source = DATASET / row["file"]
            image = Image.open(source).convert("RGB")
            image.thumbnail((270, 225))
            column = position % 5
            row_number = position // 5
            origin_x = column * 300 + (300 - image.width) // 2
            origin_y = row_number * 320
            page.paste(image, (origin_x, origin_y))
            label = f"{row['id']:03d} {Path(row['file']).stem[-31:]}"
            wrapped = "\n".join(textwrap.wrap(label, 35))
            draw.multiline_text(
                (column * 300 + 8, origin_y + 232), wrapped, fill="black", font=font, spacing=2
            )
        page.save(OUTPUT / f"no_faces_{page_start // 20 + 1:02d}.jpg", quality=92)

    print(
        f"FACE_ROWS={len(rows)} FACE_SHEETS={(len(rows) + 19) // 20} "
        f"NO_FACE_ROWS={len(no_face_rows)} NO_FACE_SHEETS={(len(no_face_rows) + 19) // 20} "
        f"OUT={OUTPUT}"
    )


if __name__ == "__main__":
    main()
