from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(r"C:\Data\AIToolkit-StagingArea")
DATASET = ROOT / r"datasets\clothing-singlet-brand-target-v3"
FACE_INDEX = ROOT / r"work\clothing\singlets\face_review_v3\faces.json"
BACKUP = ROOT / r"work\clothing\singlets\caption_backups\clothing-singlet-brand-target-v3-pre-face-20260823"
REPORT = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3.json"
AUDIT = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3-face-captions.json"


def assign(mapping: dict[int, str], ids: list[int] | range, description: str) -> None:
    for face_id in ids:
        if face_id in mapping:
            raise ValueError(f"Duplicate face description for ID {face_id}")
        mapping[face_id] = description


def descriptions() -> dict[int, str]:
    result: dict[int, str] = {}
    assign(result, [1], "only the lower face is visible at the top edge, showing light skin and a short brown beard")
    assign(result, [2], "light skin, short dark brown hair, and a neatly trimmed dark beard, shown in profile")
    assign(result, [3], "light skin, swept medium-brown hair, and a short beard, shown in three-quarter profile")
    assign(result, range(4, 7), "light skin, swept light-brown hair, and short beard stubble")
    assign(result, [7], "light-to-medium skin, closely cropped dark hair, and a neatly trimmed dark beard")
    assign(result, [8], "fair skin, short black hair, a clean-shaven face, and a neutral expression")
    assign(result, range(9, 16), "the upper face is cropped out, leaving light skin, a full brown beard, and small hoop earrings visible")
    assign(result, range(16, 18), "only a clean-shaven lower face with light skin is visible because the upper face is cropped out")
    assign(result, [18, 19, 22, 24, 26], "dark skin, a shaved head, a clean-shaven face, small hoop earrings, and a neutral expression")
    assign(result, [20, 21, 23, 27], "light skin, short curly dark hair with closely cut sides, faint facial hair, and hoop earrings")
    assign(result, [25], "fair skin, short dark hair, a clean-shaven face, and a neutral expression")
    assign(result, [28], "medium skin, curly black hair, a full black beard, and a serious expression")
    assign(result, range(29, 35), "the upper face is cropped out, leaving light skin, a full brown beard, and small hoop earrings visible")
    assign(result, [35], "light skin, dark undercut hair, faint facial hair, and a profile view with the face turned away")
    assign(result, [36], "light skin, dark undercut hair, a trimmed mustache with short stubble, and a direct gaze")
    assign(result, range(37, 40), "the upper face is cropped out, leaving light skin, a full brown beard, and small hoop earrings visible")
    assign(result, range(40, 46), "light-to-medium skin, very short dark hair, a full neatly trimmed black beard, and a direct gaze")
    assign(result, range(46, 48), "light skin, a bald head, a full dark beard, and a neutral expression")
    assign(result, range(48, 56), "the upper face is cropped out, leaving light skin and a short-to-full brown beard visible")
    assign(result, range(56, 59), "medium skin, close-cropped dark hair with shaved sides, a full black beard, and an intense expression")
    assign(result, range(59, 63), "light skin, tousled dark-brown hair, a full brown beard, and a neutral expression")
    assign(result, range(63, 67), "medium skin, close-cropped dark hair, a full black beard, and a serious expression")
    assign(result, range(67, 74), "medium skin, short black hair, a full black beard, and a direct or downward gaze")
    assign(result, range(74, 81), "the upper face is cropped out, leaving light skin, a short brown beard, and a small earring visible")
    assign(result, range(81, 87), "the upper face is cropped out, leaving dark skin, a short black beard, and small earrings visible")
    assign(result, range(87, 93), "fair skin, dark curly mullet-style hair, a clean-shaven face, and dangling hoop earrings")
    assign(result, [93], "fair skin, short textured blond hair, a clean-shaven face, and a side profile")
    assign(result, [94], "dark skin, short natural black hair, a clean-shaven face, and a neutral sideward gaze")
    assign(result, range(95, 104), "light skin, short textured brown hair, a prominent brown mustache, and light beard stubble")
    assign(result, [104], "light-to-medium skin, short straight black hair, a clean-shaven face, and a neutral expression")
    assign(result, range(105, 109), "fair skin, dark curly hair, a clean-shaven face, and small hoop earrings")
    assign(result, [109], "light-to-medium skin, very short dark hair, a full neatly trimmed black beard, and a direct gaze")
    return result


def main() -> None:
    rows = json.loads(FACE_INDEX.read_text(encoding="utf-8"))
    face_descriptions = descriptions()
    found_ids = {int(row["id"]) for row in rows}
    if found_ids != set(face_descriptions):
        missing = sorted(found_ids - set(face_descriptions))
        extra = sorted(set(face_descriptions) - found_ids)
        raise SystemExit(f"Face-description mismatch: missing={missing}, extra={extra}")
    if BACKUP.exists():
        raise SystemExit(f"Refusing to reuse existing backup directory: {BACKUP}")
    BACKUP.mkdir(parents=True)

    changed: list[dict] = []
    for row in rows:
        image_name = row["file"]
        caption_path = DATASET / f"{Path(image_name).stem}.txt"
        if not caption_path.exists():
            raise FileNotFoundError(caption_path)
        original = caption_path.read_text(encoding="utf-8").strip()
        if "Facial appearance:" in original:
            raise SystemExit(f"Caption already contains a face description: {caption_path}")
        shutil.copy2(caption_path, BACKUP / caption_path.name)
        descriptor = face_descriptions[int(row["id"])]
        base = original[:-1] if original.endswith(".") else original
        updated = f"{base}. Facial appearance: {descriptor}."
        caption_path.write_text(updated + "\n", encoding="utf-8")
        changed.append(
            {
                "id": row["id"],
                "image": image_name,
                "caption": caption_path.name,
                "description": descriptor,
                "detector_score": row["score"],
            }
        )

    if REPORT.exists():
        shutil.copy2(REPORT, BACKUP / REPORT.name)
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        current = {
            path.name: path.read_text(encoding="utf-8").strip()
            for path in DATASET.glob("*.txt")
        }
        for report_row in report.get("rows", []):
            caption_name = f"{Path(report_row['file']).stem}.txt"
            if caption_name in current:
                report_row["caption"] = current[caption_name]
        report["face_caption_count"] = len(changed)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = {
        "dataset": str(DATASET),
        "backup": str(BACKUP),
        "changed_caption_count": len(changed),
        "unchanged_caption_count": len(list(DATASET.glob("*.txt"))) - len(changed),
        "changed": changed,
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "changed"}, indent=2))


if __name__ == "__main__":
    main()
