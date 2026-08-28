from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(r"C:\Data\AIToolkit-StagingArea")
DATASET = ROOT / r"datasets\clothing-singlet-brand-target-v3"
NO_FACE_INDEX = ROOT / r"work\clothing\singlets\face_review_v3\no_faces.json"
BACKUP = ROOT / r"work\clothing\singlets\caption_backups\clothing-singlet-brand-target-v3-pre-partial-face-20260823"
REPORT = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3.json"
AUDIT = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3-face-captions.json"


def assign(mapping: dict[int, str], ids: list[int] | range, description: str) -> None:
    for image_id in ids:
        if image_id in mapping:
            raise ValueError(f"Duplicate partial-face description for ID {image_id}")
        mapping[image_id] = description


def descriptions() -> dict[int, str]:
    result: dict[int, str] = {}
    assign(result, [1, 2, 4, 8], "the upper face is cropped out, leaving light skin and a short brown beard visible")
    assign(result, [10], "the upper face is cropped out, leaving light skin and a full dark-brown beard visible")
    assign(result, [12], "the upper face is cropped out, leaving light skin, a full brown beard, and a hoop earring visible")
    assign(result, [17, 18, 19, 20], "the upper face is cropped out, leaving a light-skinned, nearly clean-shaven lower face visible")
    assign(result, [27, 28, 30, 31], "the upper face is cropped out, leaving light skin, a full brown beard, and small hoop earrings visible")
    assign(result, range(36, 44), "the upper face is cropped out, leaving light skin, a full brown beard, and small hoop earrings visible")
    assign(result, [44, 45, 46, 48, 49, 50], "the upper face is cropped out, leaving light skin and a short brown beard visible")
    assign(result, [52], "the upper face is cropped out, leaving light skin and a full brown beard visible")
    assign(result, [76, 77], "the upper face is cropped out, leaving light skin and a full dark beard visible")
    assign(result, [81, 83, 84, 85], "the upper face is cropped out, leaving a light-skinned lower face with faint stubble visible")
    assign(result, [101, 102], "the upper face is cropped out, leaving medium skin and a lower face with short dark stubble visible")
    return result


def main() -> None:
    rows = json.loads(NO_FACE_INDEX.read_text(encoding="utf-8"))
    rows_by_id = {int(row["id"]): row for row in rows}
    partial_descriptions = descriptions()
    missing_ids = sorted(set(partial_descriptions) - set(rows_by_id))
    if missing_ids:
        raise SystemExit(f"Partial-face IDs absent from review index: {missing_ids}")
    if BACKUP.exists():
        raise SystemExit(f"Refusing to reuse existing backup directory: {BACKUP}")
    BACKUP.mkdir(parents=True)

    changed: list[dict] = []
    for image_id, descriptor in partial_descriptions.items():
        image_name = rows_by_id[image_id]["file"]
        caption_path = DATASET / f"{Path(image_name).stem}.txt"
        if not caption_path.exists():
            raise FileNotFoundError(caption_path)
        original = caption_path.read_text(encoding="utf-8").strip()
        if "Facial appearance:" in original:
            raise SystemExit(f"Caption already contains a face description: {caption_path}")
        shutil.copy2(caption_path, BACKUP / caption_path.name)
        base = original[:-1] if original.endswith(".") else original
        updated = f"{base}. Facial appearance: {descriptor}."
        caption_path.write_text(updated + "\n", encoding="utf-8")
        changed.append(
            {
                "id": image_id,
                "image": image_name,
                "caption": caption_path.name,
                "description": descriptor,
                "review_source": "manual review of detector-negative contact sheets",
            }
        )

    if REPORT.exists():
        shutil.copy2(REPORT, BACKUP / REPORT.name)
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        current = {path.name: path.read_text(encoding="utf-8").strip() for path in DATASET.glob("*.txt")}
        for report_row in report.get("rows", []):
            caption_name = f"{Path(report_row['file']).stem}.txt"
            if caption_name in current:
                report_row["caption"] = current[caption_name]
        report["face_caption_count"] = sum("Facial appearance:" in caption for caption in current.values())
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    previous_audit = json.loads(AUDIT.read_text(encoding="utf-8")) if AUDIT.exists() else {"changed": []}
    all_changed = previous_audit.get("changed", []) + changed
    total_captions = len(list(DATASET.glob("*.txt")))
    audit = {
        "dataset": str(DATASET),
        "full_face_backup": previous_audit.get("backup"),
        "partial_face_backup": str(BACKUP),
        "changed_caption_count": len(all_changed),
        "unchanged_caption_count": total_captions - len(all_changed),
        "full_face_caption_count": len(previous_audit.get("changed", [])),
        "partial_face_caption_count": len(changed),
        "changed": all_changed,
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in audit.items() if key != "changed"}, indent=2))


if __name__ == "__main__":
    main()
