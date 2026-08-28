from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(r"C:\Data\AIToolkit-StagingArea")
REVIEW = ROOT / r"work\clothing\singlets\review_uncropped_20260822"
TARGET = ROOT / r"datasets\clothing-singlet-brand-target-v3"
WEB_MANIFEST = ROOT / r"work\clothing\singlets\web_dataset_20260820_v3\reports\manifest.json"
MANUAL_MANIFEST = ROOT / r"work\clothing\singlets\manual_seed_processed_20260820\reports\manifest.json"
NEW_MATERIALS = ROOT / r"work\clothing\singlets\new_material_candidates_20260822"
REPORT = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3.json"
CATALOG = ROOT / r"work\clothing\singlets\reports\clothing-singlet-brand-target-v3-catalog.md"


NEOPRENE = {
    "web__neoprene__skintwo-665-neoprene-open-crotch-wrestling-singlet-01.jpg": {
        "source": NEW_MATERIALS / "neoprene_skintwo_665_01.jpg",
        "brand": "665",
        "product": "Neoprene Wrestling Singlet",
        "product_url": "https://www.skintwo.com/products/neoprene-open-crotch-wrestling-singlet",
        "view": "front view",
        "details": "short legs, thick smooth black neoprene, deep armholes, and black binding",
    },
    "web__neoprene__skintwo-665-neoprene-open-crotch-wrestling-singlet-02.jpg": {
        "source": NEW_MATERIALS / "neoprene_skintwo_665_02.jpg",
        "brand": "665",
        "product": "Neoprene Wrestling Singlet",
        "product_url": "https://www.skintwo.com/products/neoprene-open-crotch-wrestling-singlet",
        "view": "three-quarter front view",
        "details": "short legs, thick smooth black neoprene, deep armholes, and black binding",
    },
    "web__neoprene__skintwo-665-neoprene-open-crotch-wrestling-singlet-03.jpg": {
        "source": NEW_MATERIALS / "neoprene_skintwo_665_03.jpg",
        "brand": "665",
        "product": "Neoprene Wrestling Singlet",
        "product_url": "https://www.skintwo.com/products/neoprene-open-crotch-wrestling-singlet",
        "view": "back view",
        "details": "short legs, thick smooth black neoprene, a racerback, an open-seat rear, and black binding",
    },
}


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.-")


def metadata_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for item in load_manifest(WEB_MANIFEST):
        name = f"web__{item['category']}__{Path(item['raw_file']).name}"
        index[name.lower()] = item
        index[Path(name).stem.lower()] = item
    for item in load_manifest(MANUAL_MANIFEST):
        name = f"manual__{item['category']}__{Path(item['file']).name}"
        index[name.lower()] = item
        index[Path(name).stem.lower()] = item
    return index


def skin_fraction(image: Image.Image) -> float:
    rgb = image.convert("RGB")
    rgb.thumbnail((256, 256))
    pixels = list(rgb.getdata())
    skin = 0
    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        if r > 80 and g > 35 and b > 20 and r > g and r > b and mx - mn > 15:
            skin += 1
    return skin / max(1, len(pixels))


def background_phrase(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    rgb.thumbnail((256, 256))
    width, height = rgb.size
    border = max(2, min(width, height) // 12)
    samples = []
    samples.extend(rgb.crop((0, 0, width, border)).getdata())
    samples.extend(rgb.crop((0, height - border, width, height)).getdata())
    samples.extend(rgb.crop((0, 0, border, height)).getdata())
    samples.extend(rgb.crop((width - border, 0, width, height)).getdata())
    stat = ImageStat.Stat(Image.new("RGB", (len(samples), 1))) if not samples else None
    if stat is not None:
        return "a neutral studio background"
    mean = tuple(sum(pixel[i] for pixel in samples) / len(samples) for i in range(3))
    brightness = sum(mean) / 3
    spread = max(mean) - min(mean)
    if brightness >= 238 and spread <= 14:
        return "a white studio background"
    if brightness >= 190 and spread <= 28:
        return "a light gray studio background"
    if brightness <= 72:
        return "a dark indoor studio background"
    if spread >= 55:
        return "a colored commercial studio background"
    return "a neutral commercial studio background"


def category_for(item: dict, old_category: str) -> str:
    product = f"{item.get('brand', '')} {item.get('product', '')}".lower()
    if old_category == "bareback_openseat":
        return "assless singlet"
    if old_category == "rear_zip_release":
        return "zipper singlet"
    if old_category == "mesh_sheer":
        return "mesh singlet"
    if old_category == "strappy_cutout":
        return "cutout singlet"
    if old_category == "varsity_contrast":
        return "wrestling singlet"
    if "leather" in product:
        return "leather-look singlet"
    if "nasty pig" in product or "ramrod" in product or "structural" in product:
        return "athletic singlet"
    return "glossy singlet"


def construction_for(category: str) -> str:
    return {
        "assless singlet": "short legs, an open-seat rear, and body-framing panels",
        "zipper singlet": "short legs, close-fitting panels, and zip-access construction",
        "mesh singlet": "short legs, sheer mesh panels, and contrast binding",
        "cutout singlet": "short legs, deep body cutouts, and body-framing panels",
        "wrestling singlet": "short legs, athletic coverage, and contrast trim or paneling",
        "leather-look singlet": "short legs, a matte leather-look finish, and contrast trim",
        "athletic singlet": "short legs, supportive stretch panels, and athletic contour lines",
        "glossy singlet": "short legs, a glossy coated-fabric finish, and contrast trim",
        "neoprene singlet": "short legs, thick smooth neoprene, and reinforced binding",
    }[category]


def inferred_view(item: dict) -> str | None:
    text = f"{item.get('image_url', '')} {item.get('product', '')}".lower()
    if re.search(r"(?:^|[_\-/])(back|rear)(?:[_\-.?/]|$)", text):
        return "back view"
    if re.search(r"(?:^|[_\-/])side(?:[_\-.?/]|$)", text):
        return "side view"
    if re.search(r"(?:^|[_\-/])front(?:[_\-.?/]|$)", text):
        return "front view"
    return None


def clean_product(item: dict) -> str:
    brand = normalize_spaces(str(item.get("brand", "")))
    product = normalize_spaces(str(item.get("product", "singlet")))
    brand = re.sub(r"\bCellBlock13\b", "CellBlock 13", brand, flags=re.I)
    product = re.sub(r"\bCellBlock13\b", "CellBlock 13", product, flags=re.I)
    product = re.sub(r"\bshiny latex effect\b", "glossy coated-fabric", product, flags=re.I)
    product = re.sub(r"\blatex[- ]look\b", "glossy coated-fabric", product, flags=re.I)
    product = re.sub(r"\bfetish one-piece\b", "", product, flags=re.I)
    if brand.lower() in {"", "unverified brand", "unknown"}:
        brand = ""
    if brand and product.lower().startswith(brand.lower()):
        label = product
    else:
        label = normalize_spaces(f"{brand} {product}")
    label = re.sub(r"\bmen['’]?s\b", "", label, flags=re.I)
    label = re.sub(r"\bbodysuit\b", "", label, flags=re.I)
    return normalize_spaces(label)


def caption_for(image_path: Path, item: dict, old_category: str, forced_view: str | None = None) -> str:
    category = "neoprene singlet" if old_category == "neoprene" else category_for(item, old_category)
    product = clean_product(item)
    with Image.open(image_path) as image:
        person = skin_fraction(image) >= 0.035
        background = background_phrase(image)
    subject = "a muscular adult male model wearing" if person else "a product photograph of"
    view = forced_view or inferred_view(item)
    details = item.get("details") or construction_for(category)
    parts = [f"[trigger], {category}", f"{subject} the {product}", details]
    if view:
        parts.append(view)
    parts.append(f"photographed against {background}")
    return normalize_spaces(", ".join(parts)) + "."


def main() -> None:
    if TARGET.exists():
        raise SystemExit(f"Refusing to overwrite existing target: {TARGET}")
    TARGET.mkdir(parents=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    index = metadata_index()
    rows: list[dict] = []
    missing: list[str] = []

    review_images = sorted(path for path in REVIEW.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for source in review_images:
        item = index.get(source.name.lower()) or index.get(source.stem.lower())
        if item is None:
            missing.append(source.name)
            continue
        old_category = source.name.split("__", 2)[1]
        destination = TARGET / source.name
        shutil.copy2(source, destination)
        caption = caption_for(destination, item, old_category)
        destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        rows.append({
            "file": destination.name,
            "caption": caption,
            "category": caption.split(",", 2)[1].strip(),
            "source": str(source),
            "product_url": item.get("product_url", ""),
        })

    for name, item in NEOPRENE.items():
        source = item["source"]
        destination = TARGET / name
        shutil.copy2(source, destination)
        caption = caption_for(destination, item, "neoprene", forced_view=item["view"])
        destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        rows.append({
            "file": destination.name,
            "caption": caption,
            "category": "neoprene singlet",
            "source": str(source),
            "product_url": item["product_url"],
        })

    image_names = {path.stem for path in TARGET.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    caption_names = {path.stem for path in TARGET.glob("*.txt")}
    category_counts = Counter(row["category"] for row in rows)
    report = {
        "target": str(TARGET),
        "review_image_count": len(review_images),
        "added_neoprene_count": len(NEOPRENE),
        "image_count": len(image_names),
        "caption_count": len(caption_names),
        "missing_captions": sorted(image_names - caption_names),
        "orphan_captions": sorted(caption_names - image_names),
        "unmatched_review_images": missing,
        "category_counts": dict(sorted(category_counts.items())),
        "rows": rows,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog_lines = [
        "# Singlet LoRA prompt catalog",
        "",
        "Put the desired category immediately after `[trigger]`.",
        "",
    ]
    examples = {
        "assless singlet": "[trigger], assless singlet, muscular adult man wearing a black and blue short-leg singlet with an open-seat rear",
        "zipper singlet": "[trigger], zipper singlet, muscular adult man wearing a black short-leg singlet with a rear zipper and red trim",
        "mesh singlet": "[trigger], mesh singlet, muscular adult man wearing a black sheer-mesh short-leg singlet with contrast binding",
        "cutout singlet": "[trigger], cutout singlet, muscular adult man wearing a black short-leg singlet with deep torso cutouts",
        "wrestling singlet": "[trigger], wrestling singlet, athletic adult man wearing a navy collegiate singlet with gold trim",
        "leather-look singlet": "[trigger], leather-look singlet, muscular adult man wearing a matte black short-leg singlet with red trim",
        "athletic singlet": "[trigger], athletic singlet, muscular adult man wearing a blue structural athletic singlet with contour panels",
        "glossy singlet": "[trigger], glossy singlet, muscular adult man wearing a shiny black coated-fabric singlet with blue trim",
        "neoprene singlet": "[trigger], neoprene singlet, muscular adult man wearing a thick black neoprene wrestling singlet",
    }
    for category, count in sorted(category_counts.items()):
        catalog_lines.extend([f"## {category} ({count} images)", "", examples[category], ""])
    catalog_lines.extend([
        "## Material boundary",
        "",
        "This dataset does not label any garment as rubber or latex. Those materials are reserved for a separate LoRA.",
        "",
    ])
    CATALOG.write_text("\n".join(catalog_lines), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if missing or image_names != caption_names:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
