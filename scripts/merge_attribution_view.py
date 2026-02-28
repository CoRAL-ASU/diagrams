#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ID_IN_ANSWER_RE = re.compile(r"([^,\s/]+)/")


@dataclass
class PredRecord:
    q_id: str
    annotation_path: str
    image_path: str
    pred_ids: Set[str]


def safe_load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_key_part(v: Optional[str]) -> str:
    return str(v or "").strip().lower()


def parse_answer_ids(answer: str) -> Set[str]:
    if not isinstance(answer, str):
        return set()
    return {m.group(1).strip() for m in ID_IN_ANSWER_RE.finditer(answer) if m.group(1).strip()}


def parse_pred_file(path: Path) -> List[PredRecord]:
    data = safe_load_json(path)
    if isinstance(data, dict):
        items = data.values()
    elif isinstance(data, list):
        items = data
    else:
        return []

    out: List[PredRecord] = []
    for rec in items:
        if not isinstance(rec, dict):
            continue
        q_id = str(rec.get("q_id") or "").strip()
        ann = str(rec.get("annotation") or rec.get("annotation_path") or "").strip()
        img = str(rec.get("image") or rec.get("image_path") or "").strip()
        answer = rec.get("answer") or rec.get("predicted_answer_raw") or ""
        pred_ids = parse_answer_ids(str(answer))
        if not q_id:
            continue
        out.append(PredRecord(q_id=q_id, annotation_path=ann, image_path=img, pred_ids=pred_ids))
    return out


def build_prediction_index(pred_root: Path) -> Tuple[Dict[Tuple[str, str], PredRecord], Dict[str, List[PredRecord]]]:
    by_qid_ann: Dict[Tuple[str, str], PredRecord] = {}
    by_qid: Dict[str, List[PredRecord]] = defaultdict(list)

    for file_path in sorted(pred_root.glob("*.json")):
        for rec in parse_pred_file(file_path):
            key = (normalize_key_part(rec.q_id), normalize_key_part(rec.annotation_path))
            if key not in by_qid_ann:
                by_qid_ann[key] = rec
            by_qid[normalize_key_part(rec.q_id)].append(rec)

    return by_qid_ann, by_qid


def collect_ids_from_container(value) -> Set[str]:
    out: Set[str] = set()

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                ident = item.get("id") or item.get("bbox_id") or item.get("ann_id")
                if ident:
                    out.add(str(ident))
    elif isinstance(value, dict):
        for k, item in value.items():
            if isinstance(item, dict):
                ident = item.get("bbox_id") or item.get("id")
                if ident:
                    out.add(str(ident))
                else:
                    out.add(str(k))
            else:
                out.add(str(k))

    return out


def extract_gt_ids(annotation_obj) -> Set[str]:
    ids: Set[str] = set()
    if not isinstance(annotation_obj, dict):
        return ids

    # AI2D-style
    for key in ("blobs", "arrows", "arrowHeads", "text", "texts"):
        if key in annotation_obj:
            ids |= collect_ids_from_container(annotation_obj.get(key))

    # Other common schemas
    for key in ("boxes", "bbox", "bboxes", "annotations", "bounding_boxes"):
        if key in annotation_obj:
            ids |= collect_ids_from_container(annotation_obj.get(key))

    # ChartQA-style: models -> [{ bboxes: [{bbox_id: ...}, ...] }]
    models = annotation_obj.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            if "bboxes" in model:
                ids |= collect_ids_from_container(model.get("bboxes"))

    # MapWise / MapIQ-style
    if "transferred_bboxes" in annotation_obj:
        ids |= collect_ids_from_container(annotation_obj.get("transferred_bboxes"))

    # Infographics-style OCR annotations
    for key in ("PAGE", "LINE", "WORD"):
        if key in annotation_obj:
            ids |= collect_ids_from_container(annotation_obj.get(key))

    return {x for x in ids if x}


def resolve_annotation_path(dataset_root: Path, annotation_path: str) -> Optional[Path]:
    if not annotation_path:
        return None
    p = dataset_root / annotation_path
    if p.exists():
        return p
    return None


def find_prediction_for_review(
    q_id: str,
    annotation_path: str,
    image_path: str,
    by_qid_ann: Dict[Tuple[str, str], PredRecord],
    by_qid: Dict[str, List[PredRecord]],
) -> Optional[PredRecord]:
    key = (normalize_key_part(q_id), normalize_key_part(annotation_path))
    if key in by_qid_ann:
        return by_qid_ann[key]

    q_matches = by_qid.get(normalize_key_part(q_id), [])
    if not q_matches:
        return None

    if len(q_matches) == 1:
        return q_matches[0]

    img_norm = normalize_key_part(image_path)
    for rec in q_matches:
        if normalize_key_part(rec.image_path) == img_norm:
            return rec

    return q_matches[0]


def parse_review_sources(review_obj: dict) -> Tuple[Set[str], Set[str], Set[str]]:
    pred_ids: Set[str] = set()
    gt_ids: Set[str] = set()
    new_ids: Set[str] = set()

    items = review_obj.get("bbox")
    if items is None:
        items = review_obj.get("boxes")
    if items is None:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "").strip()
        if not ident:
            continue
        source = str(item.get("source") or "").strip().lower()
        if source == "pred":
            pred_ids.add(ident)
        elif source == "gt":
            gt_ids.add(ident)
        elif source == "new":
            new_ids.add(ident)

    return pred_ids, gt_ids, new_ids


def infer_dataset_from_annotation_path(annotation_path: str) -> str:
    if not annotation_path:
        return "unknown"
    parts = annotation_path.split("/")
    return parts[0] if parts else "unknown"


def iter_review_files(review_root: Path, pattern: str) -> Iterable[Path]:
    if pattern:
        yield from sorted(review_root.glob(pattern))
        return
    yield from sorted(review_root.rglob("*.json"))


def dedupe_review_paths(paths: List[Path], mode: str) -> List[Path]:
    if mode == "none":
        return paths

    def quality_score(obj: dict) -> Tuple[int, int, int]:
        pred_ids, gt_ids, new_ids = parse_review_sources(obj)
        retained = len(pred_ids & gt_ids)
        added_gt = len(gt_ids - pred_ids)
        # Prefer files with explicit human edits first (new/added GT),
        # then more retained reviewed boxes.
        return (len(new_ids) + added_gt, added_gt, retained)

    best: Dict[Tuple[str, str], Path] = {}
    best_meta: Dict[Tuple[str, str], Tuple[Tuple[int, int, int], float, str]] = {}
    for p in paths:
        try:
            obj = safe_load_json(p)
        except Exception:
            # Keep unreadable files separate using path-based key so they are still reported.
            best[(str(p), "")] = p
            continue

        q_id = str(obj.get("q_id") or "").strip()
        ann = str(obj.get("annotation_path") or "").strip()
        key = (q_id, ann)
        score = quality_score(obj)
        st_mtime = p.stat().st_mtime
        cur_tuple = (score, st_mtime, str(p))

        if key not in best:
            best[key] = p
            best_meta[key] = cur_tuple
            continue

        if cur_tuple > best_meta[key]:
            best[key] = p
            best_meta[key] = cur_tuple

    return sorted(best.values())


def main():
    parser = argparse.ArgumentParser(description="Build a unified merged view for reviewed diagram attribution records.")
    parser.add_argument("--reviewed-root", default="Reviewed", help="Root directory containing reviewed JSON files")
    parser.add_argument("--dataset-root", default="Diagram_Attribution_Dataset", help="Root directory for GT dataset files")
    parser.add_argument("--pred-root", default="Predictions/gemini-pred", help="Directory containing Gemini prediction JSON files")
    parser.add_argument("--pattern", default="", help="Optional glob pattern under reviewed root (example: 'ai2d/*.json')")
    parser.add_argument(
        "--dedupe",
        choices=["none", "latest_mtime"],
        default="latest_mtime",
        help="How to deduplicate multiple reviewed files for the same (q_id, annotation_path)",
    )
    parser.add_argument("--out", default="merged_attribution_view.csv", help="Output CSV path")
    args = parser.parse_args()

    reviewed_root = Path(args.reviewed_root)
    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)

    by_qid_ann, by_qid = build_prediction_index(pred_root)

    review_paths = list(iter_review_files(reviewed_root, args.pattern))
    review_paths = dedupe_review_paths(review_paths, args.dedupe)

    rows: List[dict] = []
    for review_path in review_paths:
        try:
            review_obj = safe_load_json(review_path)
        except Exception as exc:
            rows.append({
                "review_file": str(review_path),
                "error": f"failed_to_load_review:{exc}",
            })
            continue

        q_id = str(review_obj.get("q_id") or "").strip()
        ann_path = str(review_obj.get("annotation_path") or "").strip()
        image_path = str(review_obj.get("image_path") or "").strip()

        pred_record = find_prediction_for_review(q_id, ann_path, image_path, by_qid_ann, by_qid)
        # Use Gemini prediction files as source of truth for predicted IDs.
        pred_ids = set(pred_record.pred_ids) if pred_record else set()

        ann_file = resolve_annotation_path(dataset_root, ann_path)
        gt_available_ids: Set[str] = set()
        ann_load_error = ""
        if ann_file:
            try:
                ann_obj = safe_load_json(ann_file)
                gt_available_ids = extract_gt_ids(ann_obj)
            except Exception as exc:
                ann_load_error = f"failed_to_load_annotation:{exc}"
        else:
            ann_load_error = "annotation_not_found"

        reviewed_pred_ids, reviewed_gt_ids, reviewed_new_ids = parse_review_sources(review_obj)

        # Cross-dataset robust definitions:
        # retained prediction = any reviewed selected ID (pred/gt source) that belongs to Gemini predictions
        reviewed_selected_ids = reviewed_pred_ids | reviewed_gt_ids
        retained_ids = reviewed_selected_ids & pred_ids
        # Explicit pair-based removal signal:
        # source: pred present but matching source: gt absent in reviewed file.
        removed_by_annotator_ids = reviewed_pred_ids - reviewed_gt_ids
        # Schema-robust effective removal:
        # Gemini prediction not present in final reviewed selected IDs (pred or gt).
        effective_removed_ids = pred_ids - retained_ids
        # added GT = reviewed GT selection not in Gemini predictions
        added_gt_ids = reviewed_gt_ids - pred_ids

        # 'predicted out of GT available' means intersection of Gemini IDs with GT IDs for the pair.
        predicted_from_gt_ids = pred_ids & gt_available_ids if gt_available_ids else set()

        rows.append(
            {
                "dataset": infer_dataset_from_annotation_path(ann_path),
                "q_id": q_id,
                "review_file": str(review_path),
                "image_path": image_path,
                "annotation_path": ann_path,
                "gt_available_count": len(gt_available_ids),
                "predicted_count": len(pred_ids),
                "predicted_from_gt_count": len(predicted_from_gt_ids),
                "reviewed_pred_count": len(reviewed_pred_ids),
                "retained_pred_count": len(retained_ids),
                "removed_by_annotator_count": len(removed_by_annotator_ids),
                "removed_by_annotator_ids": "|".join(sorted(removed_by_annotator_ids)),
                "effective_removed_count": len(effective_removed_ids),
                "effective_removed_ids": "|".join(sorted(effective_removed_ids)),
                "deselected_pred_count": max(0, len(predicted_from_gt_ids) - len(retained_ids)),
                "added_gt_count": len(added_gt_ids),
                "new_drawn_count": len(reviewed_new_ids),
                "annotation_status": ann_load_error or "ok",
                "prediction_status": "ok" if pred_record else "prediction_not_found",
                "retained_pred_ids": "|".join(sorted(retained_ids)),
                "added_gt_ids": "|".join(sorted(added_gt_ids)),
                "new_ids": "|".join(sorted(reviewed_new_ids)),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Make sure we always write a stable header even if some rows only have errors.
    header = [
        "dataset",
        "q_id",
        "review_file",
        "image_path",
        "annotation_path",
        "gt_available_count",
        "predicted_count",
        "predicted_from_gt_count",
        "reviewed_pred_count",
        "retained_pred_count",
        "removed_by_annotator_count",
        "removed_by_annotator_ids",
        "effective_removed_count",
        "effective_removed_ids",
        "deselected_pred_count",
        "added_gt_count",
        "new_drawn_count",
        "annotation_status",
        "prediction_status",
        "retained_pred_ids",
        "added_gt_ids",
        "new_ids",
        "error",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
