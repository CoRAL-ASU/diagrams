#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from merge_attribution_view import (
    build_prediction_index,
    dedupe_review_paths,
    extract_gt_ids,
    find_prediction_for_review,
    iter_review_files,
    resolve_annotation_path,
    safe_load_json,
)


def parse_review_boxes(review_obj: dict):
    items = review_obj.get("bbox")
    if items is None:
        items = review_obj.get("boxes")
    if items is None:
        items = []

    pred_ids = set()
    gt_ids = set()
    new_ids = set()
    all_ids = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "").strip()
        if not ident:
            continue
        src = str(item.get("source") or "").strip().lower()
        all_ids.add(ident)
        if src == "pred":
            pred_ids.add(ident)
        elif src == "gt":
            gt_ids.add(ident)
        elif src == "new":
            new_ids.add(ident)

    return all_ids, pred_ids, gt_ids, new_ids


def dataset_from_annotation_path(annotation_path: str) -> str:
    if not annotation_path:
        return "unknown"
    return annotation_path.split("/")[0]


def sorted_join(values):
    return "|".join(sorted(values))


def main():
    parser = argparse.ArgumentParser(
        description="Export per-dataset CSV views with GT/predicted/reviewed box IDs and counts."
    )
    parser.add_argument("--reviewed-root", default="Reviewed")
    parser.add_argument("--dataset-root", default="Diagram_Attribution_Dataset")
    parser.add_argument("--pred-root", default="Predictions/gemini-pred")
    parser.add_argument("--pattern", default="", help="Optional reviewed glob pattern, e.g. 'ai2d/*.json'")
    parser.add_argument(
        "--dedupe",
        choices=["none", "latest_mtime"],
        default="latest_mtime",
        help="Dedupe strategy for reviewed variants",
    )
    parser.add_argument("--out-dir", default="dataset_box_views")
    args = parser.parse_args()

    reviewed_root = Path(args.reviewed_root)
    dataset_root = Path(args.dataset_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_qid_ann, by_qid = build_prediction_index(pred_root)

    review_paths = list(iter_review_files(reviewed_root, args.pattern))
    review_paths = dedupe_review_paths(review_paths, args.dedupe)

    by_dataset_rows = defaultdict(list)
    for review_path in review_paths:
        try:
            review_obj = safe_load_json(review_path)
        except Exception:
            continue

        q_id = str(review_obj.get("q_id") or "").strip()
        ann_path = str(review_obj.get("annotation_path") or "").strip()
        image_path = str(review_obj.get("image_path") or "").strip()
        dataset = dataset_from_annotation_path(ann_path)

        pred_record = find_prediction_for_review(q_id, ann_path, image_path, by_qid_ann, by_qid)
        pred_ids = set(pred_record.pred_ids) if pred_record else set()

        gt_ids = set()
        ann_file = resolve_annotation_path(dataset_root, ann_path)
        if ann_file:
            try:
                gt_ids = extract_gt_ids(safe_load_json(ann_file))
            except Exception:
                gt_ids = set()

        reviewed_all, reviewed_pred, reviewed_gt, reviewed_new = parse_review_boxes(review_obj)

        by_dataset_rows[dataset].append(
            {
                "review_file": str(review_path),
                "q_id": q_id,
                "image_path": image_path,
                "annotation_path": ann_path,
                "gt_boxes_count": len(gt_ids),
                "gt_boxes": sorted_join(gt_ids),
                "predicted_boxes_count": len(pred_ids),
                "predicted_boxes": sorted_join(pred_ids),
                "reviewed_boxes_count": len(reviewed_all),
                "reviewed_boxes": sorted_join(reviewed_all),
                "reviewed_pred_boxes_count": len(reviewed_pred),
                "reviewed_pred_boxes": sorted_join(reviewed_pred),
                "reviewed_gt_boxes_count": len(reviewed_gt),
                "reviewed_gt_boxes": sorted_join(reviewed_gt),
                "reviewed_new_boxes_count": len(reviewed_new),
                "reviewed_new_boxes": sorted_join(reviewed_new),
            }
        )

    headers = [
        "review_file",
        "q_id",
        "image_path",
        "annotation_path",
        "gt_boxes_count",
        "gt_boxes",
        "predicted_boxes_count",
        "predicted_boxes",
        "reviewed_boxes_count",
        "reviewed_boxes",
        "reviewed_pred_boxes_count",
        "reviewed_pred_boxes",
        "reviewed_gt_boxes_count",
        "reviewed_gt_boxes",
        "reviewed_new_boxes_count",
        "reviewed_new_boxes",
    ]

    for dataset, rows in sorted(by_dataset_rows.items()):
        out_path = out_dir / f"{dataset}_box_view.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
