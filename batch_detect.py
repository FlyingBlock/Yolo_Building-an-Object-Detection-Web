"""
Batch detection CLI: run YOLO over every image in a folder.

    python batch_detect.py --input ./images/ --output ./results/ --model yolov8s

Writes one annotated image per input plus optional per-image JSON, a combined
results.json and a combined results.csv.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from pathlib import Path

import detector as det

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_detect.py",
        description="Detect objects across a folder of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Folder containing images.")
    parser.add_argument("--output", "-o", default="results", help="Folder for results.")
    parser.add_argument("--model", "-m", default=det.DEFAULT_MODEL, help="Model alias or .pt path.")
    parser.add_argument("--confidence", "-c", type=float, default=det.CONFIDENCE_THRESHOLD,
                        help="Minimum confidence in (0, 1].")
    parser.add_argument("--iou", type=float, default=det.IOU_THRESHOLD,
                        help="NMS IoU threshold in (0, 1].")
    parser.add_argument("--device", default=det.DEVICE, help="auto, cpu, cuda:0 or mps.")
    parser.add_argument("--max-size", type=int, default=det.MAX_IMAGE_SIZE,
                        help="Downscale the long edge to this many pixels for inference.")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Walk subfolders as well.")
    parser.add_argument("--no-annotated", action="store_true",
                        help="Skip writing annotated images (metadata only).")
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg",
                        help="Format for annotated images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many images (useful for a quick test).")
    return parser


def collect_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"error: no such folder: {input_dir}", file=sys.stderr)
        return 2

    images = collect_images(input_dir, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"error: no images found in {input_dir}", file=sys.stderr)
        return 2

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} image(s) with {args.model} -> {output_dir}")
    extension = "png" if args.image_format == "png" else "jpg"

    all_results: list[dict] = []
    csv_rows: list[dict] = []
    failures = 0
    started = time.perf_counter()

    for index, path in enumerate(images, start=1):
        relative = path.relative_to(input_dir)
        target_dir = output_dir / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            image = det.decode_image(path.read_bytes())
            result = det.detect(
                image,
                model=args.model,
                confidence=args.confidence,
                iou=args.iou,
                device=args.device,
                max_size=args.max_size,
                annotate=not args.no_annotated,
                image_format=args.image_format,
            )
        except (det.ModelError, ValueError) as exc:
            failures += 1
            print(
                f"  [{index}/{len(images)}] {relative}: FAILED - {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if not args.no_annotated:
            annotated_path = target_dir / f"{path.stem}_annotated.{extension}"
            annotated_path.write_bytes(
                base64.b64decode(result["annotated_image"].split(",", 1)[1])
            )

        counts = ", ".join(f"{k}:{v}" for k, v in result["counts_by_class"].items()) or "-"
        print(
            f"  [{index}/{len(images)}] {relative}: {result['count']} object(s) "
            f"[{counts}] in {result['inference_ms']:.1f} ms",
            flush=True,
        )

        record = {
            "image": str(relative).replace("\\", "/"),
            "width": result["image"]["width"],
            "height": result["image"]["height"],
            "count": result["count"],
            "counts_by_class": result["counts_by_class"],
            "inference_ms": result["inference_ms"],
            "detections": result["detections"],
        }
        all_results.append(record)

        for det_item in result["detections"]:
            box = det_item["box"]
            csv_rows.append(
                {
                    "image": record["image"],
                    "class_id": det_item["class_id"],
                    "class_name": det_item["class_name"],
                    "confidence": det_item["confidence"],
                    "x1": box["x1"],
                    "y1": box["y1"],
                    "x2": box["x2"],
                    "y2": box["y2"],
                }
            )

    elapsed = time.perf_counter() - started
    total_objects = sum(record["count"] for record in all_results)

    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "model": det.resolve_model_ref(args.model),
                "device": det.resolve_device(args.device),
                "confidence_threshold": args.confidence,
                "iou_threshold": args.iou,
                "images_processed": len(all_results),
                "images_failed": failures,
                "total_objects": total_objects,
                "elapsed_seconds": round(elapsed, 2),
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"\nDone: {len(all_results)} image(s), {total_objects} object(s) in "
        f"{elapsed:.1f}s ({elapsed / max(1, len(all_results)):.2f}s/image)"
    )
    if failures:
        print(f"  {failures} image(s) failed")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    return 1 if failures and not all_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
