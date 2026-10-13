"""
Single-image detection CLI.

    python detect.py --image photo.jpg --model yolov8m --save_json detections.json

Writes the annotated image next to the input (or to --output) and optionally the
detection metadata as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import detector as det


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="Detect objects in one image with YOLOv8 / YOLO11.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", "-i", required=True, help="Path to the input image.")
    parser.add_argument("--output", "-o", default=None, help="Annotated image path.")
    parser.add_argument("--model", "-m", default=det.DEFAULT_MODEL, help="Model alias or .pt path.")
    parser.add_argument("--confidence", "-c", type=float, default=det.CONFIDENCE_THRESHOLD,
                        help="Minimum confidence in (0, 1].")
    parser.add_argument("--iou", type=float, default=det.IOU_THRESHOLD,
                        help="NMS IoU threshold in (0, 1].")
    parser.add_argument("--device", default=det.DEVICE, help="auto, cpu, cuda:0 or mps.")
    parser.add_argument("--max-size", type=int, default=det.MAX_IMAGE_SIZE,
                        help="Downscale the long edge to this many pixels for inference.")
    parser.add_argument("--classes", default=None,
                        help="Comma-separated class ids to keep, e.g. 0,2,5.")
    parser.add_argument("--save-json", "--save_json", dest="save_json", default=None,
                        help="Write detection metadata to this JSON file.")
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg",
                        help="Format for the annotated image.")
    parser.add_argument("--classes-only", action="store_true",
                        help="Print only the per-class summary, not every box.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source = Path(args.image)
    if not source.is_file():
        print(f"error: no such image: {source}", file=sys.stderr)
        return 2

    class_filter = None
    if args.classes:
        try:
            class_filter = [int(part) for part in args.classes.replace(" ", "").split(",") if part]
        except ValueError:
            print("error: --classes must be a comma-separated list of ids", file=sys.stderr)
            return 2

    try:
        image = det.decode_image(source.read_bytes())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = det.detect(
            image,
            model=args.model,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
            max_size=args.max_size,
            classes=class_filter,
            image_format=args.image_format,
        )
    except (det.ModelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{source.name}: {result['count']} object(s) | model {result['model']} "
        f"| device {result['device']} | {result['inference_ms']:.1f} ms "
        f"| {result['image']['width']}x{result['image']['height']} px"
    )
    if result["counts_by_class"]:
        print("  " + ", ".join(f"{k}: {v}" for k, v in result["counts_by_class"].items()))

    summary = det.summarise(result["detections"])
    if args.classes_only:
        for row in summary:
            print(
                f"  {row['class']:<16} count={row['count']:<3} "
                f"mean={row['mean_confidence']:.3f} max={row['max_confidence']:.3f}"
            )
    else:
        for index, item in enumerate(result["detections"], start=1):
            box = item["box"]
            print(
                f"  {index:>3}. {item['class_name']:<16} {item['confidence']:.3f}  "
                f"[{box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f}]"
            )

    # Annotated image
    import base64

    extension = "png" if args.image_format == "png" else "jpg"
    annotated_path = Path(args.output) if args.output else source.with_name(
        f"{source.stem}_annotated.{extension}"
    )
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path.write_bytes(base64.b64decode(result["annotated_image"].split(",", 1)[1]))
    print(f"  annotated image -> {annotated_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in result.items() if k != "annotated_image"}
        payload["annotated_image_path"] = str(annotated_path)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  detections      -> {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
