from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_directories, write_markdown_report
from .extract_glmocr import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_ENDPOINT,
    extract_structured,
    save_structured_result,
)
from .extract_rules import extract_from_markdown
from .ingest import normalize_document
from .ocr import run_ocr
from .utils import ensure_dir, slugify, timestamp_id, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-doc-extract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr", help="Normalize input and run OCR")
    _add_input_args(ocr_parser)
    ocr_parser.add_argument("--config", default="config/local.yaml")
    ocr_parser.add_argument("--layout-device", default="cpu")
    ocr_parser.set_defaults(func=cmd_ocr)

    rules_parser = subparsers.add_parser("extract-rules", help="Run rules baseline on OCR output")
    _add_run_args(rules_parser)
    rules_parser.set_defaults(func=cmd_extract_rules)

    structured_parser = subparsers.add_parser(
        "extract-glmocr", help="Run direct structured extraction via Ollama"
    )
    _add_run_args(structured_parser)
    structured_parser.add_argument("--model", default=DEFAULT_MODEL)
    structured_parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    structured_parser.set_defaults(func=cmd_extract_glmocr)

    eval_parser = subparsers.add_parser("eval", help="Evaluate predictions against gold labels")
    eval_parser.add_argument("--gold-dir", required=True)
    eval_parser.add_argument("--pred-dir", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.set_defaults(func=cmd_eval)

    run_parser = subparsers.add_parser("run", help="Run OCR and rule extraction end to end")
    _add_input_args(run_parser)
    run_parser.add_argument("--config", default="config/local.yaml")
    run_parser.add_argument("--layout-device", default="cpu")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def cmd_ocr(args: argparse.Namespace) -> Path:
    run_dir = resolve_run_dir(args.run, args.run_root, args.input_path)
    pages = normalize_document(args.input_path, run_dir)
    result = run_ocr(
        pages,
        run_dir,
        config_path=args.config,
        layout_device=args.layout_device,
    )
    write_run_metadata(run_dir, args.input_path, pages, result)
    return Path(run_dir)


def cmd_extract_rules(args: argparse.Namespace) -> None:
    run_dir = resolve_named_run_dir(args.run, args.run_root)
    markdown_path = run_dir / "ocr.md"
    if not markdown_path.exists():
        raise FileNotFoundError(f"Missing OCR markdown: {markdown_path}")

    prediction = extract_from_markdown(markdown_path.read_text(encoding="utf-8"))
    pred_dir = ensure_dir(run_dir / "predictions")
    write_json(pred_dir / "rules.json", prediction)
    write_json(pred_dir / f"{run_dir.name}.json", prediction)


def cmd_extract_glmocr(args: argparse.Namespace) -> None:
    run_dir = resolve_named_run_dir(args.run, args.run_root)
    page_paths = sorted(str(path) for path in (run_dir / "pages").iterdir() if path.is_file())
    if not page_paths:
        raise FileNotFoundError(f"No page images found in {run_dir / 'pages'}")

    prediction, metadata = extract_structured(
        page_paths,
        model=args.model,
        endpoint=args.endpoint,
    )
    save_structured_result(run_dir, prediction, metadata)
    write_json(run_dir / "predictions" / f"{run_dir.name}.json", prediction)


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluate_directories(args.gold_dir, args.pred_dir)
    write_markdown_report(report, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_run(args: argparse.Namespace) -> None:
    run_dir = cmd_ocr(args)
    args.run = run_dir.name
    cmd_extract_rules(args)


def resolve_run_dir(run: str | None, run_root: str, input_path: str) -> Path:
    run_id = run or f"{slugify(Path(input_path).stem)}-{timestamp_id()}"
    run_dir = ensure_dir(Path(run_root) / run_id)
    return run_dir


def resolve_named_run_dir(run: str, run_root: str) -> Path:
    run_dir = Path(run_root) / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def write_run_metadata(
    run_dir: str | Path,
    input_path: str,
    page_paths: list[str],
    ocr_result: dict[str, Any],
) -> None:
    payload = {
        "input_path": str(input_path),
        "page_paths": page_paths,
        "ocr_raw_dir": ocr_result.get("raw_dir"),
    }
    write_json(Path(run_dir) / "meta.json", payload)


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--run")
    parser.add_argument("--run-root", default="data/runs")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-root", default="data/runs")


if __name__ == "__main__":
    raise SystemExit(main())
