from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate import evaluate_directories, write_markdown_report
from .extract_rules import extract_from_markdown
from .ingest import normalize_document
from .ocr import run_ocr
from .paths import RunPaths
from .settings import DEFAULT_CONFIG_PATH, DEFAULT_LAYOUT_DEVICE, DEFAULT_RUN_ROOT
from .utils import write_json
from .workflows import (
    evaluate_workflow,
    run_ocr_workflow,
    run_pipeline_workflow,
    run_rules_workflow,
    run_structured_workflow,
    write_run_metadata as workflow_write_run_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-doc-extract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ocr_parser = subparsers.add_parser("ocr", help="Normalize input and run OCR")
    _add_input_args(ocr_parser)
    ocr_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ocr_parser.add_argument("--layout-device", default=DEFAULT_LAYOUT_DEVICE)
    ocr_parser.add_argument("--layout-profile", choices=["auto", "pp_doclayout_formula", "pp_doclayout_split_formula"], default="auto")
    ocr_parser.set_defaults(func=cmd_ocr)

    rules_parser = subparsers.add_parser("extract-rules", help="Run rules baseline on OCR output")
    _add_run_args(rules_parser)
    rules_parser.set_defaults(func=cmd_extract_rules)

    structured_parser = subparsers.add_parser(
        "extract-glmocr", help="Run direct structured extraction via Ollama"
    )
    _add_run_args(structured_parser)
    structured_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    structured_parser.add_argument("--model", default=None)
    structured_parser.add_argument("--endpoint", default=None)
    structured_parser.set_defaults(func=cmd_extract_glmocr)

    eval_parser = subparsers.add_parser("eval", help="Evaluate predictions against gold labels")
    eval_parser.add_argument("--gold-dir", required=True)
    eval_parser.add_argument("--pred-dir", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.set_defaults(func=cmd_eval)

    run_parser = subparsers.add_parser("run", help="Run OCR and rule extraction end to end")
    _add_input_args(run_parser)
    run_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    run_parser.add_argument("--layout-device", default=DEFAULT_LAYOUT_DEVICE)
    run_parser.add_argument("--layout-profile", choices=["auto", "pp_doclayout_formula", "pp_doclayout_split_formula"], default="auto")
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def cmd_ocr(args: argparse.Namespace) -> Path:
    return run_ocr_workflow(
        args.input_path,
        run=args.run,
        run_root=args.run_root,
        config_path=args.config,
        layout_device=args.layout_device,
        layout_profile=args.layout_profile,
        normalize_document_fn=normalize_document,
        run_ocr_fn=run_ocr,
        write_json_fn=write_json,
    )


def cmd_extract_rules(args: argparse.Namespace) -> None:
    run_rules_workflow(
        args.run,
        run_root=args.run_root,
        extract_from_markdown_fn=extract_from_markdown,
        write_json_fn=write_json,
    )


def cmd_extract_glmocr(args: argparse.Namespace) -> None:
    from .experimental.extract_glmocr import extract_structured, save_structured_result

    run_structured_workflow(
        args.run,
        run_root=args.run_root,
        config_path=args.config,
        model=args.model,
        endpoint=args.endpoint,
        extract_structured_fn=extract_structured,
        save_structured_result_fn=save_structured_result,
        write_json_fn=write_json,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluate_workflow(
        args.gold_dir,
        args.pred_dir,
        args.output,
        evaluate_directories_fn=evaluate_directories,
        write_markdown_report_fn=write_markdown_report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_run(args: argparse.Namespace) -> None:
    run_pipeline_workflow(
        args.input_path,
        run=args.run,
        run_root=args.run_root,
        config_path=args.config,
        layout_device=args.layout_device,
        layout_profile=args.layout_profile,
        normalize_document_fn=normalize_document,
        run_ocr_fn=run_ocr,
        extract_from_markdown_fn=extract_from_markdown,
        write_json_fn=write_json,
    )


def resolve_run_dir(run: str | None, run_root: str, input_path: str) -> Path:
    return RunPaths.from_input(input_path, run=run, run_root=run_root).ensure_run_dir()


def resolve_named_run_dir(run: str, run_root: str) -> Path:
    return RunPaths.from_named_run(run, run_root=run_root).run_dir


def write_run_metadata(
    run_dir: str | Path,
    input_path: str,
    page_paths: list[str],
    ocr_result: dict[str, Any],
) -> None:
    workflow_write_run_metadata(
        run_dir,
        input_path,
        page_paths,
        ocr_result,
        write_json_fn=write_json,
    )


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--run")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
