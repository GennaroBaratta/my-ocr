from __future__ import annotations

import argparse
import json
from pathlib import Path

from my_ocr.application.commands import (
    ExtractRulesCommand,
    ExtractStructuredCommand,
    PrepareLayoutReviewCommand,
    RunPipelineCommand,
    RunReviewedOcrCommand,
)
from my_ocr.application.dto import (
    LayoutOptions,
    OcrOptions,
    RunId,
    StructuredExtractionOptions,
)
from my_ocr.application.use_cases.evaluation import (
    evaluate_directories,
    evaluate_workflow,
    write_markdown_report,
)
from my_ocr.bootstrap import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LAYOUT_DEVICE,
    DEFAULT_RUN_ROOT,
    build_backend_services,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="my-ocr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-review", help="Normalize input and prepare editable layout review"
    )
    _add_input_args(prepare_parser)
    _add_layout_args(prepare_parser)
    prepare_parser.set_defaults(func=cmd_prepare_review)

    reviewed_ocr_parser = subparsers.add_parser(
        "run-reviewed-ocr", help="Run OCR using a prepared or edited review layout"
    )
    _add_run_args(reviewed_ocr_parser)
    _add_layout_args(reviewed_ocr_parser)
    reviewed_ocr_parser.set_defaults(func=cmd_run_reviewed_ocr)

    ocr_parser = subparsers.add_parser("ocr", help="Run automatic layout detection and OCR")
    _add_input_args(ocr_parser)
    _add_layout_args(ocr_parser)
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

    run_parser = subparsers.add_parser(
        "run", help="Non-interactive automatic layout, OCR, and rules extraction"
    )
    _add_input_args(run_parser)
    _add_layout_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def cmd_prepare_review(args: argparse.Namespace) -> Path:
    services = build_backend_services(args.run_root)
    result = services.prepare_layout_review(
        PrepareLayoutReviewCommand(
            input_path=args.input_path,
            run_id=_optional_run_id(args.run),
            options=_layout_options(args),
        )
    )
    return result.snapshot.run_dir


def cmd_run_reviewed_ocr(args: argparse.Namespace) -> Path:
    services = build_backend_services(args.run_root)
    result = services.run_reviewed_ocr(
        RunReviewedOcrCommand(run_id=RunId(args.run), options=_ocr_options(args))
    )
    return result.snapshot.run_dir


def cmd_ocr(args: argparse.Namespace) -> Path:
    services = build_backend_services(args.run_root)
    prepared = services.prepare_layout_review(
        PrepareLayoutReviewCommand(
            input_path=args.input_path,
            run_id=_optional_run_id(args.run),
            options=_layout_options(args),
        )
    )
    result = services.run_reviewed_ocr(
        RunReviewedOcrCommand(run_id=prepared.snapshot.run_id, options=_ocr_options(args))
    )
    return result.snapshot.run_dir


def cmd_extract_rules(args: argparse.Namespace) -> None:
    services = build_backend_services(args.run_root)
    services.extract_rules(ExtractRulesCommand(run_id=RunId(args.run)))


def cmd_extract_glmocr(args: argparse.Namespace) -> None:
    services = build_backend_services(args.run_root)
    services.extract_structured(
        ExtractStructuredCommand(
            run_id=RunId(args.run),
            options=StructuredExtractionOptions(
                config_path=args.config,
                model=args.model,
                endpoint=args.endpoint,
            ),
        )
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


def cmd_run(args: argparse.Namespace) -> Path:
    services = build_backend_services(args.run_root)
    result = services.run_pipeline(
        RunPipelineCommand(
            input_path=args.input_path,
            run_id=_optional_run_id(args.run),
            layout_options=_layout_options(args),
            ocr_options=_ocr_options(args),
        )
    )
    return result.snapshot.run_dir


def resolve_run_dir(run: str | None, run_root: str, input_path: str) -> Path:
    run_id = run or Path(input_path).stem
    return Path(run_root) / run_id


def resolve_named_run_dir(run: str, run_root: str) -> Path:
    return Path(run_root) / run


def _optional_run_id(run: str | None) -> RunId | None:
    return RunId(run) if run else None


def _layout_options(args: argparse.Namespace) -> LayoutOptions:
    return LayoutOptions(
        config_path=args.config,
        layout_device=args.layout_device,
        layout_profile=args.layout_profile,
    )


def _ocr_options(args: argparse.Namespace) -> OcrOptions:
    return OcrOptions(
        config_path=args.config,
        layout_device=args.layout_device,
        layout_profile=args.layout_profile,
    )


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_path")
    parser.add_argument("--run")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)


def _add_layout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--layout-device", default=DEFAULT_LAYOUT_DEVICE)
    parser.add_argument(
        "--layout-profile",
        choices=["auto", "pp_doclayout_formula", "pp_doclayout_split_formula"],
        default="auto",
    )


if __name__ == "__main__":
    raise SystemExit(main())

