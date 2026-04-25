from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_ocr.domain.document import FIELD_NAMES, DocumentFields
from my_ocr.filesystem import ensure_dir, read_json as load_json, write_text


def evaluate_workflow(
    gold_dir: str,
    pred_dir: str,
    output: str,
    *,
    evaluate_directories_fn: Any = None,
    write_markdown_report_fn: Any = None,
) -> dict[str, Any]:
    evaluate = evaluate_directories_fn or evaluate_directories
    write_report = write_markdown_report_fn or write_markdown_report
    report = evaluate(gold_dir, pred_dir)
    write_report(report, output)
    return report


def evaluate_directories(gold_dir: str | Path, pred_dir: str | Path) -> dict[str, Any]:
    gold_dir = Path(gold_dir)
    pred_dir = Path(pred_dir)

    gold_files = sorted(path for path in gold_dir.glob("*.json"))
    if not gold_files:
        raise ValueError(f"No gold JSON files found in {gold_dir}")

    comparisons = []
    missing_predictions: list[str] = []
    for gold_path in gold_files:
        prediction_path = pred_dir / gold_path.name
        gold_record = DocumentFields.from_mapping(load_json(gold_path)).to_dict()
        if prediction_path.exists():
            pred_record = DocumentFields.from_mapping(load_json(prediction_path)).to_dict()
        else:
            pred_record = DocumentFields.empty().to_dict()
            missing_predictions.append(gold_path.name)
        comparisons.append(compare_records(gold_path.stem, gold_record, pred_record))

    return summarize_comparisons(comparisons, missing_predictions)


def compare_records(doc_id: str, gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    field_results: dict[str, dict[str, Any]] = {}
    for field in FIELD_NAMES:
        gold_value = gold.get(field, "")
        pred_value = pred.get(field, "")
        result = {
            "gold": gold_value,
            "pred": pred_value,
            "exact_match": normalize_value(gold_value) == normalize_value(pred_value),
            "pred_missing": is_missing(pred_value),
        }
        if field == "date":
            result["normalized_match"] = normalize_date(str(gold_value)) == normalize_date(
                str(pred_value)
            )
        field_results[field] = result

    return {"doc_id": doc_id, "fields": field_results}


def summarize_comparisons(
    comparisons: list[dict[str, Any]], missing_predictions: list[str] | None = None
) -> dict[str, Any]:
    total_docs = len(comparisons)
    field_summary: dict[str, dict[str, float]] = {}
    for field in FIELD_NAMES:
        exact = sum(1 for comp in comparisons if comp["fields"][field]["exact_match"])
        missing = sum(1 for comp in comparisons if comp["fields"][field]["pred_missing"])
        summary = {
            "exact_match_rate": exact / total_docs,
            "missing_value_rate": missing / total_docs,
        }
        if field == "date":
            normalized = sum(
                1 for comp in comparisons if comp["fields"][field].get("normalized_match")
            )
            summary["normalized_match_rate"] = normalized / total_docs
        field_summary[field] = summary

    return {
        "documents_evaluated": total_docs,
        "missing_prediction_count": len(missing_predictions or []),
        "missing_prediction_files": missing_predictions or [],
        "fields": field_summary,
        "comparisons": comparisons,
    }


def write_markdown_report(report: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    lines = [
        "# Evaluation Report",
        "",
        f"Documents evaluated: **{report['documents_evaluated']}**",
        f"Missing prediction files: **{report.get('missing_prediction_count', 0)}**",
        "",
        "| Field | Exact match | Missing value rate | Date-normalized match |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, summary in report["fields"].items():
        normalized = summary.get("normalized_match_rate")
        normalized_text = f"{normalized:.1%}" if normalized is not None else "-"
        lines.append(
            f"| {field} | {summary['exact_match_rate']:.1%} | {summary['missing_value_rate']:.1%} | {normalized_text} |"
        )
    lines.append("")
    lines.append(f"Generated at: {datetime.now(UTC).isoformat()}")
    lines.append("")
    write_text(output_path, "\n".join(lines))


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if value is None:
        return ""
    return " ".join(str(value).strip().split()).casefold()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    return str(value).strip() == ""


def normalize_date(value: str) -> str:
    value = " ".join(value.strip().split())
    if not value:
        return ""

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%Y",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%Y":
                return parsed.strftime("%Y")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value.casefold()
