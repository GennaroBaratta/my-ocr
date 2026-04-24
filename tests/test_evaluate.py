from __future__ import annotations

import json

from my_ocr.application.use_cases.evaluation import compare_records, evaluate_directories


def test_evaluate_directories_tracks_missing_predictions_and_date_normalization(tmp_path) -> None:
    gold_dir = tmp_path / "gold"
    pred_dir = tmp_path / "pred"
    gold_dir.mkdir()
    pred_dir.mkdir()

    (gold_dir / "doc-1.json").write_text(
        json.dumps(
            {
                "document_type": "report",
                "title": "A",
                "authors": [],
                "institution": "",
                "date": "2024-01-15",
                "language": "en",
                "summary_line": "x",
            }
        ),
        encoding="utf-8",
    )
    (gold_dir / "doc-2.json").write_text(
        json.dumps(
            {
                "document_type": "report",
                "title": "B",
                "authors": [],
                "institution": "",
                "date": "15 January 2024",
                "language": "en",
                "summary_line": "y",
            }
        ),
        encoding="utf-8",
    )
    (pred_dir / "doc-1.json").write_text(
        json.dumps(
            {
                "document_type": "report",
                "title": "A",
                "authors": [],
                "institution": "",
                "date": "15/01/2024",
                "language": "en",
                "summary_line": "x",
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_directories(gold_dir, pred_dir)

    assert report["missing_prediction_count"] == 1
    assert report["missing_prediction_files"] == ["doc-2.json"]
    assert report["fields"]["date"]["normalized_match_rate"] == 0.5
    assert report["fields"]["date"]["exact_match_rate"] == 0.0
    assert report["comparisons"][0]["fields"]["date"]["normalized_match"] is True


def test_compare_records_marks_missing_values() -> None:
    result = compare_records("doc", {"date": "2024"}, {"date": ""})
    assert result["fields"]["date"]["pred_missing"] is True
