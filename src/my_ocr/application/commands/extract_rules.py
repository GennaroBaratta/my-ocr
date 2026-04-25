from __future__ import annotations

from dataclasses import dataclass

from my_ocr.application.dto import RunId, WorkflowResult
from my_ocr.application.errors import OcrFailed
from my_ocr.application.ports import RulesExtractor, RunStore


@dataclass(frozen=True, slots=True)
class ExtractRulesCommand:
    run_id: RunId


class ExtractRules:
    def __init__(self, run_store: RunStore, rules_extractor: RulesExtractor) -> None:
        self._run_store = run_store
        self._rules_extractor = rules_extractor

    def __call__(self, command: ExtractRulesCommand) -> WorkflowResult:
        snapshot = self._run_store.open_run(command.run_id)
        if snapshot.ocr_result is None or not snapshot.ocr_result.markdown.strip():
            raise OcrFailed("Rules extraction requires OCR markdown.")
        prediction = self._rules_extractor.extract(snapshot.ocr_result.markdown)
        tx = self._run_store.begin_update(command.run_id)
        try:
            tx.write_rules_extraction(prediction)
            return WorkflowResult(snapshot=tx.commit())
        except Exception:
            tx.rollback()
            raise

