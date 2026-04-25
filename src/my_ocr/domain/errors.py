from __future__ import annotations


class ApplicationError(Exception):
    """Base class for user-facing workflow failures."""


class RunNotFound(ApplicationError):
    pass


class MissingInputDocument(ApplicationError):
    pass


class MissingPage(ApplicationError):
    pass


class LayoutDetectionFailed(ApplicationError):
    pass


class OcrFailed(ApplicationError):
    pass


class StructuredExtractionFailed(ApplicationError):
    pass


class RunCommitFailed(ApplicationError):
    pass
