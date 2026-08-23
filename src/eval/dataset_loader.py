"""
Eval: loads the golden question sets used by the faithfulness eval script.

Two CSVs live under resources/sample_questions/:
- positive_rag_questions.csv: 20 questions the policy documents SHOULD be
  able to answer, each with a manually-verified expected answer.
- negative_out_of_document_questions.csv: 5 questions the policy
  documents do NOT cover — the system should refuse to answer these
  (this is what Phase 5's citation enforcement is checked against).
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.config import PROJECT_ROOT

SAMPLE_QUESTIONS_DIR = PROJECT_ROOT / "resources" / "sample_questions"
POSITIVE_CSV = SAMPLE_QUESTIONS_DIR / "positive_rag_questions.csv"
NEGATIVE_CSV = SAMPLE_QUESTIONS_DIR / "negative_out_of_document_questions.csv"


@dataclass(frozen=True)
class PositiveCase:
    """A question the policy documents should be able to answer."""

    id: str
    question: str
    expected_answer: str


@dataclass(frozen=True)
class NegativeCase:
    """A question the policy documents do NOT cover — expected to be refused."""

    id: str
    question: str
    expected_note: str


def load_positive_cases(path: Path = POSITIVE_CSV) -> List[PositiveCase]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            PositiveCase(
                id=row["id"], question=row["question"], expected_answer=row["expected_answer"]
            )
            for row in csv.DictReader(f)
        ]


def load_negative_cases(path: Path = NEGATIVE_CSV) -> List[NegativeCase]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            NegativeCase(id=row["id"], question=row["question"], expected_note=row["expected_note"])
            for row in csv.DictReader(f)
        ]


if __name__ == "__main__":
    # Manual check: confirms both CSVs parse correctly and the row counts
    # match what's expected, with no model or network call needed.
    positives = load_positive_cases()
    negatives = load_negative_cases()

    print(f"{len(positives)} positive case(s):")
    print(f"  1st: {positives[0].question!r} -> {positives[0].expected_answer[:60]!r}...")
    print(f"  last: {positives[-1].question!r}")

    print(f"\n{len(negatives)} negative case(s):")
    print(f"  1st: {negatives[0].question!r} -> {negatives[0].expected_note!r}")