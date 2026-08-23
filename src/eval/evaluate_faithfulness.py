"""
Eval: offline faithfulness evaluation against the golden question sets.

Two things get measured, against the CSVs in resources/sample_questions/:

1. FAITHFULNESS on positive_rag_questions.csv (20 questions the policy
   documents should answer): for each question, runs it through the real
   pipeline, then scores whether the generated answer's claims are
   actually supported by the retrieved chunks, via DeepEval's
   FaithfulnessMetric. This is a different (more rigorous, LLM-graded)
   check than Phase 5's citation_enforcer.py — that module is a live,
   single-pass guardrail on every real query; this script is an offline,
   scored evaluation across a whole curated test set, meant to be run
   in CI (Phase 7) to catch regressions before they reach users.

2. REFUSAL CORRECTNESS on negative_out_of_document_questions.csv (5
   questions the documents do NOT cover): for each, runs it through the
   pipeline and checks whether the system correctly refused (matches
   citation_enforcer's refusal messages) rather than hallucinating an
   answer from outside knowledge.

Exits with a non-zero status code if either score falls below its
configured threshold — this is what will make a CI pipeline (Phase 7)
fail the build on a quality regression, not just print a report.

Usage:
    python -m eval.evaluate_faithfulness
"""
import logging
import sys
from dataclasses import dataclass
from typing import List

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from core.pipeline import RAGPipeline
from eval.dataset_loader import load_negative_cases, load_positive_cases
from eval.groq_deepeval_llm import GroqDeepEvalLLM
from search.citation_enforcer import NO_CITATIONS_MESSAGE, UNSUPPORTED_ANSWER_MESSAGE

logger = logging.getLogger(__name__)

FAITHFULNESS_THRESHOLD = 0.8  # per-question minimum score to count as "passing"
FAITHFULNESS_AVG_THRESHOLD = 0.85  # average across all positive cases
REFUSAL_RATE_THRESHOLD = 0.8  # fraction of negative cases correctly refused


@dataclass
class FaithfulnessResult:
    question: str
    answer: str
    score: float
    reason: str


@dataclass
class RefusalResult:
    question: str
    answer: str
    correctly_refused: bool


def _evaluate_positive_cases(pipeline: RAGPipeline, judge: GroqDeepEvalLLM) -> List[FaithfulnessResult]:
    metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge, async_mode=False)
    results = []

    for case in load_positive_cases():
        answer, citations = pipeline.generate_answer(case.question)
        test_case = LLMTestCase(
            input=case.question,
            actual_output=answer,
            retrieval_context=[c.snippet for c in citations],
        )
        metric.measure(test_case)
        results.append(
            FaithfulnessResult(
                question=case.question, answer=answer, score=metric.score, reason=metric.reason
            )
        )
        logger.info("Faithfulness %.2f for: %r", metric.score, case.question)

    return results


def _evaluate_negative_cases(pipeline: RAGPipeline) -> List[RefusalResult]:
    refusal_messages = (NO_CITATIONS_MESSAGE, UNSUPPORTED_ANSWER_MESSAGE)
    results = []

    for case in load_negative_cases():
        answer, _ = pipeline.generate_answer(case.question)
        correctly_refused = answer in refusal_messages
        results.append(
            RefusalResult(question=case.question, answer=answer, correctly_refused=correctly_refused)
        )
        logger.info(
            "Refusal %s for: %r", "OK" if correctly_refused else "MISSED", case.question
        )

    return results


def run_evaluation() -> bool:
    """Runs the full evaluation, prints a report, and returns whether it passed."""
    pipeline = RAGPipeline()
    print("Ingesting policy documents...")
    for status in pipeline.ingest_documents():
        print(f"  {status}")

    judge = GroqDeepEvalLLM(pipeline.llm)

    print("\nEvaluating faithfulness on positive cases...")
    faithfulness_results = _evaluate_positive_cases(pipeline, judge)
    avg_faithfulness = sum(r.score for r in faithfulness_results) / len(faithfulness_results)
    failing = [r for r in faithfulness_results if r.score < FAITHFULNESS_THRESHOLD]

    print("\nEvaluating refusal correctness on negative cases...")
    refusal_results = _evaluate_negative_cases(pipeline)
    refusal_rate = sum(r.correctly_refused for r in refusal_results) / len(refusal_results)
    missed_refusals = [r for r in refusal_results if not r.correctly_refused]

    print("\n" + "=" * 70)
    print("FAITHFULNESS REPORT")
    print("=" * 70)
    print(f"Average faithfulness: {avg_faithfulness:.2f} (threshold: {FAITHFULNESS_AVG_THRESHOLD})")
    print(f"Cases below per-question threshold ({FAITHFULNESS_THRESHOLD}): {len(failing)}/{len(faithfulness_results)}")
    for r in failing:
        print(f"  - {r.score:.2f}: {r.question!r}")
        print(f"      reason: {r.reason}")

    print(f"\nRefusal rate: {refusal_rate:.2f} (threshold: {REFUSAL_RATE_THRESHOLD})")
    print(f"Missed refusals: {len(missed_refusals)}/{len(refusal_results)}")
    for r in missed_refusals:
        print(f"  - {r.question!r}")
        print(f"      answered instead of refusing: {r.answer[:100]!r}")

    passed = (
        avg_faithfulness >= FAITHFULNESS_AVG_THRESHOLD and refusal_rate >= REFUSAL_RATE_THRESHOLD
    )
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    passed = run_evaluation()
    sys.exit(0 if passed else 1)