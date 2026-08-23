"""
Search: refuses to answer rather than let an unsupported claim through.

Retrieval and generation can still produce an answer that sounds
confident but isn't actually backed by the retrieved excerpts — the LLM
can paraphrase past the point of accuracy, combine two unrelated chunks
into a claim neither supports, or simply answer from its own general
knowledge when the excerpts don't contain the answer at all. This module
is a second, focused check specifically for that: given the same
excerpts and the answer that was generated, ask the LLM directly whether
every claim in the answer is actually supported — and if not, replace
the answer with an explicit refusal instead of returning it.

This is a live guardrail at answer time, distinct from the offline
faithfulness evaluation Phase 6 will add (which scores this same
question on a curated test set, over time, to catch regressions —
this module is what runs on every single real query).
"""
import logging
from typing import List, Tuple

from search.citations import Citation
from search.prompts import VERIFICATION_PROMPT

logger = logging.getLogger(__name__)

NO_CITATIONS_MESSAGE = (
    "I don't have any policy excerpts relevant to this question, so I "
    "can't answer it. Please check with HR directly, or try rephrasing "
    "your question."
)

UNSUPPORTED_ANSWER_MESSAGE = (
    "I found some potentially related policy excerpts, but I can't "
    "confidently answer this question from them — the answer I'd give "
    "isn't clearly supported by what's actually in the documents. "
    "Please review the excerpts below yourself, or check with HR directly."
)


def _build_context(citations: List[Citation]) -> str:
    """Joins citation snippets into the excerpt text the verifier reads."""
    return "\n\n".join(c.snippet for c in citations)


def _is_supported(llm, query: str, answer: str, citations: List[Citation]) -> bool:
    """
    Runs the verification LLM call and parses its SUPPORTED /
    NOT_SUPPORTED judgment from the first line of the response.

    Defaults to False (unsupported) on any parsing ambiguity — when in
    doubt, this errs toward refusing rather than letting an unverified
    answer through, since a false refusal costs the user a rephrase while
    a false pass-through costs them trusting a wrong answer.
    """
    prompt = VERIFICATION_PROMPT.format(
        context=_build_context(citations), answer=answer
    )
    response = llm.invoke(prompt)
    verdict_line = response.content.strip().splitlines()[0].strip().upper()

    logger.info("Citation verification verdict for query %r: %r", query, verdict_line)

    # Check NOT_SUPPORTED first — it contains "SUPPORTED" as a substring,
    # so checking for "SUPPORTED" alone first would misclassify it.
    if verdict_line.startswith("NOT_SUPPORTED"):
        return False
    if verdict_line.startswith("SUPPORTED"):
        return True

    logger.warning(
        "Unexpected verification response format for query %r: %r — "
        "treating as unsupported.",
        query,
        verdict_line,
    )
    return False


def enforce_citations(llm, query: str, answer: str, citations: List[Citation]) -> Tuple[str, List[Citation]]:
    """
    Checks whether `answer` is actually backed by `citations`, and
    replaces it with an explicit refusal if not.

    Two distinct refusal cases, worded differently on purpose:
    - No citations at all: nothing was even retrieved for this query —
      refuse immediately, no verification call needed, and return an
      empty citation list (there's nothing to show).
    - Citations exist but don't support the answer: something WAS
      retrieved, but the verifier judged the generated answer isn't
      actually backed by it — refuse, but still return the citations so
      the user can review what was found and judge for themselves.
    """
    if not citations:
        logger.info("No citations retrieved for query %r — refusing.", query)
        return NO_CITATIONS_MESSAGE, []

    if _is_supported(llm, query, answer, citations):
        return answer, citations

    logger.info(
        "Answer for query %r was not verified as supported — refusing.", query
    )
    return UNSUPPORTED_ANSWER_MESSAGE, citations


if __name__ == "__main__":
    # Manual check with a fake LLM — no network or real API key needed,
    # just confirms the parsing and both refusal paths behave correctly.
    class FakeLLM:
        """Returns a fixed response regardless of the prompt, for testing."""

        def __init__(self, canned_response: str):
            self._canned_response = canned_response

        def invoke(self, prompt: str):
            class _Response:
                pass

            r = _Response()
            r.content = self._canned_response
            return r

    sample_citations = [
        Citation(
            source="leave.pdf", page=0, page_label="1", chunk_index=0,
            snippet="Full-time employees receive 25 working days of annual leave.",
        )
    ]

    # Case 1: no citations at all.
    answer, cites = enforce_citations(
        FakeLLM("SUPPORTED\nirrelevant, won't be called"), "some query", "some answer", []
    )
    print("No citations case:")
    print(f"  answer: {answer[:60]}...")
    print(f"  citations: {cites}")

    # Case 2: citations exist, verifier says supported.
    answer, cites = enforce_citations(
        FakeLLM("SUPPORTED\nThe answer matches the excerpt exactly."),
        "How many leave days?",
        "Full-time employees receive 25 working days of annual leave.",
        sample_citations,
    )
    print("\nSupported case:")
    print(f"  answer: {answer}")
    print(f"  citations: {len(cites)}")

    # Case 3: citations exist, verifier says NOT supported.
    answer, cites = enforce_citations(
        FakeLLM("NOT_SUPPORTED\nThe answer claims 30 days, excerpt says 25."),
        "How many leave days?",
        "Employees receive 30 working days of annual leave.",  # deliberately wrong
        sample_citations,
    )
    print("\nUnsupported case:")
    print(f"  answer: {answer[:60]}...")
    print(f"  citations returned: {len(cites)} (should still be 1, for transparency)")