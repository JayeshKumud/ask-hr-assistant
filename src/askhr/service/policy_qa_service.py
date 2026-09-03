"""
Service: the single, interface-agnostic entry point for "answer an HR
policy question" and "reindex the policy documents".
"""
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, List

from askhr.core.pipeline import RAGPipeline
from askhr.search.citations import Citation


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: List[Citation] = field(default_factory=list)

    def has_citations(self) -> bool:
        return len(self.citations) > 0


class PolicyQAService:
    def __init__(self, pipeline: RAGPipeline = None):
        self._pipeline = pipeline or RAGPipeline()

    def ask(self, query: str) -> AnswerResult:
        answer, citations = self._pipeline.generate_answer(query)
        return AnswerResult(answer=answer, citations=citations)

    def reindex(self) -> Iterator[str]:
        yield from self._pipeline.ingest_documents()


@lru_cache(maxsize=1)
def get_policy_qa_service() -> PolicyQAService:
    return PolicyQAService()
