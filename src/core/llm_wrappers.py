"""
Core: LLM wrapper, structured logging, and multi-provider fallback chain.

Three things live here:

1. AccurateChatLiteLLM - fixes a real, previously-hit bug: LangChain
   falls back to a generic GPT-2 tokenizer for token counting when a
   chat model doesn't know how to count its own tokens (this is exactly
   what caused the "UserWarning: Using fallback GPT-2 tokenizer" warning
   seen earlier in this project, and matters because
   RetrievalQAWithSourcesChain's reduce_k_below_max_tokens mechanism
   calls get_num_tokens() to decide how many retrieved chunks fit the
   prompt). LiteLLM's token_counter() knows the correct tokenizer per
   model, across providers, so this override makes chunk-trimming
   accurate for whichever model is actually in use.

2. LLMLoggingHandler - a lightweight LangChain callback that logs which
   model handled each call, latency, and token usage. Attached to every
   model in the fallback chain (not just the primary), so when a
   fallback fires, the log sequence itself shows it: an on_llm_error for
   one model immediately followed by an on_llm_start for the next one is
   self-evidently a fallback event, with no separate "fallback detection"
   logic needed.

3. build_llm_with_fallback() - constructs the actual fallback chain from
   core/config.py's three model configs (Groq/Qwen primary, HuggingFace
   Gemma secondary, HuggingFace Mistral tertiary) using LangChain's
   built-in Runnable.with_fallbacks(), rather than LiteLLM's own
   Router/fallbacks mechanism (which lives outside the LangChain
   integration layer and isn't what ChatLiteLLM exposes). The result is
   a drop-in replacement for a single ChatLiteLLM instance everywhere
   else in the codebase — confirmed compatible with
   load_qa_with_sources_chain() and get_num_tokens() delegation before
   adopting this approach.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableWithFallbacks
from langchain_litellm import ChatLiteLLM
from litellm import token_counter

from core.config import settings

os.environ['LITELLM_LOG'] = 'DEBUG'
logger = logging.getLogger(__name__)

# Maps LangChain's internal message type strings to the role names
# LiteLLM/OpenAI-style chat APIs expect. "human" and "ai" don't match
# "user"/"assistant" directly - anything not listed here passes through
# unchanged (covers "system", "function", "tool", etc., which already
# match).
_MESSAGE_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
}


class AccurateChatLiteLLM(ChatLiteLLM):
    """ChatLiteLLM with model-accurate token counting instead of the
    generic GPT-2 fallback LangChain otherwise uses."""

    def get_num_tokens(self, text: str) -> int:
        return token_counter(model=self.model, text=text)

    def get_num_tokens_from_messages(
        self, messages: List[BaseMessage], tools: Optional[Sequence[Any]] = None
    ) -> int:
        litellm_messages = [
            {
                "role": _MESSAGE_TYPE_TO_ROLE.get(m.type, m.type),
                "content": m.content,
            }
            for m in messages
        ]
        return token_counter(model=self.model, messages=litellm_messages)


class LLMLoggingHandler(BaseCallbackHandler):
    """
    Lightweight LangChain callback: logs which model handled each call,
    latency, and token usage (when the provider reports it). Deliberately
    minimal — this is observability for "what happened", not a full
    tracing/metrics system.
    """

    def __init__(self):
        self._start_times: Dict[UUID, float] = {}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, **kwargs) -> None:
        self._start_times[run_id] = time.monotonic()
        model = kwargs.get("invocation_params", {}).get("model", "unknown")
        logger.info("LLM call starting: model=%s", model)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs) -> None:
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        usage = (response.llm_output or {}).get("token_usage", {})
        logger.info(
            "LLM call finished: elapsed=%.2fs tokens=%s",
            elapsed,
            usage or "unavailable",
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        logger.warning(
            "LLM call FAILED after %.2fs: %s: %s — falling back if another "
            "model is configured",
            elapsed,
            type(error).__name__,
            error,
        )


def _build_model(model: str, provider: str, model_kwargs: Optional[Dict[str, Any]] = None) -> AccurateChatLiteLLM:
    return AccurateChatLiteLLM(
        model=model,
        custom_llm_provider=provider,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        callbacks=[LLMLoggingHandler()],
        model_kwargs=model_kwargs or {},
    )


def build_llm_with_fallback() -> RunnableWithFallbacks[PromptValue | str | Sequence[Any], AIMessage]:
    """
    Builds the primary model (Groq/Qwen) chained with fallbacks
    (HuggingFace Gemma, then HuggingFace Mistral, both via
    featherless-ai), using LangChain's Runnable.with_fallbacks().

    Order matters: if the primary raises, LangChain tries the first
    fallback; if that also raises, it tries the second; if all three
    fail, the last exception propagates. Every model in the chain has
    its own LLMLoggingHandler attached, so the log sequence shows
    exactly which model(s) were tried and in what order.
    """
    primary = _build_model(
        settings.llm_groq_qwen_model,
        settings.llm_groq_provider,
        # qwen3 models reason internally by default and, without this,
        # put that raw reasoning inline as literal <think>...</think>
        # text in the answer. Groq-specific — not applicable to the
        # HuggingFace fallback models below, which aren't reasoning
        # models and don't accept this parameter.
        model_kwargs={"reasoning_format": "hidden"},
    )
    fallback_1 = _build_model(
        settings.llm_huggingface_google_model, settings.llm_huggingface_google_provider
    )
    fallback_2 = _build_model(
        settings.llm_huggingface_mistral_model, settings.llm_huggingface_mistral_provider
    )

    return primary.with_fallbacks([fallback_1, fallback_2])


if __name__ == "__main__":
    # Manual check: confirms the fallback chain builds correctly and that
    # get_num_tokens() delegates through to the primary's accurate
    # implementation, without making any real API call.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm = build_llm_with_fallback()
    print("Fallback chain built:", type(llm).__name__)

    token_count = llm.get_num_tokens("How many annual leave days do I get?")
    print(f"get_num_tokens() correctly delegates to primary model: {token_count} tokens")