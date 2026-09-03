# flake8: noqa
"""
Search: prompt templates used by the QA chain, built from
config/prompts.yaml via core/prompt_registry.py.

As of Phase 6, the actual prompt TEXT lives in config/prompts.yaml, not
here — this module just loads that config and builds the LangChain
PromptTemplate objects the rest of search/ needs. See config/prompts.yaml
for the versioned source of truth and its own comments on each prompt.

Three templates are built here, all consumed elsewhere in search/:

- EXAMPLE_PROMPT formats EACH individual retrieved chunk before it goes
  into the main prompt. Its input_variables ("page_content", "source")
  must match keys that exist on every retrieved Document (page_content is
  built in; "source" comes from the metadata ingestion/document_loader.py
  sets on each chunk). Used by search/qa_chain.py's build_qa_chain().

- PROMPT is the outer template: the actual instructions + persona sent to
  the LLM, with `{summaries}` (all the EXAMPLE_PROMPT-formatted chunks,
  joined together) and `{question}` slotted in. It's built by prepending
  the qa_prompt persona from config/prompts.yaml onto LangChain's built-in
  `stuff_prompt.template`, which defines the
  "FINAL ANSWER: ... \nSOURCES: ..." output format. Used by
  search/qa_chain.py's build_qa_chain().
  Note: we don't actually trust that self-reported SOURCES line for
  citations — search/citations.py builds citations directly from the
  retrieved chunks' metadata instead, which is more reliable. The persona
  still matters for tone/behavior; the SOURCES half of the format is
  mostly vestigial.

- VERIFICATION_PROMPT is used by search/citation_enforcer.py, NOT by
  build_qa_chain() — a separate, focused prompt for the second "is this
  answer actually supported?" LLM call in Phase 5's citation enforcement.
"""
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.stuff_prompt import template

from askhr.core.prompt_registry import get_prompt

_qa_config = get_prompt("qa_prompt")
updated_template = _qa_config["persona"] + template
PROMPT = PromptTemplate(template=updated_template, input_variables=["summaries", "question"])

_example_config = get_prompt("example_prompt")
EXAMPLE_PROMPT = PromptTemplate(
    template=_example_config["template"],
    input_variables=_example_config["input_variables"],
)

_verification_config = get_prompt("verification_prompt")
VERIFICATION_PROMPT = PromptTemplate(
    template=_verification_config["template"],
    input_variables=_verification_config["input_variables"],
)