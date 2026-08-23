# flake8: noqa
"""
Search: prompt templates used by the QA chain.

Two templates work together here, and they're both consumed by
search/qa_chain.py's build_qa_chain():

- EXAMPLE_PROMPT formats EACH individual retrieved chunk before it goes
  into the main prompt. Its input_variables ("page_content", "source")
  must match keys that exist on every retrieved Document (page_content is
  built in; "source" comes from the metadata ingestion/document_loader.py
  sets on each chunk). If you add a variable here that isn't actually
  present in a chunk's metadata, LangChain raises a KeyError at
  chain-invoke time, not at prompt-build time — so this list is a
  contract with whatever loader is currently in use.

- PROMPT is the outer template: the actual instructions + persona sent to
  the LLM, with `{summaries}` (all the EXAMPLE_PROMPT-formatted chunks,
  joined together) and `{question}` slotted in. It's built by prepending
  a one-line persona description onto LangChain's built-in
  `stuff_prompt.template`, which is what defines the
  "FINAL ANSWER: ... \nSOURCES: ..." output format the chain expects to
  parse the model's response into `answer` vs its self-reported sources.
  Note: as of Phase 2, we don't actually trust that self-reported SOURCES
  line for citations anymore — search/citations.py builds citations
  directly from the retrieved chunks' metadata instead, which is more
  reliable. The persona line below still matters for tone/behavior, but
  the SOURCES half of the format is now mostly vestigial.

Phase 6 will move these out of Python source entirely into a versioned
config file, since prompts are effectively part of the system's behavior
and deserve the same change tracking as code.
"""
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.stuff_prompt import template

updated_template = (
    "You are a helpful assistant for NexaCore's employee leave and visa policy. "
    "Answer only from the provided policy excerpts."
) + template

PROMPT = PromptTemplate(template=updated_template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)