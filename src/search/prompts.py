# flake8: noqa
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.qa_with_sources.stuff_prompt import template

updated_template = "You are a helpful assistant for RealEstate research." + template

PROMPT = PromptTemplate(template=updated_template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
