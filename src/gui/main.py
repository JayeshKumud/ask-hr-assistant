import streamlit as st

from core.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """
    Cached so the SAME RAGPipeline instance (and its vector store) survives
    across Streamlit reruns. Without this, `RAGPipeline()` would run again
    on every button click / text input change, wiping out anything already
    processed.
    """
    return RAGPipeline()


st.title("Company Policy Assistant")

pipeline = get_pipeline()

placeholder = st.empty()

ingest_button = st.sidebar.button("Index Policy Documents")
if ingest_button:
    for status in pipeline.ingest_documents():
        placeholder.text(status)

query = placeholder.text_input("Ask about leave or visa policy")
if query:
    try:
        answer, citations = pipeline.generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if citations:
            st.subheader("Citations:")
            for i, citation in enumerate(citations, start=1):
                # Each citation shows WHERE the answer came from (doc +
                # page) as a visible label, and the actual chunk text
                # inside an expander — so the user can verify the claim
                # without opening the PDF and hunting for the paragraph.
                with st.expander(f"{i}. {citation.display_label()}"):
                    st.write(citation.snippet)
    except RuntimeError:
        placeholder.text("You must index the policy documents first")