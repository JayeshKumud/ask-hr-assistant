import streamlit as st

from askhr.service.policy_qa_service import PolicyQAService, get_policy_qa_service


@st.cache_resource
def _cached_service() -> PolicyQAService:
    return get_policy_qa_service()


st.title("Company Policy Assistant")

service = _cached_service()

placeholder = st.empty()

ingest_button = st.sidebar.button("Index Policy Documents")
if ingest_button:
    for status in service.reindex():
        placeholder.text(status)

query = placeholder.text_input("Ask about leave or visa policy")
if query:
    try:
        result = service.ask(query)
        st.header("Answer:")
        st.write(result.answer)

        if result.has_citations():
            st.subheader("Citations:")
            for i, citation in enumerate(result.citations, start=1):
                with st.expander(f"{i}. {citation.display_label()}"):
                    st.write(citation.snippet)
    except RuntimeError:
        placeholder.text("You must index the policy documents first")
