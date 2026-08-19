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


st.title("Real Estate Research Tool")

pipeline = get_pipeline()

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url != ""]
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid url")
    else:
        for status in pipeline.process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = pipeline.generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
    except RuntimeError:
        placeholder.text("You must process urls first")