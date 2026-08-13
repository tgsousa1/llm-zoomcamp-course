import streamlit as st

from search import(load_docs,load_embedding_model,load_embeddings,DOCUMENTS_FILE,EMBEDDINGS_FILE)

from rag import rag
from monitoring import feedback_not_useful, feedback_useful

st.set_page_config(page_title="Movie Assistant", page_icon="🎬")

@st.cache_resource
def load_resources():
    docs=load_docs(DOCUMENTS_FILE)
    model=load_embedding_model()
    embeddings=load_embeddings(EMBEDDINGS_FILE)

    return docs, model, embeddings

docs, model, embeddings = load_resources()

if "answer" not in st.session_state:
    st.session_state.answer = None

if "results" not in st.session_state:
    st.session_state.results = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

st.title("🎬 Movie Assistant")
st.caption("Ask questions about movies released between 2016 and 2026.")

query = st.text_input(
"What would you like to know?",
placeholder="e.g. Who directed Oppenheimer?"
)

search = st.button("🔍 Search", type="primary")

if search and query.strip():

    with st.spinner("Searching..."):

        answer, results = rag(
            query=query,
            model=model,
            embeddings=embeddings,
            documents=docs,
        )

    st.session_state.answer = answer
    st.session_state.results = results
    st.session_state.feedback_given = False

if st.session_state.answer:

    st.subheader("Answer")

    st.write(st.session_state.answer)

    if not st.session_state.feedback_given:

        st.write("Was this answer useful?")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🟢 Y"):
                feedback_useful.add(1)
                st.session_state.feedback_given = True
                st.rerun()

        with col2:
            if st.button("🔴 N"):
                feedback_not_useful.add(1)
                st.session_state.feedback_given = True
                st.rerun()

    else:
        st.caption("Thanks for your feedback!")


    with st.expander("Retrieved movies"):
        for movie in st.session_state.results:
            st.write(movie["title"])