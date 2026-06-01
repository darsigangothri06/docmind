import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="DocMind", page_icon="\U0001f4c4", layout="wide")

# --- Sidebar: Settings & Collection Management ---
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["gemini", "openai"])
    api_key = st.text_input("API Key", type="password", help="Enter your LLM API key")
    model = st.text_input(
        "Model (optional)",
        placeholder="gemini-2.5-flash" if provider == "gemini" else "gpt-4o-mini",
    )

    st.divider()
    st.header("Collections")

    # List existing collections
    collections = []
    try:
        resp = requests.get(f"{API_BASE}/collections", params={"provider": provider, "api_key": api_key})
        if resp.status_code == 200:
            collections = resp.json()
    except requests.ConnectionError:
        st.warning("API server not running. Start with: `uvicorn src.api.main:app --reload --port 8000`")

    collection_names = [c["name"] for c in collections]

    if collection_names:
        selected_collection = st.selectbox("Select Collection", collection_names)
        selected_info = next((c for c in collections if c["name"] == selected_collection), None)
        if selected_info:
            st.caption(f"Documents: {selected_info['document_count']}")
    else:
        selected_collection = None
        st.info("No collections yet. Upload documents to create one.")

    st.divider()
    st.subheader("Create / Upload")
    new_collection_name = st.text_input("Collection Name", placeholder="my-docs")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "md", "txt"],
    )

    if st.button("Upload & Index", disabled=not (api_key and new_collection_name and uploaded_files)):
        with st.spinner("Uploading and indexing documents..."):
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            data = {"provider": provider, "api_key": api_key}
            try:
                resp = requests.post(
                    f"{API_BASE}/collections/{new_collection_name}/upload",
                    files=files,
                    data=data,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"Indexed {result['documents_loaded']} docs "
                        f"({result['chunks_created']} chunks)"
                    )
                    st.rerun()
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.error("Cannot connect to API server.")

    if selected_collection and st.button("Delete Collection", type="secondary"):
        try:
            resp = requests.delete(f"{API_BASE}/collections/{selected_collection}")
            if resp.status_code == 200:
                st.success(f"Deleted '{selected_collection}'")
                st.rerun()
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")
        except requests.ConnectionError:
            st.error("Cannot connect to API server.")

# --- Main Area: Tabs ---
tab_chat, tab_eval = st.tabs(["Chat", "Evaluate"])

with tab_chat:
    st.title("DocMind")
    st.caption("Chat with your documents.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        meta = src.get("metadata", {})
                        st.markdown(
                            f"**{i}.** {meta.get('source', 'Unknown')} "
                            f"(Page {meta.get('page', 'N/A')})"
                        )
                        st.caption(src.get("content", "")[:150])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not api_key:
            st.error("Please enter your API key in the sidebar.")
        elif not selected_collection:
            st.error("Please select or create a collection first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/collections/{selected_collection}/query",
                            json={
                                "question": prompt,
                                "provider": provider,
                                "api_key": api_key,
                                "model": model,
                            },
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.markdown(data["answer"])
                            if data["sources"]:
                                with st.expander("Sources"):
                                    for i, src in enumerate(data["sources"], 1):
                                        meta = src.get("metadata", {})
                                        st.markdown(
                                            f"**{i}.** {meta.get('source', 'Unknown')} "
                                            f"(Page {meta.get('page', 'N/A')})"
                                        )
                                        st.caption(src.get("content", "")[:150])
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data["sources"],
                            })
                        else:
                            error_msg = resp.json().get("detail", "Unknown error")
                            st.error(f"Error: {error_msg}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API server.")

with tab_eval:
    st.title("Evaluation")
    st.caption("Measure RAG pipeline quality with test datasets.")

    eval_collection = st.selectbox(
        "Collection to evaluate",
        collection_names if collection_names else ["(none)"],
        key="eval_col",
    )
    dataset_path = st.text_input(
        "Test dataset path",
        value="./data/eval/test_dataset.json",
    )

    if st.button("Run Evaluation", disabled=not (api_key and eval_collection != "(none)")):
        with st.spinner("Running evaluation (this may take a minute)..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/evaluate",
                    json={
                        "collection": eval_collection,
                        "dataset_path": dataset_path,
                        "provider": provider,
                        "api_key": api_key,
                        "model": model,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()

                    st.subheader("Average Scores")
                    cols = st.columns(len(data["average_scores"]))
                    for i, (metric, score) in enumerate(data["average_scores"].items()):
                        cols[i].metric(metric.replace("_", " ").title(), f"{score:.3f}")

                    st.subheader("Per-Question Results")
                    for r in data["results"]:
                        with st.expander(f"Q: {r['question'][:80]}..."):
                            st.markdown(f"**Answer:** {r['answer']}")
                            st.json(r["scores"])
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                st.error("Cannot connect to API server.")
