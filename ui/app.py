import os
import time

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="DocMind", page_icon="D", layout="wide")

st.title("DocMind")
st.caption(
    "RAG-powered document Q&A. Upload documents, create collections, "
    "and ask questions with source citations."
)

st.info(
    "This demo uses an open-source LLM (GPT-OSS 20B via Groq) on a free tier. "
    "Output quality may vary compared to commercial models. "
    "Responses typically take 5 to 15 seconds depending on document size."
)


def format_elapsed(elapsed: float) -> str:
    mins, secs = divmod(int(elapsed), 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def api_call_with_retry(method, url, max_retries=3, **kwargs):
    """Make an API call with retry on 429/rate-limit errors."""
    base_delay = 2
    for attempt in range(max_retries):
        try:
            resp = method(url, timeout=120, **kwargs)
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
            return resp
        except requests.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(base_delay)
                continue
            raise
    return resp


def fetch_collections() -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE}/collections", timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except (requests.ConnectionError, requests.Timeout):
        pass
    return []


collections = fetch_collections()
collection_names = [c["name"] for c in collections]

col_mgmt, col_main = st.columns([1, 2])

with col_mgmt:
    st.subheader("Collections")

    if collection_names:
        selected_collection = st.selectbox("Select Collection", collection_names)
        selected_info = next(
            (c for c in collections if c["name"] == selected_collection), None
        )
        if selected_info:
            st.caption(f"Documents: {selected_info['document_count']}")
    else:
        selected_collection = None
        st.info("No collections yet. Upload documents to create one.")

    st.divider()
    st.subheader("Upload Documents")
    new_collection_name = st.text_input("Collection Name", placeholder="my-docs")
    uploaded_files = st.file_uploader(
        "Select Files",
        accept_multiple_files=True,
        type=["pdf", "md", "txt"],
    )

    if st.button(
        "Upload and Index",
        disabled=not (new_collection_name and uploaded_files),
    ):
        with st.status("Indexing documents...", expanded=True) as status:
            start = time.time()
            st.write("Uploading files to server...")
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
            try:
                st.write("Splitting and embedding documents...")
                resp = api_call_with_retry(
                    requests.post,
                    f"{API_BASE}/collections/{new_collection_name}/upload",
                    files=files,
                )
                elapsed = time.time() - start
                if resp.status_code == 200:
                    result = resp.json()
                    status.update(
                        label=f"Indexed in {format_elapsed(elapsed)}",
                        state="complete",
                    )
                    st.success(
                        f"Indexed {result['documents_loaded']} docs "
                        f"({result['chunks_created']} chunks)"
                    )
                    st.rerun()
                elif resp.status_code == 429:
                    status.update(label="Rate limited", state="error")
                    st.error(
                        "Rate limited by API provider. "
                        "Wait 30 to 60 seconds and try again."
                    )
                else:
                    status.update(label="Failed", state="error")
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except requests.ConnectionError:
                status.update(label="Connection failed", state="error")
                st.error(
                    "Cannot connect to API server. "
                    "The server may be starting up (free tier cold start takes ~30s). "
                    "Please try again in a moment."
                )

    if selected_collection and st.button("Delete Collection", type="secondary"):
        try:
            resp = requests.delete(
                f"{API_BASE}/collections/{selected_collection}", timeout=30
            )
            if resp.status_code == 200:
                st.success(f"Deleted '{selected_collection}'")
                st.rerun()
            else:
                st.error(f"Error: {resp.json().get('detail', resp.text)}")
        except requests.ConnectionError:
            st.error("Cannot connect to API server.")

with col_main:
    tab_chat, tab_eval = st.tabs(["Chat", "Evaluate"])

    with tab_chat:
        st.subheader("Chat with your documents")

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
            if not selected_collection:
                st.error("Please select or create a collection first.")
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.status("Thinking...", expanded=True) as status:
                        start = time.time()
                        timer_placeholder = st.empty()
                        timer_placeholder.caption("Retrieving relevant documents...")

                        try:
                            resp = api_call_with_retry(
                                requests.post,
                                f"{API_BASE}/collections/{selected_collection}/query",
                                json={"question": prompt},
                            )
                            elapsed = time.time() - start
                            timer_placeholder.caption(
                                f"Completed in {format_elapsed(elapsed)}"
                            )

                            if resp.status_code == 200:
                                data = resp.json()
                                status.update(
                                    label=f"Done in {format_elapsed(elapsed)}",
                                    state="complete",
                                )
                                st.markdown(data["answer"])
                                if data["sources"]:
                                    with st.expander("Sources"):
                                        for i, src in enumerate(
                                            data["sources"], 1
                                        ):
                                            meta = src.get("metadata", {})
                                            st.markdown(
                                                f"**{i}.** {meta.get('source', 'Unknown')} "
                                                f"(Page {meta.get('page', 'N/A')})"
                                            )
                                            st.caption(
                                                src.get("content", "")[:150]
                                            )
                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": data["answer"],
                                        "sources": data["sources"],
                                    }
                                )
                            elif resp.status_code == 429:
                                status.update(label="Rate limited", state="error")
                                st.warning(
                                    "Rate limited by API provider. "
                                    "Wait 30 to 60 seconds and try again."
                                )
                            else:
                                status.update(label="Error", state="error")
                                error_msg = resp.json().get(
                                    "detail", "Unknown error"
                                )
                                st.error(f"Error: {error_msg}")
                        except requests.ConnectionError:
                            status.update(
                                label="Connection failed", state="error"
                            )
                            st.error(
                                "Cannot connect to API server. "
                                "The server may be starting up. "
                                "Please try again in a moment."
                            )

    with tab_eval:
        st.subheader("Evaluation")
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

        if st.button(
            "Run Evaluation",
            disabled=eval_collection == "(none)",
        ):
            with st.status(
                "Running evaluation...", expanded=True
            ) as status:
                start = time.time()
                timer_placeholder = st.empty()
                timer_placeholder.caption(
                    "This may take 1 to 3 minutes depending on dataset size..."
                )

                try:
                    resp = api_call_with_retry(
                        requests.post,
                        f"{API_BASE}/evaluate",
                        json={
                            "collection": eval_collection,
                            "dataset_path": dataset_path,
                        },
                    )
                    elapsed = time.time() - start
                    timer_placeholder.caption(
                        f"Completed in {format_elapsed(elapsed)}"
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        status.update(
                            label=f"Done in {format_elapsed(elapsed)}",
                            state="complete",
                        )

                        st.subheader("Average Scores")
                        cols = st.columns(len(data["average_scores"]))
                        for i, (metric, score) in enumerate(
                            data["average_scores"].items()
                        ):
                            cols[i].metric(
                                metric.replace("_", " ").title(),
                                f"{score:.3f}",
                            )

                        st.subheader("Per-Question Results")
                        for r in data["results"]:
                            with st.expander(
                                f"Q: {r['question'][:80]}..."
                            ):
                                st.markdown(f"**Answer:** {r['answer']}")
                                st.json(r["scores"])
                    elif resp.status_code == 429:
                        status.update(label="Rate limited", state="error")
                        st.warning(
                            "Rate limited by API provider. "
                            "Wait 30 to 60 seconds and try again."
                        )
                    else:
                        status.update(label="Error", state="error")
                        st.error(
                            f"Error: {resp.json().get('detail', resp.text)}"
                        )
                except requests.ConnectionError:
                    status.update(label="Connection failed", state="error")
                    st.error(
                        "Cannot connect to API server. "
                        "The server may be starting up. "
                        "Please try again in a moment."
                    )
