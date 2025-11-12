import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from dotenv import load_dotenv
from app.pipeline import AutoResearcher

load_dotenv()

st.set_page_config(page_title="Auto-Researcher", layout="wide")
st.title("🔎 Agentic RAG — Auto-Researcher")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.text_input("LLM model", value=os.getenv("LLM_MODEL", "llama3:8b"))
    max_sources = st.slider("Max sources", 3, 15, int(os.getenv("MAX_SOURCES", 8)))
    st.caption("Requires: Ollama running locally and Tavily API key in .env")

# Query input
query = st.text_input(
    "Enter your research query",
    value="Compare RAG vs fine-tuning for domain adaptation"
)

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

# Run button
if st.button("Run Auto-Research"):
    with st.spinner("Thinking…"):
        ar = AutoResearcher()
        ar.model = model
        ar.max_sources = max_sources
        st.session_state.result = ar.run(query)

# Show results if available
if st.session_state.result:
    result = st.session_state.result

    st.subheader("📚 Sources")
    for s in result["sources"]:
        st.markdown(f"- [{s['title']}]({s['url']})")

    st.subheader("🧾 Per-source Summaries")
    for i, sm in enumerate(result["summaries"], 1):
        with st.expander(f"Source {i}: {sm.get('title','(untitled)')}"):
            st.json(sm)

    st.subheader("🧠 Synthesis")
    st.write(result["synthesis"])

    st.subheader("🧪 Critic Review")
    st.json(result["review"])

    st.subheader("📄 Final Report")
    st.download_button(
        label="Download Report (Markdown)",
        data=result["report_md"],
        file_name="auto_research_report.md",
        mime="text/markdown",
    )
    st.markdown(result["report_md"])  # preview
else:
    st.info("Enter a query and click **Run Auto-Research** to begin.")
