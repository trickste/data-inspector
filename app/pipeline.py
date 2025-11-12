import os, hashlib
from typing import Dict, List
from dotenv import load_dotenv
from app.agents.retriever import WebRetriever
from app.rag.chunker import prepare_chunks
from app.agents.summarizer import SummarizerAgent
from app.agents.synthesizer import SynthesisAgent
from app.agents.critic import CriticAgent
from app.prompts import SYSTEM_SUMMARIZER, SYSTEM_SYNTHESIZER, SYSTEM_CRITIC, SYSTEM_REPORT
from langchain_community.llms import Ollama


load_dotenv()


class AutoResearcher:
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.model = os.getenv("LLM_MODEL", "llama3:8b")
        self.max_sources = int(os.getenv("MAX_SOURCES", 8))
        self.retriever = WebRetriever(self.tavily_key, self.max_sources)
        self.summarizer = SummarizerAgent(SYSTEM_SUMMARIZER, self.model)
        self.synthesizer = SynthesisAgent(SYSTEM_SYNTHESIZER, self.model)
        self.critic = CriticAgent(SYSTEM_CRITIC, self.model)
        self.report_llm = Ollama(model=self.model, temperature=float(os.getenv("TEMPERATURE", 0.2)))


    def _id(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()


    def run(self, query: str) -> Dict:
        # 1) Retrieve URLs
        results = self.retriever.search_urls(query)
        sources = [{"title": r.get("title","(untitled)"), "url": r.get("url")} for r in results]


        # 2) Fetch & summarize each
        summaries = []
        for s in sources:
            html = self.retriever.fetch_text(s["url"]) or ""
            chunks = prepare_chunks(html)

            # summarize each chunk separately, then merge
            chunk_summaries = []
            for ch in chunks[:5]:  # limit to 5 chunks to stay within token budget
                sm = self.summarizer.summarize(query, s["title"], s["url"], ch)
                chunk_summaries.append(sm)

            # merge top key points
            summary = {
                "title": s["title"],
                "url": s["url"],
                "key_points": sum([sm.get("key_points", []) for sm in chunk_summaries], []),
                "limitations": sum([sm.get("limitations", []) for sm in chunk_summaries], []),
            }

            summaries.append(summary)


        # 3) Synthesize comparative narrative
        synthesis = self.synthesizer.synthesize(query, summaries)


        # 4) Critic review / gap analysis
        review = self.critic.review(query, synthesis, summaries)


        # 5) Final report
        refs = "\n".join([f"- {s.get('title','')} — {s.get('url','')}" for s in summaries])
        report_prompt = (
            f"System:{SYSTEM_REPORT}\n\nUser: Query: {query}\n\nSYNTHESIS:\n{synthesis}\n\nCRITIC REVIEW (JSON):\n{review}\n\nREFERENCES:\n{refs}\n\nWrite the final report in Markdown."
        )
        report_md = self.report_llm.invoke(report_prompt)


        return {
            "query": query,
            "sources": sources,
            "summaries": summaries,
            "synthesis": synthesis,
            "review": review,
            "report_md": report_md,
        }