import json
from typing import Dict
from langchain_community.llms import Ollama


class SummarizerAgent:
    def __init__(self, system_prompt: str, model: str = "llama3:8b", temperature: float = 0.2):
        self.llm = Ollama(model=model, temperature=temperature)
        self.system = system_prompt


    def summarize(self, query: str, title: str, url: str, content: str) -> Dict:
        prompt = f"System:{self.system}\n\nUser: Query: {query}\nTitle: {title}\nURL: {url}\nCONTENT:\n{content[:4000]}\n\nReturn JSON."
        out = self.llm.invoke(prompt)
        try:
            return json.loads(out)
        except Exception:
            return {"title": title, "url": url, "key_points": [], "methods": "", "evidence": [], "limitations": ["parse_error"], "raw": out}