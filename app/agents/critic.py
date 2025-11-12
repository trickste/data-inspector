import json
from typing import List, Dict
from langchain_community.llms import Ollama


class CriticAgent:
    def __init__(self, system_prompt: str, model: str = "llama3:8b", temperature: float = 0.0):
        self.llm = Ollama(model=model, temperature=temperature)
        self.system = system_prompt


    def review(self, query: str, synthesis: str, summaries: List[Dict]) -> Dict:
        ctx = "\n".join([f"SRC: {s.get('title','')} — {s.get('url','')}; limits={s.get('limitations', [])}" for s in summaries])
        prompt = f"System:{self.system}\n\nUser: Query: {query}\nSYNTHESIS:\n{synthesis}\n\nSOURCES:\n{ctx}\n\nReturn JSON review."
        out = self.llm.invoke(prompt)
        try:
            return json.loads(out)
        except Exception:
            return {"overall_risk": "unknown", "raw": out}