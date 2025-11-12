from typing import List, Dict
from langchain_community.llms import Ollama


class SynthesisAgent:
    def __init__(self, system_prompt: str, model: str = "llama3:8b", temperature: float = 0.2):
        self.llm = Ollama(model=model, temperature=temperature)
        self.system = system_prompt


    def synthesize(self, query: str, summaries: List[Dict]) -> str:
        bulletized = "\n".join([f"- {s.get('title','(untitled)')}: {', '.join(s.get('key_points', [])[:5])}" for s in summaries])
        prompt = f"System:{self.system}\n\nUser: Query: {query}\nPer-source bullets:\n{bulletized}\n\nWrite the synthesis."
        return self.llm.invoke(prompt)