import os
from typing import List, Dict
from tavily import TavilyClient
import requests


class WebRetriever:
    def __init__(self, api_key: str, max_sources: int = 8):
        self.client = TavilyClient(api_key=api_key)
        self.max_sources = max_sources


    def search_urls(self, query: str) -> List[Dict]:
        res = self.client.search(query=query, max_results=self.max_sources)
        # tavily returns list with {url, title, content?}
        return res.get("results", [])[: self.max_sources]


    def fetch_text(self, url: str) -> str:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception:
            return ""