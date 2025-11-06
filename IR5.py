
# Simple PageRank using BeautifulSoup + NetworkX

import requests
from bs4 import BeautifulSoup
import networkx as nx

# Easy & interlinked demo pages (they link to each other)
pages = [
    "https://www.iana.org/domains/reserved",
    "https://www.icann.org/"
]

G = nx.DiGraph()

# Build link graph (short + clear)
for url in pages:   
    try:
        soup = BeautifulSoup(requests.get(url, timeout=5).text, "html.parser")
        for a in soup.find_all('a', href=True):
            link = a['href']
            if any(p in link for p in pages):
                G.add_edge(url, [p for p in pages if p in link][0])
    except:
        pass

# Compute PageRank
ranks = nx.pagerank(G, alpha=0.85)

print("\nPageRank Scores:")
for p, r in ranks.items():
    print(f"{p}: {r:.4f}")


