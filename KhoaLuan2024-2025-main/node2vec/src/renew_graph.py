import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial import distance
import networkx as nx

dim = 100
model = KeyedVectors.load("emb/Noun.emb.wv")
dim = model.vector_size
def getVectors(u1):
    u1 = u1.replace("-", "_")
    vs = np.zeros(dim)  # Initialize a zero vector
    found = False  # Track if any valid word is found

    if u1 in model.key_to_index:
        return model[u1].copy()  # Return early if found

    # If the whole word is not in the model, try its parts
    for i in u1.split("_"):
        if i in model.key_to_index:
            if not found:
                vs = model[i].copy()
                found = True
            else:
                vs += model[i].copy()
    
    return vs if found else np.zeros(dim)  # Ensure valid output

G = nx.Graph()
with open("graph/Noun_Graph.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:  
          word1, word2, k = parts[0], parts[1], float(parts[2])
          if word1 !=word2:  
            v1, v2 = getVectors(word1), getVectors(word2)

            if np.any(v1) and np.any(v2):
                dist = 0.5 * distance.cosine(v1, v2) + 0.5 * np.linalg.norm(v1 - v2)
            else:
                continue
            
            G.add_edge(word1, word2, weight=dist)

graph_txt = "graph/New_Graph.txt"
with open(graph_txt, "w", encoding="utf-8") as f:
    for u, v, data in G.edges(data=True):
        f.write(f"{u} {v} {data['weight']:.2f}\n")
