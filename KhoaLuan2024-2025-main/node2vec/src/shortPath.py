import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

# Load pre-trained embeddings
embedding_path = "emb/Noun.emb.wv"
model = KeyedVectors.load(embedding_path)

# Load graph từ file có sẵn (đã có trọng số là distance)
G = nx.Graph()
file_path = "graph/New_Graph.txt"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            word1, word2, dist = parts[0], parts[1], float(parts[2])  # Distance đã được tính sẵn
            if word1 in model.key_to_index and word2 in model.key_to_index:
                    G.add_edge(word1, word2, weight=dist)  



from scipy.spatial.distance import cosine


def heuristic(node1, node2):
    if node1 in model.key_to_index and node2 in model.key_to_index:
        v1, v2 = model[node1], model[node2]
        cos_dist = 1 - np.dot(v1, v2)
        euc_dist = np.linalg.norm(v1 - v2)
        return 0.5 * cos_dist + 0.5 * euc_dist  # Kết hợp cả hai metric
    return float("inf")




# 🔍 **Hàm tìm khoảng cách gần nhất từ w1 đến w2 bằng A***
def find_similarity(w1, w2):
    if w1 in G.nodes() and w2 in G.nodes() :
        try:
            cost = nx.astar_path_length(G, w1, w2, heuristic=heuristic, weight="weight")
            #cost = nx.dijkstra_path_length(G, w1, w2, weight="weight")
            #sim_score = np.exp(-cost)  # Chuyển distance về khoảng [0,1]
            return cost
        except nx.NetworkXNoPath:
            print(f"❌ Không có đường đi từ {w1} đến {w2}.")
            return None
    else:
        print(f"⚠️ Bỏ qua {w1} - {w2}: Không có trong đồ thị.")
        return None

# 📌 **Tính hệ số tương quan với Visim-400**
fSimlex = "word/Visim-400.txt"
word_pairs = []
human_scores = []  # Dữ liệu con người
model_scores = []  # Dữ liệu từ thuật toán
w1='nao_núng'
w2='kiên_định'

with open(fSimlex, "r", encoding="utf-8") as f:
    for line in f.readlines()[1:]:  # Bỏ qua tiêu đề
        parts = line.strip().split()
        if len(parts) >= 4:
            w1, w2, pos, sim1 = parts[0], parts[1], parts[2], float(parts[3])
            word_pairs.append((w1, w2, pos, sim1))

# 🔄 Chạy trên từng cặp từ trong Visim-400
for w1, w2, pos, sim1 in word_pairs:
    sim_score = find_similarity(w1, w2)
    if sim_score is not None :
        human_scores.append(sim1 / 6)  # Chuẩn hóa về [0,1]
        model_scores.append(sim_score)
Max=max(model_scores)
Min=min(model_scores)
for i in range(len(model_scores)):
    model_scores[i]=1-(model_scores[i]-Min)/(Max-Min)
# 📊 **Tính hệ số tương quan Pearson**
if len(model_scores) > 1:
    pearson_corr, p_value = pearsonr(model_scores, human_scores)
    print(f"\n📊 Hệ số tương quan Pearson: {pearson_corr:.4f}")
    print(f"📌 P-value: {p_value:.4f}")
else:
    print("❌ Không đủ dữ liệu để tính hệ số tương quan.")
