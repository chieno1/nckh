import numpy as np
from scipy.spatial import distance
import networkx as nx
from itertools import combinations

dim = 150  

# Hàm lấy vector từ model
def getVectors(u1):
    u1 = u1.replace("-", "_")
    vs = np.zeros(dim)  # Khởi tạo vector toàn số 0
    mk = 1
    
    if u1 in model:
        return model[u1]  
    
    # Nếu không tìm thấy từ, thử tìm từng phần của nó
    uu = u1.split("_")
    for i in uu:
        if i in model:
            vs = model[i] if mk else vs + model[i]
            mk = 0  
    return vs

# Tải mô hình từ file
fWord2vec = 'word/W2V_150.txt'
model = {}

try:
    with open(fWord2vec, 'r', encoding="utf-8-sig") as f:
        f.readline()  # Bỏ qua dòng đầu tiên (header)
        words_list = []

        for line in f:
            tem = line.split()
            if len(tem) != dim + 1:
                continue  

            word = tem[0]  
            try:
                vector = np.array(list(map(float, tem[1:])))
                model[word] = vector  
                words_list.append(word)  
            except ValueError:
                print(f"⚠️ Bỏ qua từ '{word}' do lỗi chuyển đổi số.")

    print(f"\n✅ Đọc xong {len(model)} từ với vector kích thước {dim}.")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{fWord2vec}'. Kiểm tra lại đường dẫn.")
    exit()

# Đọc danh sách từ từ các file khác nhau
word_list = set()
G = nx.Graph()

# Đọc file VSimlex.txt
word_txt = "word/VSimlex.txt"
try:
    with open(word_txt, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  
            if len(parts) >= 2:
                word1, word2 = parts[0].replace(' ', '_'), parts[1].replace(' ', '_')
                word_list.update([word1, word2])
except FileNotFoundError:
    print(f"❌ Không tìm thấy file: {word_txt}")
'''
# Đọc file Verbs_dn.txt
file_path = 'word/Verbs_dn.txt'
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = [item.strip().replace(' ', '_') for item in line.strip().split(',')]
            if len(words) >= 2:
                for i, w1 in enumerate(words[:-1]):
                    word_list.add(w1)
                    for w2 in words[i + 1:]:
                        word_list.add(w2)
                        G.add_edge(w1, w2, weight=0.99)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file: {file_path}")
'''
# Đọc file dongnghia.txt
def read_file(file_path):
    pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if '-' in line:
                    w1, w2 = map(lambda x: x.strip().replace(' ', '_'), line.strip().split('-'))
                    if w1 and w2 and w1 != w2:
                        pairs.append((w1, w2))
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {file_path}")
    return pairs

word_pairs = read_file('word/dongnghia.txt')

if word_pairs:
    for w1, w2 in word_pairs:
        word_list.update([w1, w2])
        G.add_edge(w1, w2, weight=0.99)
else:
    print("⚠️ Không có cặp từ nào được đọc từ file.")

# Chuyển word_list từ set về list
word_list = list(word_list)

# Tạo đồ thị dựa trên cosine similarity giữa các vector từ
for w1, w2 in combinations(word_list, 2):
    v1, v2 = getVectors(w1), getVectors(w2)
    if np.all(v1 == 0) or np.all(v2 == 0):  
        continue
    similarity = 1.0 - distance.cosine(v1, v2)
    if not G.has_edge(w1, w2):
        G.add_edge(w1, w2, weight=similarity)

# Xóa các cạnh có trọng số thấp (< 0.2)
edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] < 0.3]
G.remove_edges_from(edges_to_remove)

# Xuất đồ thị ra file
graph_txt = "graph/Noun_Graph.txt"
try:
    with open(graph_txt, "w", encoding="utf-8") as f:
        for u, v, data in G.edges(data=True):
            f.write(f"{u} {v} {data['weight']:.2f}\n")
    print(f"✅ Đồ thị đã được lưu tại {graph_txt}")
except Exception as e:
    print(f"❌ Lỗi khi ghi file: {e}")
