import numpy as np
import networkx as nx
from scipy import spatial
import gensim.downloader as api

# 📥 Tải mô hình Word2Vec từ Google News (dung lượng ~1.5GB)
print("🔄 Đang tải mô hình Word2Vec của Google News. Vui lòng chờ...")
model = api.load("word2vec-google-news-300")
dim=model.vector_size
print("✅ Mô hình đã được tải thành công!\n")
def getVectors(u1):
    u1 = u1.replace("-", "_")
    vs = np.repeat(0, dim)
    mk = 1
    if u1 in model.key_to_index:
    #if u1 in model.keys():
        vs = model[u1]
    else:
        uu = u1.split("_")
        for i in uu:
            if u1 in model.key_to_index:
            #if i in model.keys():
                if mk == 1:
                    vs = model[i]
                    mk = 0  
                else:
                    vs = vs + model[i]
    return np.array(vs)
word_txt='word/Simlex-999-english.txt'
word_list=[]
with open(word_txt, "r", encoding="utf-8") as file:
    #next(file)  
    for line in file:
        parts = line.strip().split("\t")  
        if len(parts) >= 2:
            word1 = parts[0].replace(' ', '_')  
            word2 = parts[1].replace(' ', '_')
            word_list.append(word1)  
            word_list.append(word2)
word_list = list(set(word_list))

G = nx.Graph()

for i in range(len(word_list)-1):
    w1 = word_list[i]
    v1 = getVectors(w1)  
    
    for j in range(i + 1, len(word_list)):
        w2 = word_list[j]
        v2 = getVectors(w2)  
        
        if np.all(v1 == 0) or np.all(v2 == 0):  
            continue
        k =1.0-spatial.distance.cosine(v1, v2) 
        #k =spatial.distance.cosine(v1, v2) 
        G.add_edge(w1, w2, weight=k)  
edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] <=0.35]
#edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] >0.9]
G.remove_edges_from(edges_to_remove)
file_path = 'word/dongnghia_english.txt'  # Thay bằng đường dẫn tới file của bạn
# Mở và đọc dữ liệu từ file
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
for line_number, line in enumerate(lines, start=1):
    words = [item.strip().replace(' ', '_') for item in line.strip().split(',')]
    if(len(words)>=2):
        for i in range(0,len(words)-1):
            w1=words[i]
            for j in range(i+1,len(words)):
                w2=words[j]
                if w1==w2:
                    continue
                if not G.has_edge(w1,w2):
                    G.add_edge(w1,w2,weight=0.99)
                else:
                    data=G.get_edge_data(w1,w2)
                    if(data['weight']<0.99):
                        G[w1][w2]['weight']=0.99


word_list = list(set(word_list))
graph_txt = "graph/Noun_Graph.txt"
with open(graph_txt, "w", encoding="utf-8") as f:
    for u, v, data in G.edges(data=True):
        f.write(f"{u} {v} {data['weight']:.2f}\n")
