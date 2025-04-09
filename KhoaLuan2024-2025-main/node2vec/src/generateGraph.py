import numpy as np
from scipy import spatial
import networkx as nx
from gensim.models import KeyedVectors
import gensim.downloader as api
dim = 150  

# Hàm lấy vector từ model
def getVectors(u1):
    u1 = u1.replace("-", "_")
    vs = np.repeat(0, dim)
    mk = 1
    #if u1 in model.key_to_index:
    if u1 in model.keys():
        vs = model[u1]
    else:
        uu = u1.split("_")
        for i in uu:
            #if u1 in model.key_to_index:
            if i in model.keys():
                if mk == 1:
                    vs = model[i]
                    mk = 0  
                else:
                    vs = vs + model[i]
    return np.array(vs)
# Load model từ file
'''
try:
    model = KeyedVectors.load_word2vec_format("word/fastText_4GB.vec", binary=True)
    dim = model.vector_size
    print("✅ Mô hình Word2Vec đã được tải thành công.\n")
except UnicodeDecodeError:
    print("⚠️ File có thể không phải nhị phân, thử lại với binary=False...")
    try:
        model = KeyedVectors.load_word2vec_format("word/fastText_4GB.vec", binary=False)
        dim = model.vector_size
        print("✅ Mô hình Word2Vec đã được tải thành công (dạng văn bản).\n")
    except Exception as e:
        print(f"❌ Không thể tải mô hình: {e}")
        exit()
'''
fWord2vec = 'word/W2V_150.txt'
model = {}
try:
    with open(fWord2vec, 'r', encoding="utf-8-sig") as f:
        f.readline()  # Bỏ qua dòng đầu tiên

        words_list1 = []  # Danh sách chứa các từ đã đọc

        for line in f:
            tem = line.split()

            if len(tem) != dim+1:  # 1 từ + 100 giá trị vector
                continue  

            word = tem[0]  # Lấy từ
            try:
                vector = np.array(list(map(float, tem[1:])))  # Chuyển đổi vector
                model[word] = vector  # Thêm vào dictionary
                words_list1.append(word)  # Lưu từ vào danh sách
            except ValueError:
                print(f"⚠️ Bỏ qua từ '{word}' do lỗi chuyển đổi số.")

    print(f"\n✅ Đọc xong {len(model)} từ với vector kích thước {len(vector)}.")

    # In ra danh sách các từ đã đọc
    print("\n📌 Các từ đã đọc được:")
    print(len(words_list1))  # In 50 từ đầu tiên để kiểm tra

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{fWord2vec}'. Kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"❌ Lỗi không xác định: {e}")
word_txt = "word/VSimlex.txt"
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
edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] <0.2]
#edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] >0.9]
G.remove_edges_from(edges_to_remove)
file_path = 'word/Verbs_dn.txt'  # Thay bằng đường dẫn tới file của bạn
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
                if w1!=w2:
                    if not G.has_edge(w1,w2):
                        G.add_edge(w1,w2,weight=0.99)
                    else:
                        data=G.get_edge_data(w1,w2)
                        if(data['weight']<0.99):
                            G[w1][w2]['weight']=0.99
def read_word(file_path):
    word_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if '-' in line:
                    w1, w2 = line.strip().split('-')
                    w1 = w1.strip().replace(' ', '_')
                    w2 = w2.strip().replace(' ', '_')
                    if w1!=w2:
                        word_pairs.append((w1, w2))
    except FileNotFoundError:
        print(f"Không tìm thấy file tại đường dẫn: {file_path}")
    
    return word_pairs


word_pairs=read_word('word/dongnghia.txt')
def read_word_trainghia(file_path):
    word_pairs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()  # Tách bằng dấu cách (space)
                if len(parts) == 2:  # Chỉ xử lý dòng có đúng 2 từ
                    w1, w2 = parts
                    w1 = w1.replace(' ', '_')  # Đảm bảo định dạng từ
                    w2 = w2.replace(' ', '_')
                    word_pairs.append((w1, w2))
                else:
                    print(f"⚠️ Cảnh báo: Bỏ qua dòng không hợp lệ -> {line.strip()}")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file tại: {file_path}")
    
    return word_pairs

if word_pairs:
    print("Các cặp từ đã đọc từ file:")
    for w1, w2 in word_pairs:
        if w1!=w2 and not G.has_edge(w1,w2):
            G.add_edge(w1,w2,weight=0.99)
else:
    print("Không có cặp từ nào được đọc từ file.")
word_pairs2=read_word_trainghia('word/trainghia.txt')
if word_pairs2:
    print("Các cặp từ đã đọc từ file:")
    for w1, w2 in word_pairs2:
        if w1!=w2 and not G.has_edge(w1,w2):
            G.add_edge(w1,w2,weight=0.01)
else:
    print("Không có cặp từ nào được đọc từ file.")
graph_txt = "graph/Noun_Graph.txt"
with open(graph_txt, "w", encoding="utf-8") as f:
    for u, v, data in G.edges(data=True):
        f.write(f"{u} {v} {data['weight']:.2f}\n")
