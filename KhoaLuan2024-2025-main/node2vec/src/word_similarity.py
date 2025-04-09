# Chương trình tính word similarity dùng mô hình Word2Vec và in ra kết quả
import numpy as np
from scipy import spatial
from scipy.stats import pearsonr
from gensim.models import KeyedVectors
import gensim.downloader as api

dim=150
try:
    #model = KeyedVectors.load_word2vec_format("word/fastText_4GB.vec", binary=False)
    #model = api.load("word2vec-google-news-300")
    model = KeyedVectors.load("emb/Noun.emb")
    #model = api.load("glove-wiki-gigaword-300")
    #model = KeyedVectors.load_word2vec_format("word/fastText/wiki-news-300d-1M.vec", binary=False)
    dim = model.vector_size  # Tự động lấy chiều vector từ mô hình
    print("Mô hình Word2Vec đã được tải thành công.\n")
except Exception as e:
    print(f"Không thể tải mô hình: {e}")
    exit()
print(model)
'''

#fWord2vec = 'word/W2V_150.txt'
fWord2vec = 'word/training_4GB/vectors.txt'
model = {}
try:
    with open(fWord2vec, 'r', encoding="utf-8-sig") as f:
        #f.readline()  # Bỏ qua dòng đầu tiên

        words_list = []  # Danh sách chứa các từ đã đọc

        for line in f:
            tem = line.split()

            if len(tem) != dim+1:  # 1 từ + 100 giá trị vector
                continue  

            word = tem[0]  # Lấy từ
            try:
                vector = np.array(list(map(float, tem[1:])))  # Chuyển đổi vector
                model[word] = vector  # Thêm vào dictionary
                words_list.append(word)  # Lưu từ vào danh sách
            except ValueError:
                print(f"⚠️ Bỏ qua từ '{word}' do lỗi chuyển đổi số.")

    print(f"\n✅ Đọc xong {len(model)} từ với vector kích thước {len(vector)}.")

    # In ra danh sách các từ đã đọc
    print("\n📌 Các từ đã đọc được:")
    print(len(words_list))  # In 50 từ đầu tiên để kiểm tra

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{fWord2vec}'. Kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"❌ Lỗi không xác định: {e}")
'''
# Hàm lấy vector từ mô hình
def getVectors(u1):
    u1 = u1.replace("-", "_")
    vs = np.zeros(dim)  # Tạo vector 0 với kích thước dim
    mk = 1
    if u1 in model.key_to_index:
    #if u1 in model:
        vs = model[u1].copy()  # Sử dụng copy để tránh lỗi read-only
    else:
        uu = u1.split("_")
        for i in uu:
            if i in model.key_to_index:
            #if i in model:    
                if mk == 1:
                    vs = model[i].copy()  # Sử dụng copy ở đây
                    mk = 0
                else:
                    vs += model[i].copy()  # Và cả ở đây
    return np.array(vs)
# Bước 2: Đọc dữ liệu Visim và tách thành 4 phần: w1, w2, pos, sim1
'''
fSimlex = 'word/Visim-400.txt'
vsl = []
with open(fSimlex, 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:  # Bỏ qua dòng tiêu đề nếu có
s = line.strip().split()
        if len(s) >= 4:
            w1 = s[0].strip()
            w2 = s[1].strip()
            pos = s[2].strip()
            sim1 = float(s[3].strip())
            vsl.append((w1, w2, pos, sim1))
'''

fSimlex = 'word/Visim-400.txt'
vsl = []
with open(fSimlex, 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:  # Bỏ qua dòng tiêu đề nếu có
        s = line.strip().split()
        if len(s) >= 4:
            w1 = s[0].strip()
            w2 = s[1].strip()
            pos = s[2].strip()
            sim1 = float(s[3].strip())
            vsl.append((w1, w2, pos, sim1))
rs = []  # Dãy similarity tính bằng Word2Vec
v = []   # Dãy similarity từ dữ liệu visim
cnt=0
cnt2=0

#for w1, w2, pos, sim1 in vsl:
for w1, w2,pos,sim1 in vsl:

    v1 = getVectors(w1)
    v2 = getVectors(w2)
   
    
    if (np.all(v1 == 0) or np.all(v2 == 0)):  # Kiểm tra nếu vector không tồn tại
        cnt=cnt+1
        print(f"Lỗi: Vector của '{w1}' và '{w2}' không cùng chiều.\n")
        continue

    if len(v1) != len(v2):
        print(f"Lỗi: Vector của '{w1}' và '{w2}' không cùng chiều.\n")
        continue

    similarity = 1 - spatial.distance.cosine(v1, v2)
    sim1= sim1/10
    #sim1= sim1/10
    rs.append(similarity)
    v.append(sim1)
print(len(rs))
# Bước 4: Tính hệ số tương quan Pearson
if len(rs) > 0:
    pearson_corr, p_value = pearsonr(rs, v)
    print(cnt)
    print(f"Hệ số tương quan Pearson: {pearson_corr:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("Không đủ dữ liệu để tính hệ số tương quan.")
