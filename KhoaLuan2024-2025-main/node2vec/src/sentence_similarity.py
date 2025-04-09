import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from gensim.models import KeyedVectors

# Định nghĩa số chiều vector
dim = 150  
# Câu cần so sánh
text1 = ""
text2 = "tôi đi chơi"

# Load mô hình Word2Vec
try:
    model = KeyedVectors.load("emb/Noun.emb.wv")
    dim = model.vector_size  # Cập nhật số chiều từ mô hình
    print("✅ Mô hình Node2Vec đã được tải thành công!\n")
except Exception as e:
    print(f"❌ Không thể tải mô hình: {e}")
    exit()

# Hàm lấy vector trung bình của câu
def get_sentence_vector(sentence):
    words = sentence.split()
    vector_sum = np.zeros(dim)  # Vector ban đầu = 0
    found_words = 0  # Đếm số từ có trong mô hình
    
    for word in words:
        word = word.replace("-", "_")  # Chuẩn hóa dấu gạch ngang
        if word in model.key_to_index:
            vector_sum += model[word]
            found_words += 1
        else:  # Nếu từ không có trong model, thử tách theo "_"
            subwords = word.split("_")
            for subword in subwords:
                if subword in model.key_to_index:
                    vector_sum += model[subword]
                    found_words += 1

    if found_words > 0:
        return vector_sum / found_words  
    else:
        return None  

# Lấy vector cho từng câu
vec1 = get_sentence_vector(text1)
vec2 = get_sentence_vector(text2)

# Xử lý trường hợp không tìm thấy từ nào
if vec1 is None or vec2 is None:
    print("⚠️ Không thể tính độ tương đồng vì một trong hai câu không có từ nào trong mô hình.")
else:
    # Tính Cosine Similarity
    cosine_sim = 1 - cosine(vec1, vec2) if np.any(vec1) and np.any(vec2) else 0.0  

    # Tính Pearson Correlation (chỉ khi vector không phải hằng số)
    if np.std(vec1) > 0 and np.std(vec2) > 0:
        pearson_corr, _ = pearsonr(vec1, vec2)
    else:
        pearson_corr = 0.0  # Nếu vector không có độ lệch chuẩn, Pearson không xác định

    # Kết quả
    print(f"🔹 Cosine Similarity: {cosine_sim:.4f}")
    print(f"🔹 Pearson Correlation: {pearson_corr:.4f}")
print()