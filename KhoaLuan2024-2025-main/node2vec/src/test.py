import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from scipy import spatial
from pyvi import ViTokenizer
dim=150
'''
# Tải mô hình word embeddings
try:
    model = KeyedVectors.load("emb/Noun.emb.wv")
    dim = model.vector_size  
    print("✅ Mô hình Node2Vec đã được tải thành công!\n")
except Exception as e:
    print(f"❌ Không thể tải mô hình: {e}")
    exit()
'''
fWord2vec = 'word/W2V_150.txt'
model = {}
try:
    with open(fWord2vec, 'r', encoding="utf-8-sig") as f:
        f.readline()  # Bỏ qua dòng đầu tiên

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
# Danh sách các từ phủ định cần xử lý

# Hàm lấy vector từ mô hình
def getVectors(word):
    word = word.replace("-", "_")
    vector = np.zeros(dim)  
    #if word in model.key_to_index:
    if word in model:
        vector = model[word].copy()  
    else:
        parts = word.split("_")
        for i, part in enumerate(parts):
            #if part in model.key_to_index:
            if part in model:
                vector = model[part].copy() if i == 0 else vector + model[part].copy()
    return np.array(vector) 

# Kiểm tra xem câu có từ phủ định không
def is_antonym(word1, word2, threshold=-0.5):
    v1 = getVectors(word1)
    v2 = getVectors(word2)

    if np.all(v1 == 0) or np.all(v2 == 0):  
        return False  

    similarity = 1 - spatial.distance.cosine(v1, v2)

    return similarity < threshold  

def sentence_similarity_with_antonyms(text1, text2, position_weight=0.3, antonym_penalty=0.5):
    words1 = text1.split()
    words2 = text2.split()
    similarities = []
    position_penalty = 0  
    antonym_count = 0  

    # Tính toán độ tương đồng cho từng từ trong câu 1 so với câu 2
    for i, word1 in enumerate(words1):
        v1 = getVectors(word1)
        if np.all(v1 == 0):  # Bỏ qua từ không có trong từ điển
            continue

        best_match = 0  # Khởi tạo best_match là 0 thay vì -1
        best_pos_diff = len(words1)  
        is_opposite = False  

        for j, word2 in enumerate(words2):
            v2 = getVectors(word2)
            if np.all(v2 == 0):  
                continue

            similarity = 1 - spatial.distance.cosine(v1, v2)

            if similarity > best_match:
                best_match = similarity
                best_pos_diff = abs(i - j)

            # Kiểm tra từ trái nghĩa
            if is_antonym(word1, word2):
                is_opposite = True
                antonym_count += 1  

        similarities.append(best_match)

        position_penalty += best_pos_diff * position_weight  

        if is_opposite:
            similarities[-1] -= antonym_penalty  

    if not similarities:
        return 0, 0  

    position_penalty = position_penalty / len(words1)

    final_score = np.mean(similarities) - position_penalty
    return max(0, final_score), antonym_count  
def merge_tokens(text):
    words = text.split()  # Tách thành danh sách từ
    merged_words = []
    i = 0
    while i < len(words):
        if words[i] in ["không", "chưa", "đã", "sẽ", "vừa", "mới"] and i < len(words) - 1:
            merged_words.append(words[i] + "_" + words[i + 1])  # Gộp với từ sau
            i += 2  
        else:
            merged_words.append(words[i])
            i += 1
    return " ".join(merged_words)
# Test với hai câu ví dụ
text1 = "nhà thơ không thích thi sĩ"
text2 = "thi sĩ ghét thầy thuốc"
text1=ViTokenizer.tokenize(text1)
text2=ViTokenizer.tokenize(text2)
text1 = merge_tokens(text1)
text2 = merge_tokens(text2)
score, antonym_count = sentence_similarity_with_antonyms(text1, text2)
print(f"Similarity Score (with antonyms): {score:.4f}")
print(f"Number of Antonymous Words: {antonym_count}")