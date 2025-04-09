import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from scipy import spatial
from pyvi import ViTokenizer
try:
    model = KeyedVectors.load("emb/Noun.emb.wv")
    dim = model.vector_size  
    print("✅ Mô hình Node2Vec đã được tải thành công!\n")
except Exception as e:
    print(f"❌ Không thể tải mô hình: {e}")
    exit()

def getVectors(word):
    word = word.replace("-", "_")
    vector = np.zeros(dim)  
    if word in model.key_to_index:
        vector = model[word].copy()  
    else:
        parts = word.split("_")
        for i, part in enumerate(parts):
            if part in model.key_to_index:
                vector = model[part].copy() if i == 0 else vector + model[part].copy()
    return np.array(vector) 

# Hàm kiểm tra từ trái nghĩa
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

    for i, word1 in enumerate(words1):
        v1 = getVectors(word1)
        if np.all(v1 == 0):
            continue

        best_match = 0  
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
    words = text.split()  
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
text1 = "bác sĩ không thích nhà thơ"
text2 = "thi sĩ ghét thầy thuốc"
text1=ViTokenizer.tokenize(text1)
text2=ViTokenizer.tokenize(text2)
text1 = merge_tokens(text1)
text2 = merge_tokens(text2)
score, antonym_count = sentence_similarity_with_antonyms(text1, text2)
print(f"Similarity Score (with antonyms): {score:.4f}")
print(f"Number of Antonymous Words: {antonym_count}")