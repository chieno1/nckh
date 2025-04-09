import re
from gensim.models import KeyedVectors

# Load mô hình từ file
model = KeyedVectors.load("emb/Noun.emb.wv")

def preprocess(text):
    # Bỏ dấu câu và chuyển về chữ thường
    text = text.lower()
    text = re.sub(r'[^\w\s:;]', '', text)  # bỏ tất cả ký tự không phải chữ cái/số/cách
    return text

def process(sentence, model):
    sentence = preprocess(sentence)
    words = sentence.split()
    i = 0
    new_tokens = []
    while i < len(words):
        found = False
        for j in range(len(words), i, -1):
            ngram = '_'.join(words[i:j])
            if ngram in model:
                new_tokens.append(ngram)
                i = j
                found = True
                break
        if not found:
            new_tokens.append(words[i])
            i += 1
    return ' '.join(new_tokens)
import re

def split_sentences(text):
    # Loại bỏ các ký tự thừa xung quanh và tách văn bản theo dấu câu
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Loại bỏ khoảng trắng thừa và câu rỗng
    return sentences



