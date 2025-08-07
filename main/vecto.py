import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

#Bước 1: Tiền xử lý từng câu để lấy tập từ
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text) # Loại bỏ ký tự đặc biệt
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    return  set(tokens)

# Bước 2: Tách câu từ văn bản
text = """
Natural language processing is a fascinating field. 
It deals with understanding and generating human language.
Many systems today use natural language for communication.
The language used by humans can be quite complex.
"""

sentences = sent_tokenize(text)
N = len(sentences)
print("Tổng số câu: ", N)

#Bước 3: Tiền xử lý các câu
processed_sentences = [preprocess(sentence) for sentence in sentences]

#Bước 4: Khởi tạo ma trận kề
adj_matrix = np.zeros((N,N), dtype=int)

#Bước 5: Xây dựng ma trận kề
for i in range(N):
    for j in range(N):
        if i != j:
            common_words = processed_sentences[i].intersection(processed_sentences[j]) #intersection: Dùng cho tập hợp set trong python, mục đich tìm phần tử chung giữa 2 tập hợp
            if len(common_words) >= 3:
                adj_matrix[i][j] = 1

#Bước 6: Hiển thị
print("\n Ma trận kề: ")
print(adj_matrix)

#In lại các câu tương ứng với chỉ số
print("\n Danh sách câu (tương ứng chỉ số):")
for idx, s in enumerate(sentences):
    print((f"{idx}: {s}"))