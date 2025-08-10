import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize



# Tải các gói NLTK cần thiết
nltk.download('punkt')
nltk.download('stopwords')

# Hàm tiền xử lý
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Bỏ số
    text = re.sub(r'[^a-z\s]', ' ', text)  # Chỉ giữ lại chữ cái và khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()  # Gom khoảng trắng
    # Làm sạch văn bản - xóa các ký tự đặc biệt, dấu câu, v.v.

    words = text.split()
    english_stopwords = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in english_stopwords]
    return " ".join(filtered_words)




# Thư mục chứa các file
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))  # đường dẫn của min.py
# folder_path = os.path.join(current_dir, "../test")
 # nếu test nằm cùng cấp với thư mục main
# file_names = ["D:\\Mon-hoc\\Thacsi\\xulydulieu\\doan\\test\\d112h"]   # không thêm .txt nếu file không có đuôi


# # file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# # Lưu kết quả toàn bộ văn bản đã xử lý
# all_clean_texts = []
# all_raw_texts = []

# for file_name in file_names:
#     file_path = os.path.join( file_name)
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()

#             # Tìm các đoạn <s docid=...>...</s>
#             matches = re.findall(r'<s\s[^>]*>.*?</s>', content, re.DOTALL)

#             print(f"\n>>> File: {file_name}")
#             for docid, paragraph in matches:
#                 all_raw_texts.append(paragraph.strip())  # << GIỮ nguyên đoạn gốc
#                 clean_paragraph = preprocess_text(paragraph)
#                 print(clean_paragraph)
#                 all_clean_texts.append(clean_paragraph)

#     except Exception as e:s
#         print(f"\n⚠️ Lỗi khi xử lý file {file_name}: {e}")
# # Ghép các đoạn lại thành 1 chuỗi lớn
# full_text = ' '.join(all_raw_texts)  # << Dùng đoạn gốc để tách câu
file_names = ["D:\\Mon-hoc\\Thacsi\\xulydulieu\\doan\\test\\d120i"]
# Lưu kết quả toàn bộ văn bản đã xử lý
all_clean_texts = []
all_raw_texts = []

for file_path in file_names:  # Dùng trực tiếp file_path luôn
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            matches = re.findall(r'<s\s[^>]*>.*?</s>', content, re.DOTALL)

            # print(f"\n>>> File: {file_path}")
            for match in matches:
                all_raw_texts.append(match.strip())
                clean_paragraph = preprocess_text(match)
                print(clean_paragraph)
                all_clean_texts.append(clean_paragraph)

    except Exception as e:
        print(f"\n⚠️ Lỗi khi xử lý file {file_path}: {e}")

###################### MA TRẬN KỀ #####################################
print("============================================================")
# Bước 1: Mỗi đoạn <s> là 1 node (không tách câu nữa)
sentences = [
    s for s in all_raw_texts
    if len(preprocess_text(s).split()) >= 3
]

N = len(sentences)

print("Tổng số câu: ", N)

#Bước 3: Tiền xử lý các câu
processed_sentences = [set(preprocess_text(sentence).split()) for sentence in sentences]

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
# print("\n Ma trận kề: ")
# print(adj_matrix)

# #In lại các câu tương ứng với chỉ số
# print("\n Danh sách câu (tương ứng chỉ số):")
# for idx, s in enumerate(sentences):
#     print(f"{idx}: {preprocess_text(s)}")



#======Tìm pagerank======================
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
nodes = [f"S{i}" for i in range(len(sentences))]
# print(nodes)
edges = {}
for i in range (len(sentences)):
    edges[f"S{i}"] = []
    for j in range(len(sentences)):
        if adj_matrix[i][j] == 1:
            edges[f"S{i}"].append(f"S{j}")
# for node, neighbors in edges.items(): #edges.items() → lấy ra từng cặp (key, value)
#     # print(f"{node}: {neighbors}")

N = len(nodes)
d = 0.85
# Tạo dict lưu các đỉnh liên kết đến mỗi node
adj_in = {node: [] for node in nodes}
for v in edges:
    for u in edges[v]:  # v → u
        if u in adj_in:
            adj_in[u].append(v)

# # In kết quả
# for node in nodes:
#     print(f"Các đỉnh liên kết tới {node}: {adj_in[node]}")
#
out_degree = {node: len(edges[node]) for node in nodes}
# for node in nodes:
#     print("deg_" + node + ":", out_degree[node])


# Tính pagerank(u)
sort_pg = {}
for u in nodes:
    total = 0
    # print("PageRank "+u + ": ")
    for v in adj_in[u]:  # v → u
        total += 1 / out_degree[v]
    new_pagerank_A = (1 - d) / N + d * total
    sort_pg[u] = new_pagerank_A
    # print(new_pagerank_A)


#=======Sắp xếp pagerank=========================
print("******************************************")
ranks_nodes = sorted(sort_pg.items(),key = lambda x: x[1], reverse= True)
# print(ranks_nodes)

top = max(1, int(len(ranks_nodes) * 0.15))

summary_sentences = [sentences[int(i[1:])] for i, _ in ranks_nodes[:top]]#In ra câu tóm tắt
for s in summary_sentences:
    print(s)
    print()



