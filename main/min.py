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
    text = re.sub(r'<.*?>', '', text)       # bỏ tag <s ...>
    text = re.sub(r'[^a-z0-9\s]', '', text) # bỏ ký tự đặc biệt
    text = re.sub(r'-{2,}', ' ', text)      # bỏ các chuỗi gạch ngang dài
    text_clean = re.sub(r'(\.\s*){2,}', ' ', text)  # Bỏ dấu '...' hoặc '. . .'
    text = re.sub(r'\s+', ' ', text).strip()  # Gom khoảng trắng


    # Làm sạch văn bản - xóa các ký tự đặc biệt, dấu câu, v.v.

    words = text.split()
    english_stopwords = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in english_stopwords]
    return " ".join(filtered_words)




# Thư mục chứa các file
import os

full_path = ["D:\\Mon-hoc\\Thacsi\\xulydulieu\\doan\\test\\d114h"]
# folder_path = r"D:\Mon-hoc\Thacsi\xulydulieu\doan\test"
# #Lấy tất cả file trong thư mục
# file_names = os.listdir(folder_path)
# full_path = [os.path.join(folder_path,f) for f in file_names]

# Lưu kết quả toàn bộ văn bản đã xử lý
def summarize_file(file_path):
    all_clean_texts = []
    all_raw_texts = []
    sentence_sources = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        matches = re.findall(r'<s\s[^>]*>.*?</s>', content, re.DOTALL)
        filtered_sentences = []  # reset cho mỗi file
        # print(f"\n>>> File: {file_path}")
        for match in matches:
            # Xóa các chuỗi gạch ngang bên trong <s>...</s>
            match_clean = re.sub(r'-{2,}', ' ', match)
            # Xóa các dấu "..." hoặc ". . ."
            match_clean = re.sub(r'(\.\s*){2,}', ' ', match_clean)
            all_raw_texts.append(match_clean.strip())
            clean_paragraph = preprocess_text(match)
            # print(clean_paragraph)
            all_clean_texts.append(clean_paragraph)
            # Lưu tên file cho câu này
            sentence_sources.append(os.path.basename(file_path))
                # Lọc ngay khi đọc
            if len(clean_paragraph.split()) >= 3:
                filtered_sentences.append(match_clean.strip())

    ###################### MA TRẬN KỀ #####################################
        
        # Bước 1: Mỗi đoạn <s> là 1 node (không tách câu nữa)
    
        sentences = filtered_sentences
        # for s in sentences:
        #     print(s)
        #     print()
        N = len(sentences)

        # print("Tổng số câu: ", N)

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
        
        ranks_nodes = sorted(sort_pg.items(),key = lambda x: x[1], reverse= True)
        # print(ranks_nodes)

        top = max(1, int(len(ranks_nodes) * 0.15))

        # summary_sentences = [(sentences[int(i[1:])], sentence_sources[int(i[1:])]) for i, _ in ranks_nodes[:top]]#In ra câu tóm tắt
        # for s,name_file in summary_sentences:
        #     print(f"[{name_file}] {s}")
        #     print()
    ##Lưu file vào trong txt
        output_file = "D:/Mon-hoc/Thacsi/xulydulieu/doan/main/summary_results.txt"
        with open(output_file, "w", encoding="utf-8") as out_f:
            for file_path in full_path:

                # ... xử lý ra sentences, ranks_nodes ...
                out_f.write(f"=== {os.path.basename(file_path)} ===\n")
                for i, _ in ranks_nodes[:top]:
                    idx = int(i[1:])
                    out_f.write(sentences[idx] + "\n")
                out_f.write("\n")


        print(f"✅ Kết quả đã lưu vào: {output_file}")
    # #Chạy ở terminal
    #     print(f"=== {os.path.basename(file_path)} ===")
    #     for i, _ in ranks_nodes[:top]:
    #         idx = int(i[1:])
    #         # Luu vào file txt de de quan sat
    #         print(sentences[idx])
    #     print()
                    
    
print("Kết quả đã xong")




