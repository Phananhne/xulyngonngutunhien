import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords

# Tải các gói NLTK cần thiết
nltk.download('punkt')
nltk.download('stopwords')

# Hàm tiền xử lý (dùng để đếm số từ)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)       # bỏ tag <s ...>
    text = re.sub(r'[^a-z0-9\s]', '', text) # bỏ ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()
    return text

full_path = ["D:\\Mon-hoc\\Thacsi\\xulydulieu\\doan\\test\\d120i"]

all_clean_texts = []
all_raw_texts = []
sentence_sources = []
filtered_sentences = []

for file_path in full_path:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'<s\s[^>]*>.*?</s>', content, re.DOTALL)

            for match in matches:
                all_raw_texts.append(match.strip())
                clean_paragraph = preprocess_text(match)
                all_clean_texts.append(clean_paragraph)
                sentence_sources.append(os.path.basename(file_path))

                # Lọc ngay khi đọc
                if len(clean_paragraph.split()) >= 3:
                    filtered_sentences.append(match.strip())

        print("============================================================")
        print(f"Các câu đã lọc (≥ 3 từ) từ {os.path.basename(file_path)}:")
        for s in filtered_sentences:
            print(s)

    except Exception as e:
        print(f"\n⚠️ Lỗi khi xử lý file {file_path}: {e}")
