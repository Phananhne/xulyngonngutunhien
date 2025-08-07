import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()  # Chuyển tất cả về chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Bỏ dấu câu
    words = word_tokenize(text)  # Tách văn bản thành các từ

    # Danh sách từ dừng tiếng Việt do bạn định nghĩa
    vietnamese_stopwords = [
        'và', 'là', 'các', 'có', 'cho', 'của', 'được', 'trong', 'khi', 'với',
        'một', 'những', 'đã', 'này', 'đó', 'nên', 'rằng', 'thì', 'lại', 'nữa',
        'ra', 'vẫn', 'vì', 'như', 'để', 'thế', 'tôi', 'bạn', 'anh', 'chị', 'em', 'chúng', 'ta'
    ]

    # Loại bỏ các từ stop word
    filtered_words = [word for word in words if word not in vietnamese_stopwords and word.isalnum()]
    return " ".join(filtered_words)

# Văn bản mẫu
raw_text = "A strong earthquake hit the Himalayan kingdom of Nepal and parts of eastern India on Sunday morning, triggering landslides and collapsing buildings."

# Gọi hàm
clean_text = preprocess_text(raw_text)
print(">>> Văn bản sau khi tiền xử lý:")
print(clean_text)
