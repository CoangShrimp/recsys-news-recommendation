import pandas as pd
import string
import os

# --- CẤU HÌNH DÙNG CHUNG ---
# Kích thước cố định để đưa vào model
MAX_TITLE_LENGTH = 30
MAX_HISTORY_LENGTH = 50

# Các token đặc biệt
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'

def load_news_data(filepath):
    """Đọc file news.tsv"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    
    print(f"📖 Đang đọc News: {filepath}...")
    cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    # Sử dụng quoting=3 để tránh lỗi parsing các ký tự lạ trong text
    df = pd.read_csv(filepath, sep='\t', names=cols, index_col='news_id', quoting=3)
    return df

def load_behaviors_data(filepath):
    """Đọc file behaviors.tsv"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    print(f"📖 Đang đọc Behaviors: {filepath}...")
    cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    df = pd.read_csv(filepath, sep='\t', names=cols, quoting=3)
    return df

def build_vocab(news_titles):
    """Tạo bộ từ điển từ tất cả tiêu đề bài báo"""
    print("🔨 Đang xây dựng từ điển (Vocab)...")
    word2index = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    
    for title in news_titles:
        if not isinstance(title, str): continue
        # Chuyển thường và bỏ dấu câu
        text = title.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        
        for word in words:
            if word not in word2index:
                word2index[word] = len(word2index)
    
    print(f"✅ Kích thước từ điển: {len(word2index)} từ")
    return word2index

def transform_text(text, word2index):
    """Chuyển câu văn thành dãy số (Vector)"""
    if not isinstance(text, str): 
        return [word2index[PAD_TOKEN]] * MAX_TITLE_LENGTH
    
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    
    sequence = []
    for word in words:
        sequence.append(word2index.get(word, word2index[UNKNOWN_TOKEN]))
    
    # Cắt nếu quá dài
    if len(sequence) > MAX_TITLE_LENGTH:
        sequence = sequence[:MAX_TITLE_LENGTH]
    # Thêm padding nếu quá ngắn
    else:
        sequence = sequence + [word2index[PAD_TOKEN]] * (MAX_TITLE_LENGTH - len(sequence))
        
    return sequence