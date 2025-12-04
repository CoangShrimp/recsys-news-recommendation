import pandas as pd
import string

# --- CẤU HÌNH ---
DATA_PATH = 'MIND_small_train' 
MAX_TITLE_LENGTH = 30
MAX_HISTORY_LENGTH = 50
PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'

# --- CÁC HÀM XỬ LÝ (Load Data) ---

def load_news_data(filepath):
    """Hàm chỉ lo việc đọc file News"""
    print(f"Đang đọc file news từ {filepath}...")
    cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    df = pd.read_csv(filepath, sep='\t', names=cols, index_col='news_id')
    return df

def load_behaviors_data(filepath):
    """Hàm chỉ lo việc đọc file Behaviors"""
    print(f"Đang đọc file behaviors từ {filepath}...")
    cols = ['impression_id', 'user_id', 'time', 'history', 'impressions']
    df = pd.read_csv(filepath, sep='\t', names=cols)
    return df

def build_vocab(news_titles):
    """Hàm xây dựng từ điển từ danh sách các tiêu đề"""
    print("Đang xây dựng từ điển (Vocab)...")
    word2index = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    
    for title in news_titles:
        if not isinstance(title, str): continue
        # Tách từ đơn giản
        text = title.lower().translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        
        for word in words:
            if word not in word2index:
                word2index[word] = len(word2index)
    
    print(f"--> Kích thước từ điển: {len(word2index)} từ")
    return word2index

# --- HÀM MÃ HÓA (Dùng đi dùng lại) ---
def transform_text(text, word2index):
    """Chuyển text thành list số"""
    if not isinstance(text, str): return [0] * MAX_TITLE_LENGTH
    
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    
    sequence = []
    for word in words:
        sequence.append(word2index.get(word, word2index[UNKNOWN_TOKEN]))
    
    # Cắt hoặc thêm padding
    if len(sequence) > MAX_TITLE_LENGTH:
        sequence = sequence[:MAX_TITLE_LENGTH]
    else:
        sequence = sequence + [word2index[PAD_TOKEN]] * (MAX_TITLE_LENGTH - len(sequence))
        
    return sequence