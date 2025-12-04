import preprocess as pp # Import file trên
from torch.utils.data import Dataset, DataLoader
import torch
import random
import pandas as pd
from model import NewsEncoder 

# 1. GỌI HÀM LOAD DATA (Chỉ gọi 1 lần ở đây)
df_news = pp.load_news_data(f'{pp.DATA_PATH}/news.tsv')
df_behaviors = pp.load_behaviors_data(f'{pp.DATA_PATH}/behaviors.tsv')
word2index = pp.build_vocab(df_news['title'])

# 2. CACHE DỮ LIỆU (Chuẩn bị cho Class Dataset)
print("Đang cache news titles...")
news_title_matrix = {}
for news_id, row in df_news.iterrows():
    # Gọi hàm transform bên file kia
    news_title_matrix[news_id] = pp.transform_text(row['title'], word2index)

def parse_impressions(impression_string):
    results = []
    items = impression_string.split(' ')
    for item in items:
        try:
            news_id, label = item.split('-')
            results.append((news_id, int(label)))
        except ValueError:
            continue # Bỏ qua nếu dữ liệu lỗi
    return results
# 3. ĐỊNH NGHĨA CLASS DATASET (Bắt buộc phải là Class)
class MINDDataset(Dataset):
    def __init__(self, behaviors_df, news_matrix):
        self.behaviors = behaviors_df
        self.news_matrix = news_matrix
        # Lấy tham số cấu hình từ preprocess
        self.max_history = pp.MAX_HISTORY_LENGTH 
        self.empty_news = [0] * pp.MAX_TITLE_LENGTH # Bài báo rỗng (toàn số 0)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        
        # --- 1. XỬ LÝ LỊCH SỬ (HISTORY) ---
        history_str = str(row['history'])
        if pd.isna(history_str) or history_str == 'nan':
            history_ids = []
        else:
            history_ids = history_str.split(' ')

        # Cắt bớt nếu quá dài (lấy 50 bài gần nhất)
        if len(history_ids) > self.max_history:
            history_ids = history_ids[-self.max_history:]
            
        history_seqs = []
        for nid in history_ids:
            # Lấy sequence từ cache, nếu không có thì lấy rỗng
            history_seqs.append(self.news_matrix.get(nid, self.empty_news))
            
        # Padding: Nếu chưa đủ 50 bài thì thêm bài rỗng vào ĐẦU (để đẩy bài mới nhất về cuối)
        while len(history_seqs) < self.max_history:
            history_seqs.insert(0, self.empty_news)
            
        # --- 2. XỬ LÝ ỨNG VIÊN (CANDIDATE) ---
        # Chọn ngẫu nhiên 1 cặp (Bài báo - Nhãn) để train
        impressions = parse_impressions(row['impressions'])
        if len(impressions) == 0:
            # Trường hợp hiếm: dòng không có impression nào hợp lệ -> trả về rỗng để tránh lỗi
            # (Thực tế nên lọc dòng này ra từ đầu, nhưng ở đây xử lý nhanh)
            candidate_id, label = ('', 0)
        else:
            candidate_id, label = random.choice(impressions)
        
        candidate_seq = self.news_matrix.get(candidate_id, self.empty_news)
        
        # --- 3. TRẢ VỀ TENSOR ---
        # Chuyển list thành PyTorch Tensor
        return (torch.tensor(history_seqs, dtype=torch.long),   # Shape: [50, 30]
                torch.tensor(candidate_seq, dtype=torch.long),  # Shape: [30]
                torch.tensor(label, dtype=torch.float))         # Shape: []
    
# --- PHẦN TEST (Cập nhật trong main.py) ---
from model import MINDRecModel # Import Model mới

if __name__ == "__main__":
    # 1. Dataset & DataLoader (Như cũ)
    train_dataset = MINDDataset(df_behaviors, news_title_matrix)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    batch = next(iter(train_loader))
    history, candidate, label = batch
    
    # 2. Khởi tạo Full Model
    print("\n--- KIỂM TRA FULL MODEL ---")
    vocab_size = len(word2index) + 1
    model = MINDRecModel(num_words=vocab_size)
    
    # 3. Chạy Forward
    print(f"History input: {history.shape}")     # [64, 50, 30]
    print(f"Candidate input: {candidate.shape}") # [64, 30]
    
    scores = model(history, candidate)
    
    print(f"Scores output: {scores.shape}")      # Mong đợi: [64]
    
    # In thử vài điểm số đầu tiên
    print("Vài điểm số dự đoán:", scores[:3].detach().numpy())
    
    if scores.shape == (64,):
        print("--> Mô hình hoạt động hoàn hảo! Sẵn sàng Training.")
    else:
        print("--> Lỗi kích thước đầu ra.")