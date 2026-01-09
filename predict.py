import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import preprocess as pp
from model import MINDRecModel

# --- CẤU HÌNH ---
# Folder chứa dữ liệu Test (nhớ giải nén file zip vào đây)
DIR_TEST_DATA = './mind_large_dev_data' 
# Folder Train cũ (để lấy lại bộ từ điển Vocab y hệt lúc train)
DIR_TRAIN_DATA = 'MIND_small_train'
# Đường dẫn file model tốt nhất bạn muốn dùng
MODEL_PATH = 'checkpoints/mind_model.pth' 
OUTPUT_PATH = 'prediction.txt'
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CHỈ CHO PREDICT ---
class PredictionDataset(Dataset):
    def __init__(self, behaviors_df, news_matrix):
        self.behaviors = behaviors_df
        self.news_matrix = news_matrix
        self.empty_news = [0] * pp.MAX_TITLE_LENGTH

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        history_str = str(row['history'])
        if pd.isna(history_str) or history_str == 'nan':
            history_ids = []
        else:
            history_ids = history_str.split(' ')
            
        if len(history_ids) > pp.MAX_HISTORY_LENGTH:
            history_ids = history_ids[-pp.MAX_HISTORY_LENGTH:]
            
        history_seqs = [self.news_matrix.get(nid, self.empty_news) for nid in history_ids]
        while len(history_seqs) < pp.MAX_HISTORY_LENGTH:
            history_seqs.insert(0, self.empty_news)
            
        return torch.tensor(history_seqs, dtype=torch.long), idx

def main():
    # 1. Load Từ Điển (BẮT BUỘC TỪ TẬP TRAIN)
    print("📖 Đang tái tạo từ điển từ tập Train gốc...")
    if not os.path.exists(os.path.join(DIR_TRAIN_DATA, 'news.tsv')):
        raise FileNotFoundError(f"Cần folder {DIR_TRAIN_DATA} để lấy vocab")
        
    df_news_train = pp.load_news_data(os.path.join(DIR_TRAIN_DATA, 'news.tsv'))
    word2index = pp.build_vocab(df_news_train['title'])
    
    # 2. Load Dữ Liệu Test
    print("📥 Đang đọc dữ liệu Test...")
    df_news_test = pp.load_news_data(os.path.join(DIR_TEST_DATA, 'news.tsv'))
    df_beh_test = pp.load_behaviors_data(os.path.join(DIR_TEST_DATA, 'behaviors.tsv'))
    
    # 3. Cache News Vectors (Cho tập Test)
    print("⏳ Caching News Matrix cho tập Test...")
    news_title_matrix = {}
    for nid, row in tqdm(df_news_test.iterrows(), total=len(df_news_test)):
        news_title_matrix[nid] = pp.transform_text(row['title'], word2index)
        
    # 4. Load Model
    vocab_size = len(word2index) + 1
    model = MINDRecModel(num_words=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # --- CHIẾN THUẬT FAST INFERENCE ---
    
    # Bước A: Tính trước tất cả User Vectors
    print("⚡ Đang tính toán User Vectors (Batch Processing)...")
    pred_dataset = PredictionDataset(df_beh_test, news_title_matrix)
    pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    user_vectors_map = {}
    with torch.no_grad():
        for history, indices in tqdm(pred_loader):
            history = history.to(DEVICE)
            # Chỉ chạy User Encoder
            u_vecs = model.user_encoder(history)
            # Lưu vào RAM (CPU) để đỡ tốn VRAM
            for i, idx in enumerate(indices):
                user_vectors_map[idx.item()] = u_vecs[i].cpu()

    # Bước B: Tính điểm cho từng Impression
    print("🚀 Đang chấm điểm và xếp hạng...")
    with open(OUTPUT_PATH, 'w') as f:
        for idx, row in tqdm(df_beh_test.iterrows(), total=len(df_beh_test)):
            imp_id = row['impression_id']
            
            # Lấy User Vector đã tính sẵn
            user_vec = user_vectors_map[idx].to(DEVICE)
            
            # Lấy các bài báo ứng viên
            imp_items = str(row['impressions']).strip().split(' ')
            candidates_seqs = []
            valid_indices = [] # Theo dõi xem item nào hợp lệ
            
            for i, item in enumerate(imp_items):
                nid = item.split('-')[0]
                candidates_seqs.append(news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH))
                valid_indices.append(i)
                
            if not candidates_seqs: continue

            # Chuyển thành Tensor và chạy News Encoder
            cand_tensor = torch.tensor(candidates_seqs, dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                cand_vecs = model.news_encoder(cand_tensor)
                # Dot Product để ra điểm
                scores = torch.matmul(user_vec, cand_vecs.t()).squeeze()
                
                # Xử lý trường hợp chỉ có 1 candidate (kết quả là scalar)
                if scores.ndim == 0: scores = scores.unsqueeze(0)
                scores = scores.cpu().numpy()
            
            # Ranking: Điểm cao -> Rank nhỏ (1, 2, 3...)
            # Ví dụ: Scores [0.1, 0.9, 0.4] -> Ranks [3, 1, 2]
            sorted_indices = np.argsort(-scores)
            ranks = [0] * len(scores)
            for r, sorted_idx in enumerate(sorted_indices):
                ranks[sorted_idx] = r + 1
            
            f.write(f"{imp_id} {'['+','.join(map(str, ranks))+']'}\n")

    print(f"🎉 XONG! File kết quả: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()