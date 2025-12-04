import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import preprocess as pp
from model import MINDRecModel

# --- CẤU HÌNH ---
# Quan trọng: Phải dùng tập DEV để test (Vì tập Train học rồi test không khách quan)
# Nếu bạn chưa giải nén dev, hãy giải nén MINDsmall_dev.zip
DATA_PATH_TRAIN = 'MIND_small_train' # Dùng để lấy lại từ điển cũ
DATA_PATH_TEST = 'MIND_small_dev'    # Dùng để chạy dự đoán
MODEL_PATH = 'checkpoints/mind_model.pth'
OUTPUT_PATH = 'prediction.txt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. TÁI TẠO TỪ ĐIỂN (Bắt buộc giống hệt lúc Train) ---
print("--- Đang tái tạo từ điển từ tập Train... ---")
# Phải load news train để ID của từ không bị lệch
df_news_train = pp.load_news_data(f'{DATA_PATH_TRAIN}/news.tsv')
word2index = pp.build_vocab(df_news_train['title'])
vocab_size = len(word2index) + 1

# --- 2. LOAD DỮ LIỆU TEST/DEV ---
print("--- Đang load dữ liệu Test/Dev... ---")
# Load News của tập Test (có thể có bài mới, nhưng ta dùng vocab cũ)
df_news_test = pp.load_news_data(f'{DATA_PATH_TEST}/news.tsv')
df_behaviors_test = pp.load_behaviors_data(f'{DATA_PATH_TEST}/behaviors.tsv')

# Cache title của tập Test
print("Caching Test News Titles...")
news_title_matrix = {}
for news_id, row in tqdm(df_news_test.iterrows(), total=len(df_news_test)):
    news_title_matrix[news_id] = pp.transform_text(row['title'], word2index)

# --- 3. LOAD MODEL ---
print(f"--- Đang load model từ {MODEL_PATH}... ---")
model = MINDRecModel(num_words=vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() # Chuyển sang chế độ thi (tắt dropout)

# --- 4. HÀM DỰ ĐOÁN CHO 1 USER ---
def predict_one_user(model, history_str, impressions_str):
    """
    Hàm này tính toán cho 1 dòng log:
    - history_str: "N1 N2 N3"
    - impressions_str: "N4-0 N5-1 N6-0" (Bài ứng viên)
    """
    # A. Xử lý History
    history_ids = [] if pd.isna(history_str) else history_str.split(' ')
    if len(history_ids) > pp.MAX_HISTORY_LENGTH: 
        history_ids = history_ids[-pp.MAX_HISTORY_LENGTH:]
    
    # Lấy vector số cho history
    history_seqs = [news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH) for nid in history_ids]
    while len(history_seqs) < pp.MAX_HISTORY_LENGTH:
        history_seqs.insert(0, [0]*pp.MAX_TITLE_LENGTH)
    
    # B. Xử lý Candidate List (Cả danh sách dài)
    candidates = []
    impression_items = impressions_str.split(' ')
    candidate_ids = [] # Lưu lại ID để debug nếu cần
    
    for item in impression_items:
        # item dạng "N1234-0" -> ta chỉ cần ID "N1234" để dự đoán
        nid = item.split('-')[0]
        candidate_ids.append(nid)
        candidates.append(news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH))
        
    # C. Chuyển thành Tensor
    # History: [1, 50, 30] (Batch size = 1)
    history_tensor = torch.tensor([history_seqs], dtype=torch.long).to(DEVICE)
    # Candidate: [N, 30] (N là số lượng bài được gợi ý)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(DEVICE)
    
    # D. Chạy qua Model
    with torch.no_grad():
        # 1. Lấy User Vector (Chỉ cần tính 1 lần)
        user_vector = model.user_encoder(history_tensor) # [1, 400]
        
        # 2. Lấy Candidate Vectors (Tính cho cả danh sách N bài)
        news_vectors = model.news_encoder(candidate_tensor) # [N, 400]
        
        # 3. Tính điểm (Dot Product)
        # [1, 400] * [N, 400] -> [N]
        scores = torch.matmul(user_vector, news_vectors.t()).squeeze()
        
    return scores.cpu().numpy().tolist()

# --- 5. CHẠY VÒNG LẶP VÀ GHI FILE ---
print("--- Bắt đầu dự đoán và ghi file prediction.txt ---")

with open(OUTPUT_PATH, 'w') as f:
    for idx, row in tqdm(df_behaviors_test.iterrows(), total=len(df_behaviors_test)):
        impression_id = row['impression_id']
        history_str = str(row['history'])
        impressions_str = row['impressions']
        
        # Tính điểm số (Scores)
        scores = predict_one_user(model, history_str, impressions_str)
        
        # Chuyển điểm số thành Thứ hạng (Rank)
        # Logic: Điểm càng cao -> Rank càng nhỏ (1 là nhất)
        # argsort lần 1: Ra index sắp xếp tăng dần
        # argsort lần 2: Ra thứ hạng
        # Đảo dấu scores (-scores) để sort giảm dần (điểm cao nhất đứng đầu)
        ranks = (np.argsort(np.argsort(-np.array(scores))) + 1).tolist()
        
        # Format ghi file: ImpressionID [Rank1, Rank2, ...]
        rank_str = '[' + ','.join(map(str, ranks)) + ']'
        f.write(f"{impression_id} {rank_str}\n")

print(f"\nĐã xong! File kết quả lưu tại: {OUTPUT_PATH}")
print("Hãy zip file này lại (prediction.zip) và nộp thử!")