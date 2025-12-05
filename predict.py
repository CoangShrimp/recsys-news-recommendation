import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import zipfile
import preprocess as pp
from model import MINDRecModel

# ==========================================
# CẤU HÌNH (CONFIG)
# ==========================================
# Tên file zip (chỉ dùng để check hoặc giải nén nếu chưa có data)
ZIP_TEST_PATH = 'MINDlarge_dev.zip' 

# Thư mục dữ liệu đầu vào (Code Colab sẽ giải nén Test set vào đây)
DIR_TEST_EXTRACTED = './mind_large_dev_data'

# Thư mục Train cũ (để lấy bộ từ điển)
DIR_TRAIN_DATA = 'MIND_small_train' 

# Đường dẫn Model
MODEL_PATH = 'checkpoints/mind_model.pth'

# File kết quả
OUTPUT_PATH = 'prediction.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_data_if_needed():
    # Kiểm tra folder dữ liệu trước
    if os.path.exists(DIR_TEST_EXTRACTED) and os.path.exists(os.path.join(DIR_TEST_EXTRACTED, 'news.tsv')):
        print(f"✅ Đã tìm thấy dữ liệu tại {DIR_TEST_EXTRACTED}")
        return

    # Nếu không có folder, mới tìm file zip
    if not os.path.exists(ZIP_TEST_PATH):
        print(f"⚠️ Không tìm thấy file zip {ZIP_TEST_PATH} và cũng không có folder {DIR_TEST_EXTRACTED}.")
        print("👉 Vui lòng đảm bảo bạn đã giải nén dữ liệu Test vào đúng folder 'mind_large_dev_data'.")
        return
        
    try:
        print(f"📦 Đang giải nén {ZIP_TEST_PATH}...")
        with zipfile.ZipFile(ZIP_TEST_PATH, 'r') as zip_ref:
            zip_ref.extractall(DIR_TEST_EXTRACTED)
        print(f"✅ Giải nén thành công.")
    except Exception as e:
        print(f"❌ Lỗi giải nén: {e}")

def predict_one_user(model, history_str, impressions_str, news_title_matrix):
    # --- Xử lý History (Thêm .strip() để tránh lỗi chuỗi rỗng) ---
    if pd.isna(history_str):
        history_ids = []
    else:
        history_ids = str(history_str).strip().split(' ')
        
    if len(history_ids) > pp.MAX_HISTORY_LENGTH: 
        history_ids = history_ids[-pp.MAX_HISTORY_LENGTH:]
    
    # Map ID -> Vector (Nếu ID mới thì dùng vector 0)
    history_seqs = [news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH) for nid in history_ids]
    
    # Nếu history rỗng (Cold start), thêm 1 vector 0 để tránh lỗi dimention
    if not history_seqs:
        history_seqs.append([0]*pp.MAX_TITLE_LENGTH)

    # Padding
    while len(history_seqs) < pp.MAX_HISTORY_LENGTH:
        history_seqs.insert(0, [0]*pp.MAX_TITLE_LENGTH)
    
    # --- Xử lý Candidate (Thêm .strip() cực kỳ quan trọng) ---
    candidates = []
    impression_items = str(impressions_str).strip().split(' ')
    
    for item in impression_items:
        if not item: continue # Bỏ qua item rỗng do lỗi split
        # Test set format: "N12345" (không có -0/-1)
        # Dev set format: "N12345-0"
        # split('-')[0] cân được cả hai
        nid = item.split('-')[0]
        candidates.append(news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH))
        
    # Chuyển sang Tensor
    history_tensor = torch.tensor([history_seqs], dtype=torch.long).to(DEVICE)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        user_vector = model.user_encoder(history_tensor) 
        news_vectors = model.news_encoder(candidate_tensor)
        
        scores = torch.matmul(user_vector, news_vectors.t()).squeeze()
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
            
    return scores.cpu().numpy().tolist()

def main():
    # 1. Setup dữ liệu
    extract_data_if_needed()

    # 2. Load Từ điển (Bắt buộc từ Train set)
    print(f"📖 Đang load từ điển từ {DIR_TRAIN_DATA}...")
    if not os.path.exists(os.path.join(DIR_TRAIN_DATA, 'news.tsv')):
         raise FileNotFoundError(f"❌ Cần folder '{DIR_TRAIN_DATA}' chứa news.tsv (MINDsmall_train) để tái tạo vocab.")

    df_news_train = pp.load_news_data(os.path.join(DIR_TRAIN_DATA, 'news.tsv'))
    word2index = pp.build_vocab(df_news_train['title'])
    vocab_size = len(word2index) + 1
    
    # 3. Load Dữ liệu Test
    print(f"📥 Đang đọc dữ liệu từ {DIR_TEST_EXTRACTED}...")
    news_path = os.path.join(DIR_TEST_EXTRACTED, 'news.tsv')
    beh_path = os.path.join(DIR_TEST_EXTRACTED, 'behaviors.tsv')
    
    if not os.path.exists(news_path) or not os.path.exists(beh_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu trong {DIR_TEST_EXTRACTED}.")

    df_news_dev = pp.load_news_data(news_path)
    df_beh_dev = pp.load_behaviors_data(beh_path)
    print(f"   + Số lượng logs cần dự đoán: {len(df_beh_dev)}")

    print("⏳ Đang mã hóa tiêu đề bài báo...")
    news_title_matrix = {}
    # Sử dụng file=sys.stdout để tqdm hiện mượt trên Colab
    for nid, row in tqdm(df_news_dev.iterrows(), total=len(df_news_dev), file=sys.stdout):
        news_title_matrix[nid] = pp.transform_text(row['title'], word2index)
    
    # 4. Load Model
    print(f"🤖 Đang load model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy model tại {MODEL_PATH}")

    model = MINDRecModel(num_words=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 5. Chạy Predict
    print("🚀 Bắt đầu dự đoán...")
    with open(OUTPUT_PATH, 'w') as f:
        # Sử dụng file=sys.stdout để tqdm hiện mượt trên Colab
        for _, row in tqdm(df_beh_dev.iterrows(), total=len(df_beh_dev), desc="Predicting", file=sys.stdout):
            imp_id = row['impression_id']
            try:
                scores = predict_one_user(model, row['history'], row['impressions'], news_title_matrix)
                
                # Convert Score -> Rank (1, 2, 3...)
                # argsort(-scores) -> sắp xếp index từ điểm cao xuống thấp
                sorted_indices = np.argsort(-np.array(scores))
                
                # Gán rank ngược lại: index của bài điểm cao nhất nhận rank 1
                ranks = [0] * len(scores)
                for r, idx in enumerate(sorted_indices):
                    ranks[idx] = r + 1
                
                rank_str = '[' + ','.join(map(str, ranks)) + ']'
                f.write(f"{imp_id} {rank_str}\n")
                
            except Exception as e:
                # Fallback: điền rank giả định 1->N nếu lỗi, để file không bị thiếu dòng
                try:
                    cnt = len(str(row['impressions']).strip().split(' '))
                    fallback = list(range(1, cnt + 1))
                    f.write(f"{imp_id} {'[' + ','.join(map(str, fallback)) + ']'}\n")
                except:
                    pass

    print(f"\n🎉 XONG! Kết quả lưu tại: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()