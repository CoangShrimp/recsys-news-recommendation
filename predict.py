import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import zipfile

# Import các module của bạn (đảm bảo file preprocess.py và model.py nằm cùng thư mục)
import preprocess as pp
from model import MINDRecModel

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CONFIG)
# ==========================================
# File zip cần dự đoán (Dev set)
ZIP_TEST_PATH = 'MINDlarge_dev.zip'
# Thư mục sẽ giải nén ra
DIR_TEST_EXTRACTED = './mind_large_dev_data'

# Đường dẫn tập Train cũ (Dùng để lấy lại bộ từ điển - BẮT BUỘC)
# Nếu bạn lưu tên khác, hãy sửa dòng này
DIR_TRAIN_DATA = 'MIND_small_train' 

# Đường dẫn model đã train
MODEL_PATH = 'checkpoints/mind_model.pth'
# File kết quả đầu ra
OUTPUT_PATH = 'prediction.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HÀM GIẢI NÉN (Tự động chạy)
# ==========================================
def extract_data_if_needed():
    # Kiểm tra xem folder đã tồn tại và có file chưa
    if os.path.exists(DIR_TEST_EXTRACTED) and os.path.exists(os.path.join(DIR_TEST_EXTRACTED, 'news.tsv')):
        print(f"✅ Đã tìm thấy dữ liệu tại {DIR_TEST_EXTRACTED}. Bỏ qua giải nén.")
        return

    print(f"📦 Đang giải nén {ZIP_TEST_PATH}...")
    if not os.path.exists(ZIP_TEST_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file zip: {ZIP_TEST_PATH}")
        
    try:
        with zipfile.ZipFile(ZIP_TEST_PATH, 'r') as zip_ref:
            zip_ref.extractall(DIR_TEST_EXTRACTED)
        print(f"✅ Giải nén thành công vào: {DIR_TEST_EXTRACTED}")
    except Exception as e:
        print(f"❌ Lỗi giải nén: {e}")
        raise

# ==========================================
# 3. LOGIC DỰ ĐOÁN (PREDICT)
# ==========================================
def predict_one_user(model, history_str, impressions_str, news_title_matrix):
    """
    Hàm tính toán điểm cho 1 user
    """
    # A. Xử lý History (Lịch sử đọc)
    history_ids = [] if pd.isna(history_str) else str(history_str).split(' ')
    # Cắt hoặc pad history cho đúng chiều dài quy định
    if len(history_ids) > pp.MAX_HISTORY_LENGTH: 
        history_ids = history_ids[-pp.MAX_HISTORY_LENGTH:]
    
    # Map NewsID -> Vector số (Sequence)
    history_seqs = [news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH) for nid in history_ids]
    # Padding nếu history ngắn quá
    while len(history_seqs) < pp.MAX_HISTORY_LENGTH:
        history_seqs.insert(0, [0]*pp.MAX_TITLE_LENGTH)
    
    # B. Xử lý Candidate List (Danh sách bài cần xếp hạng)
    candidates = []
    impression_items = impressions_str.split(' ')
    
    for item in impression_items:
        # item dạng "N12345-0" hoặc "N12345-1". Ta cắt lấy ID "N12345"
        nid = item.split('-')[0]
        candidates.append(news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH))
        
    # C. Chuyển thành Tensor để đưa vào GPU/CPU
    # History: [Batch=1, Max_History_Len, Title_Len] -> ví dụ [1, 50, 30]
    history_tensor = torch.tensor([history_seqs], dtype=torch.long).to(DEVICE)
    # Candidate: [Num_Candidates, Title_Len] -> ví dụ [N, 30]
    candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(DEVICE)
    
    # D. Chạy qua Model (Inference)
    with torch.no_grad():
        # 1. Mã hóa người dùng
        user_vector = model.user_encoder(history_tensor) # [1, 400]
        
        # 2. Mã hóa các bài báo ứng viên
        news_vectors = model.news_encoder(candidate_tensor) # [N, 400]
        
        # 3. Tính điểm tương đồng (Dot Product)
        # Kết quả: [N] điểm số
        scores = torch.matmul(user_vector, news_vectors.t()).squeeze()
        
        # Xử lý trường hợp chỉ có 1 candidate (squeeze làm mất chiều)
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
            
    return scores.cpu().numpy().tolist()

def main():
    # --- Bước 0: Giải nén dữ liệu ---
    extract_data_if_needed()

    # --- Bước 1: Tái tạo từ điển (Vocab) từ tập TRAIN ---
    # CẢNH BÁO: Phải dùng tập TRAIN để build vocab, không dùng tập DEV/TEST.
    print(f"📖 Đang tái tạo từ điển từ {DIR_TRAIN_DATA}...")
    if not os.path.exists(os.path.join(DIR_TRAIN_DATA, 'news.tsv')):
         raise FileNotFoundError(f"❌ Cần thư mục {DIR_TRAIN_DATA} để lấy lại bộ từ điển cũ. Hãy giải nén MINDsmall_train vào đây.")

    df_news_train = pp.load_news_data(os.path.join(DIR_TRAIN_DATA, 'news.tsv'))
    word2index = pp.build_vocab(df_news_train['title'])
    vocab_size = len(word2index) + 1
    print(f"✅ Kích thước từ điển: {vocab_size} từ.")

    # --- Bước 2: Load dữ liệu DEV (Large) ---
    print(f"📥 Đang load dữ liệu dự đoán từ {DIR_TEST_EXTRACTED}...")
    # Load news.tsv của tập Dev
    df_news_dev = pp.load_news_data(os.path.join(DIR_TEST_EXTRACTED, 'news.tsv'))
    # Load behaviors.tsv của tập Dev
    df_behaviors_dev = pp.load_behaviors_data(os.path.join(DIR_TEST_EXTRACTED, 'behaviors.tsv'))

    # Cache (lưu đệm) tiêu đề bài báo Dev thành các con số
    print("⏳ Đang mã hóa tiêu đề bài báo Dev...")
    news_title_matrix = {}
    # Kết hợp cả news train (phòng hờ history cũ) và news dev
    # Ưu tiên News Dev nếu trùng ID
    for news_id, row in tqdm(df_news_dev.iterrows(), total=len(df_news_dev)):
        news_title_matrix[news_id] = pp.transform_text(row['title'], word2index)
    
    # --- Bước 3: Load Model ---
    print(f"🤖 Đang load model từ {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file model: {MODEL_PATH}")

    model = MINDRecModel(num_words=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Chế độ đánh giá (tắt Dropout)

    # --- Bước 4: Chạy dự đoán ---
    print(f"🚀 Bắt đầu dự đoán cho {len(df_behaviors_dev)} dòng log...")
    
    with open(OUTPUT_PATH, 'w') as f:
        for idx, row in tqdm(df_behaviors_dev.iterrows(), total=len(df_behaviors_dev)):
            impression_id = row['impression_id']
            history_str = row['history']
            impressions_str = row['impressions']
            
            try:
                # Tính điểm
                scores = predict_one_user(model, history_str, impressions_str, news_title_matrix)
                
                # Chuyển điểm thành Rank (Thứ hạng)
                # argsort(-scores) -> sắp xếp index theo điểm giảm dần
                # argsort lần nữa -> lấy thứ hạng
                ranks = (np.argsort(np.argsort(-np.array(scores))) + 1).tolist()
                
                # Format: ID [rank1,rank2,...]
                rank_str = '[' + ','.join(map(str, ranks)) + ']'
                f.write(f"{impression_id} {rank_str}\n")
            except Exception as e:
                # Nếu lỗi dòng nào thì ghi log và bỏ qua để không chết chương trình
                print(f"⚠️ Lỗi tại impression {impression_id}: {e}")

    print(f"\n🎉 XONG! File kết quả lưu tại: {OUTPUT_PATH}")
    print("Mẹo: Nén file này thành zip và nộp lên hệ thống chấm điểm.")

if __name__ == "__main__":
    main()