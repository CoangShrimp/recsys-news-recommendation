import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import zipfile

# Import module nội bộ
import preprocess as pp
from model import MINDRecModel

# ==========================================
# 1. CẤU HÌNH (CONFIG)
# ==========================================
# Tên file zip (chỉ dùng nếu chưa giải nén)
ZIP_TEST_PATH = 'MINDlarge_dev.zip' 

# Thư mục chứa dữ liệu đầu vào (Quan trọng: Run_Test_Set.py sẽ đổ dữ liệu vào đây)
DIR_TEST_EXTRACTED = './mind_large_dev_data'

# Thư mục Train cũ (để lấy bộ từ điển Word2Index)
DIR_TRAIN_DATA = 'MIND_small_train' 

# Đường dẫn Model checkpoint
MODEL_PATH = 'checkpoints/mind_model.pth'

# File kết quả
OUTPUT_PATH = 'prediction.txt'

# Thiết bị (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. HÀM GIẢI NÉN (An toàn)
# ==========================================
def extract_data_if_needed():
    """
    Chỉ giải nén nếu chưa có dữ liệu. 
    Không crash nếu thiếu file zip nhưng dữ liệu đã có sẵn.
    """
    # 1. Kiểm tra nếu dữ liệu đã tồn tại (do Run_Test_Set.py tạo)
    if os.path.exists(DIR_TEST_EXTRACTED) and os.path.exists(os.path.join(DIR_TEST_EXTRACTED, 'news.tsv')):
        print(f"✅ Đã tìm thấy dữ liệu tại '{DIR_TEST_EXTRACTED}'. Sẵn sàng dự đoán!")
        return

    # 2. Nếu chưa có dữ liệu, thử giải nén
    print(f"📦 Không thấy folder dữ liệu, đang thử tìm {ZIP_TEST_PATH}...")
    if not os.path.exists(ZIP_TEST_PATH):
        # Nếu không có zip cũng không có folder -> Lỗi
        print(f"⚠️ Cảnh báo: Không tìm thấy file zip '{ZIP_TEST_PATH}' và cũng không có folder '{DIR_TEST_EXTRACTED}'.")
        print("👉 Nếu bạn đang chạy Test Set từ Drive, hãy đảm bảo Run_Test_Set.py đã chạy thành công trước bước này.")
        # Không raise lỗi ngay, để code dưới thử load file rồi mới báo lỗi chi tiết
        return
        
    try:
        with zipfile.ZipFile(ZIP_TEST_PATH, 'r') as zip_ref:
            zip_ref.extractall(DIR_TEST_EXTRACTED)
        print(f"✅ Giải nén thành công vào: {DIR_TEST_EXTRACTED}")
    except Exception as e:
        print(f"❌ Lỗi giải nén: {e}")

# ==========================================
# 3. LOGIC DỰ ĐOÁN (Đã sửa lỗi Strip)
# ==========================================
def predict_one_user(model, history_str, impressions_str, news_title_matrix):
    """
    Dự đoán điểm cho 1 user.
    """
    # --- A. Xử lý History ---
    # FIX: Thêm .strip() để loại bỏ khoảng trắng thừa
    if pd.isna(history_str):
        history_ids = []
    else:
        history_ids = str(history_str).strip().split(' ')
        
    # Lấy 50 bài gần nhất
    if len(history_ids) > pp.MAX_HISTORY_LENGTH: 
        history_ids = history_ids[-pp.MAX_HISTORY_LENGTH:]
    
    # Map ID sang Vector
    # Nếu ID không có trong dict (bài mới), dùng vector 0
    history_seqs = [news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH) for nid in history_ids]
    
    # Nếu history rỗng (User mới), thêm ít nhất 1 vector 0 để không bị lỗi dimention
    if not history_seqs:
        history_seqs.append([0]*pp.MAX_TITLE_LENGTH)

    # Padding về độ dài chuẩn (50)
    while len(history_seqs) < pp.MAX_HISTORY_LENGTH:
        history_seqs.insert(0, [0]*pp.MAX_TITLE_LENGTH)
    
    # --- B. Xử lý Candidate ---
    candidates = []
    # FIX: Thêm .strip() cực kỳ quan trọng để tránh tạo ra phần tử rỗng ''
    impression_items = str(impressions_str).strip().split(' ')
    
    for item in impression_items:
        if not item: continue # Bỏ qua nếu có item rỗng
        # item có thể là "N12345-0" (Dev) hoặc "N12345" (Test)
        # split('-')[0] xử lý được cả 2 trường hợp
        nid = item.split('-')[0]
        candidates.append(news_title_matrix.get(nid, [0]*pp.MAX_TITLE_LENGTH))
        
    # --- C. Tensor & Inference ---
    # History: [1, 50, 30]
    history_tensor = torch.tensor([history_seqs], dtype=torch.long).to(DEVICE)
    # Candidate: [N, 30]
    candidate_tensor = torch.tensor(candidates, dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        user_vector = model.user_encoder(history_tensor) # [1, 400]
        news_vectors = model.news_encoder(candidate_tensor) # [N, 400]
        
        # Dot product: [1, 400] x [400, N] -> [1, N] -> squeeze -> [N]
        scores = torch.matmul(user_vector, news_vectors.t()).squeeze()
        
        # Nếu chỉ có 1 candidate, squeeze làm nó thành scalar (0-d), cần unsqueeze lại thành 1-d array
        if scores.ndim == 0:
            scores = scores.unsqueeze(0)
            
    return scores.cpu().numpy().tolist()

def main():
    # --- Bước 0: Chuẩn bị dữ liệu ---
    extract_data_if_needed()

    # --- Bước 1: Build Vocab (Từ tập Train gốc) ---
    print(f"📖 Đang load từ điển từ {DIR_TRAIN_DATA}...")
    if not os.path.exists(os.path.join(DIR_TRAIN_DATA, 'news.tsv')):
         raise FileNotFoundError(f"❌ Cần thư mục '{DIR_TRAIN_DATA}' chứa news.tsv (MINDsmall_train) để tái tạo bộ từ điển.")

    df_news_train = pp.load_news_data(os.path.join(DIR_TRAIN_DATA, 'news.tsv'))
    word2index = pp.build_vocab(df_news_train['title'])
    vocab_size = len(word2index) + 1
    print(f"✅ Vocab size: {vocab_size}")

    # --- Bước 2: Load Data cần dự đoán ---
    print(f"📥 Đang đọc dữ liệu dự đoán từ '{DIR_TEST_EXTRACTED}'...")
    news_path = os.path.join(DIR_TEST_EXTRACTED, 'news.tsv')
    behaviors_path = os.path.join(DIR_TEST_EXTRACTED, 'behaviors.tsv')
    
    if not os.path.exists(news_path) or not os.path.exists(behaviors_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file data trong {DIR_TEST_EXTRACTED}. Hãy kiểm tra lại bước giải nén.")

    df_news_dev = pp.load_news_data(news_path)
    df_behaviors_dev = pp.load_behaviors_data(behaviors_path)
    print(f"   + Số lượng bài báo: {len(df_news_dev)}")
    print(f"   + Số lượng logs cần dự đoán: {len(df_behaviors_dev)}")

    # Mã hóa tiêu đề bài báo (Cache)
    print("⏳ Đang mã hóa tiêu đề bài báo (Embedding lookup)...")
    news_title_matrix = {}
    for news_id, row in tqdm(df_news_dev.iterrows(), total=len(df_news_dev)):
        news_title_matrix[news_id] = pp.transform_text(row['title'], word2index)
    
    # --- Bước 3: Load Model ---
    print(f"🤖 Đang load model: {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy model tại {MODEL_PATH}")

    model = MINDRecModel(num_words=vocab_size).to(DEVICE)
    # map_location đảm bảo load được trên cả CPU nếu train bằng GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Bước 4: Chạy dự đoán ---
    print("🚀 Bắt đầu dự đoán...")
    
    with open(OUTPUT_PATH, 'w') as f:
        for idx, row in tqdm(df_behaviors_dev.iterrows(), total=len(df_behaviors_dev)):
            impression_id = row['impression_id']
            history_str = row['history']
            impressions_str = row['impressions']
            
            try:
                # Lấy điểm số
                scores = predict_one_user(model, history_str, impressions_str, news_title_matrix)
                
                # Chuyển thành Rank (1 là cao nhất)
                # argsort(-scores) trả về index của phần tử lớn nhất đến nhỏ nhất
                sorted_indices = np.argsort(-np.array(scores))
                
                # Rank thực tế là vị trí trong mảng đã sort + 1? 
                # KHÔNG, MIND yêu cầu rank của từng bài theo thứ tự ban đầu.
                # Ví dụ input: [Item1, Item2] -> scores: [0.9, 0.1] -> Output: [1, 2]
                # Ví dụ input: [Item1, Item2] -> scores: [0.1, 0.9] -> Output: [2, 1]
                
                # Cách tạo rank chuẩn format MIND:
                # Ta cần xếp hạng cho từng vị trí.
                # ranks[i] = thứ hạng của item i
                
                # Dùng scipy.stats.rankdata hoặc logic đảo ngược argsort
                n = len(scores)
                ranks = [0] * n
                for rank, index in enumerate(sorted_indices):
                    ranks[index] = rank + 1
                
                # Format: [rank1,rank2,...]
                rank_str = '[' + ','.join(map(str, ranks)) + ']'
                f.write(f"{impression_id} {rank_str}\n")
                
            except Exception as e:
                print(f"\n⚠️ Lỗi Impression {impression_id}: {e}")
                # Fallback: điền random ranks để không bị chết chương trình
                # (Quan trọng để file output vẫn đủ dòng)
                try:
                    count = len(str(impressions_str).strip().split(' '))
                    fallback_ranks = list(range(1, count + 1))
                    f.write(f"{impression_id} {'[' + ','.join(map(str, fallback_ranks)) + ']'}\n")
                except:
                    pass

    print(f"\n🎉 HOÀN TẤT! Kết quả lưu tại: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()