import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # Thư viện tạo thanh tiến trình
import os

# Import các module chúng ta đã viết
import preprocess as pp
from model import MINDRecModel

# --- 1. CẤU HÌNH (HYPERPARAMETERS) ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 3 # Số lần học lặp lại toàn bộ dữ liệu (Demo để 3, thực tế cần 5-10)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Đang sử dụng thiết bị: {DEVICE}")

# --- 2. CHUẨN BỊ DỮ LIỆU ---
# Load dữ liệu thô
print("--- Bắt đầu load dữ liệu ---")
df_news = pp.load_news_data(f'{pp.DATA_PATH}/news.tsv')
df_behaviors = pp.load_behaviors_data(f'{pp.DATA_PATH}/behaviors.tsv')
word2index = pp.build_vocab(df_news['title'])

# Cache tiêu đề sang vector số
print("--- Đang Cache dữ liệu bài báo ---")
news_title_matrix = {}
for news_id, row in tqdm(df_news.iterrows(), total=len(df_news), desc="Caching News"):
    news_title_matrix[news_id] = pp.transform_text(row['title'], word2index)

# Định nghĩa lại Dataset trong file này (hoặc import từ main cũ nếu bạn đã tách file)
from torch.utils.data import Dataset
import random

# (Mình copy lại Class Dataset gọn vào đây để file này chạy độc lập được luôn)
class MINDDataset(Dataset):
    def __init__(self, behaviors_df, news_matrix):
        self.behaviors = behaviors_df
        self.news_matrix = news_matrix
        self.max_history = pp.MAX_HISTORY_LENGTH
        self.empty_news = [0] * pp.MAX_TITLE_LENGTH

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        history_str = str(row['history'])
        history_ids = [] if pd.isna(history_str) or history_str == 'nan' else history_str.split(' ')
        
        if len(history_ids) > self.max_history: history_ids = history_ids[-self.max_history:]
        
        history_seqs = [self.news_matrix.get(nid, self.empty_news) for nid in history_ids]
        while len(history_seqs) < self.max_history: history_seqs.insert(0, self.empty_news)
        
        # Parse impressions
        try:
            items = row['impressions'].split(' ')
            item = random.choice(items)
            candidate_id, label = item.split('-')
            label = int(label)
        except:
            candidate_id, label = ('', 0)
            
        candidate_seq = self.news_matrix.get(candidate_id, self.empty_news)
        
        return (torch.tensor(history_seqs, dtype=torch.long), 
                torch.tensor(candidate_seq, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))

import pandas as pd # Import lại cho chắc
print("--- Tạo DataLoader ---")
train_dataset = MINDDataset(df_behaviors, news_title_matrix)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. KHỞI TẠO MÔ HÌNH ---
print("--- Khởi tạo Model ---")
vocab_size = len(word2index) + 1
model = MINDRecModel(num_words=vocab_size).to(DEVICE) # Đẩy model sang GPU/CPU

# Định nghĩa Loss và Optimizer
criterion = nn.BCEWithLogitsLoss() # Hàm loss chuẩn cho Binary Classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. VÒNG LẶP TRAINING (THE LOOP) ---
print("--- BẮT ĐẦU TRAINING ---")

for epoch in range(EPOCHS):
    model.train() # Chuyển model sang chế độ train (bật dropout)
    total_loss = 0
    
    # Thanh tiến trình (Progress Bar)
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (history, candidate, label) in enumerate(progress_bar):
        # 1. Chuyển dữ liệu sang thiết bị (GPU/CPU)
        history = history.to(DEVICE)
        candidate = candidate.to(DEVICE)
        label = label.to(DEVICE)
        
        # 2. Xóa gradient cũ
        optimizer.zero_grad()
        
        # 3. Forward Pass (Dự đoán)
        scores = model(history, candidate)
        
        # 4. Tính lỗi (Loss)
        loss = criterion(scores, label)
        
        # 5. Backward Pass (Lan truyền ngược)
        loss.backward()
        
        # 6. Cập nhật tham số
        optimizer.step()
        
        # Cập nhật thông tin lên thanh progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})

    # Kết thúc 1 Epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Kết thúc Epoch {epoch+1} | Loss trung bình: {avg_loss:.4f}")

# --- 5. LƯU MODEL ---
print("--- Lưu Model ---")
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
torch.save(model.state_dict(), 'checkpoints/mind_model.pth')
print("Đã lưu model vào checkpoints/mind_model.pth")