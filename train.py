import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random
import pandas as pd
import preprocess as pp
from model import MINDRecModel

# --- CẤU HÌNH ---
DATA_PATH = 'MIND_small_train'  # Folder chứa dữ liệu train
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATASET CLASS ---
class MINDTrainDataset(Dataset):
    def __init__(self, behaviors_df, news_matrix):
        self.behaviors = behaviors_df
        self.news_matrix = news_matrix
        self.empty_news = [0] * pp.MAX_TITLE_LENGTH

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        
        # 1. Xử lý History
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

        # 2. Xử lý Candidate (Balanced Sampling)
        impressions = str(row['impressions']).split(' ')
        positives = [imp for imp in impressions if imp.endswith('-1')]
        negatives = [imp for imp in impressions if imp.endswith('-0')]
        
        # Chọn mẫu ngẫu nhiên cân bằng 50/50
        if positives and negatives:
            selected = random.choice(positives) if random.random() > 0.5 else random.choice(negatives)
        elif positives:
            selected = random.choice(positives)
        else:
            selected = random.choice(impressions) # Fallback
            
        try:
            candidate_id, label = selected.split('-')
            label = float(label)
        except:
            candidate_id = 'UNKNOWN'
            label = 0.0

        candidate_seq = self.news_matrix.get(candidate_id, self.empty_news)
        
        return (torch.tensor(history_seqs, dtype=torch.long), 
                torch.tensor(candidate_seq, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))

def main():
    print(f"🚀 Bắt đầu Training trên: {DEVICE}")
    
    # 1. Load Data
    df_news = pp.load_news_data(f'{DATA_PATH}/news.tsv')
    df_behaviors = pp.load_behaviors_data(f'{DATA_PATH}/behaviors.tsv')
    
    # 2. Build Vocab & Cache
    word2index = pp.build_vocab(df_news['title'])
    
    print("⏳ Đang cache dữ liệu bài báo...")
    news_title_matrix = {}
    for nid, row in tqdm(df_news.iterrows(), total=len(df_news)):
        news_title_matrix[nid] = pp.transform_text(row['title'], word2index)
        
    # 3. Setup DataLoader
    train_dataset = MINDTrainDataset(df_behaviors, news_title_matrix)
    # num_workers=2 giúp load dữ liệu nhanh hơn (đa luồng)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    
    # 4. Model & Optimizer
    model = MINDRecModel(num_words=len(word2index)+1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for history, candidate, label in progress_bar:
            history, candidate, label = history.to(DEVICE), candidate.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            scores = model(history, candidate)
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # LƯU MODEL NGAY SAU MỖI EPOCH (Quan trọng!)
        save_path = f'checkpoints/mind_model_ep{epoch+1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f"✅ Đã lưu model: {save_path}")

if __name__ == "__main__":
    main()