import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. NEWS ENCODER (Giữ nguyên logic cũ) ---
class NewsEncoder(nn.Module):
    def __init__(self, num_words, word_embed_dim=300, num_filters=400, window_size=3):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=word_embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=word_embed_dim, out_channels=num_filters, kernel_size=window_size, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.attention_linear = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length]
        emb = self.embedding(x) 
        emb = self.dropout(emb)
        
        # Bước 2: CNN
        emb = emb.permute(0, 2, 1) 
        
        # --- SỬA TẠI ĐÂY ---
        # Thay F.relu bằng torch.tanh
        feature_map = torch.tanh(self.conv(emb)) 
        # -------------------
        
        feature_map = feature_map.permute(0, 2, 1)
        
        # Attention Pooling
        att_score = self.attention_linear(feature_map) # [B, L, 1]
        att_weight = F.softmax(att_score, dim=1) 
        
        # [B, L, 400] * [B, L, 1] -> Sum -> [B, 400]
        news_vector = torch.sum(feature_map * att_weight, dim=1)
        return news_vector

# --- 2. USER ENCODER (Mới) ---
class UserEncoder(nn.Module):
    def __init__(self, news_encoder, num_filters=400):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
        # Một lớp Attention nữa để học xem trong 50 bài đã đọc, bài nào quan trọng hơn
        self.attention_linear = nn.Linear(num_filters, 1)

    def forward(self, history_input):
        # history_input shape: [Batch, 50, 30] (50 bài, mỗi bài 30 từ)
        batch_size = history_input.size(0)
        num_history = history_input.size(1) # 50
        seq_len = history_input.size(2)     # 30
        
        # Bước 1: Gộp Batch và History lại để chạy NewsEncoder một thể cho nhanh
        # [Batch, 50, 30] -> [Batch * 50, 30]
        flattened_history = history_input.view(batch_size * num_history, seq_len)
        
        # Bước 2: Chuyển thành Vector bài báo
        # -> [Batch * 50, 400]
        news_vectors = self.news_encoder(flattened_history)
        
        # Bước 3: Tách ra lại như cũ
        # -> [Batch, 50, 400]
        news_vectors = news_vectors.view(batch_size, num_history, -1)
        
        # Bước 4: Attention Pooling (Tổng hợp sở thích)
        # Tính xem bài báo nào trong lịch sử quan trọng với user nhất
        att_score = self.attention_linear(news_vectors) # [Batch, 50, 1]
        att_weight = F.softmax(att_score, dim=1)
        
        # Tổng hợp lại thành 1 Vector User duy nhất
        # [Batch, 50, 400] * [Batch, 50, 1] -> Sum -> [Batch, 400]
        user_vector = torch.sum(news_vectors * att_weight, dim=1)
        
        return user_vector

# --- 3. FULL MODEL (Kết hợp tất cả) ---
class MINDRecModel(nn.Module):
    def __init__(self, num_words, word_embed_dim=300, num_filters=400):
        super(MINDRecModel, self).__init__()
        # Khởi tạo News Encoder dùng chung
        self.news_encoder = NewsEncoder(num_words, word_embed_dim, num_filters)
        # Khởi tạo User Encoder (dùng lại News Encoder ở trên)
        self.user_encoder = UserEncoder(self.news_encoder, num_filters)

    def forward(self, history, candidate):
        # 1. Tạo Vector User từ lịch sử
        user_vector = self.user_encoder(history) # [Batch, 400]
        
        # 2. Tạo Vector Candidate từ bài báo ứng viên
        candidate_vector = self.news_encoder(candidate) # [Batch, 400]
        
        # 3. Tính điểm tương đồng (Dot Product)
        # Nhân từng cặp vector rồi cộng lại -> Ra 1 con số score
        score = torch.sum(user_vector * candidate_vector, dim=1) # [Batch]
        
        return score