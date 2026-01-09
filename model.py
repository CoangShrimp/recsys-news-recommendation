import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsEncoder(nn.Module):
    def __init__(self, num_words, word_embed_dim=300, num_filters=400, window_size=3):
        super(NewsEncoder, self).__init__()
        # padding_idx=0: Bắt buộc để model bỏ qua các phần đệm
        self.embedding = nn.Embedding(num_embeddings=num_words, 
                                      embedding_dim=word_embed_dim, 
                                      padding_idx=0)
        
        # CNN để trích xuất đặc trưng từ ngữ cảnh
        self.conv = nn.Conv1d(in_channels=word_embed_dim, 
                              out_channels=num_filters, 
                              kernel_size=window_size, 
                              padding=1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Attention Layer
        self.attention_linear = nn.Linear(num_filters, 1)

    def forward(self, x):
        # x: [Batch, Sequence_Length]
        emb = self.embedding(x) 
        emb = self.dropout(emb)
        
        # Chuyển đổi dimension cho CNN: [Batch, Embed_Dim, Seq_Len]
        emb = emb.permute(0, 2, 1) 
        
        # CNN + ReLU (ReLU tốt hơn Tanh cho bài toán này)
        feature_map = F.relu(self.conv(emb)) 
        
        # Quay lại dimension cũ: [Batch, Seq_Len, Filters]
        feature_map = feature_map.permute(0, 2, 1)
        
        # Attention Pooling
        att_score = self.attention_linear(feature_map) 
        att_weight = F.softmax(att_score, dim=1) 
        
        # Tổng hợp thành vector bài báo
        news_vector = torch.sum(feature_map * att_weight, dim=1)
        return news_vector

class UserEncoder(nn.Module):
    def __init__(self, news_encoder, num_filters=400):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
        self.attention_linear = nn.Linear(num_filters, 1)

    def forward(self, history_input):
        # history_input: [Batch, History_Length, Title_Length]
        batch_size = history_input.size(0)
        num_history = history_input.size(1)
        seq_len = history_input.size(2)
        
        # Gộp Batch và History để chạy NewsEncoder một lần cho nhanh
        flattened_history = history_input.view(batch_size * num_history, seq_len)
        
        # Lấy vector của từng bài báo
        news_vectors = self.news_encoder(flattened_history)
        
        # Tách lại thành shape User
        news_vectors = news_vectors.view(batch_size, num_history, -1)
        
        # User Attention
        att_score = self.attention_linear(news_vectors)
        att_weight = F.softmax(att_score, dim=1)
        
        user_vector = torch.sum(news_vectors * att_weight, dim=1)
        return user_vector

class MINDRecModel(nn.Module):
    def __init__(self, num_words, word_embed_dim=300, num_filters=400):
        super(MINDRecModel, self).__init__()
        self.news_encoder = NewsEncoder(num_words, word_embed_dim, num_filters)
        self.user_encoder = UserEncoder(self.news_encoder, num_filters)

    def forward(self, history, candidate):
        user_vector = self.user_encoder(history)
        candidate_vector = self.news_encoder(candidate)
        
        # Tính Dot Product Score
        score = torch.sum(user_vector * candidate_vector, dim=1)
        return score