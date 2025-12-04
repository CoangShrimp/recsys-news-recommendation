from preprocess import transform_text_to_sequence, MAX_TITLE_LENGTH
from load_data import df_news, df_behaviors, parse_impressions
class MINDTrainDataset(Dataset):
    def __init__(self, behaviors_df):
        self.behaviors = behaviors_df

    def __len__(self):
        # Tổng số mẫu dữ liệu là tổng số dòng trong file behaviors
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        
        # 1. XỬ LÝ LỊCH SỬ (HISTORY)
        history_ids = str(row['history']).split(' ')
        history_seqs = []
        
        # Lấy tối đa 50 bài gần nhất
        if len(history_ids) > MAX_HISTORY_LENGTH:
            history_ids = history_ids[-MAX_HISTORY_LENGTH:]
            
        for nid in history_ids:
            # Lấy sequence từ cache, nếu bài báo bị xóa/lỗi thì lấy rỗng
            history_seqs.append(news_title_sequences.get(nid, empty_news_sequence))
            
        # Nếu chưa đủ 50 bài thì độn thêm bài rỗng vào đầu (Padding History)
        while len(history_seqs) < MAX_HISTORY_LENGTH:
            history_seqs.insert(0, empty_news_sequence)
            
        # 2. XỬ LÝ ỨNG VIÊN (CANDIDATE)
        # Trong file behaviors, một dòng có nhiều ứng viên (impression)
        # Để đơn giản cho việc train, mỗi lần lấy dữ liệu ta chọn ngẫu nhiên 1 ứng viên để học
        impressions = parse_impressions(row['impressions'])
        selected_candidate, label = random.choice(impressions)
        
        candidate_seq = news_title_sequences.get(selected_candidate, empty_news_sequence)
        
        # 3. CHUYỂN VỀ TENSOR (Định dạng của PyTorch)
        # history_seqs: [50, 30] (50 bài, mỗi bài 30 từ)
        # candidate_seq: [30] (1 bài, 30 từ)
        # label: [1] (0 hoặc 1)
        return (torch.tensor(history_seqs, dtype=torch.long), 
                torch.tensor(candidate_seq, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float))

# --- TEST THỬ DATASET ---
train_dataset = MINDTrainDataset(df_behaviors)
sample_history, sample_candidate, sample_label = train_dataset[0]

print("\n--- KIỂM TRA MỘT MẪU DỮ LIỆU ---")
print(f"Kích thước tensor Lịch sử: {sample_history.shape} (Mong đợi: 50x30)")
print(f"Kích thước tensor Ứng viên: {sample_candidate.shape} (Mong đợi: 30)")
print(f"Nhãn (Label): {sample_label} (0: Không click, 1: Click)")