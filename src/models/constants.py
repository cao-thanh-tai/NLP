# src/models/constants.py
"""
File chứa tất cả các hằng số, hyperparameters và vocab indices
Sử dụng: from models.constants import DEVICE, SRC_PAD_IDX, HID_DIM, ...
"""
import torch
from pathlib import Path

# ====================== Device Configuration ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== Vocabulary Loading ======================
# Đường dẫn vocab (relative từ thư mục gốc dự án)
VOCAB_DIR = Path("models/vocab")
EN_VOCAB_PATH = VOCAB_DIR / "en_vocab.pt"
DE_VOCAB_PATH = VOCAB_DIR / "de_vocab.pt"

# Load vocab objects (ánh xạ token -> index)
en_vocab = torch.load(EN_VOCAB_PATH)
de_vocab = torch.load(DE_VOCAB_PATH)

# ====================== Vocabulary Dimensions ======================
INPUT_DIM = len(en_vocab)   # Kích thước vocab nguồn (English) - dùng cho Encoder embedding
OUTPUT_DIM = len(de_vocab)  # Kích thước vocab đích (German) - dùng cho Decoder embedding + output layer

# ====================== Special Token Indices ======================
# QUAN TRỌNG: Phân biệt rõ ràng giữa source và target vocab indices

# Source (English) special token indices - Dùng cho:
#   - Encoder embedding padding_idx
#   - Dataset: thêm <sos>, <eos> vào câu nguồn
#   - Collate: padding câu nguồn
SRC_PAD_IDX = en_vocab['<pad>']  # Thường là index 1
SRC_SOS_IDX = en_vocab['<sos>']  # Thường là index 2
SRC_EOS_IDX = en_vocab['<eos>']  # Thường là index 3

# Target (German) special token indices - Dùng cho:
#   - Decoder embedding padding_idx
#   - Dataset: thêm <sos>, <eos> vào câu đích
#   - Collate: padding câu đích
#   - Seq2Seq: khởi tạo input_token = TRG_SOS_IDX
#   - Seq2Seq: early stopping khi gặp TRG_EOS_IDX
#   - Loss function: ignore_index = TRG_PAD_IDX
TRG_PAD_IDX = de_vocab['<pad>']  # Thường là index 1
TRG_SOS_IDX = de_vocab['<sos>']  # Thường là index 2
TRG_EOS_IDX = de_vocab['<eos>']  # Thường là index 3

# Backwards-compatible aliases (để code cũ không bị lỗi)
# KHUYẾN NGHỊ: Dùng SRC_*/TRG_* thay vì PAD_IDX/SOS_IDX/EOS_IDX
PAD_IDX = SRC_PAD_IDX
SOS_IDX = SRC_SOS_IDX
EOS_IDX = SRC_EOS_IDX

# ====================== Model Hyperparameters ======================
# Cấu hình mô hình Seq2Seq với Attention

# --- Embedding Dimensions ---
ENC_EMB_DIM = 256  # Kích thước embedding cho Encoder (source language)
                   # Encoder input: [seq_len, batch] → Embedding: [seq_len, batch, 256]

DEC_EMB_DIM = 256  # Kích thước embedding cho Decoder (target language)
                   # Decoder input: [1, batch] → Embedding: [1, batch, 256]

# --- Hidden State Dimensions ---
HID_DIM = 512      # Kích thước hidden state của LSTM (cho cả Encoder và Decoder)
                   # Encoder hidden/cell: [n_layers, batch, 512]
                   # Decoder hidden/cell: [n_layers, batch, 512]
                   # Encoder outputs: [src_len, batch, 512]
                   # Context vector: [1, batch, 512]

# --- Network Architecture ---
N_LAYERS = 2       # Số lượng layer của LSTM (stacked LSTM)
                   # N_LAYERS = 2 nghĩa là có 2 tầng LSTM chồng lên nhau

# --- Regularization ---
ENC_DROPOUT = 0.5  # Dropout rate cho Encoder (áp dụng giữa các layer và sau embedding)
DEC_DROPOUT = 0.5  # Dropout rate cho Decoder (áp dụng giữa các layer và sau embedding)
                   # Dropout = 0.5 nghĩa là ngẫu nhiên tắt 50% neurons trong training

# --- Training Strategy ---
TEACHER_FORCING_RATIO = 0.5  # Tỷ lệ sử dụng teacher forcing khi training
                             # 0.5 = 50% thời gian dùng ground truth, 50% dùng prediction
                             # Teacher forcing: dùng target thực thay vì output của decoder ở bước trước