# src/models/encoder.py
"""
Encoder: Xử lý câu nguồn (source sentence) và tạo ra:
- Encoder outputs: trạng thái ẩn tại mỗi time step (dùng cho Attention)
- Hidden & Cell state: trạng thái cuối cùng (khởi tạo cho Decoder)
"""
import torch
import torch.nn as nn
from .constants import INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, SRC_PAD_IDX, DEVICE

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding layer: chuyển token indices thành dense vectors
        # INPUT_DIM: vocab size (~10k-20k)
        # ENC_EMB_DIM: embedding dimension (256)
        # padding_idx: index của <pad> sẽ có embedding = zero vector
        self.embedding = nn.Embedding(INPUT_DIM, ENC_EMB_DIM, padding_idx=SRC_PAD_IDX)
        
        # LSTM layers: xử lý sequence
        # input_size: ENC_EMB_DIM (256)
        # hidden_size: HID_DIM (512)
        # num_layers: N_LAYERS (2) - stacked LSTM
        # dropout: áp dụng giữa các layers (không áp dụng cho layer cuối)
        self.rnn = nn.LSTM(ENC_EMB_DIM, HID_DIM, N_LAYERS, dropout=ENC_DROPOUT)
        
        # Dropout layer: áp dụng sau embedding
        self.dropout = nn.Dropout(ENC_DROPOUT)

    def forward(self, src, src_len):
        """
        Xử lý câu nguồn qua Encoder
        
        Args:
            src: tensor [src_len, batch_size] - câu nguồn (token indices)
                 Ví dụ: [[2, 2, 2],      # <sos>
                         [45, 89, 12],    # token
                         [123, 234, 3],   # token / <eos>
                         [3, 156, 1],     # <eos> / token / <pad>
                         ...]
            
            src_len: tensor [batch_size] - độ dài thực của mỗi câu (đã sort giảm dần)
                     Ví dụ: [8, 6, 4] - câu đầu dài 8 tokens, câu 2 dài 6, câu 3 dài 4
        
        Returns:
            outputs: tensor [src_len, batch_size, hid_dim]
                     Trạng thái ẩn tại MỖI time step - dùng cho Attention
                     Shape: [src_len, batch, 512]
            
            hidden: tensor [n_layers, batch_size, hid_dim]
                    Hidden state cuối cùng của LSTM - khởi tạo cho Decoder
                    Shape: [2, batch, 512]
            
            cell: tensor [n_layers, batch_size, hid_dim]
                  Cell state cuối cùng của LSTM - khởi tạo cho Decoder
                  Shape: [2, batch, 512]
        """
        # BƯỚC 1: Embedding - chuyển token indices thành dense vectors
        # src: [src_len, batch] → embedded: [src_len, batch, 256]
        embedded = self.dropout(self.embedding(src))
        
        # BƯỚC 2: Pack sequence để LSTM bỏ qua padding (tăng tốc)
        # pack_padded_sequence loại bỏ các padding tokens khỏi computation
        # Yêu cầu: src_len phải được sort giảm dần (đã làm trong collate_fn)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=True)
        
        # BƯỚC 3: Chạy LSTM trên packed sequence
        # packed_outputs: PackedSequence chứa outputs tại mỗi time step
        # hidden, cell: [n_layers, batch, hid_dim] - trạng thái cuối cùng
        packed_outputs, (hidden, cell) = self.rnn(packed)
        
        # BƯỚC 4: Unpack để lấy lại format tensor bình thường
        # outputs: [src_len, batch_size, hid_dim] - trạng thái ẩn tại MỖI time step
        # Các vị trí padding sẽ có giá trị = 0 (zero vector)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        # outputs: [src_len, batch, 512] - dùng cho Attention trong Decoder
        # hidden: [2, batch, 512] - khởi tạo hidden state cho Decoder
        # cell: [2, batch, 512] - khởi tạo cell state cho Decoder
        return outputs, hidden, cell