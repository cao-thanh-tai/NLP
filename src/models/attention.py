# src/models/attention.py
import torch
import torch.nn as nn
from .constants import HID_DIM  # HID_DIM là kích thước hidden state của LSTM/GRU

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer tuyến tính để biến đổi trạng thái ẩn của Encoder và Decoder
        # W1: áp dụng cho hidden state của Encoder (encoder_outputs)
        self.attn_w1 = nn.Linear(HID_DIM, HID_DIM)
        # W2: áp dụng cho hidden state của Decoder (Decoder hidden state hiện tại)
        self.attn_w2 = nn.Linear(HID_DIM, HID_DIM)
        
        # Layer tuyến tính cuối cùng (V) để tính Energy Score
        self.v = nn.Linear(HID_DIM, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        Bahdanau-style additive attention.
        
        Args:
            decoder_hidden: [n_layers, batch_size, hid_dim]
                Hidden state của decoder (toàn bộ layers); chỉ dùng layer cuối cùng.
            encoder_outputs: [src_len, batch_size, hid_dim]
                Tất cả hidden states từ encoder (dùng để tính attention).
        
        Returns:
            context_vector: [1, batch_size, hid_dim]
                Kết quả weighted sum của encoder_outputs theo attention weights.
            attention_weights: [src_len, batch_size]
                Phân phối softmax theo chiều src_len.
        """
        # Lấy hidden state của layer cuối cùng của decoder
        decoder_hidden = decoder_hidden[-1].unsqueeze(0) 
        
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # Mở rộng hidden state của Decoder để khớp với chiều dài câu nguồn (src_len)
        # hidden_rep: [src_len, batch_size, hid_dim]
        hidden_rep = decoder_hidden.repeat(src_len, 1, 1)

        # Tính toán Energy Score (e)
        # e = V * tanh(W1 * Encoder_Outputs + W2 * Decoder_Hidden)
        
        # 1. Tính toán Alignment Scores (sau khi áp dụng W1 và W2)
        # attn_w1(encoder_outputs): [src_len, batch, hid_dim]
        # attn_w2(hidden_rep): [src_len, batch, hid_dim]
        
        energy = torch.tanh(self.attn_w1(encoder_outputs) + self.attn_w2(hidden_rep))
        
        # 2. Tính toán Energy (e) - ánh xạ xuống chiều 1
        # energy: [src_len, batch, 1]
        attention_energy = self.v(energy)
        
        # 3. Tính toán Trọng số Attention (Attention Weights - a)
        # attention_energy.squeeze(2): [src_len, batch_size]
        # a = softmax(e)
        attention_weights = torch.softmax(attention_energy.squeeze(2), dim=0)
        
        # 4. Tính toán Context Vector (Context Vector - c)
        # context_vector = sum(attention_weights * encoder_outputs)
        
        # attention_weights.unsqueeze(2): [src_len, batch_size, 1]
        # encoder_outputs: [src_len, batch_size, hid_dim]
        # context_vector: [src_len, batch_size, hid_dim] -> [1, batch_size, hid_dim] (sau khi sum)
        
        context_vector = (attention_weights.unsqueeze(2) * encoder_outputs).sum(dim=0).unsqueeze(0)
        
        # Trả về Context Vector và Trọng số Attention (để gỡ lỗi)
        # context_vector: [1, batch_size, hid_dim]
        # attention_weights: [src_len, batch_size]
        return context_vector, attention_weights