# src/models/model_builder.py
"""Helper xây Seq2Seq, khởi tạo trọng số và in số tham số."""
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2Seq
from .constants import DEVICE

def build_model():
    """
    Tạo encoder-decoder và khởi tạo Xavier cho các trọng số 2D+ (Linear/Embedding/LSTM).

    - Encoder/Decoder lấy cấu hình từ module riêng của chúng.
    - Xavier init áp dụng cho tensor có dim > 1 để phân phối đều phương sai.

    Returns:
        model: Seq2Seq trên đúng DEVICE
    """
    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    def init_weights(m):
        # Apply Xavier uniformly to modules with 2D+ weights (Linear/Embedding/LSTM matrices)
        if hasattr(m, "weight") and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built! Trainable parameters: {total_params:,}")

    return model