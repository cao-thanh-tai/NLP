# src/models/model_builder.py
"""Helper to build Seq2Seq model with weight init and param count."""
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2Seq
from .constants import DEVICE

def build_model():
    """
    Construct encoder-decoder model and apply Xavier init to linear/embedding weights.

    Returns:
        model: Seq2Seq on correct DEVICE
    """
    encoder = Encoder()
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder).to(DEVICE)

    def init_weights(m):
        # Apply Xavier uniformly to modules with 2D+ weights (e.g., Linear, Embedding)
        if hasattr(m, "weight") and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(init_weights)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built! Trainable parameters: {total_params:,}")

    return model