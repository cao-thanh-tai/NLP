# src/models/seq2seq.py
import torch
import torch.nn as nn
import random
from .encoder import Encoder
from .decoder import Decoder
from .constants import DEVICE, TRG_SOS_IDX, TRG_EOS_IDX, TEACHER_FORCING_RATIO

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = DEVICE

    def forward(self, src, src_len, trg=None, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        """
        Run encoder-decoder with optional teacher forcing.

        Args:
            src: [src_len, batch] - input sentence (source)
            src_len: [batch] - lengths (sorted desc)
            trg: [trg_len, batch] or None - target sentence during training
            teacher_forcing_ratio: float - prob to use ground truth token

        Returns:
            outputs: [trg_len, batch, output_dim] (padded/early-stopped)
        """
        
        batch_size = src.shape[1]
        max_len = trg.shape[0] if trg is not None else 50
        
        # Encoder trả về: outputs (cho attention), hidden, cell
        # encoder_outputs: [src_len, batch, hid_dim]
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        
        # Tensor lưu kết quả
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim, device=self.device)
        
        input_token = torch.full((batch_size,), TRG_SOS_IDX, dtype=torch.long, device=self.device)
        
        
        for t in range(max_len):
            # Decoder step: output logits and next hidden/cell
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)

            outputs[t] = output  # [batch, output_dim]

            # Greedy token
            top1 = output.argmax(1)

            # Teacher forcing decision
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = trg[t] if (trg is not None and teacher_force) else top1

            # Early stop nếu tất cả đều ra <eos>
            if not teacher_force and (input_token == TRG_EOS_IDX).all():
                break

        return outputs