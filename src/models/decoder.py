# src/models/decoder.py
"""
Decoder with additive attention for EN→DE translation.

Shapes (per step):
- input token ids: [batch]
- embedded: [1, batch, DEC_EMB_DIM]
- context vector: [1, batch, HID_DIM]
- LSTM output: [1, batch, HID_DIM]
- prediction (logits): [batch, OUTPUT_DIM]
"""
import torch
import torch.nn as nn
from .attention import Attention
from .constants import OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, TRG_PAD_IDX, DEVICE

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = OUTPUT_DIM
        # Target embedding
        self.embedding = nn.Embedding(OUTPUT_DIM, DEC_EMB_DIM, padding_idx=TRG_PAD_IDX)

        # LSTM input is concat(embedding, context)
        self.rnn = nn.LSTM(DEC_EMB_DIM + HID_DIM, HID_DIM, N_LAYERS, dropout=DEC_DROPOUT)

        # Projection to vocab
        self.fc_out = nn.Linear(HID_DIM + HID_DIM, OUTPUT_DIM)

        self.dropout = nn.Dropout(DEC_DROPOUT)
        self.attention = Attention()

    # def forward(self, input, hidden, cell):
    #     # input: [batch_size] → một token tại 1 thời điểm
    #     input = input.unsqueeze(0)  # → [1, batch_size]
        
    #     embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]
        
    #     output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
    #     # output: [1, batch, hid_dim]
        
    #     prediction = self.fc_out(output.squeeze(0))  # [batch, output_dim]
    #     return prediction, hidden, cell
    
    def forward(self, input, hidden, cell, encoder_outputs):
        """One decoding step with attention.

        Args:
            input: [batch]
            hidden, cell: [n_layers, batch, HID_DIM]
            encoder_outputs: [src_len, batch, HID_DIM]

        Returns:
            prediction: [batch, OUTPUT_DIM]
            hidden, cell: [n_layers, batch, HID_DIM]
        """
        input = input.unsqueeze(0)  # [1, batch]

        embedded = self.dropout(self.embedding(input))           # [1, batch, DEC_EMB_DIM]
        context_vector, _ = self.attention(hidden, encoder_outputs)  # [1, batch, HID_DIM]

        rnn_input = torch.cat((embedded, context_vector), dim=2) # [1, batch, DEC_EMB_DIM + HID_DIM]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell)) # output: [1, batch, HID_DIM]

        final_hidden_state = torch.cat((output.squeeze(0), context_vector.squeeze(0)), dim=1) # [batch, HID_DIM*2]
        prediction = self.fc_out(final_hidden_state) # [batch, OUTPUT_DIM]
        return prediction, hidden, cell