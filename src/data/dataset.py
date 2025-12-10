# src/data/dataset.py
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os

# ====================== Load Vocabulary và Tokenizer ======================
# Load vocab đã build ở bước trước (file build_vocab.py)
# Vocab object: ánh xạ {token -> index}
en_vocab = torch.load("models/vocab/en_vocab.pt")
de_vocab = torch.load("models/vocab/de_vocab.pt")

# Tokenizer để tách câu thành list token
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

# Source (english) indices
SRC_PAD_IDX = en_vocab['<pad>']
SRC_SOS_IDX = en_vocab['<sos>']
SRC_EOS_IDX = en_vocab['<eos>']

# Target (de) indices
TRG_PAD_IDX = de_vocab['<pad>']
TRG_SOS_IDX = de_vocab['<sos>']
TRG_EOS_IDX = de_vocab['<eos>']

def tokenize_and_numericalize(en_sent, de_sent):
    """
    Chuyển đổi câu văn (string) thành tensor chứa indices
    
    Args:
        en_sent: string - câu tiếng Anh (source)
        de_sent: string - câu tiếng Đức (target)
    
    Returns:
        en_tokens: tensor [seq_len] - indices của câu nguồn (dùng en_vocab)
        de_tokens: tensor [seq_len] - indices của câu đích (dùng de_vocab)
        
    Example:
        Input: "I love you", "Ich liebe dich"
        Output: tensor([2, 45, 123, 67, 3]), tensor([2, 89, 234, 156, 3])
                        ↑ SRC_SOS           ↑ TRG_SOS, TRG_EOS
    """
    # QUAN TRỌNG: Dùng SRC_* cho nguồn (en), TRG_* cho đích (de)
    en_tokens = [SRC_SOS_IDX] + [en_vocab[token] for token in en_tokenizer(en_sent)] + [SRC_EOS_IDX]
    de_tokens = [TRG_SOS_IDX] + [de_vocab[token] for token in de_tokenizer(de_sent)] + [TRG_EOS_IDX]
    return torch.tensor(en_tokens, dtype=torch.long), torch.tensor(de_tokens, dtype=torch.long)

# ====================== Dataset Class ======================
class Multi30kDataset(torch.utils.data.Dataset):
    """
    Wrapper cho Multi30k dataset
    
    Output mỗi sample:
        src: tensor [src_len] - câu nguồn đã numericalize
        trg: tensor [trg_len] - câu đích đã numericalize
        en_sent: string - câu tiếng Anh gốc (để debug/visualize)
        de_sent: string - câu tiếng Đức gốc (để debug/visualize)
    """
    def __init__(self, split='train'):
        """Load toàn bộ dataset vào memory (Multi30k nhỏ nên OK)"""
        self.data = list(Multi30k(split=split, language_pair=('en', 'de')))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Trả về 1 cặp câu đã được numericalize"""
        en_sent, de_sent = self.data[idx]
        src, trg = tokenize_and_numericalize(en_sent, de_sent)
        return src, trg, en_sent, de_sent
    


# ====================== Collate Function ======================
# SIÊU QUAN TRỌNG: Hàm này xử lý batch data từ DataLoader
def collate_fn(batch):
    """
    Xử lý 1 batch data: padding, sorting theo độ dài
    
    Args:
        batch: list of tuples từ Dataset.__getitem__
               [(src1, trg1, en_sent1, de_sent1), (src2, trg2, ...), ...]
               Mỗi src/trg là tensor 1D với độ dài khác nhau
    
    Returns:
        src_batch: tensor [src_len, batch_size] - câu nguồn đã pad
        trg_batch: tensor [trg_len, batch_size] - câu đích đã pad
        lengths_src: tensor [batch_size] - độ dài thực của mỗi câu nguồn
    
    Example:
        Input batch size 3:
            src1: [2, 45, 123, 3]           (len=4)
            src2: [2, 89, 234, 156, 78, 3]  (len=6)
            src3: [2, 12, 3]                (len=3)
        
        Output sau padding và sorting:
            src_batch: [[2,   2,   2  ],    # <sos>
                        [89,  45,  12 ],
                        [234, 123, 3  ],    # <eos> cho src3
                        [156, 3,   1  ],    # <eos> cho src1, <pad> cho src3
                        [78,  1,   1  ],    # <pad>
                        [3,   1,   1  ]]    # <eos> cho src2
            Shape: [6, 3] = [max_src_len, batch_size]
            lengths_src: [6, 4, 3] - đã sort giảm dần
    """
    src_batch, trg_batch = [], []
    for src, trg, _, _ in batch:
        src_batch.append(src)
        trg_batch.append(trg)
    
    # BƯỚC 1: Padding - đưa các câu khác độ dài về cùng độ dài
    # batch_first=False → output shape: [seq_len, batch_size]
    # QUAN TRỌNG: Dùng SRC_PAD_IDX cho nguồn, TRG_PAD_IDX cho đích
    src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX, batch_first=False)
    trg_batch = pad_sequence(trg_batch, padding_value=TRG_PAD_IDX, batch_first=False)
    
    # BƯỚC 2: Tính độ dài thực của mỗi câu (không tính padding)
    # Cần cho pack_padded_sequence trong Encoder
    lengths_src = torch.tensor([
        (seq != SRC_PAD_IDX).sum().item()  # Đếm số token KHÔNG phải SRC_PAD_IDX
        for seq in src_batch.T  # src_batch.T: [batch_size, seq_len]
    ], dtype=torch.long)
    
    # BƯỚC 3: Sort theo độ dài giảm dần (yêu cầu của pack_padded_sequence)
    sorted_idx = lengths_src.argsort(descending=True)
    src_batch = src_batch[:, sorted_idx]      # [src_len, batch_size]
    trg_batch = trg_batch[:, sorted_idx]      # [trg_len, batch_size]
    lengths_src = lengths_src[sorted_idx]     # [batch_size]
    
    return src_batch, trg_batch, lengths_src

# Test ngay lập tức
if __name__ == "__main__":
    print("Đang test DataLoader + 1 batch trên GPU...")
    
    train_dataset = Multi30kDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    for src, trg, lengths in train_loader:
        src = src.to(device)
        trg = trg.to(device)
        
        print(src)
        
        print(f"Batch shape  - src: {src.shape} (seq_len, batch_size)")
        print(f"Batch shape  - trg: {trg.shape}")
        print(f"Lengths src  : {lengths.tolist()}")
        print(f"Sample src[0]: {src[:, 0].tolist()[:20]}... → decode: {[en_vocab.get_itos()[i] for i in src[:, 0].tolist()[:10]]}")
        print("BATCH ĐẦU TIÊN CHẠY NGON LÀNH!")
        break