# src/data/build_vocab.py
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

# ====================== 1. Khởi tạo Tokenizer ======================
# Tokenizer dùng để tách câu thành list các token (từ)
# Input: string (câu văn)
# Output: list of strings (danh sách token)
print("Đang khởi tạo tokenizer...")
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

# Test nhanh tokenizer
print("Test tokenizer:")
# Output ví dụ: ['Two', 'young', 'guys', 'are', 'playing', 'soccer', '!']
print("EN:", en_tokenizer("Two young guys are playing soccer!"))
print("DE:", de_tokenizer("Zwei junge Männer spielen Fußball!"))
print()

# ====================== 2. Xây dựng Vocabulary ======================
def yield_tokens(data_iter, language='en'):
    """
    Generator function để yield từng câu đã tokenize
    
    Args:
        data_iter: Iterator từ Multi30k dataset
                   Mỗi phần tử là tuple (en_sentence, de_sentence)
        language: 'en' hoặc 'de'
    
    Yields:
        list of strings: Danh sách token của mỗi câu
                        Ví dụ: ['Two', 'young', 'guys', ...]
    """
    for sent in data_iter:
        if language == 'en':
            yield en_tokenizer(sent[0])   # sent[0]: câu tiếng Anh (string)
        else:
            yield de_tokenizer(sent[1])   # sent[1]: câu tiếng Đức (string)

print("Đang build vocabulary từ tập train... (khoảng 20-30 giây)")

# Lấy dataset Multi30k train split
# Multi30k chứa ~29,000 cặp câu (en, de)
train_iter = Multi30k(split='train', language_pair=('en', 'de'))

# ====================== Build EN Vocabulary ======================
# build_vocab_from_iterator sẽ:
# 1. Duyệt qua tất cả các token từ yield_tokens()
# 2. Đếm tần suất xuất hiện của mỗi token
# 3. Chỉ giữ lại token có min_freq >= 2
# 4. Thêm 4 special tokens vào đầu vocab
#
# Output: Vocab object - một ánh xạ {token -> index}
# Ví dụ: '<unk>':0, '<pad>':1, '<sos>':2, '<eos>':3, 'the':4, 'a':5, ...
en_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, 'en'), 
    min_freq=2,                     # chỉ lấy từ xuất hiện >= 2 lần
    specials=['<unk>', '<pad>', '<sos>', '<eos>'],
    special_first=True              # đặt special tokens ở đầu
)
# Set default index cho token không có trong vocab
en_vocab.set_default_index(en_vocab['<unk>'])  # index 0

# ====================== Build DE Vocabulary ======================
# Phải tạo lại iterator vì iterator đã hết sau lượt duyệt EN
train_iter = Multi30k(split='train', language_pair=('en', 'de'))
de_vocab = build_vocab_from_iterator(
    yield_tokens(train_iter, 'de'),
    min_freq=2,
    specials=['<unk>', '<pad>', '<sos>', '<eos>'],
    special_first=True
)
de_vocab.set_default_index(de_vocab['<unk>'])

# ====================== 3. In thông tin + lưu lại ======================
print(f"EN vocab size: {len(en_vocab):,}")
print(f"DE vocab size: {len(de_vocab):,}")
print(f"PAD token index : {en_vocab['<pad>']}")
print(f"UNK token index : {en_vocab['<unk>']}")
print(f"SOS token index : {en_vocab['<sos>']}")
print(f"EOS token index : {en_vocab['<eos>']}")
print()

print("10 từ đầu tiên trong EN vocab:")
print(list(en_vocab.get_itos()[:20]))
print("10 từ đầu tiên trong DE vocab:")
print(list(de_vocab.get_itos()[:20]))

# Lưu vocab để dùng sau (rất quan trọng!)
import os
os.makedirs("models/vocab", exist_ok=True)

torch.save(en_vocab, "models/vocab/en_vocab.pt")
torch.save(de_vocab, "models/vocab/de_vocab.pt")


print("\nĐÃ LƯU VOCAB VÀO:")
print("   → models/vocab/en_vocab.pt")
print("   → models/vocab/de_vocab.pt")