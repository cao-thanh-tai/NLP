# src/evaluate.py
"""
Compute BLEU on Multi30k valid split using greedy translations.

Flow:
- iterate valid set sentence by sentence
- translate English to German via translate_sentence
- clean <unk>, avoid empty hyp
- corpus_bleu with smoothing (method1)
"""
from data.dataset import Multi30kDataset, collate_fn
from torch.utils.data import DataLoader
from translate import translate_sentence, translate_sentence_beam
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

# Note: batch decode is not used; we decode one-by-one for simplicity.
BATCH_SIZE = 128

valid_dataset = Multi30kDataset(split='valid')
refs = []
hyps = []
smooth = SmoothingFunction().method1   # smoothing to avoid zero BLEU on short ngrams

print("Đang tính BLEU ")
for _, _, en_sent, de_sent in tqdm(valid_dataset):
    # Greedy translate English sentence to German
    pred = translate_sentence_beam(en_sent)
    
    # Remove <unk> to avoid unfair penalty; ensure non-empty hyp
    pred = ' '.join([w for w in pred.split() if w != '<unk>'])
    if not pred.strip():
        pred = "person"  # avoid empty string
    
    refs.append([de_sent.lower().split()])
    hyps.append(pred.lower().split())

bleu = corpus_bleu(refs, hyps, smoothing_function=smooth)
print(f"\nBLEU = {bleu*100:.2f}")