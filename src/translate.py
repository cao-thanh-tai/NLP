# src/translate.py
"""
Inference script: translate EN→DE with trained Seq2Seq + Attention.

Flow:
- tokenize English, add <sos>/<eos>
- encode → encoder_outputs, hidden, cell
- greedy decode with attention until <eos> or max_len

Shapes:
- src_tensor: [src_len, 1]
- encoder_outputs: [src_len, 1, HID_DIM]
- decoder output logits: [1, OUTPUT_DIM] per step
"""
import torch
from models.model_builder import build_model
from models.constants import DEVICE, SOS_IDX, EOS_IDX, en_vocab, de_vocab
from torchtext.data.utils import get_tokenizer

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Load model đã train
model = build_model()
model.load_state_dict(torch.load("models/best_model/best_seq2seq.pt", map_location=DEVICE))
model.eval()

def translate_sentence(sentence: str, max_len=60):
    """Greedy translate one English sentence to German."""
    tokens = [token.lower() for token in en_tokenizer(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [en_vocab.get_stoi().get(token, en_vocab['<unk>']) for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(DEVICE)  # [src_len, 1]
    src_len = torch.LongTensor([len(src_indexes)]).cpu()
    
    with torch.no_grad():
        # Encode: trả về encoder_outputs (cho attention), hidden, cell
        # encoder_outputs: [src_len, 1, hid_dim]
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    trg_indexes = [SOS_IDX]
    for _ in range(max_len):
        input_tensor = torch.LongTensor([trg_indexes[-1]]).to(DEVICE)
        
        with torch.no_grad():
            # Decode step với attention: output logits [1, output_dim]
            output, hidden, cell = model.decoder(input_tensor, hidden, cell, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == EOS_IDX:
            break
    
    translated_tokens = [de_vocab.get_itos()[i] for i in trg_indexes[1:]]  # bỏ <sos>
    return ' '.join(translated_tokens).replace(" <eos>", "")


def translate_sentence_beam(sentence: str, beam_width=3, max_len=60):
    """
    Beam search translate one English sentence to German.
    
    Giữ top-k candidates tại mỗi bước thay vì chỉ 1 (greedy).
    Giúp tìm được câu dịch tốt hơn bằng cách explore nhiều đường đi.
    
    Args:
        sentence: Câu tiếng Anh (str)
        beam_width: Số lượng candidates giữ lại mỗi bước
        max_len: Độ dài tối đa
    
    Returns:
        Câu tiếng Đức đã dịch (str)
    """
    tokens = [token.lower() for token in en_tokenizer(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [en_vocab.get_stoi().get(token, en_vocab['<unk>']) for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(DEVICE)  # [src_len, 1]
    src_len = torch.LongTensor([len(src_indexes)]).cpu()
    
    with torch.no_grad():
        # Encode 1 lần duy nhất
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)
    
    # Khởi tạo beam: (token_sequence, log_prob, hidden, cell, completed)
    beams = [([SOS_IDX], 0.0, hidden, cell, False)]
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score, h, c, completed in beams:
            # Nếu sequence đã kết thúc, giữ nguyên
            if completed:
                all_candidates.append((seq, score, h, c, True))
                continue
            
            # Token cuối cùng
            input_tensor = torch.LongTensor([seq[-1]]).to(DEVICE)
            
            with torch.no_grad():
                # Decode: output = [1, output_dim] (logits)
                output, new_h, new_c = model.decoder(input_tensor, h, c, encoder_outputs)
                # Log probabilities
                log_probs = torch.log_softmax(output, dim=1)
            
            # Lấy top-k tokens
            topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=1)
            
            # Expand beam: tạo k candidates mới
            for i in range(beam_width):
                token = topk_indices[0][i].item()
                token_log_prob = topk_log_probs[0][i].item()
                
                new_seq = seq + [token]
                new_score = score + token_log_prob
                is_completed = (token == EOS_IDX)
                
                all_candidates.append((new_seq, new_score, new_h, new_c, is_completed))
        
        # Sắp xếp và chọn top beam_width candidates
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        
        # Dừng sớm nếu tất cả beams đã completed
        if all(completed for _, _, _, _, completed in beams):
            break
    
    # Chọn sequence tốt nhất
    best_seq = beams[0][0]
    translated_tokens = [de_vocab.get_itos()[i] for i in best_seq[1:]]  # bỏ <sos>
    return ' '.join(translated_tokens).replace(" <eos>", "")

if __name__ == "__main__":
    test_sentences = [
        "A man is playing a guitar.",
        "Two young girls are playing with a dog.",
        "People are sitting on the beach.",
        "A woman is slicing a tomato.",
        "To operate a case of neurosis you need to be a professional"
    ]
    
    print("="*60)
    print("GREEDY DECODING")
    print("="*60)
    for sent in test_sentences:
        de = translate_sentence(sent)
        print(f"EN: {sent}")
        print(f"DE: {de}\n")
    
    print("="*60)
    print("BEAM SEARCH (beam_width=3)")
    print("="*60)
    for sent in test_sentences:
        de = translate_sentence_beam(sent, beam_width=3)
        print(f"EN: {sent}")
        print(f"DE: {de}\n")