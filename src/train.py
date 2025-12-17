# src/train.py
"""
Train Seq2Seq EN→DE on Multi30k with teacher forcing and val early-save.

Shapes:
- src: [src_len, batch]
- trg: [trg_len, batch]
- model output: [trg_len, batch, output_dim]
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.dataset import Multi30kDataset, collate_fn
from models.model_builder import build_model
from models.constants import DEVICE, TRG_PAD_IDX
from tqdm import tqdm
import os

# ------------------- Config -------------------
# BATCH_SIZE ảnh hưởng tốc độ/VRAM; CLIP chống exploding gradients.
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
CLIP = 1.0
SAVE_DIR = "models/best_model"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ------------------- Data -------------------
# collate_fn trả về: src[seq_len,batch], trg[trg_len,batch], lengths[src]
train_dataset = Multi30kDataset(split='train')
valid_dataset = Multi30kDataset(split='valid')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ------------------- Model -------------------
model = build_model()
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)  # ignore padding của target
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------- Training Loop -------------------
best_valid_loss = float('inf')
train_history = []
valid_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for src, trg, src_len in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, src_len, trg)  # [trg_len, batch, output_dim]

        # Reshape cho CrossEntropyLoss: ghép time/batch thành một trục
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)   # bỏ <sos>
        trg = trg[1:].reshape(-1)                     # bỏ <sos>
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_history.append(avg_train_loss)
    
    # ------------------- Validation -------------------
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for src, trg, src_len in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VALID]"):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, src_len, trg, teacher_forcing_ratio=0.0)  # không TF khi valid
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = criterion(output, trg)
            valid_loss += loss.item()
    
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_history.append(avg_valid_loss)
    
    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
    
    # Save best model
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_seq2seq.pt")
        print("Saved new best model!")


# ------------------- Plot Loss Curves -------------------
plt.figure()
plt.plot(train_history, label="Train Loss")
plt.plot(valid_history, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Valid Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_curve.png")
