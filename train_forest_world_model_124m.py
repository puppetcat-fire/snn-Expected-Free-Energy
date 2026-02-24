# -*- coding: utf-8 -*-
"""
森林冒险世界模型升级版：GPT2-small (124M) fine-tune
====================================================================
- 从 HuggingFace 加载预训练 GPT2
- resize 到你的 57 token vocab
- 继续在你的 forest_delayed_data_v2.pt 上 fine-tune
====================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

DATA_PATH = "forest_delayed_data_v2.pt"
BATCH_SIZE = 4          # 124M 模型，显存安全值
EPOCHS = 12
LR = 2e-5               # 预训练模型用小学习率

# ================== 加载数据 ==================
data_dict = torch.load(DATA_PATH, map_location=device, weights_only=True)
sequences = data_dict["sequences"]  # [8000, 36]
vocab_size = 57

class ForestDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i].long()

dataset = ForestDataset(sequences)
train_size = int(0.9 * len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ================== 加载预训练 GPT2 + resize ==================
print("正在加载预训练 GPT2-small (124M)...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.resize_token_embeddings(vocab_size)   # 关键：适配你的 57 token

print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# ================== 训练 ==================
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = batch.to(device)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    model.eval()
    total_val = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            total_val += outputs.loss.item()
    avg_val = total_val / len(val_loader)
    ppl = np.exp(avg_val)

    print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | PPL: {ppl:.2f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_forest_world_model_124m.pt")
        print("   → 保存最佳模型")

print(f"\n训练完成！最佳 Val PPL = {np.exp(best_val_loss):.2f}")
print("世界模型已保存为 best_forest_world_model_124m.pt")