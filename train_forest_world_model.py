# -*- coding: utf-8 -*-
"""
森林冒险世界模型训练 - 小型 Causal Transformer（35M）
====================================================================
- 纯 transformers + GPT2 架构（自定义小规模）
- 加载你生成的 forest_delayed_data_v2.pt
- 训练后进行完整校验（PPL / Acc + 延迟因果专项）
====================================================================
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

# ================== 配置 ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "forest_delayed_data_v2.pt"
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4
MAX_SEQ_LEN = 36

print("=== 森林冒险世界模型训练启动 ===")
print(f"设备: {device}\n")

# ================== 加载数据 ==================
data_dict = torch.load(DATA_PATH)
sequences = data_dict["sequences"]          # [8000, 36]
pending_events = data_dict["pending_events"]
vocab = data_dict["vocab"]
action_ids = data_dict["action_ids"]
pref_ids = data_dict["pref_ids"]

vocab_size = len(vocab)
print(f"加载数据: {len(sequences)} 条序列, vocab_size={vocab_size}")

# ================== Dataset ==================
class ForestLMDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx].long()

dataset = ForestLMDataset(sequences)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# ================== 小型 Causal Transformer ==================
config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=MAX_SEQ_LEN + 8,
    n_embd=256,
    n_layer=6,
    n_head=8,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    summary_first_dropout=0.1,
)

world_model = GPT2LMHeadModel(config).to(device)
world_model.resize_token_embeddings(vocab_size)   # 确保匹配

print(f"世界模型创建完成（参数量 ≈ {sum(p.numel() for p in world_model.parameters())/1e6:.1f}M）\n")

optimizer = torch.optim.AdamW(world_model.parameters(), lr=LR, weight_decay=0.01)

# ================== 训练 ==================
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    world_model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        batch = batch.to(device)
        outputs = world_model(input_ids=batch, labels=batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    world_model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = world_model(input_ids=batch, labels=batch)
            total_val_loss += outputs.loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_ppl = np.exp(avg_val_loss)

    print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(world_model.state_dict(), "best_forest_world_model.pt")
        print("   → 保存最佳模型")

print(f"\n训练完成！最佳 Val PPL = {np.exp(best_val_loss):.2f}")

# ================== 世界模型预测效果严格校验 ==================
world_model.load_state_dict(torch.load("best_forest_world_model.pt", map_location=device))
world_model.eval()

print("\n" + "="*80)
print("【世界模型预测效果严格校验】")
print("="*80)

# 1. 整体指标
total_correct = 0
total_tokens = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="整体预测准确率"):
        batch = batch.to(device)
        outputs = world_model(input_ids=batch)
        logits = outputs.logits[:, :-1, :]          # shift
        targets = batch[:, 1:]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_tokens += targets.numel()

overall_acc = total_correct / total_tokens * 100
print(f"整体 Next-Token 准确率 : {overall_acc:.2f}%")
print(f"验证集 Perplexity      : {np.exp(best_val_loss):.2f}\n")

# 2. 延迟因果专项校验（核心！）
print("延迟因果专项测试（action → delay 4-8步 pref 预测）")
correct_delay = 0
total_delay = 0
action_to_pref = {}   # 用于统计每个动作的条件概率提升

for seq, events in zip(sequences[:2000], pending_events[:2000]):   # 取前2000条加速
    seq = seq.to(device).unsqueeze(0)   # [1, 36]
    with torch.no_grad():
        outputs = world_model(input_ids=seq)
        logits = outputs.logits[0]      # [36, vocab]

    for start_pos, future_pos, expected_pid, strength in events:
        if strength <= 0: continue      # 只看正奖励
        total_delay += 1

        # 模型在 future_pos 位置预测的概率分布
        pred_probs = F.softmax(logits[future_pos-1], dim=-1)   # 前一个位置预测当前
        pred_pref_prob = pred_probs[pref_ids].sum().item()     # 所有 pref 词概率和
        pred_top = pred_probs.argmax().item()

        if pred_top in pref_ids:
            correct_delay += 1

        # 记录动作→pref 条件概率
        action = seq[0, start_pos].item()
        if action not in action_to_pref:
            action_to_pref[action] = {"total": 0, "pref_prob": 0.0}
        action_to_pref[action]["total"] += 1
        action_to_pref[action]["pref_prob"] += pred_pref_prob

delay_acc = correct_delay / total_delay * 100 if total_delay > 0 else 0
print(f"延迟正奖励预测准确率 : {delay_acc:.2f}%  ({correct_delay}/{total_delay})")

# 每个动作的条件提升
print("\n各动作延迟后 pref 预测概率：")
for aid in sorted(action_to_pref.keys()):
    stats = action_to_pref[aid]
    avg_prob = stats["pref_prob"] / stats["total"]
    token = list(vocab.keys())[list(vocab.values()).index(aid)]
    print(f"  {token:12s} → pref 条件概率 {avg_prob:.3f}")

print("\n✅ 世界模型校验完成！")
print("   • 如果延迟准确率 > 75% 说明已经学会了 4-8 步因果关系")
print("   • 模型已保存为 best_forest_world_model.pt")

# ================== 保存完整信息 ==================
torch.save({
    "model_state": world_model.state_dict(),
    "config": config,
    "vocab": vocab,
    "action_ids": action_ids,
    "pref_ids": pref_ids,
    "overall_acc": overall_acc,
    "delay_acc": delay_acc,
}, "forest_world_model_full.pt")

print("\n全部完成！下一步可以直接用这个 world_model + EFEPredictor 了。")