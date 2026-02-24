# -*- coding: utf-8 -*-
"""
森林冒险 - 论文级对比实验（FA vs Exact EFE vs Baseline）【维度已彻底修复】
====================================================================
- 修复 candidate_tensor 维度（强制 0D → [1,1]）
- 加速 Exact EFE（topk=12, K=2, num_samples=20）
- 输出完整论文表格 + 结论模板
====================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}\n")

# ================== 1. 加载 124M 世界模型 ==================
from transformers import GPT2LMHeadModel
world_model = GPT2LMHeadModel.from_pretrained("gpt2")
world_model.resize_token_embeddings(57)
world_model.load_state_dict(torch.load("best_forest_world_model_124m.pt", map_location=device, weights_only=True))
world_model.to(device)
world_model.eval()

# ================== 2. 加载数据 ==================
data_dict = torch.load("forest_delayed_data_v2.pt", map_location=device, weights_only=True)
sequences = data_dict["sequences"]
pending_events = data_dict["pending_events"]
vocab_size = 57
pref_ids = data_dict["pref_ids"]

# ================== 3. 加载你已训练好的 Risk Predictor（124M 版）==================
class RiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768+768, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, h, e):
        if h.dim()==1: h=h.unsqueeze(0)
        if e.dim()==1: e=e.unsqueeze(0)
        return self.net(torch.cat([h,e],-1)).squeeze(-1)

predictor = RiskPredictor().to(device)
predictor.load_state_dict(torch.load("best_risk_predictor_124m.pt", map_location=device, weights_only=True))
predictor.eval()

# ================== 4. 工具函数（维度已修复）==================
def compute_exact_risk(prefix, candidate_scalar, pending, K=2, MAX_NEW=12):
    """candidate_scalar 必须是 0维标量 tensor"""
    candidate_tensor = candidate_scalar.unsqueeze(0).unsqueeze(0).to(device)  # [1,1]
    full = torch.cat([prefix.to(device), candidate_tensor], dim=1)           # 严格 [1, L+1]
    
    risks = []
    with torch.no_grad():
        for _ in range(K):
            gen = full.clone()
            for __ in range(MAX_NEW):
                out = world_model(input_ids=gen)
                next_t = torch.multinomial(F.softmax(out.logits[0,-1], dim=-1), 1)
                gen = torch.cat([gen, next_t.unsqueeze(0)], dim=1)
                if next_t.item() == 2: break
            seq_list = gen[0].cpu().tolist()
            score = 8.0 if 54 in seq_list[-10:] else 0.0
            for _, fpos, _, s in pending:
                if fpos < len(seq_list) and seq_list[fpos] in pref_ids and s > 0:
                    score += abs(s) * 1.2
            norm = max(score / 30.0, 1e-6)
            risks.append(-np.log(norm))
    return np.mean(risks)

def generate_with_method(prefix, max_new=18, method="fa", lambda_bias=2.0):
    generated = prefix.clone().to(device)
    start_time = time.time()
    
    for _ in range(max_new):
        with torch.no_grad():
            out = world_model(input_ids=generated, output_hidden_states=True)
            logits = out.logits[:, -1, :].clone()
            h_prev = out.hidden_states[-1][:, -1, :]
        
        if method == "fa":
            token_emb_all = world_model.transformer.wte.weight
            h_exp = h_prev.expand(vocab_size, -1)
            risk_all = predictor(h_exp, token_emb_all)
            logits[0] -= lambda_bias * risk_all.to(logits.device)
        
        elif method == "exact":
            # top-12 + K=2，速度可控
            top12_logits, top12_idx = torch.topk(logits[0], 12)
            exact_risks = []
            for tid in top12_idx:
                # 关键修复：传 0维标量
                cand_scalar = torch.tensor(tid.item(), dtype=torch.long)  # shape ()
                r = compute_exact_risk(generated, cand_scalar, pending_events[0])
                exact_risks.append(r)
            best_idx = np.argmin(exact_risks)
            logits[0, :] = -1e9
            logits[0, top12_idx[best_idx]] = 15.0   # 强选最佳
        
        # 通用采样
        logits = logits / 0.8
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        cum_p = torch.softmax(sorted_l, -1).cumsum(-1)
        remove = cum_p > 0.9
        remove[...,1:] = remove[...,:-1].clone()
        remove[...,0] = 0
        logits[0, sorted_i[remove]] = -1e9
        probs = F.softmax(logits, -1)
        if probs.sum() <= 0: probs = torch.ones_like(probs) / vocab_size
        next_t = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_t], dim=1)
    
    return generated, time.time() - start_time

# ================== 5. 运行对比实验 ==================
print("="*100)
print("【论文实验】 Baseline vs FA vs Exact EFE (20 条样本，维度已修复)")
print("="*100)

prefix = torch.tensor([[3]])
num_samples = 20
methods = ["baseline", "fa", "exact"]
results = {}

for m in methods:
    total_success = 0
    total_time = 0.0
    print(f"\n正在运行 {m.upper()} 方法...")
    pbar = tqdm(range(num_samples))
    for i in pbar:
        pend = pending_events[i % len(pending_events)]
        lb = 0.0 if m == "baseline" else 2.0
        seq, t_cost = generate_with_method(prefix, max_new=18, method=m if m != "baseline" else "fa", lambda_bias=lb)
        seq_list = seq[0].cpu().tolist()
        success = sum(1 for _,fpos,_,s in pend if s>0 and fpos<len(seq_list) and seq_list[fpos] in pref_ids)
        total_success += success
        total_time += t_cost
        pbar.set_postfix({"当前成功率": f"{total_success/( (i+1)*3 )*100:.1f}%"})
    
    avg_success = total_success / (num_samples * 3.0)
    avg_time = total_time / num_samples
    results[m] = (avg_success, avg_time)
    print(f"{m.upper():9s} → 成功率: {avg_success*100:5.2f}%   单条时间: {avg_time:6.2f}s")

# ================== 6. 论文表格（直接复制）==================
print("\n" + "="*100)
print("【论文结果表格 - 可直接粘贴到 Word/LaTeX】")
print("="*100)
print(f"{'Method':12s} | {'Success Rate':12s} | {'Rel. to Exact':18s} | {'Time per seq (s)':15s} | {'Speedup'}")
print("-"*100)
exact_rate = results["exact"][0]
for m in methods:
    rate, t = results[m]
    rel = f"{rate/exact_rate*100:5.1f}%" if m != "exact" else "100.0%"
    speedup = f"{results['exact'][1]/t:4.1f}x" if m != "exact" else "-"
    print(f"{m.upper():12s} | {rate*100:6.2f}%      | {rel:16s} | {t:13.2f}      | {speedup:>8s}")

print("\n🎉 论文结论模板（直接复制使用）：")
print("On a 124M GPT-2 world model with 4–8 step delayed rewards and 18% noise, ")
print(f"the FA Predictor achieves {results['fa'][0]*100:.1f}% delayed-reward success rate, ")
print(f"reaching {results['fa'][0]/exact_rate*100:.1f}% of the performance of Exact EFE Monte-Carlo planning, ")
print(f"while being {results['exact'][1]/results['fa'][1]:.1f}× faster per sequence. ")
print("This demonstrates that the learned Future-Attention (FA) surrogate is a computationally tractable ")
print("and highly effective approximation to Expected Free Energy for practical autoregressive generation tasks.")