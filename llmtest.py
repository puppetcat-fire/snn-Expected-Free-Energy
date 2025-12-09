import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import math
from collections import deque
import random

# ==================== 工具函数 ====================
def calculate_pole_end_height(state):
    """计算杆子末端的绝对高度"""
    pole_length = 1.0  # CartPole杆子长度的一半为1米
    
    # 从状态中获取杆子角度（弧度）
    # CartPole中的角度是相对于垂直向上的角度
    pole_angle = state[2]  # 弧度
    
    # 计算杆子末端的y坐标
    # 当杆子垂直时，高度为1.0，当杆子水平时，高度为0.0
    y_end = pole_length * math.cos(pole_angle)  # cos(0)=1, cos(pi/2)=0
    
    # 确保高度非负（虽然理论上cos可能为负，但实际游戏中杆子不会完全倒立）
    return max(0.0, y_end)

def calculate_height_increase(prev_state, current_state):
    """计算高度增加（非负）"""
    prev_height = calculate_pole_end_height(prev_state)
    current_height = calculate_pole_end_height(current_state)
    height_increase = max(0.0, current_height - prev_height)
    return height_increase

# ==================== 自定义Transformer层（返回注意力权重）====================
class TransformerEncoderLayerWithAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu
            
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力层，返回注意力权重
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class TransformerEncoderWithAttention(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src
        all_attn_weights = []
        for layer in self.layers:
            output, attn_weights = layer(output, src_mask, src_key_padding_mask)
            all_attn_weights.append(attn_weights)  # 每层的注意力权重
        return output, all_attn_weights

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# ==================== 主模型 ====================
class CartPoleAttentionPredictor(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影：7维 -> d_model
        # 7维 = 4(state) + 2(action) + 1(height_increase)
        self.input_proj = nn.Linear(7, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 带注意力权重的Transformer编码器
        encoder_layer = TransformerEncoderLayerWithAttention(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoderWithAttention(encoder_layer, num_layers)
        
        # 输出头1: 预测未来注意力分数 (2维，对应两个动作)
        self.attn_score_head = nn.Linear(d_model, 2)
        
        # 输出头2: 预测下一状态(4维)和高度增加(1维)
        self.state_pred_head = nn.Linear(d_model, 5)  # 4 + 1
        
    def forward(self, x, return_attn_weights=False):
        """
        x: (batch, seq_len, 7)
        返回:
            future_attn_scores: (batch, 2) - 动作在未来10步中会获得的注意力分数
            next_state_pred: (batch, 4) - 下一状态预测
            height_inc_pred: (batch, 1) - 高度增加预测
            all_attn_weights: 可选，注意力权重列表
        """
        # 输入投影
        # breakpoint()
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码
        x, all_attn_weights = self.transformer_encoder(x)
        
        # 取最后一个时间步的输出（对应预测下一步）
        last_output = x[:, -1, :]  # (batch, d_model)
        
        # 预测未来注意力分数
        future_attn_scores = self.attn_score_head(last_output)  # (batch, 2)
        
        # 预测状态和高度增加
        state_height_pred = self.state_pred_head(last_output)  # (batch, 5)
        next_state_pred = state_height_pred[:, :4]  # (batch, 4)
        
        # 高度增加使用ReLU确保非负
        height_inc_pred = F.relu(state_height_pred[:, 4:])  # (batch, 1)
        
        if return_attn_weights:
            return future_attn_scores, next_state_pred, height_inc_pred, all_attn_weights
        else:
            return future_attn_scores, next_state_pred, height_inc_pred

# ==================== 训练类 ====================
class CartPoleTrainer:
    def __init__(self, env_name='CartPole-v1', seq_len=10, future_steps=10):
        self.env = gym.make(env_name)
        self.seq_len = seq_len
        self.future_steps = future_steps
        
        # 初始化模型
        self.model = CartPoleAttentionPredictor()
        self.target_model = CartPoleAttentionPredictor()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # 冻结模型
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        # 历史缓冲区（存储最近seq_len步的数据）
        self.history_buffer = deque(maxlen=seq_len)
        
        # 训练参数
        # self.temperature = 1.0  # 初始温度
        self.last_pred_error = 1.0  # 上次预测误差，用于调整温度
        
        # 训练计数器
        self.train_counter = 0
        self.update_target_every = 10  # 每100步更新一次目标网络
        
    def action_to_onehot(self, action):
        """将动作转换为one-hot编码"""
        onehot = np.zeros(2)
        onehot[action] = 1
        return onehot
    
    def collect_initial_data(self):
        """收集初始数据填充历史缓冲区"""
        state, _ = self.env.reset()
        prev_height = calculate_pole_end_height(state)
        
        while len(self.history_buffer) < self.seq_len:
            # 随机动作填充
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # 计算高度增加
            height_increase = calculate_height_increase(state, next_state)
            
            # 存储到历史
            self.history_buffer.append({
                'state': state.copy(),
                'action': self.action_to_onehot(action),
                'height_increase': height_increase
            })
            
            state = next_state
            prev_height = calculate_pole_end_height(state)
            
            if terminated or truncated:
                state, _ = self.env.reset()
                prev_height = calculate_pole_end_height(state)
    
    def select_action_with_temperature(self, future_attn_scores):
        """使用带温度的softmax选择动作"""
        # 温度与预测误差成反比：误差大则温度高，探索多
        temp = max(0.1, min(5.0, self.last_pred_error))
        
        # 计算softmax概率
        probs = F.softmax(future_attn_scores / temp, dim=-1)
        probs_np = probs.detach().cpu().numpy()
        
        # 根据概率采样动作
        action = np.random.choice([0, 1], p=probs_np.flatten())
        return action, probs
    
    def get_attention_to_action(self, attn_weights, action_idx_in_seq):
        """从注意力权重中提取对特定动作位置的注意力"""
        # attn_weights: list of (batch, nhead, seq_len, seq_len)
        # 取最后一层的注意力权重
        last_layer_attn = attn_weights[-1]  # (batch, nhead, seq_len, seq_len)
        # print(last_layer_attn.shape)
        # breakpoint()
        # 平均所有注意力头
        # avg_attn = last_layer_attn.mean(dim=1)  # (batch, seq_len, seq_len)
        
        # 取最后一个位置（当前预测步）对目标动作位置的注意力
        # 注意：这里我们假设batch_size=1
        # breakpoint()
        attn_to_action = last_layer_attn[0, -1, action_idx_in_seq].item()
        return attn_to_action
    
    def run_target_model_simulation(self, start_seq, chosen_action_idx):
        """运行冻结模型模拟未来10步，计算目标注意力分数"""
        total_attn = 0.0
        current_seq = start_seq.copy()  # 复制起始序列
        
        # 在序列中，选择的动作在最后一个位置（索引seq_len-1）
        target_action_idx = self.seq_len - 1
        
        for step in range(self.future_steps):
            # 转换为张量
            seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)  # (1, seq_len, 7)
            
            # 冻结模型前向传播（返回注意力权重）
            _, _, _, all_attn_weights = self.target_model(seq_tensor, return_attn_weights=True)
            
            # 获取当前步对目标动作的注意力
            attn = self.get_attention_to_action(all_attn_weights, target_action_idx)
            total_attn += attn
            
            # 冻结模型选择下一步动作（用于继续模拟）
            future_scores, _, _ = self.target_model(seq_tensor)
            future_scores_np = future_scores.detach().cpu().numpy().flatten()
            temp = max(0.1, min(5.0, self.last_pred_error))
            probs = np.exp(future_scores_np / temp) / np.sum(np.exp(future_scores_np / temp))
            next_action = np.random.choice([0, 1], p=probs)
            
            # 用冻结模型预测下一个状态和高度增加
            _, next_state_pred, height_pred = self.target_model(seq_tensor)
            
            # 构建下一步的输入
            next_state_np = next_state_pred.detach().cpu().numpy().flatten()
            height_np = height_pred.detach().cpu().numpy().flatten()[0]
            
            # 更新序列：移除第一步，添加新的一步
            new_step = np.concatenate([
                next_state_np,
                self.action_to_onehot(next_action),
                [height_np]
            ])
            
            current_seq = np.vstack([current_seq[1:], new_step])
            
            # 目标动作在序列中的索引向前移动一位
            target_action_idx -= 1
            if target_action_idx < 0:
                # 动作已移出窗口
                break
        
        return total_attn
    
    def train_step(self):
        """执行一次训练步骤"""
        # 1. 从历史缓冲区构建输入序列
        if len(self.history_buffer) < self.seq_len:
            return 0
        
        # 获取最近seq_len步的数据
        recent_data = list(self.history_buffer)[-self.seq_len:]
        
        # 构建输入序列
        input_seq = []
        for data in recent_data:
            input_seq.append(np.concatenate([
                data['state'],
                data['action'],
                [data['height_increase']]
            ]))
        input_seq = np.array(input_seq)  # (seq_len, 7)
        
        # 2. 当前模型预测
        seq_tensor = torch.FloatTensor(input_seq).unsqueeze(0)  # (1, seq_len, 7)
        future_scores, state_pred, height_pred = self.model(seq_tensor)
        
        # 3. 选择动作
        action, action_probs = self.select_action_with_temperature(future_scores)
        
        # 4. 执行动作，获取真实数据
        # 获取历史中的最后一个状态
        last_data = recent_data[-1]
        last_state = last_data['state']
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        
        # 计算真实高度增加
        real_height_increase = calculate_height_increase(last_state, next_state)
        
        # 5. 计算状态和高度预测损失
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        height_tensor = torch.FloatTensor([real_height_increase]).unsqueeze(0)
        
        loss_state = F.mse_loss(state_pred, next_state_tensor)
        loss_height = F.mse_loss(height_pred, height_tensor)
        
        # 6. 计算目标注意力分数（使用冻结模型模拟）
        # 构建包含当前步的序列（用于目标模型模拟）
        current_step = np.concatenate([
            last_state,
            self.action_to_onehot(action),
            [real_height_increase]
        ])
        
        # 第2步到第11步的序列
        sim_start_seq = np.vstack([input_seq[1:], current_step])  # (seq_len, 7)
        
        # 运行目标模型模拟
        target_attention = self.run_target_model_simulation(sim_start_seq, self.seq_len - 1)
        
        # 7. 计算注意力分数预测损失
        # 只计算被选择动作的损失
        target_attn_tensor = torch.FloatTensor([target_attention])
        chosen_action_score = future_scores[0, action]
        loss_attention = F.mse_loss(chosen_action_score, target_attn_tensor)
        
        # 8. 总损失
        total_loss = loss_state + loss_height + loss_attention
        
        # 9. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
        
        # 10. 更新历史缓冲区
        self.history_buffer.append({
            'state': next_state.copy(),
            'action': self.action_to_onehot(action),
            'height_increase': real_height_increase
        })
        
        # 11. 更新温度
        self.last_pred_error = total_loss.item()
        
        # 12. 更新目标网络
        self.train_counter += 1
        if self.train_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"Step {self.train_counter}: Updated target model")
        
        return total_loss.item()
    
    def train(self, num_episodes=1000, max_steps_per_episode=500):
        """主训练循环"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            # 重置环境
            state, _ = self.env.reset()
            
            # 如果是第一轮，收集初始数据
            if episode == 0:
                self.collect_initial_data()
            
            episode_reward = 0
            terminated = False
            truncated = False
            
            # breakpoint()
            for step in range(max_steps_per_episode):
                # 每20步训练一次
                if step % 20 == 0 and step > 0:
                    loss = self.train_step()
                    # breakpoint()
                    print(f"Episode {episode}, Step {step}: Loss = {loss:.4f}, Temp = {self.last_pred_error:.2f}")
                
                # breakpoint()
                
                # 执行动作（使用当前模型或随机探索）
                if len(self.history_buffer) >= self.seq_len:
                    # 使用当前模型选择动作
                    recent_data = list(self.history_buffer)[-self.seq_len:]
                    input_seq = []
                    for data in recent_data:
                        input_seq.append(np.concatenate([
                            data['state'],
                            data['action'],
                            [data['height_increase']]
                        ]))
                    input_seq = np.array(input_seq)
                    
                    seq_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
                    future_scores, _, _ = self.model(seq_tensor)
                    action, _ = self.select_action_with_temperature(future_scores)
                else:
                    # 随机探索
                    action = self.env.action_space.sample()
                
                # 执行动作
                last_state = self.history_buffer[-1]['state'] if self.history_buffer else state
                next_state, reward, _, _, _ = self.env.step(action)
                
                # 计算高度增加
                height_increase = calculate_height_increase(last_state, next_state)
                
                # 更新历史缓冲区
                self.history_buffer.append({
                    'state': next_state.copy(),
                    'action': self.action_to_onehot(action),
                    'height_increase': height_increase
                })
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            # print(f"Episode {episode}: Reward = {episode_reward}, Avg Reward = {np.mean(episode_rewards[-10:]):.1f}")
            
            # 如果连续10个episode平均奖励达到495，认为已解决
            if len(episode_rewards) >= 10 and np.mean(episode_rewards[-10:]) >= 495:
                print(f"Solved at episode {episode}!")
                break
        
        self.env.close()
        return episode_rewards

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建训练器
    trainer = CartPoleTrainer()
    
    # 开始训练
    rewards = trainer.train(num_episodes=2000, max_steps_per_episode=500)
    
    # 保存模型
    torch.save(trainer.model.state_dict(), "cartpole_attention_predictor.pth")
    print("Model saved to cartpole_attention_predictor.pth")
