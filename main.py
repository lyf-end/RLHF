#
# 实践：环境初始化和预处理
#
import torch
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import gym

#
# 实践：环境初始化和预处理
#
# --- 超参数 ---
num_steps = 128         # 每个worker收集数据的步数
num_workers = 8         # 并行环境的数量
total_updates = 10000   # 总共的更新次数

def make_env():
    def _thunk():
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, [['right'], ['right', 'A']])
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=4)
        return env
    return _thunk

# 1. 创建并行化的马里奥环境
envs = gym.vector.SyncVectorEnv([make_env() for _ in range(num_workers)])

# 此时，envs.single_observation_space.shape 就是 (4, 84, 84)，这是我们网络模型的输入
# envs.single_action_space.n 就是动作的数量

#
# 实践：构建参数共享的Actor-Critic网络
#
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        # 理论：共享的特征提取层 (Shared Trunk)
        self.conv_base = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 将卷积层输出展平后的大小
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # 理论：策略头 (Actor Head)
        # 输出每个动作的logits
        self.policy_head = nn.Linear(512, num_actions)
        
        # 理论：价值头 (Critic Head)
        # 输出一个标量值
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x的shape: (batch, 4, 84, 84)
        # x已经是tensor, 不需要再转换
        x = self.conv_base(x / 255.0) # 归一化
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        
        # 理论：π_θ(a|s) 的参数化
        action_logits = self.policy_head(x)
        # 使用Categorical分布来处理离散动作
        action_dist = Categorical(logits=action_logits)
        
        # 理论：V_φ(s) 的估计
        state_value = self.value_head(x)
        
        return action_dist, state_value
        
    def _get_conv_out(self, shape):
        # 辅助函数，用于计算卷积层输出的大小
        o = self.conv_base(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    


    #
# 实践：PPO训练主循环
#
# --- 超参数 ---
epochs = 4              # 每次收集数据后，对数据进行优化的轮数
batch_size = 256        # mini-batch的大小
gamma = 0.99            # 折扣因子
gae_lambda = 0.95       # GAE的lambda参数
clip_epsilon = 0.1      # PPO的裁剪参数
lr = 1e-4               # 学习率
entropy_coef = 0.01     # 熵奖励系数
value_loss_coef = 0.5   # 价值损失系数

# --- 设备设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
model = ActorCritic(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- 数据存储 ---
# 用来存储 num_steps * num_workers 的交互数据
states = torch.zeros((num_steps, num_workers, *envs.single_observation_space.shape)).to(device)
actions = torch.zeros((num_steps, num_workers, 1), dtype=torch.long).to(device)
log_probs = torch.zeros((num_steps, num_workers, 1)).to(device)
rewards = torch.zeros((num_steps, num_workers, 1)).to(device)
values = torch.zeros((num_steps, num_workers, 1)).to(device)
dones = torch.zeros((num_steps, num_workers, 1)).to(device)

# --- 主循环 ---
# 初始化环境状态
current_states = torch.tensor(envs.reset(), dtype=torch.float32).to(device)

for update_iteration in range(total_updates):
    #
    # === 阶段一：数据收集 (Rollout Phase) ===
    #
    for step in range(num_steps):
        with torch.no_grad(): # 在收集数据时，不计算梯度
            # 1. 模型前向传播，得到动作分布和价值
            # 理论：根据当前策略 π_θ_old 与环境交互
            action_dist, state_value = model(current_states)
            
            # 2. 从分布中采样一个动作
            action = action_dist.sample()
            
            # 3. 记录动作的对数概率和价值
            log_prob = action_dist.log_prob(action)
            
            # 4. 执行动作，与环境交互
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            
            # 5. 存储这一步的数据
            states[step] = current_states
            actions[step] = action.unsqueeze(-1)
            log_probs[step] = log_prob.unsqueeze(-1)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).unsqueeze(-1).to(device)
            values[step] = state_value
            dones[step] = torch.tensor(done, dtype=torch.float32).unsqueeze(-1).to(device)
            
            current_states = torch.tensor(next_state, dtype=torch.float32).to(device)
    
    
    print(f"Update {update_iteration + 1}/{total_updates}, Mean Reward: {rewards.sum() / num_workers}")

    #
    # === 阶段二：计算优势和回报 (GAE & Returns) ===
    #
    # 理论：计算 A_hat 和 V_target
    with torch.no_grad():
        # 需要最后一步的状态价值来启动GAE计算
        _, last_value = model(current_states)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        
        # 从后往前计算GAE
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            # 理论：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            
            # 理论：A_t^GAE = δ_t + (γλ) * A_{t+1}^GAE
            # 这是一个从后往前的递归形式，与我们之前看的求和形式等价
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        # 理论：V_t^target = A_t + V(s_t)
        returns = advantages + values
    
    #
    # === 阶段三：模型更新 (Optimization Phase) ===
    #
    # 将数据展平
    b_states = states.view(-1, *envs.single_observation_space.shape)
    b_actions = actions.view(-1)
    b_log_probs = log_probs.view(-1)
    b_advantages = advantages.view(-1)
    b_returns = returns.view(-1)
    
    # 对所有数据进行多轮(epochs)优化
    for _ in range(epochs):
        # 从所有数据中随机采样 mini-batch
        sampler = torch.randperm(num_steps * num_workers)
        for indices in sampler.split(batch_size):
            # 获取mini-batch数据
            mb_states = b_states[indices]
            mb_actions = b_actions[indices]
            mb_old_log_probs = b_log_probs[indices]
            mb_advantages = b_advantages[indices]
            mb_returns = b_returns[indices]

            # 1. 重新用当前模型计算
            # 理论：计算 π_θ(a_t|s_t) 用于构造概率比率 r_t(θ)
            new_dist, new_values = model(mb_states)
            new_log_probs = new_dist.log_prob(mb_actions)
            entropy = new_dist.entropy().mean()

            # 2. 计算Actor Loss (Policy Loss)
            # 理论：r_t(θ) = exp(log(π_θ) - log(π_θ_old))
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            
            # 理论：L^CLIP(θ) 的第一部分
            surr1 = ratio * mb_advantages
            
            # 理论：L^CLIP(θ) 的第二部分
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            
            # 理论：最终的Actor Loss (取负号因为我们要梯度上升)
            policy_loss = -torch.min(surr1, surr2).mean()

            # 3. 计算Critic Loss (Value Loss)
            # 理论：L^VF(θ) = (V_φ(s_t) - V_t^target)^2
            value_loss = (new_values.view(-1) - mb_returns).pow(2).mean()

            # 4. 计算总损失
            # 理论：L_total = L^CLIP - c1 * L^VF + c2 * S
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            # 5. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # 梯度裁剪
            optimizer.step()
