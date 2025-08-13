import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import collections
from torch.utils.tensorboard import SummaryWriter

# 超参数
REPLAY_SIZE = 2000 #经验池大小
small_BATCH_SIZE = 16 #小批量
big_BATCH_SIZE = 128 #大批量
BATCH_SIZE_door = 1000 #
GAMMA = 0.9 #折扣因子
INITIAL_EPSILON = 0.5 #初始贪婪系数
FINAL_EPSILON = 0.01 #最终贪婪系数

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) # 队列deque,先进先出
    def add(self, state, action, reward, next_state, done): # 将数据加入 buffer
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size): # 从 buffer 中采样数据,数量为 batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)

# 定义卷积神经网络模型
class DQNModel(nn.Module):
    def __init__(self, observation_height, observation_width, action_dim):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 计算展平后的尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 1, observation_height, observation_width)
            dummy = self.pool2(self.conv2(self.pool1(self.conv1(dummy))))
            self.flat_size = dummy.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, action_dim)
        self.dropout3 = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flat_size)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        return x


class DQN():
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file, learning_rate=0.001):
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.epsilon = INITIAL_EPSILON
        self.model_path = os.path.join(model_file, "dqn_model.pth")
        self.model_file = model_file
        self.log_file = log_file
        
        # 创建日志目录
        os.makedirs(log_file, exist_ok=True)
        
        # 创建 TensorBoard 记录器
        self.writer = SummaryWriter(log_dir=log_file)
        
        # 使用rl_utils.ReplayBuffer作为经验回放
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE)
        
        # 创建当前网络和目标网络
        self.current_net = DQNModel(observation_height, observation_width, action_space)
        self.target_net = DQNModel(observation_height, observation_width, action_space)
        self.target_net.load_state_dict(self.current_net.state_dict())
        self.target_net.eval()  # 目标网络不进行训练
        
        # 优化器
        self.optimizer = optim.Adam(self.current_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_net.to(self.device)
        self.target_net.to(self.device)
        
        # 加载现有模型（如果有）
        if os.path.exists(self.model_path):
            print("Model exists, loading model\n")
            checkpoint = torch.load(self.model_path)
            self.current_net.load_state_dict(checkpoint['current_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
        else:
            print("Model doesn't exist, creating new one\n")
        
        # 步数计数器
        self.step_count = 0
    
    def choose_action(self, state):
        """使用ε-贪婪策略选择动作"""
        # 将状态转换为PyTorch张量
        if state.dim() == 3:
            state_tensor = state.float().unsqueeze(0).to(self.device)
        else:
            state = state.squeeze()
            state_tensor = state.float().unsqueeze(0).unsqueeze(0).to(self.device)
        if random.random() <= self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.current_net(state_tensor)
                action = torch.argmax(q_values).item()
        
        # 衰减ε
        self.epsilon = max(FINAL_EPSILON, self.epsilon - (INITIAL_EPSILON - FINAL_EPSILON) / 10000)
        return action
    
    def store_data(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_network(self, batch_size,num_step):
        """使用经验回放训练网络"""
        if len(self.replay_buffer) < batch_size:
            return  # 经验不足，不进行训练
        
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 转换为PyTorch张量
        states = torch.stack(states,dim=0).float().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.stack(next_states,dim=0).float().to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # 动作转换为one-hot向量
        actions_tensor = torch.zeros(batch_size, self.action_dim, device=self.device)
        actions_tensor[torch.arange(batch_size), actions] = 1
        
        # 计算当前Q值
        current_q_values = self.current_net(states)
        chosen_q_values = torch.sum(current_q_values * actions_tensor, dim=1).view(-1, 1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values, _ = torch.max(next_q_values, dim=1)
            target_q_values = rewards + GAMMA * max_next_q_values.view(-1, 1) * (1 - dones)
        
        # 计算损失
        loss = self.loss_fn(chosen_q_values, target_q_values)
        
        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新步数计数器
        self.step_count += 1
        
        # 每100步记录一次损失
        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}, Loss: {loss.item():.4f}")
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', loss.item(), num_step)
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.current_net.state_dict())
    
    def action(self, state):
        """用于测试，不进行探索"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.current_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def save_model(self):
        """保存模型"""
        checkpoint = {
            'current_net': self.current_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def close(self):
        """关闭TensorBoard写入器"""
        self.writer.close()