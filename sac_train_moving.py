import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import math
import os
import pickle
############################
# 0. LiDAR Max Distance 150, False일 경우 or 50 이상일 경우 50
# 1. 5채널 360*5 1800 + 목표dxdy 2 + 내 속도 1
# 2. 앞뒤, 왼오 액션 2
# 3. 시나리오 1단계 목표 도달, 2단계 목표 도달 + 장애물 회피, 3단계 목표 도달 + 장애물 회피 + 최적 경로로
############################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

    def save(self, path='sac_model_moving/replay_buffer.pkl'):
        base, ext = os.path.splitext(path)
        counter = 0
        save_path = path
        while os.path.exists(save_path):
            counter += 1
            save_path = f"{base}_{counter}{ext}"
        data = {
            'buffer': self.buffer,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved replay buffer to {save_path}")

    def load(self, path='sac_model_moving/replay_buffer_125.pkl'):
        if not os.path.exists(path):
            print(f"No replay buffer found at {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
        print(f"Loaded replay buffer from {path}")
        return True

# Actor Network (Policy)
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action = 1):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.mu = nn.Linear(256, action_dim)
#         self.log_std = nn.Linear(256, action_dim)
#         self.max_action = max_action

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         mu = self.mu(x)
#         log_std = self.log_std(x).clamp(-20, 2)
#         std = log_std.exp()
#         return mu, std

#     def sample(self, state):
#         mu, std = self.forward(state)
#         dist = torch.distributions.Normal(mu, std)
#         action = dist.rsample()
#         log_prob = dist.log_prob(action).sum(axis=-1)
#         action = torch.tanh(action) * self.max_action
#         log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)
#         return action, log_prob

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=-1)
#         return self.net(x)

# Actor Network (CNN 추가)
class Actor(nn.Module):
    def __init__(self, state_dim=1805, action_dim=2, max_action=1):
        super(Actor, self).__init__()
        self.max_action = max_action
        # LiDAR: [batch, 5, 360] 처리
        self.conv1 = nn.Conv1d(5, 32, kernel_size=8, stride=4)  # 출력: [32, 88]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2)  # 출력: [64, 43]
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)  # 출력: [64, 41]
        self.lidar_fc = nn.Linear(64 * 41, 256)  # 평탄화 후 256차원

        # 나머지 상태(타겟 각도/거리 2, 속도 1, 좌표 2)
        self.other_fc = nn.Linear(5, 64)
        self.fc1 = nn.Linear(256 + 64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        # state: [batch, 1805] -> LiDAR [batch, 1800], other [batch, 5]
        lidar_state = state[:, :1800].reshape(-1, 5, 360)  # [batch, 5, 360]
        other_state = state[:, 1800:]  # [batch, 5]

        # CNN으로 LiDAR 처리
        x = F.relu(self.conv1(lidar_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 평탄화
        x = F.relu(self.lidar_fc(x))

        # 나머지 상태 처리
        other = F.relu(self.other_fc(other_state))

        # 결합
        x = torch.cat([x, other], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        action = torch.tanh(action) * self.max_action
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(axis=-1)
        return action, log_prob

# Critic Network (CNN 추가)
class Critic(nn.Module):
    def __init__(self, state_dim=1805, action_dim=2):
        super(Critic, self).__init__()
        # LiDAR: [batch, 5, 360] 처리
        self.conv1 = nn.Conv1d(5, 32, kernel_size=8, stride=4)  # 출력: [32, 88]
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2)  # 출력: [64, 43]
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)  # 출력: [64, 41]
        self.lidar_fc = nn.Linear(64 * 41, 256)

        # 나머지 상태(타겟 각도/거리 2, 속도 1, 좌표 2) + 액션
        self.other_fc = nn.Linear(5 + action_dim, 64)
        self.fc1 = nn.Linear(256 + 64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        # state: [batch, 1805] -> LiDAR [batch, 1800], other [batch, 5]
        lidar_state = state[:, :1800].reshape(-1, 5, 360)  # [batch, 5, 360]
        other_state = state[:, 1800:]  # [batch, 5]

        # CNN으로 LiDAR 처리
        x = F.relu(self.conv1(lidar_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 평탄화
        x = F.relu(self.lidar_fc(x))

        # 나머지 상태 + 액션 처리
        other = torch.cat([other_state, action], dim=-1)
        other = F.relu(self.other_fc(other))

        # 결합
        x = torch.cat([x, other], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q
    
# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4) # 전이학습 시 1e-4 바꾸기, 원래 3e-4로 함
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

        self.replay_buffer = ReplayBuffer(capacity=1000000)
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if deterministic:
            mu, _ = self.actor(state)
            action = torch.tanh(mu) * self.actor.max_action
            return action.cpu().detach().numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy().flatten()

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob.unsqueeze(1)
            target_q = reward + (1 - done) * self.gamma * target_q

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp() * log_prob.unsqueeze(1) - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path='sac_model_moving/sac_tank_continuous.pth'):
        base, ext = os.path.splitext(path)
        counter = 0
        save_path = path
        while os.path.exists(save_path):
            counter += 1
            save_path = f"{base}_{counter}{ext}"
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha
        }, save_path)
        print(f"Saved model to {save_path}")

    def load(self, path='sac_model_moving/sac_tank_continuous_125.pth'):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic_1.load_state_dict(checkpoint['critic_1'])
            self.critic_2.load_state_dict(checkpoint['critic_2'])
            self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
            self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            print(f"Loaded model from {path}")
        else:
            print(f"No model found at {path}")

# --- Tank Environment ---
class TankEnv:
    def __init__(self):
        self.fire_cooldown = 6.0
        self.max_distance = 50.0

    def reset(self):
        self.target_x = None
        self.target_z = None

    def update_state(self, log_data, target = None):
         # 탱크 위치
        self.tank_x = float(log_data.get("playerPos", {}).get("x"))
        self.tank_z = float(log_data.get("playerPos", {}).get("z"))
        # body 각도
        self.body_x = float(log_data.get("playerBodyX"))
        self.body_y = float(log_data.get("playerBodyY"))
        if target:
            # 목표 지점
            self.target_x = target[0]
            self.target_z = target[1]
        # 속도
        self.tank_speed = np.array([np.clip(log_data.get("playerSpeed", 0.0), 0.0, 70.0) / 70.0], dtype=np.float32)
        self.current_time = float(log_data.get("time", 0.0))
        # LiDAR
        self.flat_lidar_state = np.array([
                min(p["distance"] / self.max_distance, 1.0)
                for p in log_data["lidarPoints"]
            ], dtype=np.float32)
        return 1
    def angle_diff(self, a, b):
        return (a - b + 180) % 360 - 180
    def get_state(self):
        if self.target_x is None or self.target_z is None:
            dx = 0.0
            dz = 0.0
            target_yaw = 0
        else:
            dx = self.target_x - self.tank_x
            dz = self.target_z - self.tank_z
            target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        yaw_error = self.angle_diff(self.body_x, target_yaw) / 180.0
        distance = min(math.sqrt(dx**2 + dz**2) / 300.0, 1.0)
        yaw_dist = np.array([yaw_error, distance], dtype=np.float32)
        x_y_raw = np.array([self.tank_x, self.tank_z], dtype=np.float32) / 300.0
        x_y = np.minimum(x_y_raw, 1.0)
        state = np.concatenate((self.flat_lidar_state, x_y, self.tank_speed, yaw_dist))
        return state

    def scripted_action(self):
        return

    def step(self, action):
        action_x, action_y = action
        # Calculate reward
        reward = 0.0
        reward -= min(0.5 * np.log1p(self.current_time), 5.0)
        dx = self.target_x - self.tank_x
        dz = self.target_z - self.tank_z
        distance = min(math.sqrt(dx**2 + dz**2) / 300.0, 1.0)
        reward += (0.5-distance) * 10.0
        done = False
        # 목표 도달 조건
        if distance <= 0.01:
            reward += 10.0
            done = True
        # 시간 초과
        elif self.current_time > 60.0:
            done = True
        # 맵 경계선 침범
        elif self.tank_x <= 5 or self.tank_x >= 297 or self.tank_z <= 5 or self.tank_z >= 297:
            reward -= 10.0
            done = True
        reward = np.clip(reward, -10.0, 10.0)
        print(f"보상: {reward:.2f}, 시간: {self.current_time:.2f}, target: {self.target_x, self.target_z}, tank: {int(self.tank_x), int(self.tank_z)}")
        return reward, done