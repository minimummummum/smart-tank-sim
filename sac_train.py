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

    def save(self, path='sac_model_aiming/replay_buffer.pkl'):
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

    def load(self, path='sac_model_aiming/replay_buffer.pkl'):
        if not os.path.exists(path):
            print(f"No replay buffer found at {path}")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
        print(f"Loaded replay buffer from {path}")
        return True

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action = 1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4) # 전이학습 시 1e-4 바꾸기
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4) # 전이학습 시 1e-4 바꾸기
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4) # 전이학습 시 1e-4 바꾸기

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

    def save(self, path='sac_model_aiming/sac_tank_continuous.pth'):
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

    def load(self, path='sac_model_aiming/sac_tank_continuous.pth'):
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

    def reset(self):
        self.fire_time = 0.0
        self.turret_x = 0.0
        self.turret_y = 0.0

    def update_state(self, log_data, hit_data):
         # 탱크 위치
        self.tank_x = float(log_data.get("playerPos", {}).get("x"))
        self.tank_y = float(log_data.get("playerPos", {}).get("y"))
        self.tank_z = float(log_data.get("playerPos", {}).get("z"))
        # 포탑 각도
        self.turret_x = float(log_data.get("playerTurretX"))
        self.turret_y = float(log_data.get("playerTurretY"))
        self.enemy_turret_x = float(log_data.get("enemyTurretX"))
        self.enemy_turret_y = float(log_data.get("enemyTurretY"))
        # 적 위치
        self.enemy_x = float(log_data.get("enemyPos", {}).get("x"))
        self.enemy_y = float(log_data.get("enemyPos", {}).get("y"))
        self.enemy_z = float(log_data.get("enemyPos", {}).get("z"))
        # body 각도
        self.body_x = float(log_data.get("playerBodyX"))
        self.body_y = float(log_data.get("playerBodyY"))
        self.enemy_body_x = float(log_data.get("enemyBodyX"))
        self.enemy_body_y = float(log_data.get("enemyBodyY"))
        # 속도
        self.tank_speed = float(log_data.get("playerSpeed"))
        self.enemy_speed = float(log_data.get("enemySpeed"))
        # 적중 여부
        if hit_data:
            self.hit = 1.0 if hit_data.get("hit", "terrain") == "enemy" else 0.0
            self.hit_x = float(hit_data.get('x', 0.0))
            self.hit_y = float(hit_data.get('y', 0.0))
            self.hit_z = float(hit_data.get('z', 0.0))
        else:
            self.hit = 0.0
            self.hit_x = None
            self.hit_y = None
            self.hit_z = None
        # 발사 쿨타임
        self.current_time = float(log_data.get("time", 0.0))
        self.time_since_last_fire = self.current_time - self.fire_time if self.fire_time else 6.0
        self.cooldown_norm = np.clip(self.time_since_last_fire / self.fire_cooldown, 0.0, 1.0)
        return 1

    def get_state(self):
        relative_vx, relative_vz = self.compute_relative_state_with_enemy_motion(
            self.turret_x, self.enemy_body_x, self.enemy_speed
        )
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        distance = math.sqrt(dx**2 + dz**2)
        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        yaw_error = self.angle_diff(self.turret_x, target_yaw)
        if self.hit_x and self.hit_z:
            hit_dx = self.hit_x - self.enemy_x
            hit_dz = self.hit_z - self.enemy_z
            threshold = 5.0
            hit_dx = -1.0 if hit_dx < -threshold else (1.0 if hit_dx > threshold else 0.0)
            hit_dz = -1.0 if hit_dz < -threshold else (1.0 if hit_dz > threshold else 0.0)
        else:
            hit_dx = 0.0
            hit_dz = 0.0
        state = np.array([
            yaw_error/180.0, distance/math.sqrt(300**2 + 300**2),
            relative_vx, relative_vz,
            np.sin(np.radians(self.turret_x)), np.cos(np.radians(self.turret_x)),
            np.sin(np.radians(self.turret_y)), np.cos(np.radians(self.turret_y)),
            hit_dx, hit_dz, self.cooldown_norm
        ], dtype=np.float32)
        return state

    def normalize_angle_rad(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_relative_state_with_enemy_motion(self, my_yaw_deg, enemy_yaw_deg, enemy_speed):
        my_yaw = np.deg2rad(my_yaw_deg)
        enemy_yaw = np.deg2rad(enemy_yaw_deg)

        enemy_vx = enemy_speed * np.cos(enemy_yaw)
        enemy_vz = enemy_speed * np.sin(enemy_yaw)

        rel_vx = enemy_vx
        rel_vz = enemy_vz

        cos_yaw = np.cos(-my_yaw)
        sin_yaw = np.sin(-my_yaw)
        relative_vx = rel_vx * cos_yaw - rel_vz * sin_yaw
        relative_vz = rel_vx * sin_yaw + rel_vz * cos_yaw

        relative_vx /= 70.0
        relative_vz /= 70.0
        return relative_vx, relative_vz

    def invert_pitch_from_impact_distance(self, d, gravity=9.81, initial_speed=60):
        factor = 733.74
        arg = (2 * d) / factor
        if abs(arg) > 1.0:
            return 0.0
        pitch_rad = 0.5 * math.asin(arg)
        return math.degrees(pitch_rad)

    def angle_diff(self, a, b):
        return (a - b + 180) % 360 - 180

    def scripted_action(self):
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        distance = math.sqrt(dx**2 + dz**2)
        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        target_pitch = self.invert_pitch_from_impact_distance(distance)
        yaw_error = self.angle_diff(self.turret_x, target_yaw)
        pitch_error = target_pitch - self.turret_y
        if yaw_error > 30:
            turret_dx = 1.0
        elif yaw_error < -30:
            turret_dx = -1.0
        elif yaw_error > 10:
            turret_dx = random.uniform(0.3, 0.7)
        elif yaw_error < -10:
            turret_dx = random.uniform(-0.3, -0.7)
        elif yaw_error > 2:
            turret_dx = random.uniform(0.1, 0.3)
        elif yaw_error < -2:
            turret_dx = random.uniform(-0.1, -0.3)
        elif yaw_error > 1:
            turret_dx = random.uniform(0.0, 0.05)
        elif yaw_error < -1:
            turret_dx = random.uniform(0.0, -0.05)
        else:
            turret_dx = 0.0
        if pitch_error > 10:
            turret_dy = 1.0
        elif pitch_error < -10:
            turret_dy = -1.0
        elif pitch_error > 2:
            turret_dy = random.uniform(0.3, 0.7)
        elif pitch_error < -2:
            turret_dy = random.uniform(-0.3, -0.7)
        elif pitch_error > 1:
            turret_dy = random.uniform(0.0, 0.1)
        elif pitch_error < -1:
            turret_dy = random.uniform(0.0, -0.1)
        else:
            turret_dy = 0.0
            
        #turret_dx = random.uniform(0.1, 1.0) if yaw_error > 2.0 else (random.uniform(-1.0, -0.1) if yaw_error < -2.0 else random.uniform(-0.05, 0.05))
        turret_dy = random.uniform(0.1, 1.0) if pitch_error > 2.0 else (random.uniform(-1.0, -0.1) if pitch_error < -2.0 else random.uniform(-0.05, 0.05))
        fire = 1.0 if abs(yaw_error) < 2.0 and abs(pitch_error) < 2.0 and self.cooldown_norm == 1.0 else -1.0
        return np.array([turret_dx, turret_dy, fire], dtype=np.float32)

    def step(self, action):
        _, _, fire = action
        # Calculate reward
        reward = 0.0
        reward -= min(0.5 * np.log1p(self.current_time), 2.0)
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        aim_error = abs(self.angle_diff(self.turret_x, target_yaw))
        aim_score = min(0.9,1.0 - (aim_error / 180.0))
        reward += (aim_score - 0.5) * 5.0
        if self.hit == 1.0:
            reward += 10.0
        elif self.hit_x and self.hit_z:
            hit_dx = abs(self.enemy_x - self.hit_x)
            hit_dy = abs(self.enemy_y - self.hit_y)
            hit_dz = abs(self.enemy_z - self.hit_z)
            hit_dist = (hit_dx ** 2 + hit_dy ** 2 + hit_dz ** 2) ** 0.5
            max_dist = 70.0
            reward += 10 * (1.0 - hit_dist / max_dist)
        distance = np.sqrt(dx**2 + dz**2)
        if distance > 130.0:
            reward -= 5.0
        if fire > 0.0:
            if self.cooldown_norm == 1.0:
                self.fire_time = self.current_time
                reward += 1.0 * aim_score  # 조준 정확도에 비례
            else:
                reward -= 2.0
        reward = np.clip(reward, -10.0, 10.0)
        done = (self.hit == 1.0) or (self.current_time > 60.0) or distance > 200.0
        print(f"보상: {reward:.2f}, 시간: {self.current_time:.2f}")
        return reward, done