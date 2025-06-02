import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import os
import pickle
import numpy as np
"""
포탑 모델
ㅇ Scenario:
A 시나리오
상대와 나는 고정된 위치
B 시나리오
상대 탱크는 랜덤으로 움직이고, 나는 고정되어 있다.
C 시나리오
상대 탱크는 고정되어 있고, 나는 랜덤으로 움직인다.
D 시나리오
상대 탱크는 나와 같은 방향으로 이동하며, 속도는 랜덤이다.
E 시나리오
상대 탱크는 나와 반대 방향으로 이동하며, 속도는 랜덤이다.

Done 조건
1. 상대 탱크가 격파되면 Done
2. 60초가 지나면 Done
3. 거리가 탱크 격파 거리보다 멀어지면 Done(약 130)

ㅇ State:
relative_yaw, relative_vx, relative_vz, 상대와의 거리, 상대와의 상대적 각도, 포탑 yaw(sin, cos), pitch(sin, cos), hit_dx[-1, 0, 1], hit_dz[-1, 0, 1]
ㅇ Action:
[-1, -1]
[-1, 0]
[-1, 1]
[0, -1]
[0, 0]
[0, 1]
[1, -1]
[1, 0]
[1, 1]

ㅇ Reward:
1. 시간 패널티
2. 조준 유도 보상
3. 발사 시 거리에 따른 보상
"""
# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.pos = 0

    def push(self, transition):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = abs(td.item()) + 1e-5
    def save(self, path='dqn_model_aiming/replay_buffer.pkl'):
            """리플레이 버퍼를 파일로 저장"""
            base, ext = os.path.splitext(path)
            counter = 0
            save_path = path

            # 파일 이름이 이미 존재하면 번호를 붙여 새로운 이름 생성
            while os.path.exists(save_path):
                counter += 1
                save_path = f"{base}_{counter}{ext}"

            # 버퍼와 우선순위 데이터를 저장
            data = {
                'buffer': self.buffer,
                'priorities': self.priorities,
                'pos': self.pos
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved replay buffer to {save_path}")

    def load(self, path='dqn_model_aiming/replay_buffer.pkl'):
        """리플레이 버퍼를 파일에서 로드"""
        if not os.path.exists(path):
            print(f"No replay buffer found at {path}")
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
            self.priorities = data['priorities']
            self.pos = data['pos']
        print(f"Loaded replay buffer from {path}")
        return True
# --- Dueling Q-Network ---
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))
    
# --- Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = PrioritizedReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.action_dim = action_dim

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q = self.q_net(state)
        return q.argmax().item()

    def update(self, beta=0.4):
        if len(self.memory.buffer) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).float().to(self.device)
        weights = weights.unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # Double DQN Target
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = q_values - expected_q.detach()
        loss = (td_errors ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())
        # Soft update
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    def save(self, path='dqn_model_aiming/dqn_tank.pth'):
        base, ext = os.path.splitext(path)
        counter = 0
        save_path = path

        while os.path.exists(save_path):
            counter += 1
            save_path = f"{base}_{counter}{ext}"
        torch.save(self.q_net.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    def load(self, path='dqn_model_aiming/dqn_tank.pth'):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

# === 환경 정의 ===
class TankEnv:
    def __init__(self):
        self.fire_cooldown = 6.0  # 발사 쿨타임
        self.action_list = [[-1, -1, 0],
                    [-1, 0, 0],
                    [-1, 1, 0],
                    [0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, -1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 0, 1]]
    def reset(self):
        self.fire_time = 0.0
    
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
            self.hit_x = 0.0
            self.hit_y = 0.0
            self.hit_z = 0.0
        # 발사 쿨타임
        self.current_time = float(log_data.get("time", 0.0))
        self.time_since_last_fire = self.current_time - self.fire_time if self.fire_time else 6.0
        self.cooldown_norm = np.clip(self.time_since_last_fire / self.fire_cooldown, 0.0, 1.0)
        return 1

    def get_state(self):
        relative_yaw, relative_vx, relative_vz = self.compute_relative_state_with_enemy_motion(self.body_x, self.tank_speed, self.enemy_body_x, self.enemy_speed)

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
            relative_yaw, relative_vx, relative_vz,
            distance/424.26, yaw_error/180.0,
            np.sin(np.radians(self.turret_x)), np.cos(np.radians(self.turret_x)),
            np.sin(np.radians(self.turret_y)), np.cos(np.radians(self.turret_y)),
            hit_dx, hit_dz
        ], dtype=np.float32)
        return state

    def normalize_angle_rad(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def compute_relative_state_with_enemy_motion(self, my_yaw_deg, my_speed, enemy_yaw_deg, enemy_speed):
        # 각도를 라디안으로 변환
        my_yaw = np.deg2rad(my_yaw_deg)
        enemy_yaw = np.deg2rad(enemy_yaw_deg)

        # 내 속도 벡터 (월드 좌표계)
        my_vx = my_speed * np.cos(my_yaw)
        my_vz = my_speed * np.sin(my_yaw)

        # 적 속도 벡터 (월드 좌표계)
        enemy_vx = enemy_speed * np.cos(enemy_yaw)
        enemy_vz = enemy_speed * np.sin(enemy_yaw)

        # 상대 속도 벡터 (월드 좌표계 기준)
        rel_vx = my_vx - enemy_vx
        rel_vz = my_vz - enemy_vz

        # 상대 속도를 적 좌표계로 변환
        cos_yaw = np.cos(-enemy_yaw)
        sin_yaw = np.sin(-enemy_yaw)
        relative_vx = rel_vx * cos_yaw - rel_vz * sin_yaw
        relative_vz = rel_vx * sin_yaw + rel_vz * cos_yaw

        # Yaw 차이 (내 yaw - 적 yaw), -pi ~ pi 범위로 정규화
        relative_yaw = self.normalize_angle_rad(my_yaw - enemy_yaw)

        # 속도 정규화 (최대 속도 기준, 예: 70)
        relative_vx /= 70.0
        relative_vz /= 70.0

        return relative_yaw, relative_vx, relative_vz
    
    def invert_pitch_from_impact_distance(self, d, gravity=9.81, initial_speed=60):
        factor = 733.74  # 상수 값 (앞에서 계산한 근사치) 733.74
        arg = (2 * d) / factor
        # arg 값이 [-1, 1] 범위에 있어야 함
        if abs(arg) > 1.0:
            # 발사 불가능 거리, pitch=0 (평사)
            return 0.0
        pitch_rad = 0.5 * math.asin(arg)
        return math.degrees(pitch_rad)
    def angle_diff(self, a, b):
        """두 각도 사이의 최소 차이 (0~180도)"""
        return (a - b + 180) % 360 - 180
    def scripted_action(self):
        
        """스크립트된 행동: 포탑을 적 방향으로 조준하고 발사"""
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        distance = math.sqrt(dx**2 + dz**2)

        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        target_pitch = self.invert_pitch_from_impact_distance(distance)

        yaw_error = self.angle_diff(self.turret_x, target_yaw)
        pitch_error = target_pitch - self.turret_y

        turret_dx = 1 if yaw_error > 2.0 else (-1 if yaw_error < -2.0 else 0)
        turret_dy = 1 if pitch_error > 2.0 else (-1 if pitch_error < -2.0 else 0)
        fire = 1 if abs(yaw_error) < 2.0 and abs(pitch_error) < 2.0 and self.cooldown_norm >= 0.99 else 0
        action = [turret_dx, turret_dy, fire]

        return self.action_list.index(action)
    
    def step(self):
        reward = 0.0

        # 1. 타임 패널티
        reward -= min(0.5 * np.log1p(self.current_time), 2.0)

        # 2. 조준 유도 보상
        dx = self.enemy_x - self.tank_x
        dz = self.enemy_z - self.tank_z
        target_yaw = (math.degrees(math.atan2(dx, dz))) % 360.0
        aim_error = abs(self.angle_diff(self.turret_x, target_yaw))
        aim_score = 1.0 - (aim_error / 180.0)
        reward += (aim_score - 0.5) * 5.0  # 최대 +2.5, 최소 -2.5

        # 3. 적중 시 보상
        if self.hit == 1.0:
            reward += 10.0
        elif self.hit_x and self.hit_z:
            hit_dx = abs(self.enemy_x - self.hit_x)
            hit_dy = abs(self.enemy_y - self.hit_y)
            hit_dz = abs(self.enemy_z - self.hit_z)
            hit_dist = (hit_dx ** 2 + hit_dy ** 2 + hit_dz ** 2) ** 0.5
            max_dist = 100.0
            reward += 10 * (1.0 - hit_dist / max_dist)
        distance = np.sqrt(dx**2 + dz**2)
        if distance > 130.0:
            reward -= 5.0
        # 5. 보상 정규화
        reward = np.clip(reward, -10.0, 10.0)

        done = (self.hit == 1.0) or (self.current_time > 60.0) or reward > 9.9 or distance > 200.0
        print(f"보상: {reward:.2f}, 시간: {self.current_time:.2f}")
        return reward, done
