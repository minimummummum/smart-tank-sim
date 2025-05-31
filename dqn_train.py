import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math
import os
"""
프로토타입
state = np.array([
            self.tank_x/300, self.tank_y/300, self.tank_z/300,
            np.sin(np.radians(self.turret_x)), np.cos(np.radians(self.turret_x)),
            np.sin(np.radians(self.turret_y)), np.cos(np.radians(self.turret_y)),
            self.enemy_x/300, self.enemy_y/300, self.enemy_z/300,
            self.hit, self.cooldown_norm
        ]
action 
[-1, -1, 0]
[-1, 0, 0]
[-1, 1, 0]
[0, -1, 0]
[0, 0, 0]
[0, 1, 0]
[1, -1, 0]
[1, 0, 0]
[1, 1, 0]
[-1, -1, 1]
[-1, 0, 1]
[-1, 1, 1]
[0, -1, 1]
[0, 0, 1]
[0, 1, 1]
[1, -1, 1]
[1, 0, 1]
[1, 1, 1]

ㅇ State:
[Target 각도, Distance, 길이] x target 수,    -> Target과의 상대적 각도, Distance, 길이
포탑 yaw(sin, cos), pitch(sin, cos),         -> [0, 1] 범위로 정규화
hit_dx[-1, 0, 1], hit_dz[-1, 0, 1],         -> 왼쪽, 중앙, 오른쪽, 뒤쪽, 중앙, 앞쪽
command                                     -> 명령어

ㅇ State: -> 격파 state
Target 상대 각도, 상대 Distance,              -> Target과의 상대적 각도, Distance
포탑 yaw(sin, cos), pitch(sin, cos),         -> [0, 1] 범위로 정규화
hit_dx[-1, 0, 1], hit_dz[-1, 0, 1],         -> 왼쪽, 중앙, 오른쪽, 뒤쪽, 중앙, 앞쪽

ㅇ Action:
base_actions = [
    [0, 0],     # 정지

    [1, 0],     # 전진 1단
    [2, 0],     # 전진 2단
    [-1, 0],    # 후진

    [0, 1],     # 우회전
    [0, -1],    # 좌회전

    [1, 1],     # 전진 1단 + 우회전
    [1, -1],    # 전진 1단 + 좌회전
    [2, 1],     # 전진 2단 + 우회전
    [2, -1],    # 전진 2단 + 좌회전
]
turret_x = [-1, 0, 1]
turret_y = [-1, 0, 1]
fire = [0, 1]


# 전체 action 조합 생성
ALL_ACTIONS = list(product(base_actions, turret_x, turret_y, fire))
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
    def save(self, path='dqn_model/ppo_tank.pth'):
        base, ext = os.path.splitext(path)
        counter = 0
        save_path = path

        while os.path.exists(save_path):
            counter += 1
            save_path = f"{base}_{counter}{ext}"
        torch.save(self.q_net.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    def load(self, path='dqn_model/dqn_tank.pth'):
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
                    [-1, -1, 1],
                    [-1, 0, 1],
                    [-1, 1, 1],
                    [0, -1, 1],
                    [0, 0, 1],
                    [0, 1, 1],
                    [1, -1, 1],
                    [1, 0, 1],
                    [1, 1, 1]]
    def reset(self):
        self.fire_time = 0.0
    
    def update_state(self, log_data, hit_data = None):
        # 탱크 위치
        self.tank_x = float(log_data.get("playerPos", {}).get("x"))
        self.tank_y = float(log_data.get("playerPos", {}).get("y")) # y축은 높이이므로 z축을 사용
        self.tank_z = float(log_data.get("playerPos", {}).get("z"))
        # 포탑 각도
        self.turret_x = float(log_data.get("playerTurretX"))
        self.turret_y = float(log_data.get("playerTurretY"))
        # 적 위치
        self.enemy_x = float(log_data.get("enemyPos", {}).get("x"))
        self.enemy_y = float(log_data.get("enemyPos", {}).get("y"))
        self.enemy_z = float(log_data.get("enemyPos", {}).get("z"))
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

    def get_state(self):
        state = np.array([
            self.tank_x/300, self.tank_y/300, self.tank_z/300,
            np.sin(np.radians(self.turret_x)), np.cos(np.radians(self.turret_x)),
            np.sin(np.radians(self.turret_y)), np.cos(np.radians(self.turret_y)),
            self.enemy_x/300, self.enemy_y/300, self.enemy_z/300,
            self.hit, self.cooldown_norm
        ], dtype=np.float32)
        return state
    
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

        # if yaw_error > 2.0:
        #     turret_dx = random.uniform(0.1, 1.0)
        # elif 1.0 < yaw_error <= 2.0:
        #     turret_dx = random.uniform(0.05, 0.1)
        # elif -2.0 < yaw_error < -1.0:
        #     turret_dx = random.uniform(-0.1, -0.05)
        # elif yaw_error <= -2.0:
        #     turret_dx = random.uniform(-1.0, -0.1)
        # else:
        #     turret_dx = 0.0

        # if pitch_error > 2.0:
        #     turret_dy = random.uniform(0.1, 1.0)
        # elif 1.0 < pitch_error <= 2.0:
        #     turret_dy = random.uniform(0.05, 0.1)
        # elif -2.0 < pitch_error < -1.0:
        #     turret_dy = random.uniform(-0.1, -0.05)
        # elif pitch_error <= -2.0:
        #     turret_dy = random.uniform(-1.0, -0.1)
        # else:
        #     turret_dy = 0.0
        turret_dx = 1 if yaw_error > 2.0 else (-1 if yaw_error < -2.0 else 0)
        turret_dy = 1 if pitch_error > 2.0 else (-1 if pitch_error < -2.0 else 0)
        fire = 1 if abs(yaw_error) < 2.0 and abs(pitch_error) < 2.0 and self.cooldown_norm >= 0.99 else 0
        action = [turret_dx, turret_dy, fire]

        return self.action_list.index(action)
    
    def step(self, action):
        reward = 0.0
        turret_dx, turret_dy, fire = self.action_list[action]

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
        elif self.hit_x is not None and self.hit_y is not None and self.hit_z is not None:
            hit_dx = abs(self.enemy_x - self.hit_x)
            hit_dy = abs(self.enemy_y - self.hit_y)
            hit_dz = abs(self.enemy_z - self.hit_z)
            hit_dist = (hit_dx ** 2 + hit_dy ** 2 + hit_dz ** 2) ** 0.5
            max_dist = 60 # 현재 임의로 60
            reward += 10.0 * (1.0 - hit_dist / max_dist)

        # 4. 발사 시도 보상/패널티
        if fire > 0.0:
            if self.cooldown_norm >= 0.99:
                self.fire_time = self.current_time
                reward += 1.0 * aim_score  # 조준 정확도에 비례
            else:
                reward -= 2.0

        # 5. 보상 정규화
        reward = np.clip(reward, -10.0, 10.0)

        done = (self.hit == 1.0) or (self.current_time > 60.0)
        print(f"보상: {reward:.2f}, 시간: {self.current_time:.2f}, 액션: {action}")
        return reward, done
    

# env = gym.make("CartPole-v1")
# agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# num_episodes = 500
# epsilon_start = 1.0
# epsilon_final = 0.01
# epsilon_decay = 500

# for episode in range(num_episodes):
#     state = env.reset()
#     done = False
#     total_reward = 0
#     steps = 0

#     epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)

#     while not done:
#         action = agent.select_action(state, epsilon)
#         next_state, reward, done, _ = env.step(action)
#         agent.memory.push((state, action, reward, next_state, done))
#         state = next_state
#         total_reward += reward
#         steps += 1

#         agent.update()

#     print(f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f}")