import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import math
# === 환경 정의 ===
class TankEnv:
    def __init__(self):
        self.fire_cooldown = 6.0  # 발사 쿨타임

    def reset(self):
        self.fire_time = 0.0
    
    def update_state(self, log_data, hit_data = None):
        # 탱크 위치
        self.tank_x = float(log_data.get("playerPos", {}).get("x"))
        self.tank_y = float(log_data.get("playerPos", {}).get("z")) # y축은 높이이므로 z축을 사용
        self.tank_z = float(log_data.get("playerPos", {}).get("y"))
        # 포탑 각도
        self.turret_x = float(log_data.get("playerTurretX"))
        self.turret_y = float(log_data.get("playerTurretY"))
        # 적 위치
        self.enemy_x = float(log_data.get("enemyPos", {}).get("x"))
        self.enemy_y = float(log_data.get("enemyPos", {}).get("z"))
        self.enemy_z = float(log_data.get("enemyPos", {}).get("y"))
        # 적중 여부
        if hit_data:
            self.hit = 1.0 if hit_data.get("hit", "terrain") == "enemy" else 0.0
            self.hit_x = float(hit_data.get('x', 0.0))
            self.hit_y = float(hit_data.get('z', 0.0))
            self.hit_z = float(hit_data.get('y', 0.0))
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
            self.tank_x, self.tank_y, self.tank_z,
            self.turret_x, self.turret_y,
            self.enemy_x, self.enemy_y, self.enemy_z,
            self.hit, self.cooldown_norm
        ], dtype=np.float32)
        return state
    
    def angle_diff(self, a, b):
        """두 각도 사이의 최소 차이 (0~180도)"""
        return abs((a - b + 180) % 360 - 180)
    
    def step(self, action):
        reward = 0.0
        turret_dx, turret_dy, fire = action

        # 포탑 각도 갱신
        self.turret_x = (self.turret_x + turret_dx) % 360.0
        self.turret_y = (self.turret_y + turret_dy) % 360.0

        # 기본 타임 패널티
        reward -= 0.01 * self.current_time

        # 조준 유도 보상: 포탑 방향 vs 적 위치 각도
        dx = self.enemy_x - self.tank_x
        dy = self.enemy_y - self.tank_y
        target_yaw = (math.degrees(math.atan2(dy, dx))) % 360.0

        aim_error = self.angle_diff(self.turret_x, target_yaw)
        aim_score = 1.0 - (aim_error / 180.0)  # 0~1 사이
        reward += 0.1 * aim_score  # 조준 유도 보상

        # 적중 시 큰 보상
        if self.hit == 1.0:
            reward += 2.0
        elif self.hit_x and self.hit_y and self.hit_z:
            hit_dx = abs(self.enemy_x - self.hit_x)
            hit_dy = abs(self.enemy_y - self.hit_y)
            hit_dz = abs(self.enemy_z - self.hit_z)
            hit_dist = (hit_dx + hit_dy + hit_dz) / 3.0
            reward += max(0.0, 1.0 - hit_dist / 50.0)

        # 발사 시도
        if fire > 0.0:
            if self.cooldown_norm == 1.0:
                self.fire_time = self.current_time
                reward += 0.1  # 정당한 발사 보상
            else:
                reward -= 0.1  # 쿨타임 중 발사 패널티
        done = (self.hit == 1.0) or (self.current_time > 30.0)
        state = self.get_state()
        print("액션", action)
        print("현재 상태", state, reward)
        return state, reward, done



# === PPO 정책책 정의 ===
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 공통 레이어
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # 행동 평균값 출력 레이어
        self.actor_mean = nn.Linear(128, action_dim)
        # 행동 표준편차 로그값 (학습 가능한 파라미터)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        # 상태 가치 함수 출력 레이어
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)  # 정규분포 행동 분포 생성
        value = self.critic(x)    # 상태 가치 출력
        return dist, value

# === PPO 에이전트 ===
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99          # 할인율
        self.lam = 0.95            # GAE 감쇠 계수
        self.eps_clip = 0.2        # 클리핑 값
        self.entropy_coef = 0.01   # 엔트로피 보너스 가중치
        self.batch_size = 32
        self.epochs = 10

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            action = torch.tanh(action)
            # 로그확률 계산
            log_prob = dist.log_prob(action).sum()
            # tanh에 의한 확률 왜곡 보정
            log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6))
        return action.cpu().numpy(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0.0]  # 끝 상태 값 0으로 추가
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, memory):
        # 저장된 데이터 분리
        states, actions, log_probs, rewards, values, dones = zip(*memory)
        advantages, returns = self.compute_gae(rewards, list(values), dones)
        
        # 텐서 변환
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        # 정규화된 advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = list(zip(states, actions, old_log_probs, returns, advantages))
        for _ in range(self.epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = zip(*batch)
                b_states = torch.stack(b_states).to(self.device)
                b_actions = torch.stack(b_actions).to(self.device)
                b_old_log_probs = torch.stack(b_old_log_probs).to(self.device)
                b_returns = torch.tensor(b_returns, dtype=torch.float32).to(self.device)
                b_advantages = torch.tensor(b_advantages, dtype=torch.float32).to(self.device)
                dist, values = self.policy(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=1)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - b_old_log_probs).exp()
                surrogate1 = ratio * b_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()       # 클리핑된 surrogate objective
                critic_loss = (b_returns - values.squeeze()).pow(2).mean()   # 가치 함수 손실
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, path='ppo_tank.pth'):
        torch.save(self.policy.state_dict(), path)

    def load(self, path='ppo_tank.pth'):
        self.policy.load_state_dict(torch.load(path))


# # === 학습 실행 ===
# env = TankEnv()
# agent = PPOAgent(state_dim=10, action_dim=3)
# num_episodes = 1000

# for episode in range(num_episodes):
#     state = env.reset()
#     memory = []
#     total_reward = 0

#     for _ in range(100):
#         action, log_prob, value = agent.select_action(state)
#         next_state, reward, done = env.step(action)
#         memory.append((state, action, log_prob, reward, value, done))
#         state = next_state
#         total_reward += reward
#         if done:
#             break

#     agent.update(memory)

#     if episode % 50 == 0:
#         print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

# agent.save()

# # === 테스트 실행 ===
# agent.load()
# for _ in range(5):
#     state = env.reset()
#     print("=== Test Episode ===")
#     for _ in range(30):
#         action, _, _ = agent.select_action(state)
#         next_state, reward, done = env.step(action)
#         print(f"State: {state.round(2)}, Action: {action.round(2)}, Reward: {reward:.2f}")
#         state = next_state
#         if done:
#             print("Hit!")
#             break
