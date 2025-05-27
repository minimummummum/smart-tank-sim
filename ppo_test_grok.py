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
            self.tank_x/300, self.tank_y/300, self.tank_z/300,
            self.turret_x/360, self.turret_y/360,
            self.enemy_x/300, self.enemy_y/300, self.enemy_z/300,
            self.hit, self.cooldown_norm
        ], dtype=np.float32)
        return state
    
    def angle_diff(self, a, b):
        """두 각도 사이의 최소 차이 (0~180도)"""
        return abs((a - b + 180) % 360 - 180)
    
    def step(self, action):
        reward = 0.0
        turret_dx, turret_dy, fire = action

        # 1. 타임 패널티: 빠른 행동 유도
        reward -= min(0.002 * self.current_time, 0.05)

        # 2. 조준 유도 보상: 포탑 방향 vs 적 위치 각도
        dx = self.enemy_x - self.tank_x
        dy = self.enemy_y - self.tank_y
        target_yaw = (math.degrees(math.atan2(dx, dy))) % 360.0
        aim_error = self.angle_diff(self.turret_x, target_yaw)
        aim_score = 1.0 - (aim_error / 180.0)  # 0~1 사이
        if aim_score > 0.7:
            reward += 0.75 * aim_score  # 증가된 조준 보상
        else:
            reward -= 0.3 * (1 - aim_score) ** 2  # 감소된 패널티

        if aim_score > 0.95:  # 조준이 거의 정확할 때
            reward -= 0.05 * (abs(turret_dx) + abs(turret_dy))  # 가벼운 패널티
        
        # 4. 적중 시 보상
        if self.hit == 1.0:
            reward += 3.0  # 감소된 적중 보상
        elif self.hit_x is not None and self.hit_y is not None and self.hit_z is not None:
            # 근접 적중 보상
            hit_dx = abs(self.enemy_x - self.hit_x)
            hit_dy = abs(self.enemy_y - self.hit_y)
            hit_dz = abs(self.enemy_z - self.hit_z)
            hit_dist = (hit_dx ** 2 + hit_dy ** 2 + hit_dz ** 2) ** 0.5
            max_dist = 70 # 환경에 따라 조정
            reward += max(0.0, 2.0 * (1.0 - hit_dist / max_dist))  # 최대 2.0 보상

        # 5. 발사 시도 보상/패널티
        if fire > 0.0:
            if self.cooldown_norm >= 0.99:
                self.fire_time = self.current_time
                reward += 0.1
                if aim_score > 0.7:
                    reward += 1.5 * aim_score
                else:
                    reward -= 0.3 * (1 - aim_score) ** 2
            else:
                reward -= 0.05
        
        done = (self.hit == 1.0) or (self.current_time > 60.0)
        print(f"보상: {reward:2f} 시간: {self.current_time:2f} 액션: {action}")
        return reward, done



# === PPO 정책책 정의 ===
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh(),
            nn.Linear(action_dim, action_dim)  # 추가 선형 변환
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.2))  # std ≈ 0.223
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x)
        log_std = torch.clamp(self.actor_log_std, -2.5, -1.0)  # std: 0.082 ~ 0.368
        std = log_std.exp()
        dist = Normal(mean, std)
        action = torch.tanh(dist.sample())
        return dist, action, self.critic(x)

# === PPO 에이전트 ===
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.entropy_coef = 0.1  # 엔트로피 증가
        self.batch_size = 128
        self.epochs = 5
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist, action, value = self.policy(state)
            log_prob = dist.log_prob(action).sum()  # 스케일링 보정
            log_prob -= torch.sum(torch.log(1 - (action).pow(2) + 1e-6))
            #print(f"Mean: {dist.mean.cpu().numpy()}, Std: {dist.stddev.cpu().numpy()}, Action: {action.cpu().numpy()}")
        return action.cpu().numpy(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, memory):
        states, actions, log_probs, rewards, values, dones = zip(*memory)
        advantages, returns = self.compute_gae(rewards, list(values), dones)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
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
                dist, action, values = self.policy(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=1)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - b_old_log_probs).exp()
                surrogate1 = ratio * b_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = (b_returns - values.squeeze()).pow(2).mean()
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
