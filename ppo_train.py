import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import math
import os

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
        return 1

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
        dy = self.enemy_y - self.tank_y
        distance = math.sqrt(dx**2 + dy**2)

        target_yaw = (math.degrees(math.atan2(dx, dy))) % 360.0
        target_pitch = self.invert_pitch_from_impact_distance(distance)

        yaw_error = self.angle_diff(self.turret_x, target_yaw)
        pitch_error = target_pitch - self.turret_y

        turret_dx = random.uniform(0.1, 1.0) if yaw_error > 2.0 else (random.uniform(-1.0, -0.1) if yaw_error < -2.0 else random.uniform(-0.05, 0.05))
        turret_dy = random.uniform(0.1, 1.0) if pitch_error > 2.0 else (random.uniform(-1.0, -0.1) if pitch_error < -2.0 else random.uniform(-0.05, 0.05))
        fire = 1.0 if abs(yaw_error) < 2.0 and abs(pitch_error) < 2.0 and self.cooldown_norm >= 0.99 else 0.0
        return np.array([turret_dx, turret_dy, fire], dtype=np.float32)
    
    def step(self, action):
        reward = 0.0
        turret_dx, turret_dy, fire = action

        # 1. 타임 패널티
        reward -= min(0.5 * np.log1p(self.current_time), 2.0)

        # 2. 조준 유도 보상
        dx = self.enemy_x - self.tank_x
        dy = self.enemy_y - self.tank_y
        target_yaw = (math.degrees(math.atan2(dx, dy))) % 360.0
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
            max_dist = 60
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
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
            x = self.shared(x)
            mean = self.actor_mean(x)
            min_std = 0.3
            std = torch.exp(self.actor_log_std)
            std = torch.clamp(std, min=min_std)
            # log_std = torch.clamp(self.actor_log_std, -1.5, 0.0)  # log_std 범위 조정
            # std = log_std.exp()  # std는 약 0.223 ~ 1.0
            if not torch.all(torch.isfinite(mean)):
                print(f"Warning: Invalid mean: {mean}")
                raise ValueError("Invalid mean")
            if not torch.all(torch.isfinite(std)):
                print(f"Warning: Invalid std: {std}")
                raise ValueError("Invalid std")
            dist = Normal(mean, std)
            return dist, None, self.critic(x)

# === PPO 에이전트 ===
class PPOAgent:
    def __init__(self, state_dim, action_dim, total_episodes_max):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = Policy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.entropy_coef = 0.01
        self.batch_size = 64
        self.epochs = 5
        self.total_episodes_max = total_episodes_max
        self.warmup = int(0.1 * self.total_episodes_max) # (10%)
        self.decay_end = int(0.7 * self.total_episodes_max) # (60%)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_schedule(step))
    
    def lr_schedule(self, episode):
        """warmup 10%, 학습률 선형 감소 60%, 낮은 학습률 30%"""
        if episode < self.warmup:
            return 1.0
        elif episode < self.decay_end:
            return 1.0 - (episode - self.warmup) / (self.decay_end - self.warmup) * 0.5
        else:
            return 0.5        
        
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if not torch.all(torch.isfinite(state)):
            print(f"Warning: Invalid state in select_action: {state}")
            raise ValueError("Invalid state in select_action")
        
        with torch.no_grad():
            dist, _, value = self.policy(state)
            u = dist.rsample()  # 샘플링
            action = torch.tanh(u)
            print(f"Distribution - mean: {dist.mean.cpu().numpy()}, std: {dist.stddev.cpu().numpy()}")
            log_prob = dist.log_prob(u).sum()
            epsilon = 1e-4  # epsilon 값을 더 크게 조정
            correction = torch.sum(torch.log(torch.clamp(1 - action.pow(2), min=epsilon, max=1.0)))
            log_prob = log_prob - correction
            
            # log_prob 제한
            log_prob = torch.clamp(log_prob, min=-50.0, max=0.0)  # -inf 방지
            if not torch.isfinite(log_prob):
                print(f"Warning: Invalid log_prob: {log_prob}, u: {u}, action: {action}")
                raise ValueError("Invalid log_prob")
        return action.cpu().numpy(), log_prob.item(), value.item()



    def compute_gae(self, rewards, values, states, dones):
        advantages = []
        gae = 0
        if dones[-1]:
            values = values + [0.0]
        else:
            with torch.no_grad():
                last_state = torch.tensor(states[-1], dtype=torch.float32).to(self.device)
                _, _, last_value = self.policy(last_state)
                values = values + [last_value.item()]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, memory, total_steps):
        states, actions, log_probs, rewards, values, dones = zip(*memory)
        # log_prob 유효성 검사
        log_probs = [min(max(lp, -50.0), 0.0) for lp in log_probs]  # -inf 방지
        advantages, returns = self.compute_gae(rewards, list(values), states, dones)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = list(zip(states, actions, old_log_probs, returns, advantages))
        self.entropy_coef = max(0.01, 0.1 * (0.9999 ** total_steps))
        for _ in range(self.epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                if len(batch) < 2:
                    continue
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = zip(*batch)
                
                b_states = torch.stack(b_states).to(self.device)
                b_actions = torch.stack(b_actions).to(self.device)
                b_old_log_probs = torch.stack(b_old_log_probs).to(self.device)
                b_returns = torch.tensor(b_returns, dtype=torch.float32).to(self.device)
                b_advantages = torch.tensor(b_advantages, dtype=torch.float32).to(self.device)
                
                dist, _, values = self.policy(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=1)
                new_log_probs = torch.clamp(new_log_probs, min=-50.0, max=0.0)  # new_log_prob 제한
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - b_old_log_probs).exp()
                surrogate1 = ratio * b_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = (b_returns - values.squeeze()).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        self.scheduler.step()

    def save(self, path='ppo_model/ppo_tank.pth'):
        base, ext = os.path.splitext(path)
        counter = 0
        save_path = path

        while os.path.exists(save_path):
            counter += 1
            save_path = f"{base}_{counter}{ext}"

        torch.save(self.policy.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    def load(self, path='ppo_model/ppo_tank(0528).pth'):
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