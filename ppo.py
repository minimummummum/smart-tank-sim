import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random

# === 환경 정의 ===
class TankEnv:
    def __init__(self):
        self.fire_cooldown = 6  # 발사 쿨타임
        self.time_since_last_fire = self.fire_cooldown
        self.reset()

    def reset(self):
        # 탱크 위치 초기화 (0~50 구간)
        self.tank_x = random.uniform(0, 50)
        self.tank_y = random.uniform(0, 50)
        # 포탑 위치 초기화 (0으로 시작)
        self.turret_x = 0.0
        self.turret_y = 0.0
        # 적 위치 초기화 (100~150 구간)
        self.enemy_x = random.uniform(100, 150)
        self.enemy_y = random.uniform(100, 150)
        # 적중 여부 초기화
        self.hit = 0.0
        # 발사 쿨타임 초기화
        self.time_since_last_fire = self.fire_cooldown
        return self._get_state()

    def _get_state(self):
        cooldown_norm = self.time_since_last_fire / self.fire_cooldown
        # 현재 상태 배열 반환
        return np.array([
            self.tank_x, self.tank_y,
            self.turret_x, self.turret_y,
            self.enemy_x, self.enemy_y,
            float(self.hit), cooldown_norm
        ], dtype=np.float32)

    def step(self, action):
        turret_dx, turret_dy, fire = action
        # 포탑 각도 업데이트 및 제한
        self.turret_x = np.clip(self.turret_x + turret_dx, -180, 180)
        self.turret_y = np.clip(self.turret_y + turret_dy, -30, 30)

        reward = -0.01  # 기본 패널티(시간 지날수록 페널티)
        self.time_since_last_fire += 1  # 쿨타임 증가

        # 발사 시도 & 쿨타임 충족 시
        if fire > 0.5 and self.time_since_last_fire >= self.fire_cooldown:
            self.time_since_last_fire = 0  # 쿨타임 초기화

            # 적과 포탑 위치 차이 계산
            dx = self.enemy_x - self.tank_x
            dy = self.enemy_y - self.tank_y

            # 적중 판정 (포탑과 적 위치가 가까우면 적중)
            if abs(dx - self.turret_x) < 5 and abs(dy - self.turret_y) < 5:
                self.hit = 1.0
                reward = 1.0  # 적중 보상
            else:
                reward = -0.1  # 빗나간 페널티

        done = self.hit == 1.0  # 적중하면 에피소드 종료
        return self._get_state(), reward, done

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
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.gamma = 0.99          # 할인율
        self.lam = 0.95            # GAE 감쇠 계수
        self.eps_clip = 0.2        # 클리핑 값
        self.entropy_coef = 0.01   # 엔트로피 보너스 가중치
        self.batch_size = 32
        self.epochs = 10

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()                   # 행동 샘플링
            log_prob = dist.log_prob(action).sum()  # 로그확률 계산
        return action.numpy(), log_prob.item(), value.item()

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
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        # 정규화된 advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = list(zip(states, actions, old_log_probs, returns, advantages))
        for _ in range(self.epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), self.batch_size):
                batch = dataset[i:i + self.batch_size]
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages = zip(*batch)

                b_states = torch.stack(b_states)
                b_actions = torch.stack(b_actions)
                b_old_log_probs = torch.stack(b_old_log_probs)
                b_returns = torch.tensor(b_returns, dtype=torch.float32)
                b_advantages = torch.tensor(b_advantages, dtype=torch.float32)

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


# === 학습 실행 ===
env = TankEnv()
agent = PPOAgent(state_dim=8, action_dim=3)
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    memory = []
    total_reward = 0

    for _ in range(100):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done = env.step(action)
        memory.append((state, action, log_prob, reward, value, done))
        state = next_state
        total_reward += reward
        if done:
            break

    agent.update(memory)

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

agent.save()

# === 테스트 실행 ===
agent.load()
for _ in range(5):
    state = env.reset()
    print("=== Test Episode ===")
    for _ in range(30):
        action, _, _ = agent.select_action(state)
        next_state, reward, done = env.step(action)
        print(f"State: {state.round(2)}, Action: {action.round(2)}, Reward: {reward:.2f}")
        state = next_state
        if done:
            print("Hit!")
            break
