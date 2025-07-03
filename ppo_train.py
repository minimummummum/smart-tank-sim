
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import heapq
import multiprocessing as mp
from functools import partial
import os
# MAPPING ID, self는 sliding map에 안 들어가고 global map에만 들어감.
# sliding map에서는 항상 중앙이기 때문
MAPPING = {
    'unknown': 0, 'empty': 1, 'obstacle': 2, 'enemy': 3,
    'supply': 4, 'base': 5, 'self': 6, 'enemy_sight':7
}

class TankSimulator:
    def __init__(self, map_size=300, sight_range=int(300/4), enemy_sight_range=int(300/5)):
        self.map_size = map_size
        self.sight_range = sight_range
        self.enemy_sight_range = enemy_sight_range
        self.global_map = np.zeros((map_size, map_size), dtype=np.int32)
        self.tank_pos = [0, 0]
        self.pos_margin = 3
        self.base_pos = [0, 0]
        self.supply_pos = [0, 0]
        self.enemy_list = []
        self.obstacles_list = []
        self.enemy_previous = set()
        self.has_supply = False
        self.previous_actions_deque = deque([[0, 0, 0, 0]] * 4, maxlen=4)
        self.reset()

    def reset(self):
        self.previous_actions_deque = deque([[0, 0, 0, 0]] * 4, maxlen=4)
        self.global_map.fill(MAPPING['unknown'])
        self.enemy_previous = set()
        # 내 탱크
        x = np.random.randint(0, self.map_size)
        if x in [0, 299]:
            y = np.random.randint(0, self.map_size)
        else:
            y = np.random.choice([0, 299])
        self.tank_pos = [x, y]
        self.base_pos = self.tank_pos.copy()
        self.global_map[self.tank_pos[0], self.tank_pos[1]] = MAPPING['self']
        
        # 장애물
        self.obstacles_list = []
        num_obstacles = int(self.map_size * self.map_size * 0.01)
        obstacle_set = set()
        while len(obstacle_set) < num_obstacles:
            pos = tuple(np.random.randint(self.pos_margin, self.map_size-self.pos_margin, 2))
            if pos != tuple(self.tank_pos):
                obstacle_set.add(pos)
        for p in obstacle_set:
            self.obstacles_list.append(list(p))
            self.global_map[p[0], p[1]] = MAPPING['obstacle']
        # 보급
        max_supply_attempts = 1000 # 적절한 최대 시도 횟수 설정
        attempts = 0
        while True:
            self.supply_pos = [np.random.randint(100, self.map_size-100), np.random.randint(100, self.map_size-100)]
            attempts += 1
            if (
                self.supply_pos != self.tank_pos and
                tuple(self.supply_pos) not in obstacle_set and
                self.is_reachable(self.tank_pos, self.supply_pos, obstacle_set, self.map_size)
            ):
                break
            if attempts >= max_supply_attempts:
                return self.reset()
        self.global_map[self.supply_pos[0], self.supply_pos[1]] = MAPPING['supply']
        self.has_supply = False

        # 적 탱크
        self.enemy_list = []
        num_enemy = int(self.map_size * self.map_size * 0.0005)
        enemy_set = set()
        while len(enemy_set) < num_enemy:
            group_size = np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
            base_x, base_y = np.random.randint(self.pos_margin, self.map_size - self.pos_margin, 2)
            offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            selected_offsets = np.random.choice(len(offsets), size=group_size, replace=False)
            for idx in selected_offsets:
                dx, dy = offsets[idx]
                new_x = base_x + dx
                new_y = base_y + dy
                if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                    pos = (new_x, new_y)
                    dist_base = np.hypot(new_x - self.tank_pos[0], new_y - self.tank_pos[1])
                    dist_supply = np.hypot(new_x - self.supply_pos[0], new_y - self.supply_pos[1])

                    if (pos not in enemy_set and
                        pos not in obstacle_set and
                        pos != tuple(self.supply_pos) and
                        pos != tuple(self.tank_pos) and
                        dist_base > self.sight_range and
                        dist_supply > self.sight_range):
                        enemy_set.add(pos)

        for p in enemy_set:
            self.enemy_list.append(list(p))
            self.global_map[p[0], p[1]] = MAPPING['enemy']

        
        self.visual_map = self.global_map.copy()
        return self._get_state()

    def visualize_map(self, title="Initial Map", init_mode=False):
        color_map = {
            MAPPING['unknown']: [0.0, 0.0, 0.0],
            MAPPING['empty']: [1.0, 1.0, 1.0],
            MAPPING['obstacle']: [0.5, 0.5, 0.5],
            MAPPING['supply']: [0.0, 0.0, 1.0],
            MAPPING['enemy']: [1.0, 0.0, 0.0],
            MAPPING['enemy_sight']: [0.8, 0, 0],
            MAPPING['base']: [1.0, 1.0, 0.0],
            MAPPING['self']: [0.0, 1.0, 0.0],
        }
        map_image = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                map_image[i, j] = color_map[self.visual_map[i, j]]
        if init_mode:
            self.fig = plt.figure(figsize=(8, 8))  # 각 환경별 새로운 창
            self.ax1 = self.fig.add_subplot()
            self.im1 = self.ax1.imshow(map_image)
            self.ax1.set_title(f"map")
            self.ax1.legend(handles=[
                plt.Rectangle((0,0),1,1, color='black', label='Unknown'),
                plt.Rectangle((0,0),1,1, color='white', label='Empty'),
                plt.Rectangle((0,0),1,1, color='gray', label='Obstacle'),
                plt.Rectangle((0,0),1,1, color='green', label='Tank'),
                plt.Rectangle((0,0),1,1, color='red', label='Enemy'),
                plt.Rectangle((0,0),1,1, color='blue', label='Supply'),
                plt.Rectangle((0,0),1,1, color='yellow', label='Base')
            ], loc='upper right', fontsize=8)
            self.ax1.axis('off')
            plt.ion()  # 인터랙티브 모드 활성화
        else:
            self.im1.set_array(map_image)
            self.ax1.set_title(f"{title}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _get_state(self):
        enemy = 0.0
        found_clustered_enemy = False

        for ex, ey in self.enemy_list:
            if np.hypot(ex - self.tank_pos[0], ey - self.tank_pos[1]) <= self.sight_range:
                close_enemies = sum(
                    1 for ex2, ey2 in self.enemy_list
                    if (ex2, ey2) != (ex, ey) and np.hypot(ex2 - ex, ey2 - ey) <= 3
                )
                if close_enemies == 0:
                    enemy = 1.0
                    break
                else:
                    if (ex, ey) not in self.enemy_previous:
                        found_clustered_enemy = True

        if enemy == 0.0 and found_clustered_enemy:
            enemy = -1.0

        state = np.array([
            enemy,
            float(self.has_supply),
        ], dtype=np.float32)
        return state # 6 개 + 16 개 # 여기 하는 중

    def _is_valid_pos(self, pos):
        return 0 <= pos[0] < self.map_size and 0 <= pos[1] < self.map_size and pos not in self.obstacles

    def is_reachable(self, start, goal, obstacles, map_size):
        visited = set()
        queue = deque([tuple(start)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            current = queue.popleft()
            if current == tuple(goal):
                return True
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < map_size and 0 <= ny < map_size:
                    next_pos = (nx, ny)
                    if next_pos not in visited and next_pos not in obstacles:
                        visited.add(next_pos)
                        queue.append(next_pos)
        return False
    def get_positions_within_radius(self, center, radius, map_size):
        cx, cy = center
        positions = []
        x_min = max(0, cx - radius)
        x_max = min(map_size - 1, cx + radius)
        y_min = max(0, cy - radius)
        y_max = min(map_size - 1, cy + radius)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                dist = ((cx - x)**2 + (cy - y)**2)**0.5
                if dist <= radius:
                    self.visual_map[x, y] = MAPPING['enemy_sight']
                    positions.append((x, y))
        return positions
    def astar(self, start, goal, obstacles, map_size):
        width, height = map_size, map_size
        obstacles_set = set(tuple(pos) for pos in obstacles)

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan 거리

        def get_neighbors(pos):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 대각선 포함
            neighbors = []
            for dx, dy in directions:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) not in obstacles_set:
                        neighbors.append((nx, ny))
            return neighbors

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(tuple(start), tuple(goal)), 0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == tuple(goal):
                # 경로 복원
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # 역순으로 반환

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # 모든 이동 비용 동일
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    priority = tentative_g_score + heuristic(neighbor, tuple(goal))
                    heapq.heappush(open_set, (priority, tentative_g_score, neighbor))
                    came_from[neighbor] = current

        return None  # 경로 없음
    def step(self, action, step_count):
        reward = 0.1
        done = False
        event = False
        enemy_flag = False
        obstacles = self.obstacles_list.copy()
        self.visual_map = self.global_map.copy()
        for p in self.enemy_previous:
            positions_to_block = self.get_positions_within_radius(p, self.enemy_sight_range, self.map_size)
            obstacles.extend(positions_to_block)
        if action == 0: # Go_to_Supply
            new_action = np.array([1, 0, 0, 0])
            self.previous_actions_deque.append(new_action)
            for ex, ey in self.enemy_list:
                dist = np.hypot(ex - self.tank_pos[0], ey - self.tank_pos[1])
                if dist <= self.enemy_sight_range:
                    done = True
                    break
            if self.has_supply:
                reward -= 0.5
            else:
                path_list = self.astar(self.tank_pos, self.supply_pos, obstacles, self.map_size)
                if path_list is None or len(path_list) <= 1:
                    reward -= 0.5
                    done = True
                    # self.enemy_previous = set()
                    # print("경로 실패, 적 위험반경 해제")
                else:
                    path = deque(path_list)
                    path.popleft()
                    while not event and len(path) > 0:
                        next_pos = path.popleft()
                        for ex, ey in self.enemy_list:
                            dist = np.hypot(ex - next_pos[0], ey - next_pos[1])
                            if dist <= self.sight_range:
                                event = True
                                break
                        #if self._is_valid_pos(next_pos):
                        prev_tank_pos = self.tank_pos.copy()
                        self.tank_pos = list(next_pos)
                        self.global_map[prev_tank_pos[0], prev_tank_pos[1]] = MAPPING['empty']
                        self.global_map[self.supply_pos[0], self.supply_pos[1]] = MAPPING['supply']
                        self.global_map[self.base_pos[0], self.base_pos[1]] = MAPPING['base']
                        self.global_map[self.tank_pos[0], self.tank_pos[1]] = MAPPING['self']

                        # 보급 위치에 도달하면
                        if self.tank_pos == self.supply_pos:
                            self.global_map[self.supply_pos[0], self.supply_pos[1]] = MAPPING['empty']
                            self.has_supply = True
                            reward += 0.7  # 보급 획득 보상
                            event = True
                            print("보급 획득")
                            break
                        # else:
                        #     event = True
                        #     break # 유효하지 않은 이동

        elif action == 1: # Go_to_Base
            new_action = np.array([0, 1, 0, 0])
            self.previous_actions_deque.append(new_action)
            for ex, ey in self.enemy_list:
                dist = np.hypot(ex - self.tank_pos[0], ey - self.tank_pos[1])
                if dist <= self.enemy_sight_range:
                    done = True
                    break
            if not self.has_supply:
                reward -= 0.5
            else:
                path_list = self.astar(self.tank_pos, self.base_pos, obstacles, self.map_size)
                if path_list is None or len(path_list) <= 1:
                    reward -= 0.5
                    done = True
                    # self.enemy_previous = set()
                    # print("경로 실패, 적 위험반경 해제")
                else:
                    path = deque(path_list)
                    path.popleft()
                    while not event and len(path) > 0:
                        next_pos = path.popleft()
                        for ex, ey in self.enemy_list:
                            dist = np.hypot(ex - next_pos[0], ey - next_pos[1])
                            if dist <= self.sight_range:
                                event = True
                                break
                        #if self._is_valid_pos(next_pos):
                        prev_tank_pos = self.tank_pos.copy()
                        self.tank_pos = list(next_pos)
                        self.global_map[prev_tank_pos[0], prev_tank_pos[1]] = MAPPING['empty']
                        self.global_map[self.tank_pos[0], self.tank_pos[1]] = MAPPING['self']

                        # base 위치에 도달하면
                        if self.tank_pos == self.base_pos:
                            reward += 1.0
                            event = True
                            done = True
                            print("base 도착")
                            break
                        # else:
                        #     event = True
                        #     break # 유효하지 않은 이동

        elif action == 2:  # Fire_Enemy
            new_action = np.array([0, 0, 1, 0])
            self.previous_actions_deque.append(new_action)
            fire = False
            visible_enemies = [
                (ex, ey) for (ex, ey) in self.enemy_list
                if np.hypot(ex - self.tank_pos[0], ey - self.tank_pos[1]) <= self.sight_range
            ]
            for ex, ey in visible_enemies:
                # 주변 3m 안에 다른 적이 있는지 확인
                close_enemies = sum(
                    1 for (ex2, ey2) in self.enemy_list
                    if (ex2, ey2) != (ex, ey) and np.hypot(ex2 - ex, ey2 - ey) <= 3
                )
                if close_enemies == 0:
                    self.global_map[ex, ey] = MAPPING['empty']
                    self.enemy_list.remove([ex, ey])
                    self.enemy_previous.discard((ex, ey))
                    reward += 0.5
                    fire = True
                    break  # 한 명만 공격하고 종료
            if not fire:
                reward -= 0.5  # 공격 실패 패널티

        elif action == 3: # Detour_to_Waypoint
            new_action = np.array([0, 0, 0, 1])
            self.previous_actions_deque.append(new_action)
            for ex, ey in self.enemy_list:
                dist = np.hypot(ex - self.tank_pos[0], ey - self.tank_pos[1])
                if dist <= self.sight_range:
                    enemy_tuple = (ex, ey)
                    if enemy_tuple not in self.enemy_previous:
                        self.enemy_previous.add(enemy_tuple)
                        break
            else:
                reward -= 0.5
        else:
            print("action 오류")
        return self._get_state(), reward, done

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_size) # 액션 확률 출력
        self.value_head = nn.Linear(128, 1) # 상태 가치 출력
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    
# PPO 클래스 (병렬 환경 지원하도록 수정)
class PPO:
    def __init__(self, state_dim=2, action_size=4, lr=1e-4, gamma=0.99, clip_eps=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.memory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "dones": [],
            "next_states": [],
        }

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, _ = self.policy(state)
            dist = torch.distributions.Categorical(logits=policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, states, actions, rewards, log_probs, dones, next_states):
        # next_state를 메모리에 추가
        self.memory["states"].append(states)
        self.memory["actions"].append(actions)
        self.memory["rewards"].append(rewards)
        self.memory["log_probs"].append(log_probs)
        self.memory["dones"].append(dones)
        self.memory["next_states"].append(next_states)

    def update(self, next_states):
        # 메모리에서 데이터 추출 (next_state 포함)
        states = torch.FloatTensor(np.array(self.memory["states"])).to(self.device)
        actions = torch.LongTensor(np.array(self.memory["actions"])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.memory["rewards"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.memory["log_probs"])).to(self.device)
        dones = torch.FloatTensor(np.array(self.memory["dones"])).to(self.device)
        next_states = torch.FloatTensor(np.array(self.memory["next_states"])).to(self.device)

        with torch.no_grad():
            _, values = self.policy(states)
            values = values.squeeze(-1)  # (N,)
            _, next_values = self.policy(next_states)
            next_values = next_values.squeeze(-1)  # (N,)

            returns = []
            gae = 0
            for step in reversed(range(len(rewards))):
                mask = 1 - dones[step]
                delta = rewards[step] + self.gamma * next_values[step] * mask - values[step]
                gae = delta + self.gamma * 0.95 * mask * gae
                returns.insert(0, gae + values[step])

        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(5):
            states = states.detach()
            policy, value = self.policy(states)
            dist = torch.distributions.Categorical(logits=policy)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantages = returns - value.squeeze()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            else:
                advantages = advantages - advantages.mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value.view(-1), returns)

            entropy_coef = 0.1
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "dones": [],
            "next_states": [],
        }

def load_checkpoint(ppo, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=ppo.device)
    ppo.policy.load_state_dict(checkpoint['model_state_dict'])
    ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint['episode']
    rewards = checkpoint['rewards']
    avg_rewards = checkpoint['avg_rewards']
    print(f"Loaded checkpoint from {checkpoint_path} at episode {start_episode}")
    return start_episode, rewards, avg_rewards

def worker_process(env_id, conn, map_size=300, sight_range=int(300/4), enemy_sight_range=int(300/5)):
    """
    각 프로세스에서 실행되는 환경 워커 함수
    """
    env = TankSimulator(map_size, sight_range, enemy_sight_range)
    env.reset()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action, step_count = data
            state, reward, done = env.step(action, step_count)
            conn.send((state, reward, done))
        elif cmd == "reset":
            state = env.reset()
            conn.send(state)
        elif cmd == "exit":
            break

# 학습 루프 (병렬 환경 지원)
def train(num_envs=12, num_episodes=300000, max_steps=5000):
    mp.set_start_method('spawn', force=True)  # 멀티프로세싱 시작 방식 설정
    ppo = PPO()
    processes = []
    parent_conns = []
    window_size = 50

    # 환경별 프로세스 생성
    for env_id in range(num_envs):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(env_id, child_conn))
        p.start()
        processes.append(p)
        parent_conns.append(parent_conn)
        
    # 체크포인트 로드
    checkpoint_path = "ppo_model/checkpoint_step_32000.pth"  # 마지막 체크포인트 파일 경로
    if os.path.exists(checkpoint_path):
        start_episode, rewards, avg_rewards = load_checkpoint(ppo, checkpoint_path)
    print(start_episode, rewards, avg_rewards)
    
    # 리워드 플롯 설정
    print("Initializing reward plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = cm.get_cmap('tab20', num_envs)
    colors = [colors(i / num_envs) for i in range(num_envs)]
    reward_lines = []
    avg_reward_lines = []
    for i in range(num_envs):
        line, = ax.plot([], [], label=f'Env {i+1} Reward', color=colors[i], linestyle='-')
        avg_line, = ax.plot([], [], label=f'Env {i+1} Avg Reward', color=colors[i], linestyle='--')
        reward_lines.append(line)
        avg_reward_lines.append(avg_line)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress - All Environments')
    ax.legend()
    ax.grid(True)
    plt.ion()
    plt.show(block=False)
    print("Reward plot initialized.")

    # 초기 상태 수집
    states = []
    for conn in parent_conns:
        conn.send(("reset", None))
    for conn in parent_conns:
        state = conn.recv()
        states.append(state)

    episode_rewards = [0.0] * num_envs
    step_counts = [0] * num_envs
    dones = [False] * num_envs
    episode_counts = [0] * num_envs
    rewards = [[] for _ in range(num_envs)]
    avg_rewards = [[] for _ in range(num_envs)]
    global_step = 0

    while max(episode_counts) < num_episodes:
        global_step += 1
        actions = []
        log_probs = []
        for i in range(num_envs):
            if dones[i]:
                continue
            action, log_prob = ppo.select_action(states[i])
            actions.append(action)
            log_probs.append(log_prob)

        next_states = []
        for i in range(num_envs):
            if dones[i]:
                next_states.append(states[i])
                continue
            start_time = time.time()
            parent_conns[i].send(("step", (actions[i], step_counts[i])))
            next_state, reward, done = parent_conns[i].recv()
            step_time = time.time() - start_time
            if i == 0:
                print(f"env{i+1} action: {actions[i]}, state: {next_state[0]}, reward: {reward}, done: {done}, step_time: {step_time:.4f}s")
            ppo.store_transition(states[i], actions[i], reward, log_probs[i], done, next_state)
            episode_rewards[i] += reward
            step_counts[i] += 1
            next_states.append(next_state)
            if done or step_counts[i] >= max_steps:
                dones[i] = True
                rewards[i].append((episode_rewards[i]*100.0)/step_counts[i] if step_counts[i] else 0.0)
                avg_reward = np.mean(rewards[i][-window_size:]) if len(rewards[i]) >= window_size else np.mean(rewards[i])
                avg_rewards[i].append(avg_reward)
                with open(f"training_log_env{i+1}.txt", "a") as f:
                    f.write(f"Episode {episode_counts[i] + 1}, max_steps: {step_counts[i]}, Reward: {episode_rewards[i]:.2f}, Avg Reward: {avg_reward:.2f}\n")
                print(f"Env {i+1}, Episode {episode_counts[i] + 1}, max_steps: {step_counts[i]}, Reward: {episode_rewards[i]:.2f}, Avg Reward: {avg_reward:.2f}")
                # 리워드 플롯 업데이트
                reward_lines[i].set_xdata(range(len(rewards[i])))
                reward_lines[i].set_ydata(rewards[i])
                avg_reward_lines[i].set_xdata(range(len(avg_rewards[i])))
                avg_reward_lines[i].set_ydata(avg_rewards[i])
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
                parent_conns[i].send(("reset", None))
                states[i] = parent_conns[i].recv()
                episode_rewards[i] = 0.0
                step_counts[i] = 0
                dones[i] = False
                episode_counts[i] += 1
            else:
                states[i] = next_state

        if len(ppo.memory["states"]) >= 64 * num_envs:
            start_time = time.time()
            ppo.update(next_states)
            update_time = time.time() - start_time
            if global_step % 50 == 0:
                print(f"Global Step {global_step}, Update Time: {update_time:.2f}s")

        if global_step % 1000 == 0:
            model_path = f"ppo_model/ppo_model_step_{global_step}.pth"
            torch.save(ppo.policy.state_dict(), model_path)
            print(f"Model saved at step {global_step} to {model_path}")
            checkpoint = {
                'episode': max(episode_counts),
                'model_state_dict': ppo.policy.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'rewards': rewards,
                'avg_rewards': avg_rewards
            }
            checkpoint_path = f"ppo_model/checkpoint_step_{global_step}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at step {global_step} to {checkpoint_path}")

    # 프로세스 종료
    for conn in parent_conns:
        conn.send(("exit", None))
    for p in processes:
        p.join()

    torch.save(ppo.policy.state_dict(), "ppo_model/ppo_model_final.pth")
    checkpoint = {
        'episode': max(episode_counts),
        'model_state_dict': ppo.policy.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'rewards': rewards,
        'avg_rewards': avg_rewards
    }
    torch.save(checkpoint, "ppo_model/ppo_checkpoint_final.pth")
    print("Final model saved to ppo_model_final.pth")

    plt.ioff()
    fig.savefig("training_progress_all_envs.png")
    print("Saved training progress plot to training_progress_all_envs.png")
    plt.close('all')


def test(checkpoint_path="ppo_model/checkpoint_step_44000.pth", num_episodes=100, max_steps=5000, map_size=300, sight_range=int(300/4), enemy_sight_range=int(300/5)):
    """
    학습된 PPO 모델을 사용하여 TankSimulator 환경에서 테스트를 수행하는 함수
    
    Parameters:
    - checkpoint_path: 테스트에 사용할 모델 체크포인트 경로
    - num_episodes: 테스트할 에피소드 수
    - max_steps: 한 에피소드의 최대 스텝 수
    - map_size: 환경 맵 크기
    - sight_range: 탱크의 시야 범위
    - enemy_sight_range: 적의 시야 범위
    """
    # PPO 모델 초기화
    ppo = PPO(state_dim=2, action_size=4)
    device = ppo.device

    # 체크포인트 로드
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ppo.policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

    # 환경 초기화
    env = TankSimulator(map_size=map_size, sight_range=sight_range, enemy_sight_range=enemy_sight_range)
    
    # 결과 저장용 리스트
    episode_rewards = []
    episode_steps = []
    success_count = 0  # 보급 획득 후 기지 도착 성공 횟수
    
    # 시각화 설정
    env.visualize_map(title="Test Initial Map", init_mode=True)
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        
        print(f"\n=== Test Episode {episode + 1} ===")
        
        while not done and step_count < max_steps:
            # 행동 선택
            action, _ = ppo.select_action(state)
            next_state, reward, done = env.step(action, step_count)
            
            # 보상 및 상태 업데이트
            total_reward += reward
            state = next_state
            step_count += 1
            
            # 시각화 업데이트
            env.visualize_map(title=f"Test Episode {episode + 1}, Step {step_count}, Reward: {total_reward:.2f}")
            plt.pause(0.01)  # 시각화 업데이트를 위해 잠시 대기
            
            # 로그 출력
            action_names = ["Go_to_Supply", "Go_to_Base", "Fire_Enemy", "Detour_to_Waypoint"]
            print(f"Step {step_count}, Action: {action_names[action]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Done: {done}")
            
            # 성공 여부 확인 (기지에 도착하면 성공)
            if done and env.tank_pos == env.base_pos and reward >= 1.0:
                success_count += 1
                print("Success: Reached base with supply!")
            
        # 에피소드 결과 저장
        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        
        # 에피소드 결과 출력
        avg_reward = total_reward / step_count if step_count > 0 else 0
        with open("test_log.txt", "a") as f:
            f.write(f"Episode {episode + 1}, Steps: {step_count}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}\n")
        print(f"Episode {episode + 1} finished, Steps: {step_count}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
    
    # 최종 결과 출력
    avg_episode_reward = np.mean(episode_rewards)
    avg_episode_steps = np.mean(episode_steps)
    success_rate = success_count / num_episodes * 100
    print(f"\n=== Test Summary ===")
    print(f"Average Episode Reward: {avg_episode_reward:.2f}")
    print(f"Average Episode Steps: {avg_episode_steps:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    with open("test_log.txt", "a") as f:
        f.write(f"\nTest Summary\n")
        f.write(f"Average Episode Reward: {avg_episode_reward:.2f}\n")
        f.write(f"Average Episode Steps: {avg_episode_steps:.2f}\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n")
    
    # 시각화 종료
    plt.ioff()
    plt.show()
    
    
if __name__ == "__main__":
    print(f"GPU available: {torch.cuda.is_available()}")
    test()