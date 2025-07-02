import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

class PPO:
    def __init__(self, state_dim=2, action_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_size).to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.policy(state)
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()
        return action.item()

class Agent:
    def __init__(self, state_dim=2, action_size=4, checkpoint_path="ppo_model/model2.pth"):
        self.ppo = PPO(state_dim, action_size)
        self.device = self.ppo.device
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.ppo.policy.load_state_dict(checkpoint['model_state_dict'])
            print(f"모델 로드 성공: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"모델 로드 실패: {checkpoint_path}")

    def select_action(self, state):
        return self.ppo.select_action(state)