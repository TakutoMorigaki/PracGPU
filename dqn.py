import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import game2048_env
from collections import deque


# Qネットワーク
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # 隠れ層(y=Wx+bのやつ)
        self.fc1 = nn.Linear(state_dim, 128)    # 線形結合1
        self.fc2 = nn.Linear(128, 128)          # 線形結合2
        self.fc3 = nn.Linear(128, action_dim)   # 線形結合3

    # 順伝搬
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# バッファー(この中からデータを学習)
class RepalyBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.stack(next_states),
            torch.tensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ダブルDQN(QネットワークとTargetネットワーク)
class DoubleDQN:
    def __init__(self, state_dim, action_dim):
        self.device = "cuda"

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=1e-4
        )

        self.buffer = RepalyBuffer()

        self.gamma = 0.99
        self.batch_size = 96

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(4)

        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.q_net(state)

        return q.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 現在のQ値
        q_values = self.q_net(states)
        q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 行動選択(main)
        next_actions = self.q_net(next_states).argmax(1)

        # 評価(target)
        next_q = self.target_net(next_states)\
            .gather(1, next_actions.unsqueeze(1))\
            .squeeze(1)

        target = rewards + self.gamma * next_q * (1 - dones)
        target = target.to(self.device)

        loss = F.mse_loss(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )


env = game2048_env.Env2048()

agent = DoubleDQN(16, 4)

epsilon = 1.0

for episode in range(500):
    state = env.reset()

    done = False

    while not done:
        state_t = torch.FloatTensor(state)
        action = agent.select_action(state_t, epsilon)

        next_state, reward, done = env.step(action)

        agent.buffer.push(
            state_t,
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float),
            torch.FloatTensor(next_state),
            torch.tensor(done, dtype=torch.float)
        )

        agent.train()
        state = next_state

    if episode % 10 == 0:
        agent.update_target()

    print(episode)

    epsilon *= 0.999

print("done")
